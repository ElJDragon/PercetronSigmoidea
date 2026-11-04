"""Seis ejecuciones prefijadas de Inventario; sin SQL, reconstrucción, test o PCA."""
import argparse
from collections import Counter
from contextlib import ExitStack
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.db.auditoria_dw import BASE,guardar,sha
from src.db.conexion import Configuracion
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual,binary_cross_entropy_logits,sigmoid
from src.models.logistic_inventario import LogisticRegressionInventarioPonderada,weighted_loss_gradients,balanced_weights
from src.evaluation.metricas import metricas_binarias,clave_seleccion
from src.experiments.entrenar_transacciones import average_precision,convergence
from src.datamarts.inventario.caracteristicas import FEATURES
from src.datamarts.inventario.preprocesamiento import PreprocesadorInventario
from src.datamarts.inventario.construir import validate_alignment

DATA=BASE/'datos/Procesados/inventario'
PREVIOUS=BASE/'resultados/extension_datamarts/inventario'
OUT=PREVIOUS/'entrenamiento'
NEW={'src/experiments/entrenar_inventario.py','src/experiments/reportes_inventario.py','src/models/logistic_inventario.py','tests/test_entrenamiento_inventario.py'}
PLAN=dict(version=1,process='inventory_high_receipt_frequency',features=FEATURES,
    primary_metric='unweighted validation balanced_accuracy, all 2628 observations',
    selection='BA descending, F1 descending, epoch ascending, experiment index ascending',
    initialization='fresh zero weights and bias for each run; no warm start or previous Data Mart parameters',
    thresholds=dict(perceptron='z>=0',logistic=.5,candidates=[.5],search=False),
    weighting_amendment='This user request explicitly authorizes an isolated weighted logistic variant, superseding the earlier no-class-weight training default only for that variant. Dataset and validation remain unweighted.',
    class_weights='n_train/(2*n_train_class), only y_train, no resampling',
    experiments=[
        dict(id='p_seed42',algorithm='perceptron',variant='standard',params=dict(learning_rate=.01,max_epochs=100,seed=42,shuffle=True)),
        dict(id='p_seed7',algorithm='perceptron',variant='standard',params=dict(learning_rate=.01,max_epochs=100,seed=7,shuffle=True)),
        dict(id='l_eta005',algorithm='logistic',variant='standard',params=dict(learning_rate=.05,max_epochs=400,seed=42,initialization='zeros',tolerance=1e-6,threshold=.5)),
        dict(id='l_eta010',algorithm='logistic',variant='standard',params=dict(learning_rate=.1,max_epochs=400,seed=42,initialization='zeros',tolerance=1e-6,threshold=.5)),
        dict(id='lw_eta005',algorithm='logistic',variant='weighted',params=dict(learning_rate=.05,max_epochs=400,seed=42,initialization='zeros',tolerance=1e-6,threshold=.5)),
        dict(id='lw_eta010',algorithm='logistic',variant='weighted',params=dict(learning_rate=.1,max_epochs=400,seed=42,initialization='zeros',tolerance=1e-6,threshold=.5))],
    budget=dict(runs=6,online_epochs_max=200,batch_epochs_max=1600,adaptive_expansion=False),
    perceptron_rate_rationale='One eta and two predefined permutations; from zero eta rescaling is redundant',
    baselines=dict(majority='class most frequent in train, tie 0',historical='1[N(t-1)>median(N(t-1),N(t-2),N(t-3))]',history_months=3),
    support_policy='principal and common identical when all three-month baselines available; never drop principal rows',
    AP='descriptive only, tie-aware ranking, not precision at cutoff and not checkpoint criterion',
    model_policy='one best perceptron; best standard and best weighted logistic documented; select final logistic across both variants by same validation rule',
    test_reserved=True,PCA=False,SQL=False)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def protected():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe'] for p in (BASE/root).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents and str(p.relative_to(BASE)).replace('\\','/') not in NEW}

def verify_prior():
    before=read(OUT/'integridad_antes.json')
    if before!={n:sha(BASE/n) for n in before}:raise RuntimeError('Regresión: archivo protegido modificado')
    return before

def preflight():
    manifest=read(PREVIOUS/'dataset/manifiesto.json')
    for rel,h in manifest['artifacts_sha256'].items():
        if sha(BASE/rel)!=h:raise RuntimeError('Artefacto de dataset alterado: '+rel)
    if len(FEATURES)!=17 or read(PREVIOUS/'catalogo_caracteristicas.json')!=read(PREVIOUS/'dataset/caracteristicas.json')['approved_catalog']:
        raise RuntimeError('Catálogo incompatible')
    state=read(PREVIOUS/'dataset/preprocesador.json')
    if state['feature_names']!=FEATURES or state['fit_partition']!='train' or state['train_rows']!=4599:raise RuntimeError('Preprocesamiento no aprobado')
    meta=pd.read_csv(DATA/'metadatos.csv.gz',keep_default_na=False);raw=pd.read_csv(DATA/'caracteristicas_sin_escalar.csv.gz',float_precision='round_trip')
    labels=pd.read_csv(DATA/'etiquetas.csv.gz');validate_alignment(meta,raw,labels)
    prep=PreprocesadorInventario.from_dict(state);dev={}
    for split,n,pos in [('train',4599,86),('val',2628,35)]:
        mask=meta.partition.eq(split).to_numpy();x=np.load(DATA/f'X_{split}.npy',allow_pickle=False);y=np.load(DATA/f'y_{split}.npy',allow_pickle=False)
        if x.shape!=(n,17) or y.shape!=(n,) or x.dtype!=np.float64 or y.dtype!=np.int8 or not np.isfinite(x).all() or int(y.sum())!=pos:raise RuntimeError('Matrices inválidas')
        if not np.array_equal(prep.transform(raw.loc[mask,FEATURES]),x) or not np.array_equal(labels.loc[mask,'y'],y):raise RuntimeError('X/y fuera de alineación')
        md=meta.loc[mask].reset_index(drop=True)
        if not (md.month<'2016-01').all() or not (md.latest_history_edited_at<md.prediction_at).all():raise RuntimeError('Fuga temporal')
        if pd.read_csv(DATA/f'claves_{split}.csv').observation_id.tolist()!=md.observation_id.tolist():raise RuntimeError('Claves incompatibles')
        historical=(raw.loc[mask,'recepciones_lag_1'].to_numpy()>raw.loc[mask,['recepciones_lag_1','recepciones_lag_2','recepciones_lag_3']].median(axis=1).to_numpy()).astype(np.int8)
        if not np.array_equal(historical,labels.loc[mask,'historical_baseline']):raise RuntimeError('Baseline alterado')
        dev[split]=dict(X=x,y=y,meta=md,historical=historical)
    if list(DATA.glob('*test*')):raise RuntimeError('Test ya construido: detener y revisar')
    diagnosis=read(PREVIOUS/'dataset/discrepancias_suite_previas.json');old_suite=read(PREVIOUS/'dataset/suite_existente.json')
    if len(old_suite['failures'])!=5 or len(old_suite['errors'])!=4 or not diagnosis['all_differences_predate_this_task']:raise RuntimeError('Fallos previos sin clasificación')
    for group in diagnosis['groups']:
        for d in group['differences']:
            if not d['path'].replace('\\','/').startswith('informe/') or sha(BASE/d['path'])!=d['current'] or d['current']!=d['before_inventory_dataset']:
                raise RuntimeError('Diferencia histórica no demostrada')
    return dev,state

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',protected())
    verify_prior();dev,state=preflight();path=OUT/'configuracion_experimental.json'
    if path.exists() and read(path)!=PLAN:raise RuntimeError('Plan ya registrado; no ampliar tras métricas')
    if (OUT/'registro_previo.json').exists():
        if read(OUT/'registro_previo.json')['plan_sha256']!=sha(path):raise RuntimeError('Registro previo inconsistente')
        print(json.dumps(dict(plan_already_fixed=True,plan_sha256=sha(path))));return
    guardar(path,PLAN)
    guardar(OUT/'registro_previo.json',dict(recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),plan_sha256=sha(path),
        class_weights_train=balanced_weights(dev['train']['y']),runs=6,results_existed_at_registration=(OUT/'experimentos.json').exists()))
    guardar(OUT/'preflight.json',dict(status='successful',dataset_manifest_sha256=sha(PREVIOUS/'dataset/manifiesto.json'),
        preprocessor_sha256=sha(PREVIOUS/'dataset/preprocesador.json'),shapes={s:list(v['X'].shape) for s,v in dev.items()},
        historical_failures_confirmed=9,historical_differences_report_only=True,SQL_queries=0,test_loaded=False,
        matrices_rebuilt=False,features=FEATURES,weighted_variant_explicitly_authorized=True))
    print(json.dumps(dict(plan_fixed=True,runs=6,plan_sha256=sha(path))))

def metrics(y,p):
    p=np.asarray(p)
    return dict(metricas_binarias(y,p),positive_prediction_proportion=float(np.mean(p)),only_majority_predicted=bool(np.all(p==0)))

class Trace:
    """Observa predict al final de época; no modifica actualizaciones o selección."""
    def fit(self,X,y,*,X_val,y_val,feature_names):
        self.trace_=[];self._trace_labels=[y,y_val];self._trace_calls=0;self._trace_active=True
        try:super().fit(X,y,X_val=X_val,y_val=y_val,feature_names=feature_names)
        finally:self._trace_active=False
        if len(self.trace_)!=len(self.history_):raise RuntimeError('Historial incompleto')
        for h,t in zip(self.history_,self.trace_):
            h.update({f'{split}_{key}':m[key] for split,m in t['metrics'].items() for key in ['precision','recall','f1','specificity','balanced_accuracy','positive_prediction_proportion']})
            if 'unweighted_train_BCE' in t:h['unweighted_train_BCE']=t['unweighted_train_BCE']
        return self

    def predict(self,X,*args,**kwargs):
        pred=super().predict(X,*args,**kwargs)
        if getattr(self,'_trace_active',False):
            side=self._trace_calls%2;m=metrics(self._trace_labels[side],pred)
            if side==0:
                self._trace_current=dict(metrics=dict(train=m))
                if isinstance(self,LogisticRegressionManual):self._trace_current['unweighted_train_BCE']=binary_cross_entropy_logits(self._trace_labels[0],self.decision_function(X))
            else:
                self._trace_current['metrics']['val']=m
                self._trace_current.update(weights=self.weights_.tolist(),bias=float(self.bias_));self.trace_.append(self._trace_current)
            self._trace_calls+=1
        return pred

class TracedPerceptron(Trace,PerceptronSimple):pass
class TracedLogistic(Trace,LogisticRegressionManual):pass
class TracedWeighted(Trace,LogisticRegressionInventarioPonderada):pass

def load_model(path):
    alg=read(path)['algorithm']
    cls={'manual_binary_perceptron':PerceptronSimple,'manual_numpy_binary_logistic_regression':LogisticRegressionManual,
         'inventory_manual_weighted_binary_logistic':LogisticRegressionInventarioPonderada}[alg]
    model=cls.load(path)
    if model.feature_names_!=FEATURES or len(model.weights_)!=17:raise RuntimeError('Modelo fuera del catálogo Inventario')
    return model

def fit_experiments(dev,plan,out):
    runs=[];models={};chosen={}
    for i,c in enumerate(plan['experiments']):
        cls=TracedPerceptron if c['algorithm']=='perceptron' else TracedWeighted if c['variant']=='weighted' else TracedLogistic
        model=cls(**c['params']);model.fit(dev['train']['X'],dev['train']['y'],X_val=dev['val']['X'],y_val=dev['val']['y'],feature_names=FEATURES)
        path=out/'runs'/c['id'];path.mkdir(parents=True,exist_ok=True);model.save(path/'modelo.json')
        measured={s:metrics(v['y'],model.predict(v['X'],feature_names=FEATURES)) for s,v in dev.items()}
        last={}
        for s,v in dev.items():
            z=v['X']@model.last_weights_+model.last_bias_
            last[s]=metrics(v['y'],(z>=0).astype(np.int8) if c['algorithm']=='perceptron' else (sigmoid(z)>=.5).astype(np.int8))
        row=dict(id=c['id'],index=i,algorithm=c['algorithm'],variant=c['variant'],params=json.loads(json.dumps(model.get_params())),
            selected_epoch=model.best_epoch_,metrics=measured,last_epoch_metrics=last,epochs_run=len(model.history_),history=model.history_,
            last_weights=model.last_weights_.tolist(),last_bias=float(model.last_bias_),
            termination='gradient_tolerance' if c['algorithm']=='logistic' and model.converged_ else 'max_epochs')
        if c['algorithm']=='logistic':
            row.update(initial_train_loss=model.initial_train_loss_,initial_val_loss=model.initial_val_loss_,
                validation_average_precision=average_precision(dev['val']['y'],model.predict_proba(dev['val']['X'])),
                loss_definition='weighted train BCE / unweighted validation BCE' if c['variant']=='weighted' else 'unweighted train and validation BCE')
        t=model.trace_[model.best_epoch_-1]
        if not np.array_equal(t['weights'],model.weights_) or t['bias']!=model.bias_ or t['metrics']!=measured:raise RuntimeError('Checkpoint no reproducido por traza')
        guardar(path/'historial.json',model.history_);guardar(path/'traza_parametros.json',model.trace_)
        pd.DataFrame(model.history_).to_csv(path/'historial.csv',index=False,lineterminator='\n',float_format='%.17g')
        guardar(path/'resumen.json',{k:v for k,v in row.items() if k!='history'})
        runs.append(row);models[c['id']]=model
        groups=['perceptron'] if c['algorithm']=='perceptron' else ['logistic','logistic_'+c['variant']]
        for group in groups:
            old=chosen.get(group)
            if old is None or clave_seleccion(measured['val'],model.best_epoch_,i)>clave_seleccion(old['metrics']['val'],old['selected_epoch'],old['index']):chosen[group]=row
    return runs,chosen,{g:models[r['id']] for g,r in chosen.items()}

def evaluate(dev,models):
    full={};monthly=[];frames=[]
    majority=int(dev['train']['y'].mean()>.5)
    for split,v in dev.items():
        preds={name:models[name].predict(v['X'],feature_names=FEATURES) for name in ['perceptron','logistic_standard','logistic_weighted']}
        preds.update(majority=np.full(len(v['y']),majority,dtype=np.int8),historical=v['historical'])
        full[split]={name:metrics(v['y'],p) for name,p in preds.items()}
        frame=v['meta'][['observation_id','item_id','month']].copy();frame['partition']=split;frame['y']=v['y']
        for name,p in preds.items():frame[name]=p
        frame['perceptron_margin']=models['perceptron'].decision_function(v['X'])
        for name in ['logistic_standard','logistic_weighted']:frame[name+'_score']=models[name].predict_proba(v['X'])
        frames.append(frame)
        for month in sorted(set(frame.month)):
            mask=frame.month.eq(month).to_numpy()
            for name,p in preds.items():monthly.append(dict(partition=split,month=month,method=name,metrics=metrics(v['y'][mask],p[mask])))
    return dict(primary=full,common=full,common_equals_primary=True,missing_historical=0,majority_train_class=majority,
        historical_formula=PLAN['baselines']['historical']),monthly,pd.concat(frames,ignore_index=True)

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return dict(low=max(0,float(c-h)),high=min(1,float(c+h)),nominal_level=.95,
        caveat='Descriptive binomial Wilson interval; independence not established for item-months; optimistic under clustering, not inferential evidence')

def concentration(predictions):
    val=predictions[predictions.partition=='val'];result={}
    for name in ['perceptron','logistic_standard','logistic_weighted','majority','historical']:
        items=[]
        for item,r in val.groupby('item_id',sort=True):
            y=r.y.to_numpy();p=r[name].to_numpy();m=metrics(y,p)
            items.append(dict(item_id=int(item),observations=len(r),positive=int(y.sum()),negative=int((y==0).sum()),metrics=m))
        m=metrics(val.y.to_numpy(),val[name].to_numpy());cm=m['confusion_matrix']
        result[name]=dict(positive_items=int(val.loc[val.y==1,'item_id'].nunique()),all_negative_items=sum(r['positive']==0 for r in items),
            false_alert_items=sum(r['metrics']['confusion_matrix'][0][1]>0 for r in items),
            true_positive_items=sum(r['metrics']['confusion_matrix'][1][1]>0 for r in items),by_item=items,
            recall_wilson_descriptive=wilson(cm[1][1],cm[1][0]+cm[1][1]))
    return result

def freeze(selected,models,comparison,state):
    folder=OUT/'modelos';folder.mkdir(exist_ok=True);frozen={}
    for name,model in models.items():
        path=folder/(name+'.json');model.save(path);loaded=load_model(path)
        row=selected[name]
        frozen[name]=dict(path=str(path.relative_to(BASE)),sha256=sha(path),run=row['id'],variant=row['variant'],
            selected_epoch=model.best_epoch_,params=model.get_params(),validation_metrics=row['metrics']['val'],
            validation_average_precision=row.get('validation_average_precision'),n_features=len(loaded.weights_),scores_calibrated=False)
    guardar(OUT/'congelacion.json',dict(status='frozen_development_test_reserved',models=frozen,final_models=['perceptron','logistic'],
        feature_names=FEATURES,preprocessor=state,preprocessor_sha256=sha(PREVIOUS/'dataset/preprocesador.json'),
        source_artifacts_sha256={str(p.relative_to(BASE)):sha(p) for p in DATA.iterdir() if p.is_file()},
        plan_sha256=sha(OUT/'configuracion_experimental.json'),prior_registration_sha256=sha(OUT/'registro_previo.json'),
        dataset_manifest_sha256=sha(PREVIOUS/'dataset/manifiesto.json'),majority_class=comparison['majority_train_class'],
        thresholds=PLAN['thresholds'],baselines=PLAN['baselines'],common_equals_primary=True,test_2016_loaded=False,
        training_on_validation=False,retraining_train_plus_validation=False))

def manifest():
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifiesto.json']+[BASE/n for n in sorted(NEW)]
    guardar(OUT/'manifiesto.json',dict(scope='Inventory development training and freezing only',
        sha256={str(p.relative_to(BASE)):sha(p) for p in sorted(paths) if p.exists()}))

def train():
    if not (OUT/'configuracion_experimental.json').exists() or read(OUT/'configuracion_experimental.json')!=PLAN:raise RuntimeError('Preparar plan antes de resultados')
    if (OUT/'experimentos.json').exists():raise RuntimeError('Resultados ya congelados; usar verificación, no ampliar/reentrenar silenciosamente')
    verify_prior();dev,state=preflight()
    with patch.object(Configuracion,'conectar',side_effect=AssertionError('SQL prohibido')):
        from src.visualization.pca import PCA2D
        with patch.object(PCA2D,'fit',side_effect=AssertionError('PCA prohibido')):
            runs,selected,models=fit_experiments(dev,PLAN,OUT)
    comp,monthly,predictions=evaluate(dev,models)
    guardar(OUT/'experimentos.json',runs);guardar(OUT/'metricas.json',comp);guardar(OUT/'metricas_mensuales.json',monthly)
    predictions.to_csv(OUT/'predicciones_desarrollo.csv',index=False,lineterminator='\n',float_format='%.17g')
    conv={r['id']:dict(convergence(r),variant=r['variant'],loss_definition=r.get('loss_definition','online errors, not BCE')) for r in runs}
    guardar(OUT/'convergencia.json',conv);guardar(OUT/'concentracion_positivos.json',concentration(predictions))
    freeze(selected,models,comp,state)
    from src.experiments.reportes_inventario import plots,reports,protocol
    plots(selected,comp,monthly,OUT);reports(runs,selected,comp,conv);protocol()
    guardar(OUT/'integridad_despues.json',verify_prior());manifest()
    print(json.dumps(dict(selected={k:dict(id=r['id'],epoch=r['selected_epoch'],variant=r['variant'],val=r['metrics']['val']) for k,r in selected.items()})))

def verify():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    verify_prior();dev,state=preflight();frozen=read(OUT/'congelacion.json')
    check('Plan coincide con registro previo',read(OUT/'configuracion_experimental.json')==PLAN and sha(OUT/'configuracion_experimental.json')==read(OUT/'registro_previo.json')['plan_sha256'])
    check('Registro anterior a resultados',read(OUT/'registro_previo.json')['results_existed_at_registration'] is False)
    from src.visualization.pca import PCA2D
    with patch.object(Configuracion,'conectar',side_effect=AssertionError('SQL prohibido')),patch.object(PCA2D,'fit',side_effect=AssertionError('PCA prohibido')):
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_entrenamiento_inventario')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (OUT/'pruebas_focalizadas.txt').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
        check('Pruebas focalizadas aprobadas',result.wasSuccessful() and result.testsRun>=19)
        with tempfile.TemporaryDirectory(prefix='inventario-training-replay-') as tmp:
            tmp=Path(tmp);runs,selected,models=fit_experiments(dev,PLAN,tmp)
            check('Seis ejecuciones reproducen historiales y métricas',runs==read(OUT/'experimentos.json'))
            comp,monthly,preds=evaluate(dev,models)
            check('Evaluación global reproduce exactamente',comp==read(OUT/'metricas.json'))
            check('Evaluación mensual reproduce exactamente',monthly==read(OUT/'metricas_mensuales.json'))
            from src.experiments.reportes_inventario import plots
            plots(selected,comp,monthly,tmp)
            hashes={str(p.relative_to(tmp)):sha(p) for p in tmp.rglob('*') if p.is_file()}
            check('Modelos, trazas, historiales y gráficos idénticos byte a byte',all(sha(OUT/n)==h for n,h in hashes.items()))
            guardar(OUT/'reproducibilidad.json',dict(status='identical',logical_budget_runs=6,replayed_runs=6,replay_is_fixed_plan_verification_not_new_search=True,
                files_compared=len(hashes),hashes=hashes,metrics_exact=True,histories_exact=True,SQL_queries=0,test_loaded=False,PCA=False))
    for name,record in frozen['models'].items():
        path=BASE/record['path'];check('Hash de modelo '+name,sha(path)==record['sha256']);model=load_model(path)
        for split,v in dev.items():check('Predicciones persistidas '+name+'/'+split,metrics(v['y'],model.predict(v['X'],feature_names=FEATURES))==selected[name]['metrics'][split])
    expected=read(OUT/'experimentos.json')
    for row in expected:
        history=row['history'];traces=read(OUT/'runs'/row['id']/'traza_parametros.json')
        check('Checkpoint óptimo prefijado '+row['id'],row['selected_epoch']==max(range(1,len(history)+1),key=lambda e:clave_seleccion(traces[e-1]['metrics']['val'],e)))
        check('Historial finito '+row['id'],all(np.isfinite(v) for h in history for v in h.values() if isinstance(v,(int,float))))
    check('Población común igual a principal sin exclusiones',comp['common_equals_primary'] and comp['missing_historical']==0 and comp['common']==comp['primary'])
    check('Archivos protegidos no alterados',protected()==verify_prior())
    check('Datos, preprocesador y modelos matemáticos anteriores intactos',all(sha(BASE/n)==h for n,h in frozen['source_artifacts_sha256'].items()))
    check('Test no cargado ni construido',not frozen['test_2016_loaded'] and not list(DATA.glob('*test*')) and not read(OUT/'protocolo_test.json')['outcomes_queried'])
    guardar(OUT/'validaciones.json',dict(status='successful_training_verification',checks_passed=len(checks),checks=checks,
        new_tests=result.testsRun,new_failures=len(result.failures),new_errors=len(result.errors),preserved_files=len(verify_prior()),
        SQL_queries=0,dataset_rebuilt=False,test_loaded=False,PCA_experimental=False,protected_hashes_overwritten=False,
        manual_base_models_unchanged=True,weighted_variant_isolated=True,validation_weighted=False,threshold_search=False,
        synthetic_business_training_data=False,full_suite='See suite_global.json; nine historical cases must remain explicitly distinguished'))
    guardar(OUT/'integridad_despues.json',verify_prior());manifest()
    print(json.dumps(dict(status='verified',checks=len(checks),tests=result.testsRun,replayed_runs=6,files_reproduced=len(hashes),preserved_files=len(verify_prior()))))

def run_suite():
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(BASE/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'suite_global.txt').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    prior=read(PREVIOUS/'dataset/suite_existente.json')
    def key(test):return str(test).replace('tests.','')
    expected={key(r['test']) for category in ['failures','errors'] for r in prior[category]}
    actual={key(test) for test,trace in result.failures+result.errors}
    no_new=actual==expected and len(actual)==9
    data=dict(status='known_historical_failures_only' if no_new else 'new_regressions_or_changed_failure_set',
        tests_run=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failures=[dict(test=str(test),traceback=trace) for test,trace in result.failures],
        errors=[dict(test=str(test),traceback=trace) for test,trace in result.errors],
        skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],
        exact_prior_nine_cases=no_new,new_regressions=not no_new,global_suite_passed=result.wasSuccessful(),
        old_tests_modified=False,preserved_files=len(verify_prior()),historical_cause='old report version hashes, proven before Inventory dataset',
        synthetic_unit_fixture_training_or_PCA_may_run=True,experimental_test_or_PCA_not_executed=True)
    guardar(OUT/'suite_global.json',data)
    validation=read(OUT/'validaciones.json');validation['global_suite']=dict(status=data['status'],tests_run=data['tests_run'],passed=data['passed'],failures=len(result.failures),errors=len(result.errors),new_regressions=not no_new,global_suite_passed=result.wasSuccessful())
    validation['overall_status']='training_verified_with_nine_preexisting_suite_cases' if no_new else 'new_regressions'
    guardar(OUT/'validaciones.json',validation);manifest()
    print(json.dumps({k:data[k] for k in ['status','tests_run','passed','exact_prior_nine_cases','preserved_files']}))
    if not no_new:raise RuntimeError('La suite presenta una diferencia nueva: investigar antes de continuar')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['preparar','entrenar','verificar','suite']);args=p.parse_args()
    {'preparar':prepare,'entrenar':train,'verificar':lambda:verify(),'suite':lambda:run_suite()}[args.action]()
