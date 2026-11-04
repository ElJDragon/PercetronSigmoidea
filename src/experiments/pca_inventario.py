"""PCA train-only y auditoría offline de Inventario, sin reentrenar ni abrir test."""
import argparse
from collections import defaultdict,Counter
from contextlib import ExitStack
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
from src.experiments import entrenar_inventario as training
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual
from src.models.logistic_inventario import LogisticRegressionInventarioPonderada
from src.datamarts.inventario.caracteristicas import FEATURES,features
from src.datamarts.inventario.extraccion import read_snapshot,SOURCE_SQL
from src.datamarts.inventario.preprocesamiento import PreprocesadorInventario
from src.datamarts.inventario_viabilidad import add_month
from src.visualization.pca import PCA2D

DATA=training.DATA
PREVIOUS=training.OUT
OUT=training.PREVIOUS/'pca'
NEW={'src/experiments/pca_inventario.py','src/experiments/reportes_pca_inventario.py','tests/test_pca_inventario.py'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def snapshot():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe'] for p in (BASE/root).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents and str(p.relative_to(BASE)).replace('\\','/') not in NEW}

def verify_prior():
    old=read(OUT/'integridad_antes.json')
    if old!={n:sha(BASE/n) for n in old}:raise RuntimeError('Artefacto previo modificado')
    return old

def metric_audit(y,pred):
    m=training.metrics(y,pred);tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
    return dict(m,TN=tn,FP=fp,FN=fn,TP=tp,false_positive_rate=fp/(fp+tn) if fp+tn else None)

def verify_models(dev):
    for n,h in read(PREVIOUS/'manifiesto.json')['sha256'].items():
        if sha(BASE/n)!=h:raise RuntimeError('Congelación alterada: '+n)
    frozen=read(PREVIOUS/'congelacion.json');models={};details={}
    runs=read(PREVIOUS/'experimentos.json')
    for name,epoch in [('perceptron',30),('logistic',156)]:
        r=frozen['models'][name];path=BASE/r['path']
        if sha(path)!=r['sha256']:raise RuntimeError('Modelo no coincide con SHA congelado')
        model=training.load_model(path)
        if model.best_epoch_!=epoch or model.feature_names_!=FEATURES or len(model.weights_)!=17:raise RuntimeError('Checkpoint/catálogo incorrecto')
        if name=='logistic' and (model.threshold!=.5 or model.learning_rate!=.05):raise RuntimeError('Umbral/tasa modificados')
        if name=='perceptron' and (model.seed!=42 or model.learning_rate!=.01):raise RuntimeError('Configuración modificada')
        source=next(v for v in runs if v['id']==r['run']);trace=read(PREVIOUS/'runs'/r['run']/'traza_parametros.json')[epoch-1]
        if not np.array_equal(model.weights_,trace['weights']) or model.bias_!=trace['bias']:raise RuntimeError('Pesos/sesgo no corresponden al checkpoint')
        models[name]=model;details[name]=dict(path=r['path'],sha256=sha(path),weights=model.weights_.tolist(),bias=model.bias_,epoch=epoch,
            params=model.get_params(),feature_names=FEATURES,metrics_match_training=True,fit_calls=0,checkpoint_equals_trace=True)
    saved=pd.read_csv(PREVIOUS/'predicciones_desarrollo.csv',keep_default_na=False,float_precision='round_trip')
    comp=read(PREVIOUS/'metricas.json');audit={};baseline={}
    for split,v in dev.items():
        rows=saved[saved.partition==split].reset_index(drop=True)
        if v['meta'].observation_id.tolist()!=rows.observation_id.tolist() or not np.array_equal(v['y'],rows.y):raise RuntimeError('Predicciones fuera de alineación')
        vals={}
        for name,model in models.items():
            col='perceptron' if name=='perceptron' else 'logistic_standard';p=model.predict(v['X'],feature_names=FEATURES)
            if not np.array_equal(p,rows[col]):raise RuntimeError('Predicciones no reproducidas')
            key='perceptron_margin' if name=='perceptron' else 'logistic_standard_score'
            z=model.decision_function(v['X']) if name=='perceptron' else model.predict_proba(v['X'])
            if not np.array_equal(z,rows[key].to_numpy(np.float64)):raise RuntimeError('Scores no reproducidos')
            m=metric_audit(v['y'],p);original=comp['primary'][split][col]
            if any(m[k]!=value for k,value in original.items()):raise RuntimeError('Métricas congeladas discrepantes')
            vals[name]=m
        majority=np.zeros(len(v['y']),dtype=np.int8)
        if not np.array_equal(majority,rows.majority) or not np.array_equal(v['historical'],rows.historical):raise RuntimeError('Baseline no reproducido')
        for name,p in [('majority',majority),('historical',v['historical'])]:
            m=training.metrics(v['y'],p)
            if m!=comp['primary'][split][name]:raise RuntimeError('Métrica baseline discrepante')
        audit[split]=vals
        baseline[split]=dict(primary=len(v['y']),common=len(v['y']),positive=int(v['y'].sum()),negative=int((v['y']==0).sum()),
            majority_support=len(majority),historical_support=len(v['historical']),missing_historical=0,
            same_ordered_keys_for_all=True,primary_selection_by_errors=False,
            methods={n:comp['primary'][split][n] for n in ['perceptron','logistic_standard','majority','historical']})
    baseline.update(formula=training.PLAN['baselines']['historical'],history_months=3,common_equals_primary=True)
    return models,audit,baseline,saved,details

def row_features(meta,lookup):
    history=[lookup[int(meta['item_id']),add_month(meta['month'],-k)] for k in (1,2,3)]
    return features(history,meta['month'],meta['first_event_at'][:7])

def causal_audit(dev):
    monthly=pd.read_csv(DATA/'agregados_articulo_mes.csv.gz',keep_default_na=False)
    raw=pd.read_csv(DATA/'caracteristicas_sin_escalar.csv.gz',float_precision='round_trip');labels=pd.read_csv(DATA/'etiquetas.csv.gz')
    meta=pd.read_csv(DATA/'metadatos.csv.gz',keep_default_na=False);lookup={(int(r.item_id),r.month):r._asdict() for r in monthly.itertuples(index=False)}
    if monthly[['item_id','month']].duplicated().any() or meta[['item_id','month']].duplicated().any():raise RuntimeError('Identidad artículo-mes no única')
    receipts=read_snapshot(DATA/'recepciones_desarrollo.csv.gz');group=defaultdict(list)
    for r in receipts:group[r['item_id'],r['occurred_at'][:7]].append(r)
    for r in monthly.itertuples():
        events=group[int(r.item_id),r.month]
        if r.coverage_status!='covered':raise RuntimeError('Cobertura real requiere investigación')
        if r.receipt_count!=len(events) or r.receipt_units!=sum(int(e['quantity']) for e in events):raise RuntimeError('Agregado no corresponde a eventos')
    # Primera aparición contrastada con la evidencia de cualquier movimiento, nunca con ventas futuras.
    first={}
    for r in read(training.PREVIOUS/'eventos_desarrollo.json'):
        first[r['item_id']]=min(first.get(r['item_id'],r['occurred_at']),r['occurred_at'])
    errors={n:0. for n in FEATURES};source_catalog=read(training.PREVIOUS/'catalogo_caracteristicas.json')
    for i,r in enumerate(meta.to_dict('records')):
        if r['first_event_at']!=first[int(r['item_id'])] or r['first_event_at']>=r['prediction_at']:raise RuntimeError('Primera aparición no causal')
        hs=[add_month(r['month'],-k) for k in (1,2,3)]
        if hs!=[r['history_month_1'],r['history_month_2'],r['history_month_3']]:raise RuntimeError('Desplazamiento de historia')
        for month in hs:
            h=lookup[int(r['item_id']),month]
            if month>=r['month'] or h['latest_edited_at']>=r['prediction_at']:raise RuntimeError('Evento futuro en historia')
        expected=row_features(r,lookup)
        for n,value in expected.items():
            error=abs(value-raw.iloc[i][n]);errors[n]=max(errors[n],float(error))
            if not np.isclose(value,raw.iloc[i][n],atol=1e-12,rtol=0):raise RuntimeError('Fuga/fórmula incorrecta: '+n)
        ref=float(np.median([lookup[int(r['item_id']),m]['receipt_count'] for m in hs]));target=lookup[int(r['item_id']),r['month']]['receipt_count']
        if labels.iloc[i].historical_reference!=ref or labels.iloc[i].target_count!=target or labels.iloc[i].y!=int(target>ref):raise RuntimeError('Etiqueta/referencia no corresponde a ventana')
    def dependencies(name):
        if '_lag_' in name:return [-int(name.rsplit('_',1)[1])]
        if name=='recepciones_cambio_1':return [-1,-2]
        if name=='mes_anterior_sin_recepciones':return [-1]
        if name in ['antiguedad_observada_meses','mes_sin','mes_cos','dias_mes_objetivo']:return []
        return [-1,-2,-3]
    result=[dict(**f,verified_observations=len(meta),max_absolute_formula_error=errors[f['name']],
        historical_month_offsets_used=dependencies(f['name']),
        actual_information_window=('known target calendar; no receipt events' if f['name'] in ['mes_sin','mes_cos','dias_mes_objetivo'] else
            'first event in any movement known strictly before t' if f['name']=='antiguedad_observada_meses' else 'explicit prior-month receipt aggregates only'),
        technical_availability='Event and LastEditedWhen precede prediction; monthly closure is an explicit academic assumption',
        source_event_snapshot_sha256=sha(DATA/'recepciones_desarrollo.csv.gz'),target_events_used=False) for f in source_catalog]
    return dict(status='no_temporal_leak_detected',observations=len(meta),features=result,
        aggregation_verified_from_receipt_snapshot=True,identity='operational item_id, not dimension version/name',
        implementation='Explicit (item_id, month) lookups; no positional shift/rolling across articles',
        chronological_order='item ascending then month ascending',zero_policy='only under event_ledger_academic coverage',
        first_appearance_verified_from_any_movement=True,independent_creation_log_available=False,
        limitations='Retrospective non-temporal event archive; conservation/closure assumed academically, not certified'),lookup,raw

def concentration(saved,dev,raw):
    val=saved[saved.partition=='val'].reset_index(drop=True);train=saved[saved.partition=='train']
    train_counts=train.groupby('item_id').y.sum().to_dict();positive_ids=sorted(int(i) for i in val.loc[val.y==1,'item_id'].unique())
    items=[]
    for item,rows in val.groupby('item_id',sort=True):
        items.append(dict(item_id=int(item),train_positive=int(train_counts.get(item,0)),val_positive=int(rows.y.sum()),
            val_negative=int((rows.y==0).sum()),perceptron=metric_audit(rows.y,rows.perceptron),logistic=metric_audit(rows.y,rows.logistic_standard)))
    p=val.perceptron.to_numpy();l=val.logistic_standard.to_numpy();y=val.y.to_numpy();p_fp=(y==0)&(p==1);l_fp=(y==0)&(l==1)
    # Diagnóstico único prefijado, sin optimizar ventana/umbral ni sustituir baselines.
    vraw=raw.set_index('observation_id').loc[val.observation_id,FEATURES]
    activity=(vraw[['recepciones_lag_1','recepciones_lag_2','recepciones_lag_3']].sum(axis=1)>0).to_numpy(dtype=np.int8)
    diag=dict(formula='1[N(t-1)+N(t-2)+N(t-3)>0]',status='post_freeze_descriptive_only_not_new_baseline_or_selection',
        metrics=metric_audit(y,activity),agreement_perceptron=float(np.mean(activity==p)),agreement_logistic=float(np.mean(activity==l)),
        months_window_chosen=3,threshold=0,no_search=True)
    return dict(positive_item_ids=positive_ids,positive_items=len(positive_ids),positive_observations=int(y.sum()),
        all_negative_items=sum(i['val_positive']==0 for i in items),positive_items_without_train_positives=[i['item_id'] for i in items if i['val_positive'] and not i['train_positive']],
        by_item=items,positive_by_month={str(k):int(v) for k,v in val.groupby('month').y.sum().items()},
        shared_false_positives=int((p_fp&l_fp).sum()),perceptron_only_false_positives=int((p_fp&~l_fp).sum()),logistic_only_false_positives=int((l_fp&~p_fp).sum()),
        disagreements=int((p!=l).sum()),both_correct=int(((p==y)&(l==y)).sum()),both_wrong=int(((p!=y)&(l!=y)).sum()),
        shared_true_positives=int(((y==1)&(p==1)&(l==1)).sum()),
        error_keys=dict(shared_fp=val.loc[p_fp&l_fp,'observation_id'].tolist(),p_only_fp=val.loc[p_fp&~l_fp,'observation_id'].tolist(),l_only_fp=val.loc[l_fp&~p_fp,'observation_id'].tolist()),
        historical_activity_diagnostic=diag,population_redefined=False,models_reselected=False)

def fit_pca(train,val):
    pca=PCA2D().fit(train,feature_names=FEATURES)
    return pca,dict(train=pca.transform(train,feature_names=FEATURES),val=pca.transform(val,feature_names=FEATURES))

def math_checks(pca,x,z):
    centered=x-pca.mean_;cov=np.cov(centered,rowvar=False,ddof=1);eig=np.linalg.eigvalsh(cov)[::-1]
    projected_cov=np.cov(z,rowvar=False,ddof=1);residual=float(np.sum((centered-z@pca.components_.T)**2)/(len(x)-1))
    checks=dict(orthogonality_error=float(np.max(np.abs(pca.components_.T@pca.components_-np.eye(2)))),
        eigenvalue_error=float(np.max(np.abs(eig-pca.explained_variance_all_))),
        projection_covariance_error=float(np.max(np.abs(projected_cov-np.diag(pca.explained_variance_)))),
        explained_ratio_sum=float(pca.explained_variance_ratio_all_.sum()),reconstruction_residual=residual,
        discarded_variance=float(pca.explained_variance_all_[2:].sum()),rank=pca.rank_)
    if not (checks['orthogonality_error']<1e-12 and checks['eigenvalue_error']<1e-10 and checks['projection_covariance_error']<1e-10
        and np.isclose(checks['explained_ratio_sum'],1,atol=1e-12) and np.isclose(residual,checks['discarded_variance'],atol=1e-10)):
        raise RuntimeError('PCA no cumple coherencia matemática')
    return checks

def convergence_audit():
    runs=read(PREVIOUS/'experimentos.json');recorded=read(PREVIOUS/'convergencia.json');selected={}
    for name,epoch in [('perceptron',30),('logistic',156)]:
        run=read(PREVIOUS/'congelacion.json')['models'][name]['run'];r=next(v for v in runs if v['id']==run)
        c=training.convergence(r)
        if any(c[k]!=recorded[run][k] for k in c):raise RuntimeError('Convergencia no reproducida')
        if r['selected_epoch']!=epoch or c['termination']!='max_epochs':raise RuntimeError('Checkpoint/terminación discrepantes')
        history=read(PREVIOUS/'runs'/run/'historial.json')
        if history!=r['history']:raise RuntimeError('Historial original discrepante')
        if name=='perceptron' and (c['last_online_errors']<=0 or c['zero_error_epochs']):raise RuntimeError('Supuesto de no convergencia discrepante')
        if name=='logistic' and (c['gradient_tolerance_reached'] or c['last_gradient_norm']<=1e-6):raise RuntimeError('Diagnóstico de gradiente discrepante')
        selected[name]=c
    return selected

def protocol():
    frozen=read(PREVIOUS/'congelacion.json')
    return dict(status='closed_before_first_test_opening',test_executed=False,sql_queries_now=0,
        months=['2016-01','2016-02','2016-03','2016-04','2016-05'],event_interval=['2016-01-01','2016-06-01'],
        label_knowledge_cut_exclusive='2016-06-01 00:00:00',receipt_SELECT=SOURCE_SQL,receipt_parameters=[11,'2016-01-01','2016-06-01'],
        first_appearance_SELECT='SELECT st.StockItemID AS item_id,MIN(st.TransactionOccurredWhen) AS first_event_at FROM Warehouse.StockItemTransactions st WHERE st.TransactionOccurredWhen < ? GROUP BY st.StockItemID',
        first_appearance_parameter='prediction_at for each target month; verify edit knowledge before cut; any movement, not first receipt',
        identity='StockItemID / WWI Stock Item ID; exact source event StockItemTransactionID; DW bijection on item,type,date,quantity and documents',
        coverage='event_ledger_academic explicitly on; only complete covered months after first known appearance may be zero; unknown/incomplete excluded with reason',
        eligibility='three consecutive prior covered months; identity and first event known before target; never filter by target activity, future survival, prediction or error',
        history='initial October-December 2015 from frozen snapshot; subsequent closed prior test months may supply history; no refit',
        label='1[N(t)>median(N(t-1),N(t-2),N(t-3))], ties 0; target counted after closure and known strictly before label cut',
        feature_names=FEATURES,feature_catalog_sha256=sha(training.PREVIOUS/'catalogo_caracteristicas.json'),
        preprocessing_sha256=sha(training.PREVIOUS/'dataset/preprocesador.json'),preprocessing='transform only; same 17 features, mean and population std, no new masking or fitting',
        models={n:frozen['models'][n] for n in ['perceptron','logistic']},thresholds=dict(perceptron='z>=0',logistic=.5),
        majority_class=0,historical='1[N(t-1)>median(N(t-1),N(t-2),N(t-3))]',
        primary='all otherwise eligible item-month classifier observations with known label',common='exact stable-key intersection of perceptron, logistic, majority and historical, expected equal to primary',
        missing_historical='keep principal row, secondary common support separate; no invented fourth lag or silent row dropping',
        metrics=['n','class support','accuracy','balanced_accuracy','precision','recall','F1','specificity','FPR','confusion [[TN,FP],[FN,TP]]','predicted-positive rate','logistic Average Precision from grouped tied scores'],
        denominator_policy='precision/F1 zero when no predicted positive (annotate); recall absent positive None, specificity absent negative None, BA None if either real class absent; keep single-class months; AP None with no positives',
        breakdowns=['global','every calendar month','per operational item','seen in train','seen only in val','new eligible in test','insufficient history exclusions','positive events, TP and false alerts'],
        opening_record='persist UTC, immutable protocol/model/dataset/source/code hashes and purpose BEFORE any outcome query; write-once failure if already opened',
        reproducibility='canonical snapshots, SQL logs and before/after fingerprints, keys, raw/processed features, labels and scores separately, all SHA256; regenerate offline without fit',
        no_selection_on_test=True,no_train_plus_val_refit=True,PCA='train-fit visualization only; never classifier input',
        diagnostics_not_baselines=['1[any receipt in past three months] is descriptive only; not substituted for frozen historical'],
        prior_protocol_sha256=sha(PREVIOUS/'protocolo_test.json'),frozen_models_manifest_sha256=sha(PREVIOUS/'congelacion.json'),
        decision='first test evaluation methodologically recommended after successful audit; not executed or automatically opened')

def export_pca(dev,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);pca,z=fit_pca(dev['train']['X'],dev['val']['X'])
    pca.save(out/'pca_parametros.json');checks=math_checks(pca,dev['train']['X'],z['train'])
    for split,coords in z.items():
        np.save(out/f'proyeccion_{split}.npy',coords,allow_pickle=False)
        frame=dev[split]['meta'][['observation_id','item_id','month']].copy();frame['PC1']=coords[:,0];frame['PC2']=coords[:,1];frame['y']=dev[split]['y']
        frame.to_csv(out/f'proyeccion_{split}.csv',index=False,lineterminator='\n',float_format='%.17g')
    guardar(out/'comprobaciones_matematicas.json',checks)
    guardar(out/'varianza_explicada.json',dict(PC1_percent=float(pca.explained_variance_ratio_[0]*100),PC2_percent=float(pca.explained_variance_ratio_[1]*100),
        retained_percent=float(pca.explained_variance_ratio_.sum()*100),lost_percent=float((1-pca.explained_variance_ratio_.sum())*100),
        all_ratios=pca.explained_variance_ratio_all_.tolist(),rank=pca.rank_,fit_samples=4599,fit_features=17))
    loadings=pd.DataFrame(dict(feature=FEATURES,PC1=pca.components_[:,0],PC2=pca.components_[:,1]))
    loadings.to_csv(out/'coeficientes_componentes.csv',index=False,lineterminator='\n',float_format='%.17g')
    tops={f'PC{j+1}':sorted([dict(feature=n,coefficient=float(w),squared_loading_share=float(w*w)) for n,w in zip(FEATURES,pca.components_[:,j])],key=lambda r:-abs(r['coefficient']))[:6] for j in range(2)}
    groups={};centroids=[]
    for split,coords in z.items():
        items=dev[split]['meta'].item_id.to_numpy();total=np.sum((coords-coords.mean(axis=0))**2,axis=0);between=np.zeros(2)
        for item in sorted(set(items)):
            mask=items==item;mean=coords[mask].mean(axis=0);between+=mask.sum()*(mean-coords.mean(axis=0))**2
            centroids.append(dict(partition=split,item_id=int(item),n=int(mask.sum()),positive=int(dev[split]['y'][mask].sum()),PC1_centroid=float(mean[0]),PC2_centroid=float(mean[1])))
        groups[split]=dict(between_item_SS_fraction_PC1=float(between[0]/total[0]),between_item_SS_fraction_PC2=float(between[1]/total[1]),
            caveat='descriptive repeated-item grouping, not supervised separation or causal inference')
    guardar(out/'interpretacion_componentes.json',dict(top_coefficients=tops,grouping=groups,
        definition='unit eigenvector coefficients on standardized inputs; squared loadings sum to 1 per PC, not causal effects'))
    pd.DataFrame(centroids).to_csv(out/'centroides_articulo.csv',index=False,lineterminator='\n',float_format='%.17g')
    from src.experiments.reportes_pca_inventario import figures
    figures(pca,z,{s:v['y'] for s,v in dev.items()},out)
    return pca,z

def manifest():
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifiesto.json']+[BASE/n for n in sorted(NEW)]
    guardar(OUT/'manifiesto.json',dict(scope='Inventory metric/causal audit and train-only SVD PCA; test not opened',
        sha256={str(p.relative_to(BASE)):sha(p) for p in sorted(paths) if p.exists()},
        frozen_training_manifest_sha256=sha(PREVIOUS/'manifiesto.json')))

def guards(stack):
    for cls in [PerceptronSimple,LogisticRegressionManual,LogisticRegressionInventarioPonderada,PreprocesadorInventario]:
        stack.enter_context(patch.object(cls,'fit',side_effect=AssertionError('Ajuste experimental prohibido')))
    stack.enter_context(patch.object(Configuracion,'conectar',side_effect=AssertionError('SQL prohibido')))
    stack.enter_context(patch.object(training,'fit_experiments',side_effect=AssertionError('Reentrenamiento prohibido')))

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',snapshot())
    verify_prior()
    try:
        with ExitStack() as stack:
            guards(stack);dev,state=training.preflight();models,audit,baseline,saved,details=verify_models(dev)
            causal,lookup,raw=causal_audit(dev);conc=concentration(saved,dev,raw);conv=convergence_audit()
            guardar(OUT/'auditoria_metricas.json',audit);guardar(OUT/'verificacion_modelos.json',details)
            guardar(OUT/'revision_causal.json',causal);guardar(OUT/'concentracion_positivos.json',conc)
            guardar(OUT/'conciliacion_baselines.json',baseline);guardar(OUT/'verificacion_convergencia.json',conv)
            pca,z=export_pca(dev,OUT);guardar(OUT/'protocolo_test_definitivo.json',protocol())
            from src.experiments.reportes_pca_inventario import reports
            reports(audit,causal,conc,baseline,conv,pca)
        guardar(OUT/'integridad_despues.json',verify_prior());manifest()
        print(json.dumps(dict(status='audit_and_PCA_completed',variance=read(OUT/'varianza_explicada.json'),positive_items=conc['positive_item_ids'])))
    except Exception as error:
        guardar(OUT/'bloqueo.json',dict(status='blocked',reason=str(error),test_authorized=False,no_silent_correction=True));raise

def verify():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    verify_prior()
    with ExitStack() as stack:
        guards(stack);dev,state=training.preflight()
        models,audit,baseline,saved,details=verify_models(dev)
        check('Métricas originales y auditoría coinciden exactamente',audit==read(OUT/'auditoria_metricas.json'))
        check('Modelos congelados y trazas intactos',details==read(OUT/'verificacion_modelos.json'))
        check('Baselines congelados y soportes exactos',baseline==read(OUT/'conciliacion_baselines.json'))
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_pca_inventario')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (OUT/'pruebas_focalizadas.txt').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
        check('Pruebas focalizadas satisfactorias',result.wasSuccessful() and result.testsRun==20)
        with tempfile.TemporaryDirectory(prefix='inventory-pca-replay-') as tmp:
            root=Path(tmp);pca,z=export_pca(dev,root);hashes={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
            check('PCA, proyecciones, coeficientes y gráficos reproducibles byte a byte',all(sha(OUT/n)==h for n,h in hashes.items()))
            guardar(OUT/'reproducibilidad.json',dict(status='identical',files_compared=len(hashes),hashes=hashes,
                PCA_fit_partition='train_only',classifier_fit_calls=0,scaler_fit_calls=0,SQL_queries=0,test_loaded=False,
                numerical_environment='see entorno.json; bitwise replay in same environment'))
        loaded=PCA2D.load(OUT/'pca_parametros.json')
        for split,v in dev.items():check('Persistencia y proyección '+split,np.array_equal(loaded.transform(v['X'],feature_names=FEATURES),np.load(OUT/f'proyeccion_{split}.npy',allow_pickle=False)))
    check('No hay matrices test',not list(DATA.glob('*test*')))
    check('Todos los archivos anteriores preservados',snapshot()==verify_prior())
    check('No hay bloqueo causal o de integridad',(OUT/'bloqueo.json').exists() is False)
    import platform
    import matplotlib
    guardar(OUT/'entorno.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,
        PCA_implementation='unchanged src/visualization/pca.py',PCA_source_sha256=sha(BASE/'src/visualization/pca.py'),dtypes='float64',variance_ddof=1))
    guardar(OUT/'validaciones.json',dict(status='successful_focused_audit',checks_passed=len(checks),checks=checks,new_tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),preserved_files=len(verify_prior()),
        classifier_training_runs=0,scaler_fit_calls=0,dataset_rebuilt=False,SQL_queries=0,test_outcomes_loaded=False,
        causal_audit='no_temporal_leak_detected_under_explicit_academic_event_conservation_and_closure',
        principal_validation=2628,common_validation=2628,missing_historical=0,
        preliminary_dictamen='methodologically ready for first test evaluation only after complete suite audit and persisted opening record; no test executed'))
    guardar(OUT/'integridad_despues.json',verify_prior());manifest()
    print(json.dumps(dict(status='verified',new_tests=result.testsRun,checks=len(checks),files_reproduced=len(hashes),preserved_files=len(verify_prior()))))

def run_suite():
    stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(BASE/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'suite_global.txt').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    prior=read(PREVIOUS/'suite_global.json')
    def key(value):return str(value).replace('tests.','')
    expected={key(r['test']) for category in ['failures','errors'] for r in prior[category]}
    actual={key(test) for test,trace in result.failures+result.errors}
    no_new=actual==expected and len(actual)==9
    check_integrity=verify_prior()
    data=dict(status='only_nine_preexisting_cases' if no_new else 'new_regression_or_changed_failure_set',tests_run=result.testsRun,
        passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failures=[dict(test=str(test),traceback=trace) for test,trace in result.failures],errors=[dict(test=str(test),traceback=trace) for test,trace in result.errors],
        skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],exact_prior_nine_cases=no_new,new_regressions=not no_new,
        global_suite_passed=result.wasSuccessful(),old_tests_modified=False,protected_hashes_updated=False,preserved_files=len(check_integrity),
        note='Existing global tests may fit synthetic unit fixtures; no new Inventory classifier training or experimental test execution')
    guardar(OUT/'suite_global.json',data)
    validation=read(OUT/'validaciones.json');validation['global_suite']=dict(status=data['status'],tests_run=data['tests_run'],passed=data['passed'],failures=len(result.failures),errors=len(result.errors),new_regressions=not no_new,global_suite_passed=result.wasSuccessful())
    validation['dictamen']='ready_for_first_test_evaluation_under_frozen_protocol_and_opening_record' if no_new else 'blocked_new_regressions'
    validation['test_executed']=False
    guardar(OUT/'validaciones.json',validation)
    guardar(OUT/'dictamen.json',dict(status=validation['dictamen'],test_executed=False,first_test_methodologically_recommended=no_new,
        causal_leak_detected=False,models_unchanged=True,PCA_classifier_input=False,
        protocol_path=str((OUT/'protocolo_test_definitivo.json').relative_to(BASE)),protocol_sha256=sha(OUT/'protocolo_test_definitivo.json'),
        prerequisites_before_any_outcome_query=['Verify all frozen hashes','Persist a write-once opening record before the first SQL query','Respect coverage, cutoff, eligible population and frozen models'],
        historical_failures=9 if no_new else len(actual),new_regressions=not no_new))
    with (OUT/'dictamen_pretest.md').open('a',encoding='utf-8',newline='\n') as f:
        f.write(f'\nResultado final de suite: {data["tests_run"]} pruebas, {data["passed"]} correctas; {len(result.failures)} fallos y {len(result.errors)} errores. Coincidencia exacta con los nueve casos históricos: {no_new}. Nuevas regresiones: {not no_new}. Archivos previos preservados: {len(check_integrity)}. No se abrió test.\n')
    manifest()
    print(json.dumps({k:data[k] for k in ['status','tests_run','passed','exact_prior_nine_cases','preserved_files']}))
    if not no_new:raise RuntimeError('No autorizar test: nueva regresión')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['ejecutar','verificar','suite']);args=p.parse_args()
    {'ejecutar':run,'verificar':lambda:verify(),'suite':lambda:run_suite()}[args.action]()
