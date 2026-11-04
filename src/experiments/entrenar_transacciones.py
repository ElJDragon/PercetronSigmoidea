"""Desarrollo financiero: cuatro ejecuciones predefinidas, sin SQL, test o PCA."""
import argparse
import datetime as dt
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.auditoria_dw import BASE,guardar,sha
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual,sigmoid,binary_cross_entropy_logits
from src.evaluation.metricas import metricas_binarias,clave_seleccion,validar_etiquetas
from src.datamarts.transacciones.caracteristicas import FEATURES,ordinal
from src.datamarts.transacciones.preprocesamiento import PreprocesadorFinanciero

DATA=BASE/'datos/Procesados/transacciones'
PREVIOUS=BASE/'resultados/extension_datamarts/transacciones'
OUT=PREVIOUS/'entrenamiento'
NEW={'src/experiments/entrenar_transacciones.py','tests/test_entrenamiento_transacciones.py'}
PLAN={
 'version':1,'process':'financial_customer_payment_frequency','input_features':FEATURES,
 'selection':'validation balanced_accuracy descending; F1 descending; earliest epoch; earliest experiment index',
 'selection_population':'all approved validation observations, n=2943; no test',
 'threshold_policy':{'perceptron':'z>=0','logistic':0.5,'threshold_candidates':[0.5],'threshold_search':False},
 'initialization':'new zero weights and zero bias; no weights from Ventas',
 'experiments':[
  {'id':'p_seed42','algorithm':'perceptron','params':{'learning_rate':.01,'max_epochs':100,'seed':42,'shuffle':True}},
  {'id':'p_seed7','algorithm':'perceptron','params':{'learning_rate':.01,'max_epochs':100,'seed':7,'shuffle':True}},
  {'id':'l_eta005','algorithm':'logistic','params':{'learning_rate':.05,'max_epochs':400,'seed':42,'initialization':'zeros','tolerance':1e-6,'threshold':.5}},
  {'id':'l_eta010','algorithm':'logistic','params':{'learning_rate':.1,'max_epochs':400,'seed':42,'initialization':'zeros','tolerance':1e-6,'threshold':.5}}],
 'budget':{'runs':4,'max_online_epochs_total':200,'max_batch_epochs_total':800,'adaptive_expansion':False},
 'perceptron_order':'numpy default_rng(seed).permutation(train rows) each epoch; training rows in frozen dataset order',
 'logistic_order':'full batch gradient descent; seed recorded but zero initialization is deterministic',
 'perceptron_rate_rationale':'One rate with two predefined permutations; with zero initialization scaling eta scales parameters, so no redundant rate sweep.',
 'reference':'Existing manual class defaults/Ventas supply technical starting rates, not transferred weights or optimality claims.',
 'baselines':{'majority':'majority y_train, tie=0','history':'1[N(t-1)>median(N(t-2),N(t-3),N(t-4))]',
              'common_comparison':'all four methods on identical rows with four covered prior months; full model metrics reported separately; no fourth-lag imputation'},
 'average_precision':'Report logistic validation scores with grouped ties, noninterpolated AP; descriptive only, not selection',
 'stopping':{'perceptron':'existing implementation runs max_epochs; zero-error epochs reported without assuming convergence',
             'logistic':'gradient norm<=1e-6 or max_epochs, as implemented; report actual reason'},
 'test_2016':'sealed; no SQL or test array access','pca':False}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def protected():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
            for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents
            and str(p.relative_to(BASE)).replace('\\','/') not in NEW}


def verify_prior():
    prior=read(OUT/'integridad_antes.json')
    if any(sha(BASE/n)!=h for n,h in prior.items()):raise RuntimeError('Se modificó un artefacto previo')
    return prior


def preflight():
    """Solo carga las cuatro matrices aprobadas de desarrollo; nunca construye datos."""
    m=read(PREVIOUS/'dataset/manifiesto.json')
    for section in ['implementation_sha256','artifacts_sha256']:
        for name,h in m[section].items():
            if sha(BASE/name)!=h:raise RuntimeError('Dataset/implementación congelada alterada: '+name)
    if not read(PREVIOUS/'contrato_datos.json')['allowed_future_construction']:raise RuntimeError('Contrato financiero no autorizado')
    state=read(DATA/'preprocesador.json')
    if state['fit_partition']!='train' or state['train_rows']!=4612 or state['feature_names']!=FEATURES or len(FEATURES)!=16:raise RuntimeError('Esquema o fit del preprocesador incorrectos')
    prep=PreprocesadorFinanciero.from_dict(state);dev={}
    for split,n,p in [('train',4612,1798),('val',2943,1198)]:
        x=np.load(DATA/f'X_{split}.npy',allow_pickle=False);y=np.load(DATA/f'y_{split}.npy',allow_pickle=False)
        meta=pd.read_csv(DATA/f'metadatos_{split}.csv',keep_default_na=False)
        raw=pd.read_csv(DATA/f'caracteristicas_crudas_{split}.csv',float_precision='round_trip')
        audit=pd.read_csv(DATA/f'etiquetas_{split}.csv')
        if x.shape!=(n,16) or x.dtype!=np.float64 or y.shape!=(n,) or y.dtype!=np.int8 or not np.isfinite(x).all() or int(y.sum())!=p:raise RuntimeError('Matrices incompatibles')
        validar_etiquetas(y)
        if len(meta)!=n or not meta.observation_id.is_unique or meta.observation_id.tolist()!=raw.observation_id.tolist() or meta.observation_id.tolist()!=audit.observation_id.tolist():raise RuntimeError('Observaciones desalineadas')
        if not np.array_equal(y,audit.y.to_numpy(dtype=np.int8)):raise RuntimeError('y desalineada')
        if list(raw.columns)!=['observation_id']+FEATURES or not np.allclose(prep.transform(raw[FEATURES]),x,rtol=0,atol=1e-14):raise RuntimeError('X no corresponde al catálogo/preprocesador')
        if not (meta.month<'2016-01').all() or set(meta.partition)!={split}:raise RuntimeError('Partición/test incompatibles')
        if not (meta.history_known_at<meta.prediction_time).all() or not (meta.target_known_at<meta.label_cut_exclusive).all():raise RuntimeError('Disponibilidad temporal incompatible')
        dev[split]={'X':x,'y':y,'meta':meta,'raw':raw}
    if set(dev['train']['meta'].observation_id)&set(dev['val']['meta'].observation_id):raise RuntimeError('Claves compartidas entre splits')
    return dev,state


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',protected())
    verify_prior();dev,state=preflight()
    path=OUT/'configuracion_experimental.json'
    if path.exists() and read(path)!=PLAN:raise RuntimeError('Presupuesto ya fijado; no cambiar tras resultados')
    guardar(path,PLAN)
    guardar(OUT/'preflight.json',dict(status='successful',shapes={s:list(v['X'].shape) for s,v in dev.items()},
        dataset_manifest_sha256=sha(PREVIOUS/'dataset/manifiesto.json'),preprocessor_sha256=sha(DATA/'preprocesador.json'),
        feature_names=FEATURES,sql_queries=0,test_loaded=False,matrices_rebuilt=False,initialization_from_sales=False))
    (OUT/'presupuesto.md').write_text('''# Presupuesto fijado antes de entrenamiento

Cuatro ejecuciones: perceptrón eta=0,01, 100 épocas, semillas 42 y 7, permutación por época; logística eta=0,05 y 0,10, 400 épocas, seed=42, pesos cero y lote completo. Todas parten de pesos/sesgo cero; ninguna reutiliza pesos de Ventas. En logística la semilla no altera la inicialización cero. Una única eta del perceptrón evita una búsqueda redundante de escalas con inicialización cero.

Selección por balanced accuracy de validación completa, después F1, época más temprana y orden de experimento. Umbral logístico fijo 0,5; no se realiza búsqueda posterior. Perceptrón conserva z>=0. No se amplía el presupuesto por resultados. Parada logística por norma de gradiente <=1e-6 o 400 épocas; perceptrón ejecuta 100 según su clase actual. AP logística descriptiva, no selección.

El baseline aprobado requiere cuatro meses: 1[N(t-1)>mediana(N(t-2),N(t-3),N(t-4))]. No se sustituye por la fórmula candidata de tres meses del nuevo texto porque no coincide con el diseño aprobado. Se comparan los cuatro métodos en una misma intersección con cuarto lag disponible; se conservan además métricas de modelos y mayoría en todo el soporte. No se imputa historia para el baseline.

No hay acceso SQL, test 2016 o PCA. La configuración JSON es la fuente estructurada del presupuesto y se escribe antes de obtener resultados nuevos.
''',encoding='utf-8',newline='\n')
    print(json.dumps({'plan_fixed':True,'runs':4,'features':len(state['feature_names']),'config_sha256':sha(path)}))


def metrics(y,pred):
    return dict(metricas_binarias(y,pred),positive_prediction_proportion=float(np.mean(pred)))


def average_precision(y,scores):
    y=validar_etiquetas(y);scores=np.asarray(scores,dtype=np.float64)
    if scores.shape!=y.shape or not np.isfinite(scores).all():raise ValueError('Scores inválidos')
    positives=int(y.sum())
    if not positives:return None
    order=np.argsort(-scores,kind='stable');sorted_scores=scores[order];labels=y[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_scores)!=0),len(y)-1]
    tp=np.cumsum(labels)[ends];precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0,tp])/positives*precision))


def baseline_history(meta,monthly):
    lookup={(int(r.customer_id),r.month):(int(r.n),r.known_at) for r in monthly.itertuples()}
    mask=[];pred=[]
    for r in meta.itertuples():
        from src.datamarts.transacciones_viabilidad import add_month
        months=[add_month(r.month,-k) for k in [1,2,3,4]]
        available=ordinal(r.month)-ordinal(r.first_observed_month)>=4 and all((int(r.customer_id),m) in lookup for m in months)
        values=[lookup[int(r.customer_id),m] for m in months] if available else []
        available=available and all(k<r.prediction_time for n,k in values)
        mask.append(available);pred.append(int(values[0][0]>np.median([v[0] for v in values[1:]])) if available else -1)
    return np.asarray(mask,dtype=bool),np.asarray(pred,dtype=np.int8)


def fit_experiments(dev,plan,out):
    """Entrena solo train; validación se entrega a las clases para checkpoints."""
    runs=[];selected={};models={}
    for i,config in enumerate(plan['experiments']):
        algorithm=config['algorithm'];cls=PerceptronSimple if algorithm=='perceptron' else LogisticRegressionManual
        model=cls(**config['params'])
        model.fit(dev['train']['X'],dev['train']['y'],X_val=dev['val']['X'],y_val=dev['val']['y'],feature_names=FEATURES)
        path=out/'runs'/config['id'];path.mkdir(parents=True,exist_ok=True);model.save(path/'modelo.json')
        history=model.history_
        if not all(np.isfinite(v) for h in history for v in h.values() if isinstance(v,(float,int))):raise RuntimeError('Historial no finito')
        measured={s:metrics(v['y'],model.predict(v['X'],feature_names=FEATURES)) for s,v in dev.items()}
        last={}
        for s,v in dev.items():
            z=v['X']@model.last_weights_+model.last_bias_
            pred=(z>=0).astype(np.int8) if algorithm=='perceptron' else (sigmoid(z)>=model.threshold).astype(np.int8)
            last[s]=metrics(v['y'],pred)
        row=dict(id=config['id'],index=i,algorithm=algorithm,params=model.get_params(),selected_epoch=model.best_epoch_,
            metrics=measured,last_epoch_metrics=last,epochs_run=len(history),history=history,
            last_weights=model.last_weights_.tolist(),last_bias=model.last_bias_,
            termination='gradient_tolerance' if algorithm=='logistic' and model.converged_ else 'max_epochs')
        if algorithm=='logistic':
            row.update(initial_train_loss=model.initial_train_loss_,initial_val_loss=model.initial_val_loss_,
                       validation_average_precision=average_precision(dev['val']['y'],model.predict_proba(dev['val']['X'])))
        guardar(path/'historial.json',history)
        pd.DataFrame(history).to_csv(path/'historial.csv',index=False,lineterminator='\n',float_format='%.17g')
        guardar(path/'resumen.json',{k:v for k,v in row.items() if k!='history'})
        runs.append(row);models[config['id']]=model
        old=selected.get(algorithm)
        if old is None or clave_seleccion(measured['val'],model.best_epoch_,i)>clave_seleccion(old['metrics']['val'],old['selected_epoch'],old['index']):selected[algorithm]=row
    return runs,selected,{k:models[v['id']] for k,v in selected.items()}


def evaluate(dev,models):
    majority=int(np.count_nonzero(dev['train']['y']==1)>np.count_nonzero(dev['train']['y']==0))
    if majority!=0:raise RuntimeError('Mayoría de train difiere del contrato')
    monthly=pd.read_csv(DATA/'cuentas_meses.csv.gz',keep_default_na=False)
    full={};common={};by_month=[];predictions=[]
    for split,v in dev.items():
        mask,hist=baseline_history(v['meta'],monthly)
        preds={name:model.predict(v['X'],feature_names=FEATURES) for name,model in models.items()}
        preds['majority']=np.full(len(v['y']),majority,dtype=np.int8)
        full[split]={name:metrics(v['y'],p) for name,p in preds.items()}
        preds['historical']=hist
        common[split]=dict(n=int(mask.sum()),excluded_without_fourth_month=int((~mask).sum()),
                           metrics={name:metrics(v['y'][mask],p[mask]) for name,p in preds.items()})
        frame=v['meta'][['observation_id','customer_id','month']].copy();frame['partition']=split;frame['y']=v['y'];frame['historical_available']=mask
        for name,p in preds.items():frame[name]=[int(n) if n>=0 else None for n in p]
        frame['perceptron_margin']=models['perceptron'].decision_function(v['X'])
        frame['logistic_probability']=models['logistic'].predict_proba(v['X'])
        predictions.append(frame)
        for month in sorted(set(v['meta'].month)):
            for support,scope in [('full',np.ones(len(mask),dtype=bool)),('common_four_months',mask)]:
                local=scope&(v['meta'].month.to_numpy()==month)
                if not local.any():continue
                for name,p in preds.items():
                    if name=='historical' and support=='full':continue
                    by_month.append(dict(split=split,month=month,support=support,method=name,metrics=metrics(v['y'][local],p[local])))
    return dict(majority_train_class=majority,full_support=full,common_four_month_support=common,
                historical_formula=PLAN['baselines']['history'],validation_logistic_average_precision=average_precision(dev['val']['y'],models['logistic'].predict_proba(dev['val']['X']))),by_month,pd.concat(predictions,ignore_index=True)


def convergence(row):
    h=row['history'];last=h[-1];selected=h[row['selected_epoch']-1]
    base=dict(run=row['id'],epochs_run=len(h),selected_epoch=row['selected_epoch'],termination=row['termination'],
              selected_checkpoint_metrics=row['metrics'],last_epoch_metrics=row['last_epoch_metrics'],last_weights_norm=last['weights_norm'])
    if row['algorithm']=='perceptron':
        errors=np.array([r['online_errors'] for r in h]);base.update(first_online_errors=int(errors[0]),last_online_errors=int(errors[-1]),
            selected_online_errors=selected['online_errors'],last_end_epoch_train_errors=last['train_errors_end'],
            updates_total=int(errors.sum()),zero_error_epochs=[r['epoch'] for r in h if r['online_errors']==0],
            error_increase_steps=int(np.sum(np.diff(errors)>0)),online_error_min=int(errors.min()),online_error_max=int(errors.max()),
            perfect_convergence_observed=bool(errors[-1]==0))
    else:
        losses=np.r_[row['initial_train_loss'],[r['train_loss'] for r in h]];deltas=np.diff(losses)
        base.update(initial_train_loss=row['initial_train_loss'],initial_val_loss=row['initial_val_loss'],last_train_loss=last['train_loss'],
            last_val_loss=last['val_loss'],selected_train_loss=selected['train_loss'],selected_val_loss=selected['val_loss'],
            loss_increases=int(np.sum(deltas>1e-12)),last_loss_delta=float(deltas[-1]),last_gradient_norm=last['gradient_norm'],
            selected_gradient_norm=selected['gradient_norm'],gradient_tolerance_reached=row['termination']=='gradient_tolerance',
            selected_train_validation_loss_gap=selected['val_loss']-selected['train_loss'])
    return base


def plots(selected,comparison,out):
    from src.evaluation.graficos import graficar_perceptron,plt,PercentFormatter
    from src.evaluation.graficos_logistica import graficar_logistica
    root=out/'graficos';root.mkdir(exist_ok=True)
    p=selected['perceptron'];l=selected['logistic']
    graficar_perceptron(p['history'],p['metrics']['val'],p['selected_epoch'],16,root/'perceptron')
    graficar_logistica(l['history'],l['metrics']['val'],l['selected_epoch'],.5,16,root/'logistica')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.plot([h['epoch'] for h in p['history']],[h['updates'] for h in p['history']],color='#236b8e')
    ax.axvline(p['selected_epoch'],color='#b86e19',linestyle='--',label='Checkpoint seleccionado')
    ax.set(title='Transacciones: actualizaciones del perceptrón',xlabel='Época',ylabel='Actualizaciones');ax.grid(alpha=.2);ax.legend()
    fig.savefig(root/'perceptron/actualizaciones.png',dpi=170,metadata={'Software':'PerceptronReal Transacciones'});plt.close(fig)
    scores=comparison['common_four_month_support']['val']['metrics'];methods=list(scores)
    for filename,keys,title in [('comparacion_metricas.png',['accuracy','balanced_accuracy','precision','recall','f1'],'Métricas de validación: soporte común con cuatro meses'),
                                ('sensibilidad_especificidad.png',['recall','specificity'],'Sensibilidad y especificidad: soporte común')]:
        fig,ax=plt.subplots(figsize=(11,5),layout='constrained');x=np.arange(len(methods));width=.8/len(keys)
        for j,key in enumerate(keys):ax.bar(x+(j-(len(keys)-1)/2)*width,[scores[n][key] for n in methods],width,label=key)
        ax.set(xticks=x,xticklabels=methods,ylim=(0,1),title=title,xlabel='Método',ylabel='Proporción')
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncols=3,fontsize=9);ax.grid(axis='y',alpha=.2)
        fig.savefig(root/filename,dpi=170,metadata={'Software':'PerceptronReal Transacciones'});plt.close(fig)


def freeze(selected,models,comparison):
    paths={};(OUT/'modelos').mkdir(exist_ok=True)
    for name,model in models.items():
        path=OUT/'modelos'/f'{name}.json';model.save(path);paths[name]=str(path.relative_to(BASE))
        loaded=(PerceptronSimple if name=='perceptron' else LogisticRegressionManual).load(path)
        if loaded.feature_names_!=FEATURES or len(loaded.weights_)!=16:raise RuntimeError('Modelo persistido incompatible')
    sources={str(p.relative_to(BASE)):sha(p) for p in DATA.glob('*') if p.is_file()}
    guardar(OUT/'congelacion.json',dict(status='frozen_development_only_test_not_evaluated',configuration_sha256=sha(OUT/'configuracion_experimental.json'),
        feature_names=FEATURES,n_features=16,preprocessor=read(DATA/'preprocesador.json'),preprocessor_sha256=sha(DATA/'preprocesador.json'),
        source_artifacts_sha256=sources,models={name:dict(path=path,sha256=sha(BASE/path),selected_run=selected[name]['id'],
            selected_epoch=selected[name]['selected_epoch'],params=models[name].get_params(),validation_metrics=selected[name]['metrics']['val']) for name,path in paths.items()},
        baselines=PLAN['baselines'],majority_class=comparison['majority_train_class'],thresholds={'perceptron':'z>=0','logistic':.5},
        test_2016_loaded=False,training_with_validation=False,retraining_train_plus_validation=False,
        future_evaluation='Apply frozen models/preprocessor and compare all methods on identical four-month intersection; report model full support separately'))


def reports(runs,selected,comparison,convergence_data):
    chosen='\n'.join(f"| {name} | {r['id']} | {r['params']['learning_rate']} | {r['params']['seed']} | {r['selected_epoch']} |" for name,r in selected.items())
    table=[]
    for split,values in comparison['full_support'].items():
        for method,m in values.items():table.append(f"| {split} | {method} | {m['n']} | {m['accuracy']:.6f} | {m['balanced_accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['specificity']:.6f} |")
    common=[]
    for split,v in comparison['common_four_month_support'].items():
        for name,m in v['metrics'].items():common.append(f"| {split} | {name} | {m['n']} | {m['accuracy']:.6f} | {m['balanced_accuracy']:.6f} | {m['f1']:.6f} |")
    text='''# Entrenamiento financiero reproducible

Se entrenan las clases matemáticas propias sin modificarlas, con pesos/sesgo nuevos a cero, 16 columnas congeladas y fit exclusivamente en train. Los checkpoints y la configuración ganadora de cada algoritmo se seleccionan solo con validación completa por BA, F1, menor época y orden predefinido. No hay SQL, reconstrucción del dataset, apertura de test, PCA o cambio del informe. Los supuestos académicos de cobertura y conservación del libro permanecen vigentes: esto mide frecuencia de cobros registrados y no demuestra generalización bancaria o utilidad monetaria.

## Presupuesto y selección

Antes de resultados se fijaron dos perceptrones (eta 0,01, 100 épocas, semillas 42/7) y dos logísticas (eta 0,05/0,10, 400 épocas, semilla 42, inicialización cero, tolerancia de norma de gradiente 1e-6). Sin ampliación adaptativa ni selección de umbral: logística usa 0,5 y perceptrón z>=0. La semilla logística se conserva por trazabilidad pero no cambia pesos cero. Los modelos de Ventas no se cargan ni se transfieren.

| Algoritmo | Ejecución elegida | Learning rate | Semilla | Época |
|---|---|---:|---:|---:|
'''+chosen+'''

## Métricas en soporte completo

| Partición | Método | n | Accuracy | BA | Precision | Recall | F1 | Especificidad |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''+'\n'.join(table)+'''

La mayoría se verifica solo con y_train y es 0. El histórico aprobado requiere cuarto mes: 1[N(t-1)>mediana(N(t-2),N(t-3),N(t-4))]. No se sustituye por 1[N(t-1)>R(t)], que usa otra referencia. No se imputa el cuarto lag ni se mezclan soportes al comparar:

| Partición | Método | n común | Accuracy | BA | F1 |
|---|---|---:|---:|---:|---:|
'''+'\n'.join(common)+f'''

Soporte perdido por falta de cuarto mes/control de disponibilidad: train={comparison['common_four_month_support']['train']['excluded_without_fourth_month']}, val={comparison['common_four_month_support']['val']['excluded_without_fourth_month']}. Los cuatro métodos se evalúan en idénticas claves dentro de esa intersección; selección de checkpoints se mantiene en validación completa. métricas.json conserva matrices de confusión [0,1], soporte, proporción de predicciones positivas y convenciones de denominador cero. métricas_mensuales.json desglosa ambos soportes, sin ocultar subconjuntos monoclase.

Average Precision de logística en validación completa: **{comparison['validation_logistic_average_precision']:.6f}**. Se calcula desde probabilidades sigmoid con grupos de scores empatados y sin interpolación; no participa en selección. El margen del perceptrón no se llama probabilidad calibrada. Tampoco se asume calibración perfecta de logística.

## Artefactos y reproducción

runs/ conserva los cuatro historiales completos JSON/CSV, parámetros, métricas seleccionadas y última época, además de cada modelo candidato. modelos/ guarda los dos checkpoints seleccionados y congelacion.json sus hashes, umbrales, nombres y preprocesador. predicciones_desarrollo.csv permite comprobar alineación y la ausencia explícita del histórico sin cuarto lag. Se exportan 12 gráficos reales: curvas de errores/actualizaciones/accuracy/BA y confusión del perceptrón; pérdidas/gradiente/accuracy/BA y confusión de logística; dos comparaciones en soporte común. Las funciones de gráficos existentes se reutilizan; su metadato Software histórico se conserva sin alterar Ventas.

```powershell
python -m src.experiments.entrenar_transacciones --preparar
python -m src.experiments.entrenar_transacciones --entrenar
python -m unittest discover -s tests -p test_entrenamiento_transacciones.py -v
python -m unittest discover -s tests -v
```

Preparar verifica los hashes y fija el plan antes de entrenar; entrenar rechaza ausencia o cambio del presupuesto. No se reescala, reimputa o regenera el dataset. El preflight aplica en memoria el estado ya congelado para comprobar X, sin fit o escritura de matrices. La reproducción de entrenamiento se compara con pesos, historiales y métricas en el mismo entorno numérico registrado, sin añadir nuevas configuraciones al presupuesto.

La validación fue utilizada repetidamente para seleccionar checkpoint: sus resultados son de desarrollo y no estimadores independientes del desempeño final. Comparar con los baselines no certifica significación, valor comercial o independencia entre cuentas/meses. La evaluación futura sigue sellada y no se inicia automáticamente.
'''
    (OUT/'reporte_entrenamiento.md').write_text(text,encoding='utf-8',newline='\n')
    lines=['# Convergencia observada\n']
    for name,r in selected.items():
        c=convergence_data[r['id']]
        lines.append(f"## {name}: {r['id']}\n\nCheckpoint: época {c['selected_epoch']}; última época: {c['epochs_run']}; terminación: {c['termination']}.")
        if name=='perceptron':lines.append(f"Errores online inicial/final: {c['first_online_errors']}/{c['last_online_errors']}; mínimo/máximo: {c['online_error_min']}/{c['online_error_max']}; actualizaciones totales: {c['updates_total']}. Aumentos entre épocas: {c['error_increase_steps']}; épocas sin errores: {c['zero_error_epochs']}. Errores train al final de la última época: {c['last_end_epoch_train_errors']}. Actualizar parámetros al final impide declarar convergencia perfecta. Los errores online y los errores al cierre de época son medidas distintas.")
        else:lines.append(f"Pérdida train inicial/final: {c['initial_train_loss']:.9f}/{c['last_train_loss']:.9f}; val inicial/final: {c['initial_val_loss']:.9f}/{c['last_val_loss']:.9f}. Incrementos de pérdida train (>1e-12): {c['loss_increases']}; último cambio: {c['last_loss_delta']:.9g}; gradiente final: {c['last_gradient_norm']:.9g}; tolerancia alcanzada: {c['gradient_tolerance_reached']}. Pérdidas del checkpoint train/val: {c['selected_train_loss']:.9f}/{c['selected_val_loss']:.9f}. Descenso de pérdida no demuestra convergencia completa si no se alcanza el criterio. El gradiente final corresponde a la última época, no necesariamente al checkpoint restaurado.")
        lines.append(f"BA train/val seleccionada: {r['metrics']['train']['balanced_accuracy']:.6f}/{r['metrics']['val']['balanced_accuracy']:.6f}. BA train/val de última época: {r['last_epoch_metrics']['train']['balanced_accuracy']:.6f}/{r['last_epoch_metrics']['val']['balanced_accuracy']:.6f}. Norma final de pesos: {c['last_weights_norm']:.9f}. Se conservan por separado ambas épocas; no se presenta la última como la elegida.\n")
    lines.append('Los cuatro historiales y sus diagnósticos están en convergencia.json. No se demuestra separabilidad lineal ni falta de ella con estos ensayos; no hay ampliación del presupuesto por oscilaciones o por llegar al máximo de épocas.\n')
    (OUT/'reporte_convergencia.md').write_text('\n\n'.join(lines),encoding='utf-8',newline='\n')


def protocol():
    text='''# Protocolo financiero futuro de test, aún sellado

No ejecutar en este encargo. Ventana predefinida: enero-mayo de 2016 según cobertura ya documentada, intervalo de eventos [2016-01-01,2016-06-01), corte propuesto de conocimiento de etiquetas exclusivo 2016-06-01 00:00. Confirmar cierre de cada mes; un mes incompleto o de cobertura no confirmada queda indeterminado, nunca cero automático ni ampliación de ventana por resultados.

Antes de acceder: verificar todos los hashes de congelacion.json, el dataset/contrato, modelos seleccionados, umbral 0,5, catálogo/orden y preprocesador. Congelar baselines y este protocolo. No seleccionar ni reajustar nada con test, ni entrenar con train+val antes de comparar estos modelos ya fijados.

Extraer por SELECT únicamente CustomerTransactions tipo 3 con la proyección mínima aprobada, IDs únicos y CustomerID operacional. Cotejar IDs/fechas con DW sin agrupar miembros desconocidos ni multiplicar eventos por joins. Conservar snapshot/procedencia/huellas de lectura. Mantener los supuestos académicos explícitos de conservación y cobertura del libro; última edición no es fecha de creación ni prueba absoluta de completitud. No usar saldos, finalización o importes como X.

Concatenar desarrollo congelado con eventos ya concluidos y conocidos antes de cada t. Construir X al inicio del mes a partir de t-1,t-2,t-3, calendario y primera observación anterior al instante; no acceder a pagos de t para decidir X o elegibilidad. Cuentas nuevas requieren tres meses completos desde primera observación; cuentas sin historia se registran como indeterminadas. No seleccionar solo cuentas con pagos en t o con supervivencia futura. Aplicar únicamente transform del preprocesador congelado; no fit, nuevas constantes, nuevas columnas o cambio de escala. Obtener y después del cierre de t, respetando corte y disponibilidad.

Predecir con pesos/sesgos congelados: perceptrón z>=0 y logística sigmoid(z)>=0,5. Mayoritaria fija=0. Baseline histórico contractual=1[N(t-1)>mediana(N(t-2),N(t-3),N(t-4))], con cuarto mes cubierto y conocido; ausencia de ese lag no se imputa. Comparar los cuatro métodos sobre idéntica intersección de claves, informando además resultados completos de los modelos y la mayoría y soporte excluido.

Reportar accuracy, balanced accuracy, precision, recall, F1, especificidad, matriz [0,1], soporte y proporción de predicciones positivas globales y por mes. AP logística desde scores positivos con misma convención de empates. No interpretar márgenes del perceptrón como probabilidades. Mantener convenciones para denominador cero y meses monoclase. Separar cuentas vistas en train, vistas solo en validación y nuevas del test cuando el soporte lo permita, sin usar los grupos para filtrar entrenamiento. Reconocer dependencia entre meses/cuentas y límites de la simulación; no inventar costes financieros o significación.

Conservar predicciones, claves alineadas, hashes y registro de apertura. Ningún resultado de test puede alterar modelos, umbral, cobertura, ventanas, features o elección de mejor modelo; un nuevo diseño requeriría otro experimento explícito. Este archivo y especificacion_test.json del dataset son instrucciones, no una ejecución ni una autorización de acceso actual.
'''
    (OUT/'protocolo_test.md').write_text(text,encoding='utf-8',newline='\n')


def manifest():
    guardar(OUT/'manifiesto.json',dict(implementation_sha256={n:sha(BASE/n) for n in sorted(NEW) if (BASE/n).exists()},
        artifacts_sha256={str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['manifiesto.json','validaciones.json','reproducibilidad.json']},
        classifier_implementation_sha256={str(p.relative_to(BASE)):sha(p) for p in [BASE/'src/models/perceptron.py',BASE/'src/models/logistic_regression.py',BASE/'src/evaluation/metricas.py']}))


def train():
    if not (OUT/'configuracion_experimental.json').exists() or read(OUT/'configuracion_experimental.json')!=PLAN:raise RuntimeError('Ejecutar --preparar y respetar el presupuesto fijado antes de resultados')
    verify_prior();dev,state=preflight();runs,selected,models=fit_experiments(dev,PLAN,OUT)
    comparison,monthly,preds=evaluate(dev,models)
    guardar(OUT/'experimentos.json',runs);guardar(OUT/'metricas.json',comparison);guardar(OUT/'metricas_mensuales.json',monthly)
    preds.to_csv(OUT/'predicciones_desarrollo.csv',index=False,lineterminator='\n',float_format='%.17g',na_rep='')
    convergence_data={r['id']:convergence(r) for r in runs};guardar(OUT/'convergencia.json',convergence_data)
    plots(selected,comparison,OUT);freeze(selected,models,comparison);reports(runs,selected,comparison,convergence_data);protocol()
    guardar(OUT/'entorno.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,executions=4,models_reused_without_changes=True))
    guardar(OUT/'integridad_despues.json',verify_prior());manifest()
    print(json.dumps(dict(selected={k:dict(id=r['id'],epoch=r['selected_epoch'],train=r['metrics']['train'],val=r['metrics']['val']) for k,r in selected.items()},
        common=comparison['common_four_month_support'],AP=comparison['validation_logistic_average_precision'],figures=len(list((OUT/'graficos').rglob('*.png')))),ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--preparar',action='store_true');g.add_argument('--entrenar',action='store_true');args=p.parse_args()
    prepare() if args.preparar else train()
