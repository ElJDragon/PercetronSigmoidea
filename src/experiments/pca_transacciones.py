"""Conciliación financiera y PCA train-only, sin entrenar clasificadores o abrir test."""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.auditoria_dw import BASE,guardar,sha
from src.experiments import entrenar_transacciones as training
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual
from src.visualization.pca import PCA2D
from src.datamarts.transacciones.caracteristicas import FEATURES,ordinal
from src.datamarts.transacciones_viabilidad import add_month

DATA=training.DATA
PREVIOUS=training.OUT
OUT=training.PREVIOUS/'pca'
NEW={'src/experiments/pca_transacciones.py','tests/test_pca_transacciones.py'}
EXPECTED_MODELS={'perceptron':'19aca080e0e38d007dfb6b8e4050c7b371a5b7665e93bcc6a7463aa70b5109b0',
                 'logistic':'11e86463138a284caa4cac90f4ae10fa4448b6cccb5d2a4d8f6c9070d4e8f726'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def snapshot():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
            for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents
            and str(p.relative_to(BASE)).replace('\\','/') not in NEW}


def verify_prior():
    previous=read(OUT/'integridad_antes.json')
    if any(sha(BASE/n)!=h for n,h in previous.items()):raise RuntimeError('Cambió un artefacto anterior al PCA')
    return previous


def verify_models(dev):
    manifest=read(PREVIOUS/'manifiesto.json')
    for section in ['implementation_sha256','artifacts_sha256','classifier_implementation_sha256']:
        for path,h in manifest[section].items():
            if sha(BASE/path)!=h:raise RuntimeError('Congelación de entrenamiento alterada: '+path)
    frozen=read(PREVIOUS/'congelacion.json');models={};details={}
    for name,cls,epoch in [('perceptron',PerceptronSimple,3),('logistic',LogisticRegressionManual,399)]:
        record=frozen['models'][name];path=BASE/record['path']
        if sha(path)!=EXPECTED_MODELS[name] or record['sha256']!=EXPECTED_MODELS[name]:raise RuntimeError('SHA del modelo no coincide con el aprobado')
        model=cls.load(path)
        if model.best_epoch_!=epoch or model.feature_names_!=FEATURES or model.weights_.shape!=(16,) or not np.isfinite(model.weights_).all() or not np.isfinite(model.bias_):raise RuntimeError('Checkpoint o dimensiones incorrectas')
        if name=='logistic' and model.threshold!=.5:raise RuntimeError('Umbral alterado')
        if name=='perceptron' and (model.learning_rate!=.01 or model.seed!=7):raise RuntimeError('Configuración de perceptrón alterada')
        if name=='logistic' and model.learning_rate!=.1:raise RuntimeError('Configuración logística alterada')
        models[name]=model
        details[name]=dict(sha256=sha(path),weights=model.weights_.tolist(),bias=model.bias_,selected_epoch=model.best_epoch_,
            feature_names=model.feature_names_,n_features=16,params=model.get_params(),classification_input='frozen 16-dimensional X, never PCA projections')
    comparison,monthly,predicted=training.evaluate(dev,models)
    if comparison!=read(PREVIOUS/'metricas.json') or monthly!=read(PREVIOUS/'metricas_mensuales.json'):raise RuntimeError('Métricas de desarrollo no reproducibles')
    saved=pd.read_csv(PREVIOUS/'predicciones_desarrollo.csv',keep_default_na=False,float_precision='round_trip')
    if predicted.observation_id.tolist()!=saved.observation_id.tolist():raise RuntimeError('Claves de predicción desalineadas')
    for key in ['y','perceptron','logistic','majority','historical_available']:
        if not np.array_equal(predicted[key].to_numpy(),saved[key].to_numpy()):raise RuntimeError('Predicciones no reproducibles: '+key)
    for key in ['perceptron_margin','logistic_probability']:
        if not np.array_equal(predicted[key].to_numpy(),saved[key].to_numpy(dtype=np.float64)):raise RuntimeError('Scores no reproducibles: '+key)
    historical=pd.to_numeric(saved.historical,errors='coerce').to_numpy(dtype=float)
    if not np.array_equal(predicted.historical.to_numpy(dtype=float),historical,equal_nan=True):raise RuntimeError('Baseline histórico desalineado')
    return models,comparison,dict(status='successful',models=details,predictions_reproduced=True,scores_reproduced=True,
        metrics_full_common_monthly_reproduced=True,classifier_training_runs=0,sql_queries=0,test_loaded=False)


def reconcile(dev):
    meta=dev['val']['meta'];labels=pd.read_csv(DATA/'etiquetas_val.csv')
    saved=pd.read_csv(PREVIOUS/'predicciones_desarrollo.csv',keep_default_na=False,float_precision='round_trip')
    saved=saved[saved.partition=='val'].reset_index(drop=True)
    if meta.observation_id.tolist()!=saved.observation_id.tolist() or meta.observation_id.tolist()!=labels.observation_id.tolist():raise RuntimeError('Alineación inválida en conciliación')
    monthly=pd.read_csv(DATA/'cuentas_meses.csv.gz',keep_default_na=False)
    eligibility=pd.read_csv(DATA/'elegibilidad.csv.gz')
    mask,hist=training.baseline_history(meta,monthly)
    if not np.array_equal(mask,saved.historical_available.to_numpy()):raise RuntimeError('La regla histórica no explica el subconjunto congelado')
    excluded=meta.loc[~mask].copy()
    excluded=excluded.merge(labels,on='observation_id',validate='one_to_one',sort=False)
    excluded=excluded.merge(saved[['observation_id','perceptron','logistic','majority','historical']],on='observation_id',validate='one_to_one',sort=False)
    excluded=excluded.merge(eligibility[['observation_id','state']],on='observation_id',validate='one_to_one',sort=False)
    excluded['age_months']=excluded.apply(lambda r:ordinal(r['month'])-ordinal(r['first_observed_month']),axis=1)
    excluded['fourth_month']=excluded.month.map(lambda m:add_month(m,-4))
    available_keys=set(zip(monthly.customer_id,monthly.month))
    excluded['fourth_month_in_population']=[(int(c),m) in available_keys for c,m in zip(excluded.customer_id,excluded.fourth_month)]
    excluded['label_valid']=(excluded.y== (excluded.target_count>excluded.historical_reference).astype(np.int8))
    for name in ['perceptron','logistic','majority']:excluded[name+'_prediction_valid']=excluded[name].isin([0,1])
    excluded['historical_defined']=excluded.historical!=''
    excluded['exclusion_reason']='fourth_month_precedes_first_observed_payment; no imputation before appearance'
    consistent=(len(meta)==2943 and int(mask.sum())==2927 and len(excluded)==16
        and bool((excluded.age_months==3).all()) and bool((excluded.state=='eligible').all())
        and bool(excluded.label_valid.all()) and not bool(excluded.fourth_month_in_population.any())
        and not bool(excluded.historical_defined.any())
        and all(bool(excluded[name+'_prediction_valid'].all()) for name in ['perceptron','logistic','majority']))
    report=dict(status='consistent' if consistent else 'blocked_discrepancy',primary_validation=2943,common_validation=int(mask.sum()),
        excluded=len(excluded),excluded_positive=int(excluded.y.sum()),excluded_negative=int(len(excluded)-excluded.y.sum()),
        excluded_by_month=excluded.groupby('month').size().to_dict(),
        all_exclusions_only_fourth_history_missing=consistent,
        primary_population='all eligible classifier rows with 3 prior covered months and valid label; n=2943',
        common_population='secondary comparison of all four methods where 4 prior months are covered and known; n=2927',
        historical_formula=training.PLAN['baselines']['history'],selection_by_model_errors=False,
        previous_predictions_modified=False,alignment_discrepancy_detected=not consistent)
    return report,excluded


def mathematical_checks(pca,x,z):
    centered=x-pca.mean_;cov=np.cov(centered,rowvar=False,ddof=1)
    eigenvalues=np.linalg.eigvalsh(cov)[::-1]
    projected_cov=np.cov(z,rowvar=False,ddof=1)
    reconstruction=z@pca.components_.T
    residual=float(np.sum((centered-reconstruction)**2)/(len(x)-1))
    discarded=float(pca.explained_variance_all_[2:].sum())
    checks=dict(orthogonality_max_error=float(np.max(np.abs(pca.components_.T@pca.components_-np.eye(2)))),
        total_variance=pca.total_variance_,covariance_trace=float(np.trace(cov)),ratio_sum=float(pca.explained_variance_ratio_all_.sum()),
        projected_variance=z.var(axis=0,ddof=1).tolist(),sample_variance_ddof=1,
        reconstruction_discarded_variance=residual,spectral_discarded_variance=discarded,
        eigenvalues_vs_svd_max_error=float(np.max(np.abs(eigenvalues-pca.explained_variance_all_))),
        train_projected_mean_max_abs=float(np.max(np.abs(z.mean(axis=0)))),rank=pca.rank_,n_features=16)
    np.testing.assert_allclose(pca.components_.T@pca.components_,np.eye(2),rtol=0,atol=1e-12)
    np.testing.assert_allclose(eigenvalues,pca.explained_variance_all_,rtol=0,atol=1e-12)
    np.testing.assert_allclose(projected_cov,np.diag(pca.explained_variance_),rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.trace(cov),pca.total_variance_,rtol=0,atol=1e-12)
    np.testing.assert_allclose(residual,discarded,rtol=0,atol=1e-12)
    np.testing.assert_allclose(pca.explained_variance_ratio_all_.sum(),1,rtol=0,atol=1e-12)
    checks['status']='successful'
    return checks


def fit_pca(train_x,val_x):
    # La interfaz no recibe y: etiquetas no intervienen en media/SVD/componentes.
    pca=PCA2D().fit(train_x,feature_names=FEATURES)
    return pca,pca.transform(train_x,feature_names=FEATURES),pca.transform(val_x,feature_names=FEATURES)


def load_projection(split):
    if split not in ['train','val']:raise ValueError('Solo proyecciones de desarrollo')
    return pd.read_csv(OUT/f'proyeccion_{split}.csv',float_precision='round_trip')[['PC1','PC2']].to_numpy(dtype=np.float64).copy(order='C')


def verify_convergence():
    runs=read(PREVIOUS/'experimentos.json');approved=read(PREVIOUS/'convergencia.json')
    reconstructed={r['id']:training.convergence(r) for r in runs}
    if reconstructed!=approved:raise RuntimeError('Los historiales no reproducen el diagnóstico de convergencia')
    frozen=read(PREVIOUS/'congelacion.json')
    chosen={name:reconstructed[frozen['models'][name]['selected_run']] for name in ['perceptron','logistic']}
    p=chosen['perceptron'];l=chosen['logistic']
    if p['selected_epoch']!=3 or p['last_online_errors']<=0 or p['zero_error_epochs']:raise RuntimeError('Convergencia de perceptrón no coincide con la congelación')
    if l['selected_epoch']!=399 or l['termination']!='max_epochs' or l['gradient_tolerance_reached'] or not l['last_train_loss']<l['initial_train_loss']:raise RuntimeError('Convergencia logística no coincide con el historial')
    return chosen


def figures(pca,projections,labels,out):
    from src.evaluation.graficos import plt,PercentFormatter
    root=out/'graficos';root.mkdir(exist_ok=True)
    ratios=pca.explained_variance_ratio_*100
    all_z=np.vstack(list(projections.values()));minimum=all_z.min(axis=0);maximum=all_z.max(axis=0)
    margin=np.maximum(.05*(maximum-minimum),.1)
    for split,z in projections.items():
        fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
        # Todos los puntos, sin jitter ni descarte por clase; transparencia para solapamiento.
        for cls,color,label in [(0,'#246aa0','0: frecuencia no superior'),(1,'#d66b26','1: frecuencia superior')]:
            mask=labels[split]==cls
            ax.scatter(z[mask,0],z[mask,1],s=13,alpha=.38,c=color,edgecolors='none',label=f'{label} (n={int(mask.sum())})',zorder=2)
        ax.set(xlabel=f'PC1 ({ratios[0]:.2f}% varianza train)',ylabel=f'PC2 ({ratios[1]:.2f}% varianza train)',
               title=f'Transacciones: {"entrenamiento" if split=="train" else "validación"} por clase\nPCA ajustado exclusivamente con train; representación visual',
               xlim=(minimum[0]-margin[0],maximum[0]+margin[0]),ylim=(minimum[1]-margin[1],maximum[1]+margin[1]))
        ax.legend(fontsize=9);ax.grid(alpha=.18)
        fig.savefig(root/f'dispersion_{split}.png',dpi=220,metadata={'Software':'PerceptronReal Transacciones PCA'});plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');index=np.arange(1,17)
    ax.bar(index,pca.explained_variance_ratio_all_,color='#246aa0',label='Varianza individual (train)')
    ax.plot(index,np.cumsum(pca.explained_variance_ratio_all_),marker='o',color='#d66b26',label='Acumulada (train)')
    ax.axvline(2,color='#665275',linestyle='--',label=f'Dos PC: {ratios.sum():.2f}%')
    ax.set(title='Transacciones: varianza explicada del PCA de entrenamiento',xlabel='Componente principal',ylabel='Proporción de varianza',xticks=index,ylim=(0,1.03))
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.grid(axis='y',alpha=.18)
    fig.savefig(root/'varianza_explicada.png',dpi=220,metadata={'Software':'PerceptronReal Transacciones PCA'});plt.close(fig)


def protocol():
    return dict(status='frozen_before_first_test_opening',test_executed=False,sql_queries_now=0,
        planned_months=['2016-01','2016-02','2016-03','2016-04','2016-05'],event_interval_inclusive_exclusive=['2016-01-01','2016-06-01'],
        proposed_label_knowledge_cut_exclusive='2016-06-01 00:00:00',
        prior_protocol_sha256=sha(PREVIOUS/'protocolo_test.md'),prior_test_spec_sha256=sha(training.PREVIOUS/'dataset/especificacion_test.json'),
        primary_population='All eligible account-month rows: 3 consecutive prior covered months, stable operational identity and target label known by cut; no fourth-lag requirement',
        primary_methods=['perceptron','logistic','majority'],secondary_population='Exact intersection where contractual 4-month historical baseline is defined',
        secondary_methods=['perceptron','logistic','majority','historical'],
        events='Only Customer Payment Received type 3, operational CustomerTransactionID unique; minimal approved projection and OLTP-DW bijection, read-only extraction',
        identity='CustomerTransactions.CustomerID (Bill To), never substitute unknown DW keys or current names',
        first_seen='First payment observable before prediction, no rows before appearance and no selection by future activity/survival',
        history='N(t-1),N(t-2),N(t-3) and known calendar at start(t); development frozen plus prior test months already closed, never t for X',
        coverage='Explicit business_ledger_academic and documented conservation assumptions; unknown/incomplete months indeterminate, never silent zeros',
        y='1[N(t)>median(N(t-1),N(t-2),N(t-3))], ties 0; only after complete month and before cut; never select by prediction/error',
        preprocessing='Frozen 16-column preprocessor: transform only, no fit, new constants, scaling or feature selection',
        classifiers='Frozen perceptron epoch 3 seed 7 eta .01 z>=0; frozen logistic epoch 399 eta .10 threshold .5; no refit train+val',
        pca='Visualization only, not classifier input; never fit on validation/test',
        majority='Constant 0 from train only',historical='1[N(t-1)>median(N(t-2),N(t-3),N(t-4))]',
        historical_undefined='Keep row in primary evaluation if otherwise eligible; mark historical prediction missing; no imputation or classifier/majority exclusion from primary',
        exclusions='Log stable ID, month, first appearance, primary eligibility, label availability, coverage state, fourth-month availability and separate reasons. Never overwrite old predictions.',
        metrics='Full frozen conventions: accuracy, BA, precision, recall, F1, specificity, confusion [0,1], class support and predicted-positive rate; logistic AP from scores with grouped ties',
        comparisons='Global and monthly, with exact stable-key equality for all methods within each named support. Do not compare n of one support with n of another.',
        generalization='Report accounts seen in train, seen only in validation, and new eligible in test when support permits; no filtering based on these future outcomes',
        before_opening=['Check all original dataset/model/implementation hashes and new manifest/protocol','Check thresholds, checkpoint, feature order and preprocessor hashes','Record date and scope of first opening and preserve snapshot/provenance','Execute only after an explicit subsequent evaluation request'],
        after_opening='No changes of model, threshold, label, horizon, coverage, support rules or baselines in response to outcomes; limitations remain documented')


def reports(reconciliation,excluded,pca,variance,verification):
    columns=['observation_id','month','first_observed_month','fourth_month','y','perceptron','logistic','majority']
    table='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    table+='\n'.join('| '+' | '.join(str(row[c]) for c in columns)+' |' for _,row in excluded.iterrows())
    text=f'''# Conciliación de validación financiera

**Dictamen: evaluación consistente; no hay discrepancia de alineación.** La población principal de clasificadores y mayoría contiene 2.943 observaciones elegibles con tres meses históricos completos y etiqueta válida. La comparación de los cuatro métodos contiene 2.927: el baseline histórico requiere un cuarto mes, según el diseño fijado antes del entrenamiento. La diferencia es exclusivamente disponibilidad del baseline, no falta de etiqueta ni error de los clasificadores.

Las 16 observaciones tienen antigüedad observada de tres meses, estado eligible, etiqueta válida (7 positivas, 9 negativas), predicción de perceptrón/logística/mayoría y ningún cuarto mes anterior a su primera aparición en la población. No se rellena ese mes con cero ni se fabrica historia previa a la primera observación. No se filtró por acierto, error o clase. Reconciliación reconstruida desde metadatos, elegibilidad, etiquetas, mensualidad de cobros y predicciones mediante lecturas/joins uno a uno; sin tocar los archivos originales.

{table}

La lista completa con estados, cortes, conteo objetivo, referencia y predicciones está en observaciones_excluidas.csv. Distribución por mes: {json.dumps(reconciliation['excluded_by_month'],ensure_ascii=False)}. historical_defined=false en todas; la regla exacta es edad>=4 más cuatro meses cubiertos y conocidos antes de inicio(t). Aquí edad=3 y t-4 precede primera aparición. No hay otra limitación de disponibilidad para estas filas.

Evaluación principal original: perceptrón BA=66,782699%, logística BA=62,081956%, n=2.943. Comparación secundaria común: perceptrón BA=66,840058%, logística BA=62,114500%, mayoría=50%, histórico=47,788231%, n=2.927. Las pequeñas diferencias se explican por soporte, no por nuevos modelos o corrección de predicciones. En train se conservan igualmente n=4.612 principal y n=4.376 común. No se reemplazan las métricas ni los checkpoints seleccionados con validación completa.

Predicciones, scores y todas las métricas completas/comunes/mensuales se reprodujeron con los modelos cargados, sin fit. El umbral logístico es 0,5; checkpoints 3 y 399; ambos reciben exactamente las mismas 16 características estandarizadas y ordenadas. Los hashes coinciden con los aprobados. No se modifica el manifiesto anterior para adaptar este dictamen.
'''
    (OUT/'conciliacion_validacion.md').write_text(text,encoding='utf-8',newline='\n')
    loadings=pd.DataFrame({'feature':FEATURES,'PC1':pca.components_[:,0],'PC2':pca.components_[:,1]})
    loadings.to_csv(OUT/'coeficientes_componentes.csv',index=False,lineterminator='\n',float_format='%.17g')
    tops={f'PC{k+1}':sorted([dict(feature=n,coefficient=float(w)) for n,w in zip(FEATURES,pca.components_[:,k])],key=lambda r:-abs(r['coefficient']))[:6] for k in range(2)}
    guardar(OUT/'interpretacion_componentes.json',dict(top_absolute_coefficients=tops,coefficients_are='unit eigenvector coefficients on standardized inputs, not causal effects or correlations',sign_convention=pca.to_dict()['sign_convention']))
    top_table='\n'.join(f"| {pc} | {r['feature']} | {r['coefficient']:.6f} |" for pc,rows in tops.items() for r in rows)
    text=f'''# PCA financiero: representación bidimensional

Se reutiliza PCA2D por SVD NumPy sin modificaciones. Fit solo con X_train 4.612×16 ya estandarizada; no nuevo escalado ni etiquetas. Validación 2.943×16 recibe transform con la misma media y W2. No se cambian las entradas de los clasificadores: conservan 16 dimensiones y sus checkpoints originales. Las proyecciones son artefactos nuevos de visualización, no sustitutos del dataset.

Las proyecciones se guardan en CSV UTF-8 con 17 cifras significativas, claves estables y etiquetas de desarrollo para auditoría gráfica. Su lectura con float_precision=round_trip recupera los valores float64. No se crean matrices .npy en resultados: se respeta la restricción histórica de esa carpeta y se mantienen las matrices originales de clasificadores en su ubicación aprobada.

- PC1: **{variance['PC1_percent']:.6f}%** de varianza de train.
- PC2: **{variance['PC2_percent']:.6f}%**.
- Acumulada en dos PC: **{variance['retained_percent']:.6f}%**.
- No representada: **{variance['discarded_percent']:.6f}%**.
- Rango numérico observado: {pca.rank_} de 16, sin retirar columnas ni alterar los modelos.

Varianza muestral ddof=1. Se verifican ortogonalidad, espectro SVD frente a autovalores de covarianza, traza de covarianza, varianza de las proyecciones, suma de ratios y residuo de reconstrucción 2D frente a varianza descartada. Resultados numéricos en comprobaciones_matematicas.json. El rango menor que 16, si ocurre, es compatible con características históricas derivadas y redundantes; no exige cambiar el experimento congelado.

## Componentes y límites de interpretación

| Componente | Característica | Coeficiente |
|---|---|---:|
{top_table}

Estos son coeficientes de vectores unitarios en coordenadas estandarizadas, no importancia causal, pesos de los clasificadores ni correlaciones automáticamente. PC1/PC2 resumen combinaciones de frecuencia y variación histórica según los coeficientes anteriores; el signo se fija haciendo positivo el coeficiente de mayor valor absoluto. No se optimiza la separación de clases. Medias, diferencias, tendencias y lags pueden representar varias veces una misma estructura correlacionada; la varianza no equivale a utilidad predictiva.

En este ajuste, PC1 tiene sus mayores coeficientes positivos en media, mediana y los tres lags de cobros: describe principalmente un nivel conjunto de actividad histórica. PC2 contrapone aumentos recientes (cambio, variación relativa y tendencia con coeficientes positivos) a ausencia de cobros el mes anterior (coeficiente negativo): describe variación reciente e intermitencia. Son descripciones de las combinaciones observadas, no factores económicos identificados. Las dos clases se solapan en ambas particiones; los puntos extremos se mantienen visibles y no se recortan para aparentar separación.

Los tres gráficos muestran todos los puntos, transparencia para el solapamiento, mismos límites de ejes en ambas particiones y porcentajes de varianza de entrenamiento. No hay jitter, separación artificial, eliminación de atípicos o frontera 2D presentada como clasificador de 16 entradas. Las etiquetas solo colorean las proyecciones. No se deduce separabilidad o convergencia de una neurona de estos dos componentes; queda fuera de la representación el porcentaje indicado de varianza.

## Congelación y convergencia conservadas

Perceptrón: SHA-256 {EXPECTED_MODELS['perceptron']}; checkpoint 3, eta=0,01, seed=7. Logística: SHA-256 {EXPECTED_MODELS['logistic']}; checkpoint 399, eta=0,10, umbral 0,5. Pesos, sesgos y orden de features se verifican en verificacion_modelos.json. Ningún entrenamiento nuevo de clasificadores.

Se leen los historiales aprobados: el perceptrón sigue actualizando al terminar sus 100 épocas y no registra épocas sin errores; logística reduce pérdida train de 0,693147 a aproximadamente 0,616973, termina por 400 épocas y no alcanza norma de gradiente <=1e-6. La pérdida decreciente no certifica convergencia completa. Se conservan curvas y modelos anteriores intactos; no se reemplaza checkpoint por última época.

## Reproducción y siguiente decisión

```powershell
python -m src.experiments.pca_transacciones
python -m unittest discover -s tests -p test_pca_transacciones.py -v
python -m unittest discover -s tests -v
```

El runner solo lee desarrollo y ajusta PCA, no consulta SQL, no carga outcomes de 2016, no reconstruye matrices originales ni entrena clasificadores. Las pruebas unitarias sintéticas de la suite existente son verificaciones de componentes, no nuevas ejecuciones experimentales sobre WWI.

Puede autorizarse la primera evaluación de 2016 en un encargo posterior **bajo el contrato académico de eventos y el protocolo definitivo congelado aquí**, siempre que pasen de nuevo integridad, cobertura/cierre y disponibilidad. No es certificación bancaria/operativa ni completitud absoluta del libro. Esta tarea no abre test ni ejecuta la evaluación final. La población principal nunca se reduce por falta del cuarto lag; el histórico se compara aparte en la intersección exacta.
'''
    (OUT/'reporte_pca.md').write_text(text,encoding='utf-8',newline='\n')


def manifest():
    guardar(OUT/'manifiesto.json',dict(implementation_sha256={n:sha(BASE/n) for n in sorted(NEW) if (BASE/n).exists()},
        pca_implementation_sha256=sha(BASE/'src/visualization/pca.py'),
        artifacts_sha256={str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['manifiesto.json','validaciones.json','reproducibilidad.json']},
        frozen_model_sha256=EXPECTED_MODELS))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',snapshot())
    verify_prior()
    try:
        dev,_=training.preflight();models,comparison,verification=verify_models(dev)
        reconciliation,excluded=reconcile(dev);guardar(OUT/'conciliacion.json',reconciliation)
        excluded.to_csv(OUT/'observaciones_excluidas.csv',index=False,lineterminator='\n',float_format='%.17g')
        if reconciliation['alignment_discrepancy_detected']:raise RuntimeError('Conciliación inconsistente; no autorizar test')
        pca,train_z,val_z=fit_pca(dev['train']['X'],dev['val']['X'])
        checks=mathematical_checks(pca,dev['train']['X'],train_z)
        pca.save(OUT/'parametros_pca.json');guardar(OUT/'comprobaciones_matematicas.json',checks)
        for split,z in [('train',train_z),('val',val_z)]:
            frame=dev[split]['meta'][['observation_id','customer_id','month']].copy();frame['y']=dev[split]['y'];frame['PC1']=z[:,0];frame['PC2']=z[:,1]
            frame.to_csv(OUT/f'proyeccion_{split}.csv',index=False,lineterminator='\n',float_format='%.17g')
        variance=dict(PC1_percent=float(pca.explained_variance_ratio_[0]*100),PC2_percent=float(pca.explained_variance_ratio_[1]*100),
            retained_percent=float(pca.explained_variance_ratio_.sum()*100),discarded_percent=float((1-pca.explained_variance_ratio_.sum())*100),
            ratios_all=pca.explained_variance_ratio_all_.tolist(),variance_all=pca.explained_variance_all_.tolist(),fit_partition='train_only',labels_used_in_fit=False)
        guardar(OUT/'varianza_explicada.json',variance);guardar(OUT/'verificacion_modelos.json',verification)
        guardar(OUT/'convergencia_verificada.json',verify_convergence())
        figures(pca,{'train':train_z,'val':val_z},{s:d['y'] for s,d in dev.items()},OUT)
        reports(reconciliation,excluded,pca,variance,verification)
        final_protocol=protocol();guardar(OUT/'protocolo_test_definitivo.json',final_protocol)
        (OUT/'protocolo_test_definitivo.md').write_text('''# Protocolo definitivo de primera apertura de test

## Estado y condiciones de apertura

Reglas congeladas antes de acceder a outcomes de 2016. Esta tarea no abre test. La primera evaluación podrá ejecutarse en un encargo posterior explícito si los controles de integridad, cobertura, cierre y disponibilidad pasan nuevamente. La fuente estructurada y sus referencias a protocolos previos están en protocolo_test_definitivo.json; no se sobrescribe ningún documento anterior.

Antes de abrir, comprobar los hashes de dataset, snapshot de desarrollo, preprocesador, catálogo, modelos, implementaciones y manifiesto actual. Verificar checkpoints 3/399, perceptrón eta=0,01/seed=7, logística eta=0,10/umbral=0,5, pesos, sesgos y orden de 16 columnas. Registrar fecha, alcance y huellas de la primera apertura. No reentrenar con train+val ni seleccionar configuraciones con test.

## Extracción y cobertura

Ventana predefinida: enero-mayo de 2016, establecida a partir de la cobertura ya auditada, no de etiquetas nuevas. Intervalo [2016-01-01,2016-06-01), corte propuesto de conocimiento exclusivo 2016-06-01 00:00. Confirmar cierre de cada mes: objetivo incompleto o no confirmado queda indeterminado; no ampliar la ventana según resultados.

Extraer únicamente cobros tipo 3 Customer Payment Received por SELECT y proyección mínima aprobada: ID único operacional, CustomerID, TransactionDate, tipo, última edición como control de versión y comprobación de signo sin exportar importes. Cotejar ID/fecha/tipo con DW sin duplicar eventos por joins. La cuenta es CustomerTransactions.CustomerID (Bill To), nunca un miembro desconocido DW, un nombre actual o una clave sustituta tratada como magnitud. Guardar snapshot, procedencia, consultas y huellas; no interpretar última edición como alta auditada.

Mantener explícitamente business_ledger_academic y sus supuestos de cobertura/conservación. Cero significa ausencia de eventos registrados en un mes cubierto posterior a primera observación. Falta de cobertura/extracto o mes incompleto significa desconocido, no cero ni clase negativa. No se afirma completitud absoluta, ausencia universal de borrados/backdating o conciliación bancaria.

## Construcción causal y etiqueta

Predicción al inicio de cada mes t. Usar snapshot de desarrollo congelado más eventos de test ya concluidos y conocidos antes de t para formar N(t-1),N(t-2),N(t-3), estadísticas autorizadas, antigüedad observada y calendario. Primera aparición se determina con cobros anteriores al instante; no usar t, supervivencia o última aparición futura para seleccionar cuentas. Nuevas cuentas requieren tres meses de historia. Saldo, finalización, importe y atributos condicionados siguen prohibidos.

Aplicar exclusivamente transform del preprocesador congelado: sin fit, nuevas constantes, reescalado ni selección de columnas. Ambos clasificadores reciben 16 entradas; PCA queda fuera de sus predicciones y solo sirve para visualización. Tras cerrar t, obtener y=1[N(t)>mediana(N(t-1),N(t-2),N(t-3))], empate 0, respetando disponibilidad antes del corte. No usar y o aciertos/errores para decidir elegibilidad.

## Poblaciones y baselines

**Principal:** todas las cuentas-mes elegibles con tres meses anteriores cubiertos y etiqueta válida. Evaluar perceptrón, logística y mayoría en exactamente esas claves. No exigir cuarto lag a los clasificadores ni a la mayoría. Perceptrón usa z>=0; logística sigmoid(z)>=0,5; mayoría fija=0 aprendida de train.

**Secundaria común:** intersección exacta donde además está disponible el baseline histórico contractual 1[N(t-1)>mediana(N(t-2),N(t-3),N(t-4))]. Evaluar los cuatro métodos sobre idénticas claves. Cuando el cuarto mes no existe antes de primera aparición, no está cubierto o no era conocido, dejar histórico indeterminado, sin imputarlo a cero ni eliminar esa fila de la evaluación principal si cumple sus reglas.

Registrar por clave estable y mes: primera aparición, elegibilidad principal, disponibilidad de etiqueta, cobertura, presencia/fecha de conocimiento de cuatro lags, elegibilidad común y motivos separados. Conservar filas excluidas y no sobrescribir predicciones anteriores. No seleccionar observaciones según error de modelo, clase o pago futuro.

## Métricas y cierre

Informar accuracy, BA, precision, recall, F1, especificidad, matriz [0,1], soporte por clase y proporción predicha positiva, globales y mensuales para cada soporte claramente nombrado. AP logística usa scores positivos y grupos de empates, sin interpolación ni papel en selección. Mantener convenciones de denominadores cero y reconocer subconjuntos monoclase. Nunca comparar n de un soporte con métricas de otro como si fueran la misma población.

Desglosar cuentas vistas en train, vistas solo en validación y nuevas elegibles de test cuando el soporte lo permita, sin filtrar por sus outcomes. Reconocer dependencia entre cuentas/meses y límites de la simulación; no inventar significación o costes monetarios. Después de abrir, ningún resultado puede alterar modelos, umbral, etiqueta, horizonte, cobertura, features, soportes o baselines. Un cambio exigiría otro experimento explícito. Conservar predicciones alineadas, exclusiones, hashes y registro de ejecución.
''',encoding='utf-8',newline='\n')
        guardar(OUT/'dictamen.json',dict(status='development_consistent_pca_verified_test_readiness_confirmed_under_academic_contract',
            first_test_evaluation_can_be_authorized_in_subsequent_task=True,unconditional_operational_certification=False,
            primary_validation=2943,secondary_common_validation=2927,exclusions_only_for_historical_baseline=True,
            models_unchanged=True,classifier_inputs_remain_16D=True,pca_fit_partition='train',test_opened=False,final_evaluation_executed=False,
            new_classifier_training_runs=0,sql_queries=0,conditions_before_test=final_protocol['before_opening']))
    except Exception as error:
        guardar(OUT/'dictamen.json',dict(status='blocked_do_not_open_test',first_test_evaluation_can_be_authorized_in_subsequent_task=False,reason=str(error),test_opened=False))
        raise
    guardar(OUT/'entorno.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pca_fit_partition='train',classifier_training_runs=0))
    guardar(OUT/'integridad_despues.json',verify_prior());manifest()
    print(json.dumps(dict(reconciliation=reconciliation,variance=variance,test_readiness=True,test_opened=False),ensure_ascii=True))


if __name__=='__main__':run()
