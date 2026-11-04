"""Primera evaluación financiera 2016: contrato congelado, sin fit ni selección."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from src.db.auditoria_dw import BASE,guardar,sha
from src.db.conexion import Configuracion,Lector
from src.experiments import entrenar_transacciones as train
from src.experiments import pca_transacciones as review
from src.datamarts.transacciones.extraccion import SOURCE_SQL,SOURCE_FP,DW_SQL,DW_FP,FIELDS,csv_bytes,read_snapshot
from src.datamarts.transacciones.caracteristicas import FEATURES,feature_row,ordinal,COVERAGE
from src.datamarts.transacciones.preprocesamiento import PreprocesadorFinanciero
from src.datamarts.transacciones_viabilidad import add_month,start

OUT=train.PREVIOUS/'evaluacion_final'
NEW={'src/experiments/evaluar_transacciones.py','tests/test_evaluacion_transacciones.py'}
MONTHS=['2016-01','2016-02','2016-03','2016-04','2016-05']
PARAMS=(3,dt.date(2016,1,1),dt.date(2016,6,1))
CUT='2016-06-01 00:00:00'
HISTORICAL='1[N(t-1)>median(N(t-2),N(t-3),N(t-4))]'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def protected():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
            for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents
            and str(p.relative_to(BASE)).replace('\\','/') not in NEW}


def verify_prior():
    previous=read(OUT/'integridad_antes.json')
    if any(sha(BASE/n)!=h for n,h in previous.items()):raise RuntimeError('Artefacto previo modificado')
    return previous


def preflight():
    dev,state=train.preflight();models,comparison,verified=review.verify_models(dev)
    m=read(review.OUT/'manifiesto.json')
    for key in ['implementation_sha256','artifacts_sha256']:
        for name,h in m[key].items():
            if sha(BASE/name)!=h:raise RuntimeError('Revisión PCA/protocolo alterada: '+name)
    if sha(BASE/'src/visualization/pca.py')!=m['pca_implementation_sha256']:raise RuntimeError('Implementación PCA alterada')
    protocol=read(review.OUT/'protocolo_test_definitivo.json')
    frozen=read(train.OUT/'congelacion.json')
    if protocol['historical']!=HISTORICAL or frozen['baselines']['history']!=HISTORICAL or train.PLAN['baselines']['history']!=HISTORICAL:raise RuntimeError('Contradicción del baseline histórico: no abrir test')
    if protocol['planned_months']!=MONTHS or protocol['proposed_label_knowledge_cut_exclusive']!=CUT:raise RuntimeError('Ventana/corte no congelados')
    if not read(review.OUT/'dictamen.json')['first_test_evaluation_can_be_authorized_in_subsequent_task']:raise RuntimeError('Dictamen previo no autoriza apertura')
    reconciliation,_=review.reconcile(dev)
    if reconciliation['alignment_discrepancy_detected']:raise RuntimeError('Validación desalineada; test permanece reservado')
    return dev,state,models,dict(status='successful',verified_models=verified,validation_comparison_reproduced=True,
        baseline_kind='A: persistence of previous label',baseline_formula=HISTORICAL,requires_four_prior_months=True,
        rejected_alternative='1[N(t-1)>median(N(t-1),N(t-2),N(t-3))], not the frozen rule',
        feature_names=FEATURES,preprocessor_fit_partition=state['fit_partition'],test_data_accessed=False)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',protected())
    verify_prior()
    try:_,_,_,checks=preflight()
    except Exception as error:
        guardar(OUT/'bloqueo_previo.json',dict(status='blocked_before_test_opening',reason=str(error),test_queried=False));raise
    guardar(OUT/'verificacion_previa.json',checks)
    path=OUT/'apertura_previa.json'
    if path.exists():
        verify_opening();print('Registro previo existente verificado; no se sustituye');return
    frozen=read(train.OUT/'congelacion.json');protocol=read(review.OUT/'protocolo_test_definitivo.json')
    record=dict(recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='preflight_passed_before_test_SQL',
        protocol_version='PCA definitive protocol frozen before first test opening',protocol_sha256=sha(review.OUT/'protocolo_test_definitivo.json'),
        protocol_markdown_sha256=sha(review.OUT/'protocolo_test_definitivo.md'),protocol=protocol,
        models_sha256=review.EXPECTED_MODELS,preprocessor_sha256=frozen['preprocessor_sha256'],
        source_matrices_sha256={n:h for n,h in frozen['source_artifacts_sha256'].items() if n.endswith('.npy')},
        feature_names=FEATURES,thresholds=frozen['thresholds'],baseline=HISTORICAL,majority=0,
        source_sql=SOURCE_SQL,source_fingerprint_sql=SOURCE_FP,dw_sql=DW_SQL,dw_fingerprint_sql=DW_FP,parameters=list(PARAMS),
        coverage_mode=COVERAGE,coverage_assumptions=read(train.PREVIOUS/'cobertura_y_supuestos.json')['assumptions'],
        month_evaluation_rule='Complete calendar month before frozen cut; prior audit covers Jan-May; whole source interval reconciled OLTP-DW. Require observed monthly payment ledger continuity. Missing global month or failed reconciliation => unknown, never zero. Closure is academic, not inferred from last event date alone or bank-certified.',
        primary_population='3 prior covered consecutive months, stable identity, X available at start(t), y known before cut',
        common_population='Primary plus contractual 4 prior covered months for historical baseline, no imputation',
        metrics=['accuracy','balanced_accuracy','precision','recall','f1','specificity','false_alarm_rate','confusion_matrix','support','positive_prediction_proportion','logistic_average_precision'],
        eligibility='First payment before prediction, age>=3; no selection using future account activity or prediction errors',
        classifier_training_allowed=False,scaler_fit_allowed=False,pca_used_for_classification=False,
        evaluator_sha256_before_opening=sha(Path(__file__)),test_outcomes_accessed_when_record_written=False)
    guardar(path,record)
    print(json.dumps({'preflight':'successful','opening_record_created_before_SQL':True,'record_sha256':sha(path),'baseline':HISTORICAL}))


def verify_opening():
    r=read(OUT/'apertura_previa.json')
    if r['protocol_sha256']!=sha(review.OUT/'protocolo_test_definitivo.json') or r['models_sha256']!=review.EXPECTED_MODELS or r['preprocessor_sha256']!=sha(train.DATA/'preprocesador.json'):raise RuntimeError('Cambió una decisión fijada antes de abrir')
    if r['parameters']!=[3,'2016-01-01','2016-06-01'] or r['source_sql']!=SOURCE_SQL or r['dw_sql']!=DW_SQL or r['baseline']!=HISTORICAL:raise RuntimeError('Extracción/baseline cambiados tras apertura')
    return r


def normalize_test(rows):
    result=[];seen=set()
    for row in rows:
        if set(row)!=set(FIELDS) or any(row[k] is None for k in FIELDS):raise ValueError('Snapshot test incompatible o nulo')
        r={k:str(row[k]) for k in FIELDS}
        for k in ['event_id','customer_id','type_id','negative_payment']:r[k]=int(r[k])
        d=dt.date.fromisoformat(r['event_date']);edited=dt.datetime.fromisoformat(r['edited_at'])
        if not PARAMS[1]<=d<PARAMS[2] or r['type_id']!=3 or r['negative_payment']!=1:raise ValueError('Evento fuera de ventana/semántica congelada')
        if r['event_id']<=0 or r['customer_id']<=0 or r['event_id'] in seen or edited.date()<d:raise ValueError('Evento duplicado o identidad/fecha inválida')
        seen.add(r['event_id']);result.append(r)
    if not result:raise ValueError('Extracción de test vacía: no fabricar etiquetas')
    return sorted(result,key=lambda r:r['event_id'])


def save_test(path,rows):
    raw=csv_bytes(normalize_test(rows))
    with path.open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=9) as gz:gz.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_test(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:return normalize_test(list(csv.DictReader(f)))


def extract(server):
    verify_opening();verify_prior();preflight()
    if (OUT/'snapshot_test.csv.gz').exists():raise RuntimeError('Test ya extraído; reproducir desde snapshot, no volver a abrir sin registro')
    log=[];sources={};fingerprints={}
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    guardar(OUT/'inicio_consulta_test.json',dict(started_at_utc=started,opening_record_sha256=sha(OUT/'apertura_previa.json'),opening_record_precedes_SQL=True))
    try:
        for db,sql,fp in [('WideWorldImporters',SOURCE_SQL,SOURCE_FP),('WideWorldImportersDW',DW_SQL,DW_FP)]:
            reader=Lector(Configuracion(server,db).conectar())
            try:
                if db=='WideWorldImporters':
                    kind=reader.select('SELECT TransactionTypeName FROM Application.TransactionTypes WHERE TransactionTypeID=?',(3,))
                    if kind!=[{'TransactionTypeName':'Customer Payment Received'}]:raise RuntimeError('Semántica tipo 3 no confirmada')
                a=reader.select(fp,PARAMS);sources[db]=reader.select(sql,PARAMS);b=reader.select(fp,PARAMS)
                if a!=b:raise RuntimeError('La fuente cambió durante la lectura de test')
                fingerprints[db]={'before':a,'after':b}
            finally:log.extend(dict(database=db,**q) for q in reader.consultas);reader.close()
    finally:guardar(OUT/'consultas_test.json',log)
    rows=normalize_test(sources['WideWorldImporters'])
    src=[(r['event_id'],r['event_date'],r['type_id']) for r in rows]
    dw=[(r['event_id'],str(r['event_date']),r['type_id']) for r in sources['WideWorldImportersDW']]
    if src!=dw or len(set(dw))!=len(dw):raise RuntimeError('Test no forma biyección OLTP-DW')
    canonical=save_test(OUT/'snapshot_test.csv.gz',rows)
    guardar(OUT/'procedencia_test.json',dict(started_at_utc=started,completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),server=server,
        source_database='WideWorldImporters',verification_database='WideWorldImportersDW',params=list(PARAMS),events=len(rows),accounts=len({r['customer_id'] for r in rows}),
        source_dw_bijection=True,fingerprints=fingerprints,snapshot_sha256=sha(OUT/'snapshot_test.csv.gz'),canonical_csv_sha256=canonical,
        opening_record_sha256=sha(OUT/'apertura_previa.json'),opening_evaluator_sha256=read(OUT/'apertura_previa.json')['evaluator_sha256_before_opening'],
        all_payment_versions_edited_on_event_day=all(r['edited_at'][:10]==r['event_date'] for r in rows),absolute_completeness_certified=False))
    print(json.dumps({'extracted_events':len(rows),'accounts':len({r['customer_id'] for r in rows}),'source_dw_bijection':True}))


def coverage_report(rows):
    result={}
    for month in MONTHS:
        local=[r for r in rows if r['event_date'].startswith(month)]
        closed=start(add_month(month,1))<=CUT
        covered=bool(local) and closed
        result[month]=dict(covered_under_academic_contract=covered,calendar_closed_before_cut=closed,event_count=len(local),
            min_event_date=min((r['event_date'] for r in local),default=None),max_event_date=max((r['event_date'] for r in local),default=None),
            status='academic_closed_reconciled_ledger' if covered else 'unknown_do_not_fill_zero',
            justification='Frozen Jan-May scope, complete interval extraction matched to DW, monthly continuity, conservation assumptions; not absolute certification or last-event-only closure')
    return result


def build_test(development,test_rows,coverage):
    """Reutiliza feature_row aprobado; extiende el calendario, no el constructor sellado de desarrollo."""
    all_rows=development+test_rows;seen=set();groups=defaultdict(lambda:{'n':0,'known_at':''});first={}
    for r in all_rows:
        if r['event_id'] in seen:raise ValueError('Cobro duplicado entre desarrollo/test')
        if r['type_id']!=3 or r['customer_id']<=0:raise ValueError('Evento financiero no autorizado')
        seen.add(r['event_id']);c=r['customer_id'];m=r['event_date'][:7]
        first[c]=min(first.get(c,m),m);g=groups[c,m];g['n']+=1;g['known_at']=max(g['known_at'],r['edited_at'])
    known_development={r['customer_id'] for r in development};metadata=[];raw=[];targets=[];audit=[]
    def covered(m):return '2013-01'<=m<'2016-01' or bool(coverage.get(m,{}).get('covered_under_academic_contract',False))
    for month in MONTHS:
        for c,fm in sorted(first.items()):
            if fm>month:continue
            oid=f'customer:{c}:{month}';hist_months=[add_month(month,-k) for k in [1,2,3,4]]
            h=[groups.get((c,m),{'n':0,'known_at':''}) for m in hist_months];current=groups.get((c,month),{'n':0,'known_at':''})
            age=ordinal(month)-ordinal(fm);reason='eligible'
            if age<3:reason='insufficient_history'
            elif not all(covered(m) for m in hist_months[:3]):reason='unknown_historical_coverage'
            elif any(g['known_at']>=start(month) for g in h[:3]):reason='historical_version_not_available_at_prediction'
            feature_eligible=reason=='eligible'
            if feature_eligible and not covered(month):reason='unknown_target_month_coverage'
            elif feature_eligible and current['known_at']>=CUT:reason='target_not_known_before_cut'
            primary=reason=='eligible'
            fourth=age>=4 and all(covered(m) for m in hist_months) and all(g['known_at']<start(month) for g in h)
            history_reason='available' if fourth else ('fourth_month_before_first_observation' if age<4 else 'fourth_history_unknown_or_unavailable')
            record=dict(observation_id=oid,customer_id=c,month=month,first_observed_month=fm,age_months=age,prediction_time=start(month),
                primary_features_eligible=feature_eligible,primary_evaluation_eligible=primary,primary_reason=reason,
                fourth_month=hist_months[3],historical_available=fourth,historical_reason=history_reason,common_evaluation_eligible=primary and fourth,
                history_known_at=max(g['known_at'] for g in h[:3]),target_known_at=current['known_at'],label_cut_exclusive=CUT,
                coverage=COVERAGE,cohort='known_development' if c in known_development else 'new_in_test')
            audit.append(record)
            if not primary:continue
            ns=[g['n'] for g in h[:3]];reference=median(ns)
            metadata.append(record);raw.append(dict(observation_id=oid,**feature_row(ns,month,fm)))
            targets.append(dict(observation_id=oid,y=int(current['n']>reference),target_count=current['n'],historical_reference=reference,
                historical_prediction=int(h[0]['n']>median(g['n'] for g in h[1:])) if fourth else None))
    if not metadata:raise RuntimeError('No hay filas evaluables bajo las reglas congeladas')
    return pd.DataFrame(metadata),pd.DataFrame(raw),pd.DataFrame(targets),pd.DataFrame(audit)


def metrics(y,pred):
    y=np.asarray(y,dtype=np.int8);pred=np.asarray(pred,dtype=np.int8)
    if not len(y):return dict(n=0,not_estimable_reason='no eligible observations')
    m=train.metrics(y,pred);tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
    return dict(m,TN=tn,FP=fp,FN=fn,TP=tp,false_alarm_rate=fp/(tn+fp) if tn+fp else None)


def evaluate_frame(frame):
    full={n:metrics(frame.y.to_numpy(),frame[n].to_numpy()) for n in ['perceptron','logistic','majority']}
    common=frame[frame.historical_available]
    secondary={n:metrics(common.y.to_numpy(),common[n].to_numpy()) for n in ['perceptron','logistic','majority','historical']}
    return dict(primary_n=len(frame),common_n=len(common),excluded_only_from_historical=len(frame)-len(common),
        primary=full,common=secondary,logistic_AP_primary=train.average_precision(frame.y.to_numpy(),frame.logistic_probability.to_numpy()) if len(frame) else None,
        logistic_AP_common=train.average_precision(common.y.to_numpy(),common.logistic_probability.to_numpy()) if len(common) else None)


def error_analysis(frame):
    p=frame.perceptron.to_numpy();l=frame.logistic.to_numpy();y=frame.y.to_numpy()
    flags=frame[['observation_id','customer_id','month','y','perceptron','logistic','majority','historical_available','historical']].copy()
    flags['shared_correct']=(p==y)&(l==y);flags['shared_error']=(p!=y)&(l!=y);flags['disagreement']=p!=l
    for method in ['perceptron','logistic']:
        flags[method+'_FP']=(frame[method]==1)&(frame.y==0);flags[method+'_FN']=(frame[method]==0)&(frame.y==1)
    keys=['shared_correct','shared_error','disagreement','perceptron_FP','perceptron_FN','logistic_FP','logistic_FN']
    return dict(global_counts={k:int(flags[k].sum()) for k in keys},monthly={m:{k:int(g[k].sum()) for k in keys} for m,g in flags.groupby('month',sort=True)},
                no_parameter_changes_from_errors=True),flags


def plots(global_metrics,monthly,comparison):
    from src.evaluation.graficos import plt,PercentFormatter
    root=OUT/'graficos';root.mkdir(exist_ok=True)
    names=['perceptron','logistic','majority','historical'];common=global_metrics['common']
    for filename,keys,title in [('01_balanced_accuracy.png',['balanced_accuracy'],'Test: balanced accuracy de cuatro métodos, soporte común'),
        ('02_metricas.png',['accuracy','precision','recall','f1'],'Test: métricas en soporte común'),
        ('07_sensibilidad_especificidad.png',['recall','specificity'],'Test: sensibilidad y especificidad, soporte común')]:
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(4);w=.8/len(keys)
        for j,key in enumerate(keys):ax.bar(x+(j-(len(keys)-1)/2)*w,[common[n].get(key,np.nan) for n in names],w,label=key)
        ax.set(xticks=x,xticklabels=names,ylim=(0,1),title=title,ylabel='Proporción');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.grid(axis='y',alpha=.2)
        fig.savefig(root/filename,dpi=190,metadata={'Software':'PerceptronReal test financiero'});plt.close(fig)
    for i,method in enumerate(['perceptron','logistic'],3):
        cm=np.array(global_metrics['primary'][method]['confusion_matrix']);fig,ax=plt.subplots(figsize=(6,5),layout='constrained');im=ax.imshow(cm,cmap='Blues',vmin=0)
        for a in range(2):
            for b in range(2):ax.text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=15,color='white' if cm[a,b]>cm.max()/2 else 'black')
        ax.set(title=f'Test: {method}, población principal n={global_metrics["primary_n"]}',xlabel='Predicción',ylabel='Clase real',xticks=[0,1],yticks=[0,1]);fig.colorbar(im,ax=ax,label='Observaciones')
        fig.savefig(root/f'0{i}_confusion_{method}.png',dpi=190,metadata={'Software':'PerceptronReal test financiero'});plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for n in names:ax.plot(MONTHS,[monthly[m]['common'][n].get('balanced_accuracy',np.nan) for m in MONTHS],marker='o',label=n)
    ax.set(title='Test: balanced accuracy mensual, soporte común de cuatro métodos',xlabel='Mes 2016',ylabel='Balanced accuracy',ylim=(0,1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.grid(alpha=.2)
    fig.savefig(root/'05_BA_mensual.png',dpi=190,metadata={'Software':'PerceptronReal test financiero'});plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(3)
    for j,n in enumerate(['perceptron','logistic','majority']):ax.bar(x+(j-1)*.24,[comparison[s][n]['balanced_accuracy'] for s in ['train','val','test']],.24,label=n)
    ax.set(xticks=x,xticklabels=['Train','Validación','Test'],ylim=(0,1),ylabel='Balanced accuracy',title='Generalización temporal: población principal de cada partición');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'06_train_validacion_test.png',dpi=190,metadata={'Software':'PerceptronReal test financiero'});plt.close(fig)


def reports(summary,monthly,cohorts,errors,coverage,frame):
    concentration={}
    for method in ['perceptron','logistic']:
        rows=[]
        for month,v in monthly.items():
            m=v['primary'][method]
            if m['n']:rows.append(dict(month=month,n=m['n'],FP=m['FP'],FN=m['FN'],errors=m['FP']+m['FN'],error_rate=(m['FP']+m['FN'])/m['n']))
        concentration[method]=dict(by_error_count=sorted(rows,key=lambda r:(-r['errors'],r['month'])),by_error_rate=sorted(rows,key=lambda r:(-r['error_rate'],r['month'])))
    guardar(OUT/'concentracion_errores.json',concentration)
    worst='\n'.join(f"- {name}: mayor conteo de errores en {v['by_error_count'][0]['month']}, {v['by_error_count'][0]['errors']}/{v['by_error_count'][0]['n']}; mayor tasa en {v['by_error_rate'][0]['month']}, {v['by_error_rate'][0]['error_rate']*100:.2f}%." for name,v in concentration.items())
    table=[]
    for support in ['primary','common']:
        for method,m in summary[support].items():
            if not m['n']:continue
            table.append(f"| {support} | {method} | {m['n']} | {m['accuracy']:.6f} | {m['balanced_accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['specificity']:.6f} |")
    month_table='\n'.join(f"| {m} | {v['primary_n']} | {v['primary']['perceptron'].get('balanced_accuracy')} | {v['primary']['logistic'].get('balanced_accuracy')} | {v['common_n']} |" for m,v in monthly.items())
    cohort_table='\n'.join(f"| {name} | {v['accounts']} | {v['metrics']['primary_n']} | {v['metrics']['primary']['perceptron'].get('balanced_accuracy')} | {v['metrics']['primary']['logistic'].get('balanced_accuracy')} |" for name,v in cohorts.items())
    old=read(train.OUT/'metricas.json');deltas={n:summary['primary'][n]['balanced_accuracy']-old['full_support']['val'][n]['balanced_accuracy'] for n in ['perceptron','logistic']}
    text=f'''# Evaluación final de Transacciones: test 2016

## Integridad y apertura

La verificación previa reprodujo validación completa, común y mensual, scores y predicciones, y confirmó modelos, 16 columnas, checkpoints 3/399, umbral logístico 0,5 y preprocesador train-only. Se verificaron manifiestos de dataset, entrenamiento y PCA. apertura_previa.json se escribió antes de la primera consulta de eventos 2016; inicio_consulta_test.json y procedencia_test.json registran la lectura posterior. No se reentrenó, reajustó, reconstruyó desarrollo ni seleccionó parámetros con test. El PCA congelado no interviene en X ni se modifica.

El baseline real es **A, persistencia de la etiqueta anterior**: {HISTORICAL}. Requiere N(t-1) y N(t-2),N(t-3),N(t-4). No se reemplaza por B, que compara N(t-1) con una mediana que lo incluye y solo necesita tres meses. Se verificó consistencia entre implementación, congelación y protocolo antes de abrir.

## Fuente y población

Ventana congelada enero-mayo 2016, [2016-01-01,2016-06-01), conocimiento de y antes de 2016-06-01 00:00. Extracción focalizada de cobros tipo 3, IDs y CustomerID operacional, TransactionDate, edición como control de versión y comprobación de signo sin exportar importes. Solo SELECT, con huellas antes/después y biyección OLTP-DW. El snapshot gzip usa mtime=0, orden por ID y CSV UTF-8/LF; los hashes y consultas están conservados.

Meses evaluables bajo cobertura académica: {[m for m,v in coverage.items() if v['covered_under_academic_contract']]}. Cierre por calendario dentro del corte, alcance de la auditoría previa, extracción de intervalo completo conciliada y continuidad mensual apoyan el libro académico; no certifican completitud absoluta, ausencia de borrados/backdating o conciliación bancaria. No se deduce cierre solamente de la fecha del último evento. Meses sin cobertura o incompletos se registran desconocidos, sin fabricar ceros ni extender la ventana por resultados.

Población principal: **{summary['primary_n']}** filas, {int(frame.y.sum())} positivas y {int(len(frame)-frame.y.sum())} negativas, {frame.customer_id.nunique()} cuentas. Comparación común: **{summary['common_n']}** filas; {summary['excluded_only_from_historical']} quedan fuera exclusivamente del histórico. La falta de cuarto mes no reduce la evaluación principal. elegibilidad_test.csv registra también insuficiencia de tres meses, cobertura o disponibilidad; exclusiones_historico.csv identifica las filas primarias sin baseline. Todos los métodos de cada soporte usan idénticas claves, sin selección por error/clase.

Las características de t solo usan tres meses anteriores y calendario conocido; meses de test ya concluidos sirven para meses posteriores. Se reutiliza feature_row del pipeline aprobado, sin tocar el constructor sellado de desarrollo. La primera aparición no incorpora cuentas futuras en períodos anteriores. Se aplica únicamente transform del preprocesador y los modelos congelados sobre 16 columnas float64 finitas. X_test.csv, y_test.csv y metadatos_test.csv se guardan alineados con claves y precisión de 17 cifras; no se crean NPY dentro de resultados, preservando la convención histórica de esa carpeta.

## Métricas globales

| Soporte | Método | n | Accuracy | BA | Precision | Recall | F1 | Especificidad |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+'\n'.join(table)+f'''

Average Precision logística: principal={summary['logistic_AP_primary']}, común={summary['logistic_AP_common']}. Se usa el algoritmo ya fijado con grupos de scores empatados, sin interpolación ni selección. Las probabilidades no se presentan como calibradas; los márgenes del perceptrón no son probabilidades. metricas_globales.json incluye TN/FP/FN/TP, tasa de falsas alertas FP/(TN+FP), soporte, proporción positiva predicha y convenciones de denominador cero. No se imputa un histórico ausente.

## Generalización temporal y por cuentas

Diferencia BA test menos validación completa: perceptrón={deltas['perceptron']*100:.4f} puntos porcentuales; logística={deltas['logistic']*100:.4f}. comparacion_desarrollo_test.json conserva métricas completas de train, validación y test; las diferencias no autorizan elegir otro checkpoint o umbral. Los soportes de cada período son diferentes: no es una comparación emparejada de las mismas cuentas-mes ni una prueba de significación.

| Mes | n principal | BA perceptrón | BA logística | n común |
|---|---:|---:|---:|---:|
{month_table}

Se incluyen todos los meses previstos, aun desfavorables o sin casos estimables. Las métricas globales se calculan desde todas las predicciones, no como media simple de porcentajes mensuales; el promedio mensual no ponderado se conserva aparte en resumen_temporal.json. Subconjuntos sin una clase usan las convenciones aprobadas; BA/recall/especificidad pueden no ser estimables.

| Cohorte | Cuentas | n principal | BA perceptrón | BA logística |
|---|---:|---:|---:|---:|
{cohort_table}

Se distingue observado en desarrollo de primera aparición en test usando el snapshot anterior, no el resultado futuro. cohortes_detalladas.json separa cuentas elegibles en train, solo validación, observadas en desarrollo sin elegibilidad previa y nuevas de test. Cohortes pequeñas no permiten afirmar estabilidad de generalización. No se borran cuentas sin actividad actual ni se presume alta real a partir de primera observación.

## Errores y convergencia previamente observada

Conteos: {json.dumps(errors['global_counts'],ensure_ascii=False)}. analisis_errores.json y errores_por_observacion.csv conservan FP, FN, aciertos/errores compartidos y desacuerdos, globales y mensuales. Los meses con mayor concentración de errores se identifican por conteos y tasas en métricas mensuales, sin omitirlos ni utilizar esos resultados para cambiar clasificadores o baselines. No se inventan costes monetarios de errores ni se declara mejora estadísticamente significativa.

{worst}

En la población principal, perceptrón tiene recall={summary['primary']['perceptron']['recall']*100:.2f}% y tasa de falsas alertas={summary['primary']['perceptron']['false_alarm_rate']*100:.2f}%; logística tiene recall={summary['primary']['logistic']['recall']*100:.2f}% y tasa de falsas alertas={summary['primary']['logistic']['false_alarm_rate']*100:.2f}%. El primero detecta más positivos con más FP; la segunda reduce FP y conserva más FN. Son comportamientos de los checkpoints fijados, no una nueva elección por test. La prevalencia positiva test es {frame.y.mean()*100:.2f}%; no se atribuye causalmente el cambio de comportamiento a una variable particular ni se ajusta el umbral para compensarlo.

Se mantienen los historiales anteriores: perceptrón continuaba actualizando a 100 épocas, con checkpoint elegido en 3; logística reducía pérdida train de 0,6931 a 0,6170, alcanzaba el máximo de 400 épocas y no su tolerancia de gradiente, con checkpoint 399. No se afirma convergencia completa ni se generan curvas de entrenamiento sobre test. Los reportes/gráficos previos de convergencia y PCA siguen intactos y pueden consultarse como contexto, sin reinterpretar la proyección 2D como frontera del clasificador.

## Reproducción y límites

```powershell
python -m src.experiments.evaluar_transacciones --preparar
python -m src.experiments.evaluar_transacciones --extraer
python -m src.experiments.evaluar_transacciones --evaluar
python -m unittest discover -s tests -p test_evaluacion_transacciones.py -v
python -m unittest discover -s tests -v
```

Extraer exige el registro previo y no sustituye un snapshot existente. Para reproducciones técnicas, ejecutar solo --evaluar desde el snapshot con hash verificado; no abrir nuevamente SQL ni reinterpretar la búsqueda. El código y el protocolo inicial quedan registrados; correcciones técnicas, si ocurren, deben registrarse sin alterar reglas o configuraciones. Todos los resultados describen el libro académico de WWI y cinco meses de una simulación, no una certificación bancaria o utilidad operacional. Test ya fue observado: cualquier diseño nuevo requeriría un experimento distinto y no podría presentar este período como reserva nunca vista.

Se detiene aquí: sin cambios de pesos, umbrales, escaladores, features, etiquetas, baselines o ventanas; sin modificación de Ventas, Pedidos, Compras o LaTeX, operaciones Git o avance automático a Inventario.
'''
    (OUT/'reporte_evaluacion_final.md').write_text(text,encoding='utf-8',newline='\n')


def manifest():
    guardar(OUT/'manifiesto.json',dict(implementation_sha256={n:sha(BASE/n) for n in sorted(NEW) if (BASE/n).exists()},
        artifacts_sha256={str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['manifiesto.json','validaciones.json','reproducibilidad.json']},
        frozen_models_sha256=review.EXPECTED_MODELS,opening_record_sha256=sha(OUT/'apertura_previa.json')))


def evaluate():
    verify_prior();verify_opening();dev,state,models,_=preflight()
    provenance=read(OUT/'procedencia_test.json')
    if sha(OUT/'snapshot_test.csv.gz')!=provenance['snapshot_sha256']:raise RuntimeError('Snapshot de test alterado')
    rows=load_test(OUT/'snapshot_test.csv.gz');development=read_snapshot(train.DATA/'cobros_desarrollo.csv.gz');coverage=coverage_report(rows)
    meta,raw,labels,audit=build_test(development,rows,coverage)
    if meta.observation_id.tolist()!=raw.observation_id.tolist() or meta.observation_id.tolist()!=labels.observation_id.tolist() or not meta.observation_id.is_unique:raise RuntimeError('Test desalineado')
    x=PreprocesadorFinanciero.from_dict(state).transform(raw[FEATURES]);y=labels.y.to_numpy(dtype=np.int8)
    if x.shape!=(len(y),16) or x.dtype!=np.float64 or not np.isfinite(x).all():raise RuntimeError('X test inválida')
    frame=meta[['observation_id','customer_id','month','cohort','historical_available']].copy();frame['y']=y
    for name,model in models.items():frame[name]=model.predict(x,feature_names=FEATURES)
    frame['majority']=0;frame['historical']=labels.historical_prediction
    frame['perceptron_margin']=models['perceptron'].decision_function(x);frame['logistic_probability']=models['logistic'].predict_proba(x)
    metrics_global=evaluate_frame(frame);monthly={m:evaluate_frame(frame[frame.month==m]) for m in MONTHS}
    cohorts={c:dict(accounts=int(g.customer_id.nunique()),metrics=evaluate_frame(g)) for c,g in frame.groupby('cohort',sort=True)}
    train_ids=set(dev['train']['meta'].customer_id);val_ids=set(dev['val']['meta'].customer_id);old_ids={r['customer_id'] for r in development}
    def detail(c):return 'eligible_in_train' if c in train_ids else 'eligible_in_validation_only' if c in val_ids else 'observed_development_not_previously_eligible' if c in old_ids else 'first_observed_in_test'
    detailed={c:dict(accounts=int(g.customer_id.nunique()),metrics=evaluate_frame(g)) for c,g in frame.assign(detailed_cohort=frame.customer_id.map(detail)).groupby('detailed_cohort',sort=True)}
    errors,error_frame=error_analysis(frame);old=read(train.OUT/'metricas.json');comparison=dict(train=old['full_support']['train'],val=old['full_support']['val'],test=metrics_global['primary'])
    for name,df in [('metadatos_test.csv',meta),('caracteristicas_crudas_test.csv',raw),('y_test.csv',labels),('elegibilidad_test.csv',audit),('predicciones_test.csv',frame),('errores_por_observacion.csv',error_frame),('exclusiones_historico.csv',meta[~meta.historical_available])]:df.to_csv(OUT/name,index=False,lineterminator='\n',float_format='%.17g',na_rep='')
    standardized=pd.DataFrame(x,columns=FEATURES);standardized.insert(0,'observation_id',meta.observation_id.tolist());standardized.to_csv(OUT/'X_test.csv',index=False,lineterminator='\n',float_format='%.17g')
    for name,data in [('metricas_globales.json',metrics_global),('metricas_mensuales.json',monthly),('cohortes.json',cohorts),('cohortes_detalladas.json',detailed),('analisis_errores.json',errors),('cobertura_test.json',coverage),('comparacion_desarrollo_test.json',comparison)]:guardar(OUT/name,data)
    temporal={n:dict(global_BA=metrics_global['primary'][n]['balanced_accuracy'],unweighted_monthly_mean_BA=float(np.mean([monthly[m]['primary'][n]['balanced_accuracy'] for m in MONTHS if monthly[m]['primary'][n].get('balanced_accuracy') is not None]))) for n in ['perceptron','logistic','majority']}
    guardar(OUT/'resumen_temporal.json',temporal)
    guardar(OUT/'conteos.json',dict(source_events=len(rows),accounts_in_extraction=len({r['customer_id'] for r in rows}),primary=len(meta),common=metrics_global['common_n'],positive=int(y.sum()),negative=int(len(y)-y.sum()),shape=list(x.shape),dtype=str(x.dtype),eligibility_reasons=dict(Counter(audit.primary_reason)),evaluable_months=[m for m,v in coverage.items() if v['covered_under_academic_contract']],input_features=FEATURES))
    plots(metrics_global,monthly,comparison);reports(metrics_global,monthly,cohorts,errors,coverage,frame)
    guardar(OUT/'integridad_despues.json',verify_prior());manifest()
    print(json.dumps({'counts':read(OUT/'conteos.json'),'metrics':metrics_global,'cohorts':{c:{'accounts':v['accounts'],'n':v['metrics']['primary_n']} for c,v in cohorts.items()}},ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preparar',action='store_true');g.add_argument('--extraer',action='store_true');g.add_argument('--evaluar',action='store_true');p.add_argument('--server',default='lpc:.\\AMBATO');a=p.parse_args()
    prepare() if a.preparar else extract(a.server) if a.extraer else evaluate()
