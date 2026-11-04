"""Construcción reproducible train/val; frontera fija <2016, sin modelos ni PCA."""
import argparse
from collections import Counter
import datetime as dt
import gzip
import hashlib
import json
import platform
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.db.auditoria_dw import BASE,guardar,sha
from src.evaluation.metricas import metricas_binarias
from .extraccion import extract,normalize,save_snapshot,read_snapshot,SOURCE_SQL,DW_SQL,PARAMS
from .caracteristicas import build,FEATURES,COVERAGE
from .preprocesamiento import PreprocesadorInventario

APPROVED=BASE/'resultados/extension_datamarts/inventario'
OUT=APPROVED/'dataset'
DATA=BASE/'datos/Procesados/inventario'
CODE=BASE/'src/datamarts/inventario'
SNAPSHOT='recepciones_desarrollo.csv.gz'
TEST_FILE=BASE/'tests/test_inventario_dataset.py'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def protected_files():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
        for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and not any(x in p.parents for x in [OUT,DATA,CODE]) and p!=TEST_FILE}

def verify_prior():
    before=read(OUT/'integridad_antes.json')
    if before!={n:sha(BASE/n) for n in before}:raise RuntimeError('Cambió un archivo protegido')
    return before

def approved_inputs():
    manifest=read(APPROVED/'manifiesto.json')['artifacts']
    names=['semantica_movimientos.md','analisis_etiquetas.md','contrato_datos.md','diseno_evaluacion.md',
        'validaciones.json','catalogo_caracteristicas.json','eventos_desarrollo.json','exploracion_etiquetas.json','cobertura_articulos.json']
    for name in names:
        p=APPROVED/name;key=str(p.relative_to(BASE))
        if sha(p)!=manifest.get(key):raise RuntimeError('Cambió evidencia aprobada: '+name)
    catalog=read(APPROVED/'catalogo_caracteristicas.json')
    if [f['name'] for f in catalog]!=FEATURES:raise RuntimeError('Catálogo incompatible')
    raw=read(APPROVED/'eventos_desarrollo.json');first={}
    for r in raw:
        item=r['item_id'];when=r['occurred_at']
        if not '2013-01-01'<=when<'2016-01-01':raise RuntimeError('Auditoría fuera de desarrollo')
        first[item]=min(first.get(item,when),when)
    receipts=normalize([{k:r[k] for k in ['event_id','item_id','type_id','occurred_at','quantity','edited_at','invoice_id','purchase_order_id','customer_id','supplier_id']} for r in raw if r['type_id']==11])
    return receipts,[dict(item_id=i,first_event_at=t,origin='approved any-movement audit; not first receipt') for i,t in sorted(first.items())],catalog

def write_frame(path,frame):
    raw=frame.to_csv(index=False,lineterminator='\n',float_format='%.17g',na_rep='').encode('utf-8')
    if path.suffix=='.gz':
        with path.open('wb') as f:
            with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=9) as gz:gz.write(raw)
    else:path.write_bytes(raw)

def validate_alignment(meta,x,labels):
    ids=meta.observation_id.tolist()
    if len(ids)!=len(set(ids)) or ids!=x.observation_id.tolist() or ids!=labels.observation_id.tolist():raise ValueError('Desalineación o identidad duplicada')
    if meta[['item_id','month']].duplicated().any():raise ValueError('Artículo-mes repetido')
    if list(x.columns)!=['observation_id']+FEATURES:raise ValueError('Entradas no autorizadas')
    if not np.isin(labels.y,[0,1]).all() or not np.array_equal(labels.y,(labels.target_count>labels.historical_reference).astype(np.int8)):
        raise ValueError('Etiqueta incompatible')
    if (meta.month>='2016-01').any() or not ((meta.partition=='train')==(meta.month<'2015-01')).all():raise ValueError('Particiones incompatibles')

def summarize(meta,labels):
    merged=meta[['observation_id','item_id','month','partition']].merge(labels,on='observation_id',validate='one_to_one')
    majority=int(merged.loc[merged.partition=='train','y'].mean()>.5)
    summary={}
    for split in ['train','val']:
        rows=merged[merged.partition==split];pos=int(rows.y.sum())
        monthly=[]
        for month,rs in rows.groupby('month',sort=True):
            p=int(rs.y.sum());monthly.append(dict(month=month,observations=len(rs),positive=p,negative=len(rs)-p,prevalence=p/len(rs)))
        peritem=[dict(item_id=int(i),observations=len(rs),positive=int(rs.y.sum()),negative=int((rs.y==0).sum()),class_variation=rs.y.nunique()==2)
            for i,rs in rows.groupby('item_id',sort=True)]
        summary[split]=dict(observations=len(rows),positive=pos,negative=len(rows)-pos,prevalence=pos/len(rows),entities=int(rows.item_id.nunique()),
            positive_entities=sum(r['positive']>0 for r in peritem),entities_varying=sum(r['class_variation'] for r in peritem),monthly=monthly,by_item=peritem,
            majority_class_from_train=majority,majority=metricas_binarias(rows.y,np.full(len(rows),majority,dtype=np.int8)),
            historical=metricas_binarias(rows.y,rows.historical_baseline))
    return summary

def export(rows,first,catalog,*,data_dir=DATA,out_dir=OUT,coverage=None,month_status=None,reconcile_expected=True):
    data_dir=Path(data_dir);out_dir=Path(out_dir);data_dir.mkdir(parents=True,exist_ok=True);out_dir.mkdir(parents=True,exist_ok=True)
    meta,x,labels,monthly,excluded=build(rows,first,coverage=coverage,month_status=month_status)
    validate_alignment(meta,x,labels);summary=summarize(meta,labels)
    if reconcile_expected:
        expected=read(APPROVED/'exploracion_etiquetas.json')['A']
        for split in ['train','val']:
            for key in ['observations','positive','negative','entities','positive_entities','entities_varying']:
                if summary[split][key]!=expected['summary'][split][key]:raise RuntimeError(f'Discrepancia {split}.{key}; investigar sin cambiar reglas')
            if summary[split]['monthly']!=[m for m in expected['monthly'] if (m['month']<'2015-01')==(split=='train')]:
                # La auditoría usa la clave adicional entities y la misma prevalencia.
                for actual in summary[split]['monthly']:
                    original=next(m for m in expected['monthly'] if m['month']==actual['month'])
                    if any(actual[k]!=original[k] for k in ['observations','positive','negative','prevalence']):raise RuntimeError('Discrepancia mensual')
    train=meta.partition.eq('train').to_numpy();val=~train
    prep=PreprocesadorInventario().fit(x.loc[train,FEATURES],partition='train');state=prep.to_dict();shapes={}
    write_frame(data_dir/'metadatos.csv.gz',meta);write_frame(data_dir/'caracteristicas_sin_escalar.csv.gz',x)
    write_frame(data_dir/'etiquetas.csv.gz',labels);write_frame(data_dir/'agregados_articulo_mes.csv.gz',monthly)
    write_frame(data_dir/'exclusiones.csv',excluded);write_frame(data_dir/'primera_aparicion.csv',pd.DataFrame(first))
    for split,mask in [('train',train),('val',val)]:
        xx=prep.transform(x.loc[mask,FEATURES]);yy=labels.loc[mask,'y'].to_numpy(dtype=np.int8)
        np.save(data_dir/f'X_{split}.npy',xx,allow_pickle=False);np.save(data_dir/f'y_{split}.npy',yy,allow_pickle=False)
        write_frame(data_dir/f'claves_{split}.csv',meta.loc[mask,['observation_id','item_id','month','partition']])
        write_frame(data_dir/f'baseline_historico_{split}.csv',labels.loc[mask,['observation_id','historical_baseline']])
        shapes[split]=dict(X=list(xx.shape),y=list(yy.shape),X_dtype=str(xx.dtype),y_dtype=str(yy.dtype))
    guardar(out_dir/'preprocesador.json',state);guardar(out_dir/'caracteristicas.json',dict(approved_catalog=catalog,input_features=FEATURES,
        feature_names=state['feature_names'],discarded=state['discarded'],all_features_required=True,no_imputation_needed=True))
    guardar(out_dir/'conteos.json',summary);guardar(out_dir/'dimensiones.json',shapes)
    guardar(out_dir/'cobertura_y_exclusiones.json',dict(mode=coverage,assumed_event_conservation=True,assumed_month_closure=True,
        zero_policy='zero only in covered months after first known appearance; unknown/incomplete excluded',
        item_months=len(monthly),months_without_receipts=int(monthly.zero_assumed_under_coverage.sum()),
        coverage_states=dict(Counter(monthly.coverage_status)),exclusions=dict(Counter(excluded.reason)),excluded_rows=len(excluded),
        no_stock_levels_reconstructed=True,first_appearance_source='approved all-movement development audit'))
    return summary,shapes,state

def test_spec(state):
    return dict(status='reserved_not_executed',outcomes_queried=False,X_test_built=False,y_test_built=False,period='2016; only complete certified/assumed-covered months at future opening',
        target_months=['2016-01','2016-02','2016-03','2016-04','2016-05'],
        target_window_origin='Coverage endpoint from prior general audit; not selected using test labels',
        event_type_id=11,event_time_field='TransactionOccurredWhen',history_for_first_target=['2015-12','2015-11','2015-10'],
        observation_order=['item_id ascending','target_month ascending'],majority_class_from_train=0,
        prerequisites=['Freeze preprocessing, models, checkpoint, threshold and protocol before opening outcomes','Persist opening record before querying test'],
        recipe=['Extract only type 11 using the same canonical schema and SQL SELECT after explicit opening authorization',
            'Use operational item identity, event IDs and reconcile bijectively with DW; recover first appearance causally without future survival filters',
            'For each t, use only t-1,t-2,t-3 closed covered months; earlier concluded test months may supply subsequent history',
            'Unknown/incomplete months are not zero; exclude new items until three covered months after first appearance',
            'Compute y with strict excess over three-month median, ties zero; keep target count separate',
            'Apply exactly the frozen catalog, mask, mean and population std below, without fit or imputation of essential history',
            'Preserve aligned keys, observations, exclusions, hashes, monthly supports and both causal baselines'],
        preprocessing=state,required_features=FEATURES,baseline='1[n1>median(n1,n2,n3)]; majority fit on development train only',
        test_sql_implemented=False,no_test_dates_exposed_in_current_CLI=True)

def report(summary,shapes,state):
    text=f'''# Dataset de desarrollo de Inventario: frecuencia de recepciones

Objetivo aprobado: N(i,t) es COUNT de IDs distintos tipo 11; y=1[N(i,t)>mediana(N(i,t-1),N(i,t-2),N(i,t-3))], empate 0. Predicción al comienzo de t. Se usaron 7298 recepciones reales de desarrollo, sin salidas, ajustes, eventos sintéticos ni rebalanceo. No hay campo de anulación en el contrato/origen auditado: no se inventó un filtro de cancelación ni se usaron estados actuales de compras.

## Procedencia y cobertura

Extracción SELECT focalizada de Warehouse.StockItemTransactions anterior a 2016, reconciliada por ID, identidad operativa, tipo, fecha, cantidad y documentos con Fact.Movement. Snapshot compacto en `{SNAPSHOT}`, SQL, parámetros y huellas antes/después en procedencia.json y consultas_sql.json. La primera aparición por cualquier movimiento procede de la auditoría aprobada, sin repetirla ni consultar nuevos outcomes de salida. Las versiones DW se reúnen por WWI Stock Item ID; no se usan nombres, claves sustitutas ni IDs como magnitudes en X.

Modo obligatorio `{COVERAGE}`: conservación del archivo de eventos y cierre mensual supuestos académicamente. Solo meses cubiertos sin tipo 11 valen cero. Sin modo, el pipeline bloquea; estados unknown/incomplete provocan exclusión y registro, nunca imputación de etiquetas. Los tres primeros meses por artículo no son elegibles para predicción. Para nuevos artículos no se rellena historia anterior a su primera aparición. No se reconstruyen existencias y no se consulta Fact.Stock Holding.

## Reconciliación y matrices

| Partición | X | y | Positivos | Negativos | Prevalencia |
|---|---|---|---:|---:|---:|
| Train abril 2013–diciembre 2014 | {shapes['train']['X']} | {shapes['train']['y']} | {summary['train']['positive']} | {summary['train']['negative']} | {100*summary['train']['prevalence']:.6f}% |
| Validación enero–diciembre 2015 | {shapes['val']['X']} | {shapes['val']['y']} | {summary['val']['positive']} | {summary['val']['negative']} | {100*summary['val']['prevalence']:.6f}% |

219 artículos por partición. Conteos y prevalencias mensuales reconciliados exactamente con exploración A, sin cambiar target ni población. Solo diez artículos tienen positivos en train y nueve en val; las demás filas reales se conservan. Pocos positivos y dependencia por artículo/mes limitan inferencias y utilidad. Baseline histórico causal y mayoritario se conservan por clave; no son predicciones de clasificadores. BA histórica train/val: {summary['train']['historical']['balanced_accuracy']:.9f}/{summary['val']['historical']['balanced_accuracy']:.9f}. En meses monoclase BA no se define. Las métricas previstas siguen siendo BA primaria, accuracy, precision, recall, specificity, F1 y confusiones; no se optimizan umbrales aquí.

## Entradas y preprocesado

17 características aprobadas, mismo orden y fórmulas del catálogo. Dimensión retenida: {len(state['feature_names'])}; descartadas por constantes en train: {state['discarded']}. Historia esencial completa, sin imputación; faltantes/infinitos bloquean. Adaptador propio del esquema train-only existente, sin parchear los catálogos globales de otros procesos. Ajuste único con train, media y desviación poblacional (ddof=0), misma máscara y escala en val, sin selección por validación, PCA, remuestreo ni ponderación. X float64, y int8. Medias, escalas, máscara, orden, fórmulas y trazabilidad en preprocesador.json y caracteristicas.json.

## Artefactos, reproducción y reserva

Agregados artículo-mes, exclusiones, primeras apariciones, metadatos, objetivos y entradas crudas están separados; N(i,t) y R(i,t) no se filtran en X. Se exportan claves train/val alineadas con los arrays. Cada fila detalla sus tres meses históricos y última fecha real de evento/edición de historia. El cierre académico cubre meses sin eventos; no se presenta como timestamp técnico de observación.

Desde la raíz:

```powershell
python -m src.datamarts.inventario.construir --origen sql --cobertura {COVERAGE}
python -m src.datamarts.inventario.construir --origen snapshot --cobertura {COVERAGE}
python -m src.datamarts.inventario.construir --verificar --cobertura {COVERAGE}
python -m unittest discover -s tests -v
```

`reproducibilidad.json` contrasta dos exportaciones offline independientes byte a byte y contra los artefactos canónicos. `validaciones.json` recoge causalidad, matrices y preservación; `manifiesto.json` incluye hashes SHA-256 de fuentes y artefactos. Las huellas SQL COUNT/CHECKSUM son comprobaciones focalizadas de estabilidad, no prueba criptográfica de toda la BD ni garantía frente a escrituras externas concurrentes. Todas las consultas propias son SELECT.

`especificacion_test.json` define una construcción futura determinista; el CLI actual no extrae ni construye 2016, X_test o y_test. No se entrenan clasificadores ni se ejecuta PCA experimental. Las pruebas de la suite existente usan fixtures sintéticos; no son entrenamiento de nuevos modelos empresariales. El informe académico y los experimentos anteriores permanecen intactos. Detenerse después de la verificación de desarrollo.
'''
    (OUT/'reporte_dataset.md').write_text(text,encoding='utf-8',newline='\n')

def manifest():
    paths=[p for root in [DATA,OUT,CODE] for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifiesto.json']+[TEST_FILE]
    guardar(OUT/'manifiesto.json',dict(scope='Inventory development train/val only; no test, training or PCA',
        artifacts_sha256={str(p.relative_to(BASE)):sha(p) for p in sorted(paths) if p.exists()},
        approved_contract_sha256=sha(APPROVED/'contrato_datos.md'),approved_catalog_sha256=sha(APPROVED/'catalogo_caracteristicas.json')))

def run(*,origin,coverage,server='lpc:.\\AMBATO'):
    if coverage!=COVERAGE:raise ValueError('Modo académico obligatorio')
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',protected_files())
    verify_prior();approved,first,catalog=approved_inputs();snapshot=DATA/SNAPSHOT
    if origin=='sql':
        log=[]
        try:rows,fingerprints,mapping=extract(server,log)
        finally:guardar(OUT/'consultas_sql.json',log)
        if rows!=approved:raise RuntimeError('Recepciones SQL difieren de auditoría; investigar')
        if snapshot.exists() and read_snapshot(snapshot)!=rows:raise RuntimeError('No sobrescribir snapshot diferente')
        save_snapshot(snapshot,rows)
        guardar(OUT/'identidad_dw.json',[dict(item_key=k,item_id=i,valid_from=f,valid_to=t) for k,i,f,t in mapping])
        guardar(OUT/'procedencia.json',dict(executed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),server=server,
            source='WideWorldImporters.Warehouse.StockItemTransactions',verification='WideWorldImportersDW.Fact.Movement',
            source_sql=SOURCE_SQL,verification_sql=DW_SQL,parameters=list(PARAMS),events=len(rows),items=len(first),
            source_sql_sha256=hashlib.sha256(SOURCE_SQL.encode()).hexdigest(),fingerprints=fingerprints,
            snapshot_sha256=sha(snapshot),approved_receipts_match=True,coverage_mode=coverage,
            approved_first_appearance_sha256=sha(APPROVED/'eventos_desarrollo.json'),
            source_audit_manifest_sha256=sha(APPROVED/'manifiesto.json'),first_appearance_is_not_first_receipt=True))
    elif origin=='snapshot':
        provenance=read(OUT/'procedencia.json')
        if sha(snapshot)!=provenance['snapshot_sha256'] or sha(APPROVED/'manifiesto.json')!=provenance['source_audit_manifest_sha256']:raise RuntimeError('Procedencia alterada')
        rows=read_snapshot(snapshot)
        if rows!=approved:raise RuntimeError('Snapshot no corresponde a auditoría')
    else:raise ValueError('Origen inválido')
    summary,shapes,state=export(rows,first,catalog,coverage=coverage)
    guardar(OUT/'especificacion_test.json',test_spec(state));guardar(OUT/'entorno.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        serialization='CSV LF %.17g; gzip level9 mtime0 empty name; npy allow_pickle=False; fixed recorded numerical environment'))
    report(summary,shapes,state);guardar(OUT/'integridad_despues.json',verify_prior());manifest()
    print(json.dumps(dict(shapes=shapes,discarded=state['discarded'],test_reserved=True)))

def verify():
    import io
    import unittest
    from contextlib import ExitStack
    from unittest.mock import patch
    import importlib
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    rows,first,catalog=approved_inputs();snapshot=read_snapshot(DATA/SNAPSHOT)
    check('Snapshot de recepciones coincide con auditoría',rows==snapshot)
    recorded=read(OUT/'procedencia.json');check('Hash snapshot coincide',sha(DATA/SNAPSHOT)==recorded['snapshot_sha256'])
    guarded=[]
    with ExitStack() as stack:
        for mod,cls,methods in [('src.models.perceptron','PerceptronSimple',['fit','predict','decision_function']),
            ('src.models.logistic_regression','LogisticRegressionManual',['fit','predict','predict_proba','decision_function']),
            ('src.visualization.pca','PCA2D',['fit','transform']),('src.db.conexion','Configuracion',['conectar'])]:
            obj=getattr(importlib.import_module(mod),cls)
            for method in methods:
                stack.enter_context(patch.object(obj,method,side_effect=AssertionError('Operación experimental prohibida')))
                guarded.append(cls+'.'+method)
        with tempfile.TemporaryDirectory(prefix='inventario-repro-') as tmp:
            dirs=[]
            for name in ['a','b']:
                root=Path(tmp)/name;export(snapshot,first,catalog,data_dir=root/'data',out_dir=root/'out',coverage=COVERAGE);dirs.append(root)
            a,b=dirs;ha={str(p.relative_to(a)):sha(p) for p in a.rglob('*') if p.is_file()};hb={str(p.relative_to(b)):sha(p) for p in b.rglob('*') if p.is_file()}
            check('Dos exportaciones offline idénticas byte a byte',ha==hb)
            for name,h in ha.items():
                area,rel=Path(name).parts[0],Path(*Path(name).parts[1:]);canonical=(DATA if area=='data' else OUT)/rel
                check('Canónico idéntico '+name,sha(canonical)==h)
            guardar(OUT/'reproducibilidad.json',dict(status='identical',exports=2,files_compared=len(ha),file_hashes=ha,SQL_queries=0,
                classifier_training=False,classifier_predictions=False,PCA=False,canonical_files_identical=True))
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_inventario_dataset')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (OUT/'pruebas_focalizadas.txt').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    check('Pruebas focalizadas satisfactorias',result.wasSuccessful() and result.testsRun>0)
    state=read(OUT/'preprocesador.json');meta=pd.read_csv(DATA/'metadatos.csv.gz');x=pd.read_csv(DATA/'caracteristicas_sin_escalar.csv.gz',float_precision='round_trip');labels=pd.read_csv(DATA/'etiquetas.csv.gz')
    validate_alignment(meta,x,labels);check('Alineación única y etiqueta exacta',True)
    prep=PreprocesadorInventario.from_dict(state)
    for split in ['train','val']:
        mask=meta.partition.eq(split).to_numpy();xx=np.load(DATA/f'X_{split}.npy',allow_pickle=False);yy=np.load(DATA/f'y_{split}.npy',allow_pickle=False)
        check(split+' tipos y finitud',xx.dtype==np.float64 and yy.dtype==np.int8 and np.isfinite(xx).all() and np.isin(yy,[0,1]).all())
        check(split+' X/y reproducen claves',np.array_equal(prep.transform(x.loc[mask,FEATURES]),xx) and np.array_equal(labels.loc[mask,'y'],yy))
        check(split+' claves CSV coinciden',pd.read_csv(DATA/f'claves_{split}.csv').observation_id.tolist()==meta.loc[mask,'observation_id'].tolist())
    for database,fp in recorded['fingerprints'].items():check('Huella SQL estable '+database,fp['before']==fp['after'])
    logs=read(OUT/'consultas_sql.json')
    from src.db.conexion import validar_select
    check('Solo SELECT focalizados de desarrollo',all(validar_select(q['sql']) and q['parameters']==[11,'2013-01-01','2016-01-01'] and q['status']=='successful' for q in logs))
    check('Test no construido',not list(DATA.glob('*test*')) and not test_spec(state)['outcomes_queried'])
    before=verify_prior();check('Artefactos previos íntegros',protected_files()==before)
    validation=dict(status='successful',checks_passed=len(checks),checks=checks,focused_tests=result.testsRun,
        focused_failures=len(result.failures),focused_errors=len(result.errors),preserved_files=len(before),
        test_outcomes_queried=False,test_matrices_built=False,experimental_classifiers_trained=False,PCA_executed=False,
        SQL_mode='SELECT only',guarded_operations=guarded,coverage_mode=COVERAGE,resampling=False,
        prior_suite='See suite_existente.txt and suite_existente.json; never alter previous tests to suppress failures')
    if (OUT/'suite_existente.json').exists():
        suite=read(OUT/'suite_existente.json')
        validation['full_suite']=dict(status=suite['status'],tests_run=suite['tests_run'],failures=len(suite['failures']),errors=len(suite['errors']),skipped=len(suite['skipped']),report='suite_existente.json')
        validation['overall_status']='dataset_verified' if suite['status']=='successful' else 'dataset_verified_with_full_suite_failures'
    guardar(OUT/'validaciones.json',validation)
    guardar(OUT/'integridad_despues.json',before);manifest()
    print(json.dumps(dict(status='successful',checks=len(checks),tests=result.testsRun,preserved_files=len(before))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--origen',choices=['sql','snapshot']);p.add_argument('--verificar',action='store_true')
    p.add_argument('--cobertura',required=True,choices=[COVERAGE]);p.add_argument('--server',default='lpc:.\\AMBATO');args=p.parse_args()
    if args.verificar:
        if args.origen:p.error('--verificar es offline; no combinar con --origen')
        verify()
    elif args.origen:run(origin=args.origen,coverage=args.cobertura,server=args.server)
    else:p.error('Especificar --origen o --verificar')
