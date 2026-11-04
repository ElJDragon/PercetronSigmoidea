"""Construye únicamente train/val de cobros, desde SQL local o snapshot verificado."""
import argparse
import datetime as dt
import gzip
import io
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.db.auditoria_dw import BASE,guardar,sha
from .extraccion import extract,normalize,save_snapshot,read_snapshot,csv_bytes,SOURCE_SQL,DW_SQL,PARAMS
from .caracteristicas import build,FEATURES,OPTIONAL,COVERAGE
from .preprocesamiento import PreprocesadorFinanciero

APPROVED=BASE/'resultados/extension_datamarts/transacciones'
OUT=APPROVED/'dataset'
DATA=BASE/'datos/Procesados/transacciones'
CODE=BASE/'src/datamarts/transacciones'
SNAPSHOT='cobros_desarrollo.csv.gz'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def protected_files():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
        for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and not any(x in p.parents for x in [OUT,DATA,CODE]) and p!=BASE/'tests/test_transacciones_dataset.py'}


def verify_prior():
    before=read(OUT/'integridad_antes.json');after={n:sha(BASE/n) for n in before}
    if before!=after:raise RuntimeError('Cambió un artefacto anterior a este dataset')
    return before


def approved_projection():
    raw=read(APPROVED/'eventos_customer_desarrollo.json')
    return normalize([dict(event_id=r['event_id'],customer_id=r['entity_id'],event_date=r['event_date'],type_id=r['type_id'],
                           edited_at=r['edited_at'],negative_payment=int(float(r['amount'])<0)) for r in raw if r['type_id']==3])


def write_frame(path,frame):
    payload=frame.to_csv(index=False,lineterminator='\n',float_format='%.17g',na_rep='').encode('utf-8')
    if path.suffix=='.gz':
        with path.open('wb') as fh:
            with gzip.GzipFile(filename='',mode='wb',fileobj=fh,mtime=0,compresslevel=9) as gz:gz.write(payload)
    else:path.write_bytes(payload)


def summarize(meta,labels):
    merged=meta[['observation_id','partition','month','customer_id']].merge(labels,on='observation_id',validate='one_to_one')
    result={}
    for split in ['train','val']:
        v=merged[merged.partition==split];p=int(v.y.sum())
        monthly=[]
        for m,rows in v.groupby('month',sort=True):
            pos=int(rows.y.sum());monthly.append(dict(month=m,eligible=len(rows),positive=pos,negative=len(rows)-pos,positive_proportion=pos/len(rows)))
        result[split]=dict(eligible=len(v),positive=p,negative=len(v)-p,positive_proportion=p/len(v),entities=int(v.customer_id.nunique()),monthly=monthly)
    return result


def validate_alignment(meta,x,y):
    ids=meta.observation_id.tolist()
    if len(set(ids))!=len(ids) or ids!=x.observation_id.tolist() or ids!=y.observation_id.tolist():raise ValueError('Identidad o alineación inválida')
    if list(x.columns)!=['observation_id']+FEATURES:raise ValueError('X incluye columnas ajenas al contrato')
    if not np.isin(y.y.to_numpy(),[0,1]).all():raise ValueError('Etiqueta no binaria')
    if not (y.y.to_numpy()==(y.target_count.to_numpy()>y.historical_reference.to_numpy()).astype(np.int8)).all():raise ValueError('Etiqueta no corresponde a fórmula')


def catalog(approved):
    selected=[r for r in approved['features'] if r['estado'] in ['permitido','permitido_contrato_academico']]
    if [r['nombre'] for r in selected]!=FEATURES:raise RuntimeError('Cambió el catálogo aprobado; no adaptar silenciosamente')
    return [dict(r,missing_treatment='mediana ajustada exclusivamente en train; indicador base_variacion_cero' if r['nombre'] in OPTIONAL else 'No permitido faltar; bloqueo si NaN',
                 output_dtype='float64') for r in selected]


def manifest():
    implementation=list(sorted(CODE.glob('*.py')))+[BASE/'tests/test_transacciones_dataset.py']
    guardar(OUT/'manifiesto.json',dict(
        scope='Dataset de desarrollo financiero train/val; test 2016 sellado; sin entrenamiento ni PCA',
        implementation_sha256={str(p.relative_to(BASE)):sha(p) for p in implementation if p.exists()},
        artifacts_sha256={str(p.relative_to(BASE)):sha(p) for root in [DATA,OUT] for p in sorted(root.rglob('*'))
                          if p.is_file() and p.name not in ['manifiesto.json','validaciones.json','reproducibilidad.json']},
        approved_contract_sha256=sha(APPROVED/'contrato_datos.json')))


def export(rows,*,data_dir=DATA,out_dir=OUT):
    """Transformación offline determinista. Usable con destinos temporales en QA."""
    approved=read(APPROVED/'contrato_datos.json');features_catalog=catalog(approved)
    meta,x,labels,eligibility,monthly=build(rows,coverage=COVERAGE)
    validate_alignment(meta,x,labels);summary=summarize(meta,labels)
    expected=approved['decision']['development_support']
    checks={}
    for split,key in [('train','train'),('val','validation')]:
        checks[split]={name:{'expected':expected[key][name],'actual':summary[split][name]} for name in ['eligible','positive','negative']}
    guardar(out_dir/'reconciliacion.json',dict(partitions=checks,matches_approved=all(v['actual']==v['expected'] for split in checks.values() for v in split.values()),
        source_events=len(rows),accounts=len({r['customer_id'] for r in rows}),population_slots=len(eligibility),eligibility_states=dict(Counter(eligibility.state))))
    if any(v['actual']!=v['expected'] for split in checks.values() for v in split.values()):raise RuntimeError('Discrepancia frente al contrato; investigar sin forzar coincidencias')
    prep=PreprocesadorFinanciero().fit(x.loc[meta.partition=='train',FEATURES],partition='train')
    data_dir.mkdir(parents=True,exist_ok=True);out_dir.mkdir(parents=True,exist_ok=True)
    shapes={}
    for split in ['train','val']:
        mask=meta.partition==split;fm=meta.loc[mask].reset_index(drop=True);fx=x.loc[mask].reset_index(drop=True);fy=labels.loc[mask].reset_index(drop=True)
        transformed=prep.transform(fx[FEATURES]);target=fy.y.to_numpy(dtype=np.int8)
        if transformed.shape[0]!=len(target) or transformed.dtype!=np.float64 or not np.isfinite(transformed).all():raise RuntimeError('Matriz inválida')
        np.save(data_dir/f'X_{split}.npy',transformed,allow_pickle=False);np.save(data_dir/f'y_{split}.npy',target,allow_pickle=False)
        write_frame(data_dir/f'metadatos_{split}.csv',fm);write_frame(data_dir/f'etiquetas_{split}.csv',fy)
        write_frame(data_dir/f'caracteristicas_crudas_{split}.csv',fx)
        shapes[split]={'X':list(transformed.shape),'y':list(target.shape),'X_dtype':str(transformed.dtype),'y_dtype':str(target.dtype)}
    write_frame(data_dir/'elegibilidad.csv.gz',eligibility);write_frame(data_dir/'cuentas_meses.csv.gz',monthly)
    guardar(data_dir/'preprocesador.json',prep.to_dict());guardar(data_dir/'feature_names.json',prep.state['feature_names'])
    guardar(out_dir/'caracteristicas.json',features_catalog);guardar(out_dir/'conteos.json',summary)
    guardar(out_dir/'matrices.json',dict(shapes=shapes,authorized_input_features=FEATURES,final_features=prep.state['feature_names'],discarded_train_only=prep.state['discarded']))
    guardar(out_dir/'cobertura_aplicada.json',dict(mode=COVERAGE,explicitly_enabled=True,scope=['2013-01','2015-12'],
        assumptions=approved['coverage']['assumptions'],limitations=approved['coverage']['unknowns'],absolute_completeness_certified=False,
        evidence_sha256=sha(APPROVED/'cobertura_y_supuestos.json'),last_edited_is_creation_timestamp=False))
    return summary,shapes,prep.state


def test_spec(state):
    return dict(status='sealed_not_constructed',queries_executed=False,X_test_created=False,y_test_created=False,
        scope='2016 reservado; ventana propuesta enero-mayo según cobertura ya documentada en la auditoría general, sin consultar nuevas etiquetas',
        planned_target_months=['2016-01','2016-02','2016-03','2016-04','2016-05'],
        proposed_event_interval_inclusive_exclusive=['2016-01-01','2016-06-01'],
        proposed_label_availability_cut_exclusive='2016-06-01 00:00:00',
        window_source='resultados/extension_datamarts/auditoria_dw/reporte_auditoria.md; cobertura previa, no selección por nuevos outcomes',
        incomplete_or_unconfirmed_month='indeterminado; no rellenar cero ni ampliar la ventana usando resultados; cualquier revisión debe congelarse antes de evaluar',
        preprocessor_sha256=sha(DATA/'preprocesador.json'),development_snapshot_sha256=sha(DATA/SNAPSHOT),
        prerequisites=['Congelar contrato, cobertura, features y hashes del preprocesador','Congelar modelos, hiperparámetros, umbral, baselines y protocolo de evaluación antes de abrir outcomes','Autorizar la evaluación final en un encargo posterior; esta especificación no la ejecuta'],
        extraction='Reutilizar proyección mínima tipo 3 y unión de IDs fuente/DW, con intervalo de test explícito tras congelación; no existe opción de ejecución de test en este CLI',
        history='Concatenar snapshot de desarrollo con eventos de meses de test ya concluidos, deduplicados por ID; no consultar t ni meses futuros para X(c,t)',
        first_seen='Mínimo mes observado de cobro hasta el instante, nunca catálogo o última aparición futura; nuevas cuentas elegibles tras tres meses de historia',
        target='y=1[N(c,t)>median(N(c,t-1),N(c,t-2),N(c,t-3))], empate 0; obtener solo tras cierre y comprobar disponibilidad',
        order='mes objetivo ascendente, CustomerID ascendente; observation_id=customer:{CustomerID}:{YYYY-MM}',
        features=state['feature_names'],preprocessing='Cargar preprocesador financiero congelado; solo transform; ningún fit ni nueva selección de constantes',
        known_history='Ventana consecutiva de tres meses; ceros solo en cobertura académica explícita, meses sin cobertura indeterminados',
        eligibility='Separar elegibilidad por historia antes de t de disponibilidad posterior de y; no excluir cuentas por no pagar en t',
        maturity='Excluir objetivo mensual incompleto; no tratar el último mes observado parcialmente como cerrado',
        cutoffs='Documentar cierre y fecha de conocimiento del test antes de su evaluación; no deducir completitud de la mera presencia de filas',
        prohibited=['Ajustar con test','Construir X_test/y_test ahora','Usar saldo/finalización/importe/atributos condicionados','Elegir entidades según actividad futura'])


def report(summary,shapes,state):
    rows='\n'.join(f"| {p} | {v['eligible']} | {v['positive']} | {v['negative']} | {shapes[p]['X']} |" for p,v in summary.items())
    text=f'''# Dataset financiero de desarrollo

Construcción completada exclusivamente para train y validación. Fuente: Sales.CustomerTransactions, CustomerID operacional, cobros tipo 3 con signo contable confirmado; correspondencia de IDs con Fact.Transaction. No se exportan importes, saldos, documentos, nombres ni atributos de pago. La última edición se guarda como control de versión y no certifica fecha de alta ni completitud absoluta.

| Partición | Observaciones | Positivos | Negativos | Forma X |
|---|---:|---:|---:|---|
{rows}

El catálogo aprobado aporta {len(FEATURES)} columnas de entrada; quedan {len(state['feature_names'])} tras eliminar constantes solo con train. Descartadas: {json.dumps(state['discarded'],ensure_ascii=False)}. No se fija una dimensión arbitraria ni se añade CustomerID, y, conteo objetivo, importe, estado o atributo dimensional a X. La mediana histórica R sí entra como cobros_mediana_3, porque solo depende del pasado. Se mantienen redundancias exactas autorizadas; no se seleccionan columnas por validación.

X usa float64, y int8 binario. La única característica que puede faltar es variacion_relativa_cobros cuando su denominador histórico es cero: se imputa con la mediana de train y conserva base_variacion_cero. Toda historia esencial debe existir; no se imputa para fabricar elegibilidad. Se ajustan media y desviación poblacional exclusivamente en train; la validación solo recibe transform. preprocesador.json guarda parámetros y orden.

Snapshot cobros_desarrollo.csv.gz: CSV UTF-8/LF, orden de event_id, compresión gzip nivel 9 con mtime=0 y nombre vacío. Sus bytes son reproducibles; procedencia.json separa el instante real de extracción de los artefactos deterministas. No se guarda una copia de todas las transacciones. Las consultas y sus parámetros/huellas se conservan en consultas_sql.json y procedencia.json. El snapshot se contrasta con la proyección de cobros de la auditoría aprobada antes de construir matrices.

Población: meses desde primera observación de cobro, sin usar última aparición o actividad futura. Se exigen tres meses consecutivos previos y control de disponibilidad antes de inicio(t); etiquetas conocidas antes de 2015-01-01 para train y 2016-01-01 para val. El modo {COVERAGE} está activado explícitamente. Solo permite ceros del libro cubierto después de primera observación; no afirma cero ventas, impago o conciliación bancaria. Sin ese modo se bloquea el constructor. Los supuestos y limitaciones están en cobertura_aplicada.json y se conservan los originales sin cambios.

Orden de todas las observaciones: mes, CustomerID. metadatos_train/val.csv contienen identidad, fechas y cortes. etiquetas_train/val.csv contienen por separado y, conteo objetivo y R. caracteristicas_crudas_train/val.csv permiten auditar X antes de transformar, con una columna de ID que se excluye mediante catálogo cerrado. elegibilidad.csv.gz explica también los meses sin historia; cuentas_meses.csv.gz separa actividad observada de ceros por cobertura académica.

El código matemático y los artefactos de Ventas permanecen intactos. El adaptador financiero conserva las reglas numéricas del preprocesador existente, con catálogo propio y sin parchear sus variables globales. No hay entrenamiento ni PCA nuevo. Las pruebas con datos sintéticos son solo fixtures de verificación, no sustituyen eventos empresariales.

## Reproducción

Desde la raíz, primera extracción focalizada local:

```powershell
python -m src.datamarts.transacciones.construir --origen sql --cobertura business_ledger_academic
```

Reproducción sin SQL usando el snapshot y la procedencia ya verificados:

```powershell
python -m src.datamarts.transacciones.construir --origen snapshot --cobertura business_ledger_academic
python -m unittest discover -s tests -p test_transacciones_dataset.py -v
python -m unittest discover -s tests -v
```

No modificar la auditoría para ajustar discrepancias. El pipeline aborta si cambia la proyección aprobada de cobros, el contrato, el catálogo o la reconciliación esperada. Las huellas COUNT/CHECKSUM antes/después son controles acotados no criptográficos, sin aislamiento snapshot ni prueba de inexistencia de escrituras externas; todas las consultas propias son SELECT y no se modifica SQL Server.

## Test reservado

No hay X_test ni y_test, ni consultas a eventos de 2016. especificacion_test.json define su construcción posterior, con preprocesador congelado, historial secuencial y verificación de cierre. El CLI actual solo admite desarrollo y no expone fechas de test. No se avanza automáticamente al entrenamiento o a otro Data Mart.
'''
    (OUT/'reporte_dataset.md').write_text(text,encoding='utf-8',newline='\n')


def run(*,origin,coverage,server='lpc:.\\AMBATO'):
    if coverage!=COVERAGE:raise ValueError('Declarar explícitamente el modo académico aprobado')
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',protected_files())
    verify_prior();approved=read(APPROVED/'contrato_datos.json');catalog(approved)
    if not approved['allowed_future_construction']:raise RuntimeError('Contrato sin autorización')
    snapshot=DATA/SNAPSHOT;approved_rows=approved_projection()
    if origin=='sql':
        log=[]
        try:rows,fingerprints=extract(server,log)
        finally:guardar(OUT/'consultas_sql.json',log)
        if rows!=approved_rows:raise RuntimeError('Cobros actuales difieren del snapshot aprobado; investigar')
        if snapshot.exists() and read_snapshot(snapshot)!=rows:raise RuntimeError('No sobrescribir snapshot distinto')
        uncompressed=save_snapshot(snapshot,rows)
        import hashlib
        guardar(OUT/'procedencia.json',dict(executed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),server=server,
            source_database='WideWorldImporters',source_table='Sales.CustomerTransactions',verification_database='WideWorldImportersDW',
            source_sql=SOURCE_SQL,source_sql_sha256=hashlib.sha256(SOURCE_SQL.encode()).hexdigest(),verification_sql=DW_SQL,
            parameters=list(PARAMS),events=len(rows),accounts=len({r['customer_id'] for r in rows}),fingerprints=fingerprints,
            compressed_snapshot_sha256=sha(snapshot),canonical_csv_sha256=uncompressed,snapshot_fields=list(rows[0]),
            approved_contract_sha256=sha(APPROVED/'contrato_datos.json'),approved_audit_projection_matches=True,
            coverage_mode=coverage,assumptions=approved['coverage']['assumptions'],last_edit_is_audit_of_creation=False))
    elif origin=='snapshot':
        provenance=read(OUT/'procedencia.json')
        if sha(snapshot)!=provenance['compressed_snapshot_sha256']:raise RuntimeError('Snapshot alterado')
        if provenance['approved_contract_sha256']!=sha(APPROVED/'contrato_datos.json'):raise RuntimeError('Cambió el contrato fuente')
        rows=read_snapshot(snapshot)
        if rows!=approved_rows:raise RuntimeError('Snapshot no corresponde a auditoría aprobada')
    else:raise ValueError('Origen inválido')
    summary,shapes,state=export(rows)
    guardar(OUT/'especificacion_test.json',test_spec(state))
    guardar(OUT/'entorno.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,gzip='level9,mtime0,empty_name',
                                  arrays='npy allow_pickle=False; float64 X, int8 y; deterministic within recorded numeric environment'))
    report(summary,shapes,state);before=verify_prior();guardar(OUT/'integridad_despues.json',before);manifest()
    print(json.dumps(dict(shapes=shapes,summary={k:{n:v for n,v in s.items() if n!='monthly'} for k,s in summary.items()},
                         input_features=len(FEATURES),final_features=len(state['feature_names']),discarded=state['discarded'],test_sealed=True),ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--origen',required=True,choices=['sql','snapshot'])
    p.add_argument('--cobertura',required=True,choices=[COVERAGE]);p.add_argument('--server',default='lpc:.\\AMBATO');args=p.parse_args()
    run(origin=args.origen,coverage=args.cobertura,server=args.server)
