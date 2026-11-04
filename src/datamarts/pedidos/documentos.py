"""Alternativa por documento: análisis de desarrollo, sin líneas ni matrices."""
import argparse
import datetime as dt
import json
from pathlib import Path

from src.db.conexion import Configuracion,Lector
from src.db.auditoria_dw import BASE,guardar,sha
from src.db.especificar_pedidos import Desarrollo,clasificar,resumir

OUT=BASE/'resultados/extension_datamarts/pedidos/documentos'
COHORTS=('original_without_backorder','original_with_backorder','derived_terminal','derived_with_backorder','ambiguous_lineage')


def tipo(incoming,outgoing):
    if incoming>1:return 'ambiguous_lineage'
    if incoming:return 'derived_with_backorder' if outgoing else 'derived_terminal'
    return 'original_with_backorder' if outgoing else 'original_without_backorder'


def base_sql():
    return """SELECT o.OrderID,o.OrderDate,o.PickingCompletedWhen AS pick_at,o.LastEditedWhen AS edited_at,
    o.BackorderOrderID,COALESCE(p.parents,0) AS parent_count,
    CASE WHEN COALESCE(p.parents,0)>1 THEN 'ambiguous_lineage'
      WHEN p.parents IS NOT NULL AND o.BackorderOrderID IS NOT NULL THEN 'derived_with_backorder'
      WHEN p.parents IS NOT NULL THEN 'derived_terminal'
      WHEN o.BackorderOrderID IS NOT NULL THEN 'original_with_backorder'
      ELSE 'original_without_backorder' END AS cohort,
    DATEADD(day,3,CAST(o.OrderDate AS datetime2)) AS horizon_end,
    CASE WHEN o.OrderDate<'20150101' THEN CAST('20150101' AS datetime2) ELSE CAST('20160101' AS datetime2) END AS cutoff
    FROM Sales.Orders o LEFT JOIN (SELECT BackorderOrderID,COUNT_BIG(*) AS parents FROM Sales.Orders WHERE BackorderOrderID IS NOT NULL GROUP BY BackorderOrderID) p ON o.OrderID=p.BackorderOrderID
    WHERE o.OrderDate < ?"""


def estado_sql():
    return """CASE WHEN OrderDate IS NULL OR pick_at<CAST(OrderDate AS datetime2) THEN 'invalid'
    WHEN horizon_end>cutoff THEN 'insufficient_followup'
    WHEN pick_at IS NULL THEN 'missing_pick_indeterminate'
    WHEN pick_at>=cutoff THEN 'pick_not_observed_by_cut'
    WHEN edited_at IS NULL OR edited_at>=cutoff THEN 'row_not_available_by_cut'
    WHEN CAST(pick_at AS date)>DATEADD(day,2,OrderDate) THEN 'after_two_days'
    ELSE 'within_two_days' END"""


def obtener(r):
    dev=Desarrollo(r);b=base_sql();s='SELECT b.*,'+estado_sql()+' AS state FROM ('+b+') b';out={}
    out['identity_coverage']=dev.select('SELECT COUNT_BIG(*) AS documents,COUNT_BIG(DISTINCT OrderID) AS distinct_documents,MIN(OrderDate) AS min_date,MAX(OrderDate) AS max_date,SUM(CAST(CASE WHEN pick_at IS NULL THEN 1 ELSE 0 END AS bigint)) AS current_pending_or_unobserved,SUM(CAST(CASE WHEN pick_at IS NOT NULL THEN 1 ELSE 0 END AS bigint)) AS current_completed_flag,SUM(CAST(CASE WHEN pick_at<CAST(OrderDate AS datetime2) THEN 1 ELSE 0 END AS bigint)) AS pick_before_order FROM ('+b+') q')[0]
    out['monthly_states_by_cohort']=dev.select('SELECT CONVERT(char(7),OrderDate,126) AS month,cohort,state,COUNT_BIG(*) AS documents FROM ('+s+') q GROUP BY CONVERT(char(7),OrderDate,126),cohort,state ORDER BY month,cohort,state')
    out['cohort_counts']=dev.select('SELECT cohort,COUNT_BIG(*) AS documents FROM ('+b+') q GROUP BY cohort ORDER BY cohort')
    out['backorder_dates']=dev.select('SELECT COUNT_BIG(*) AS edges,SUM(CAST(CASE WHEN c.OrderID IS NULL THEN 1 ELSE 0 END AS bigint)) AS orphan_child,SUM(CAST(CASE WHEN c.OrderDate<>o.OrderDate THEN 1 ELSE 0 END AS bigint)) AS different_dates,SUM(CAST(CASE WHEN c.OrderID<=o.OrderID THEN 1 ELSE 0 END AS bigint)) AS non_increasing_ids FROM Sales.Orders o LEFT JOIN Sales.Orders c ON c.OrderID=o.BackorderOrderID WHERE o.OrderDate < ? AND o.BackorderOrderID IS NOT NULL')[0]
    out['header_delay_histogram']=dev.select('SELECT cohort,DATEDIFF(day,OrderDate,CAST(pick_at AS date)) AS days_to_pick,COUNT_BIG(*) AS documents FROM ('+b+') q WHERE pick_at IS NULL OR (pick_at<cutoff AND edited_at<cutoff) GROUP BY cohort,DATEDIFF(day,OrderDate,CAST(pick_at AS date)) ORDER BY cohort,days_to_pick')
    out['fingerprint']=dev.select('SELECT COUNT_BIG(*) AS documents,CHECKSUM_AGG(BINARY_CHECKSUM(OrderID,OrderDate,pick_at,edited_at,BackorderOrderID,parent_count)) AS checksum FROM ('+b+') q')[0]
    return out


def resumir_cohortes(rows):
    result={'all_header_diagnostic':resumir(rows,'documents')}
    for cohort in COHORTS:result[cohort]=resumir([r for r in rows if r['cohort']==cohort],'documents')
    return result


def dictamen(summary):
    base=summary['original_without_backorder']
    both=all(base.get(s,{}).get('positive',0)>0 and base.get(s,{}).get('negative',0)>0 for s in ['train','validation'])
    return {'retrospective_business_anchor':'OrderDate propio del documento original no derivado; fecha de negocio, no timestamp de inserción auditado',
        'cohort_with_defensible_full_order_interpretation':'original_without_backorder',
        'can_build_dataset_now':False,
        'sufficient_class_support_in_both_development_splits':both,
        'verdict':'insufficient_evidence_or_events_for_requested_document_target' if not both else 'conditional_retrospective_document_candidate_requires_explicit_cohort_acceptance',
        'reason':'Los backorders heredan fecha y pueden cerrar un documento tras transferir pendientes. La cohorte original sin traslado se evalúa aparte; no se incorporan otras cohortes para conseguir positivos.',
        'fallback':'Continuar con Fact.Movement bajo un contrato propio; no iniciar esa investigación ni cambiar objetivo silenciosamente aquí.',
        'test_outcomes_read':False,'matrices_generated':False,'training_performed':False}


def snapshot():
    result={}
    new={BASE/'src/datamarts/pedidos/documentos.py',BASE/'src/datamarts/pedidos/reportes_documentos.py',BASE/'tests/test_pedidos_documentos.py'}
    for root in ['datos','referencias','resultados','src','tests','informe']:
        for p in (BASE/root).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents and p not in new:result[str(p.relative_to(BASE))]=sha(p)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--server');args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',snapshot())
    cfgpath=BASE/'resultados/extension_datamarts/auditoria_dw/conexion_verificada.json'
    server=args.server or 'lpc:'+json.loads(cfgpath.read_text(encoding='utf-8'))['connection'][0]['server_name']
    r=Lector(Configuracion.entorno(server,'WideWorldImporters').conectar())
    try:
        meta=r.select("SELECT t.name AS table_name,c.name AS column_name,ty.name AS data_type,c.is_nullable,dc.definition AS default_definition,t.temporal_type_desc FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON ty.user_type_id=c.user_type_id LEFT JOIN sys.default_constraints dc ON c.default_object_id=dc.object_id WHERE s.name=? AND t.name=? ORDER BY c.column_id",('Sales','Orders'))
        guardar(OUT/'campos_cabecera.json',meta);guardar(OUT/'consultas_metadatos.json',r.consultas);r.consultas=[]
        pre=Desarrollo(r).select('SELECT COUNT_BIG(*) AS documents,CHECKSUM_AGG(BINARY_CHECKSUM(OrderID,OrderDate,pick_at,edited_at,BackorderOrderID,parent_count)) AS checksum FROM ('+base_sql()+') q')[0]
        guardar(OUT/'huella_antes.json',pre)
        data=obtener(r);guardar(OUT/'agregados_documentos.json',data)
        if pre!=data['fingerprint']:raise RuntimeError('Cambió la huella de documentos durante la consulta')
        guardar(OUT/'huella_despues.json',data['fingerprint'])
        summary=resumir_cohortes(data['monthly_states_by_cohort']);guardar(OUT/'etiquetas_desarrollo.json',summary)
        decision=dictamen(summary);guardar(OUT/'dictamen.json',decision)
        print(json.dumps({'identity':data['identity_coverage'],'cohort_counts':data['cohort_counts'],'cohort_labels':{c:{s:v[s] for s in ['train','validation'] if s in v} for c,v in summary.items()},'decision':decision},default=str))
        from .reportes_documentos import exportar
        exportar(data,summary,decision)
        before=json.loads((OUT/'integridad_antes.json').read_text(encoding='utf-8'));after={n:sha(BASE/n) for n in before}
        if before!=after:raise RuntimeError('Cambió una evidencia anterior')
        guardar(OUT/'integridad_despues.json',after)
    finally:guardar(OUT/'consultas_desarrollo.json',r.consultas);r.close()
    manifiesto()


def manifiesto():
    files=[BASE/'src/datamarts/pedidos/documentos.py',BASE/'src/datamarts/pedidos/reportes_documentos.py',BASE/'tests/test_pedidos_documentos.py']
    guardar(OUT/'manifiesto.json',{'scope':'Viabilidad por documento sin matrices ni entrenamiento; desarrollo anterior a 2016',
        'implementation_sha256':{str(p.relative_to(BASE)):sha(p) for p in files if p.exists()},
        'artifacts_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'manifiesto.json','validaciones.json'}}})


if __name__=='__main__':main()
