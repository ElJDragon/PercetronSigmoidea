"""Verifica existencia al registro antes de autorizar cualquier matriz de Pedidos.

Solo metadatos y agregados de desarrollo. Nunca infiere fecha de creación a
partir de picking, última edición, secuencia o fecha heredada de un encabezado.
"""
import argparse
import calendar
import datetime as dt
import json
import math
from pathlib import Path

from src.db.conexion import Configuracion,Lector
from src.db.auditoria_dw import BASE,guardar,sha
from src.db.especificar_pedidos import Desarrollo,base_sql,line_sql

OUT=BASE/'resultados/extension_datamarts/pedidos/dataset'
SPEC=OUT.parent
PROCESSED=BASE/'datos/Procesados/pedidos'
FEATURES=('mes_sin','mes_cos','semana_sin','semana_cos','dia_mes','dias_mes')


def calendario(fecha):
    """Seis valores definidos por contrato; no consulta ningún otro campo."""
    if type(fecha) is not dt.date:raise ValueError('Fecha de referencia date requerida')
    m,w=fecha.month-1,fecha.weekday()
    return dict(zip(FEATURES,(math.sin(2*math.pi*m/12),math.cos(2*math.pi*m/12),math.sin(2*math.pi*w/7),math.cos(2*math.pi*w/7),fecha.day,calendar.monthrange(fecha.year,fecha.month)[1])))


def verificar_existencia(fecha_referencia,evidencia=None):
    """Ejemplos de evidencia independiente para una puerta cerrada por defecto.

    Un campo CreatedAt arbitrario tampoco certifica procedencia: se requieren
    origen validado y asociación por línea, no una fecha inferida de su outcome.
    Las fuentes reales consultadas no aportan actualmente esta evidencia.
    """
    if not evidencia or not evidencia.get('provenance_verified') or evidencia.get('kind') not in {'creation_audit','asof_row_version'}:
        return 'creation_not_proven'
    created=evidencia.get('created_at')
    if not isinstance(created,dt.datetime):return 'creation_not_proven'
    if created.date()>fecha_referencia:return 'not_existing_on_reference_day'
    if created.date()!=fecha_referencia:return 'creation_day_differs_from_reference'
    return 'verified_reference_day'


def autorizar_matrices(evidencias):
    if not evidencias or any(e!='verified_reference_day' for e in evidencias):
        raise ValueError('Construcción bloqueada: falta evidencia independiente de existencia al registro')
    return True


def metadata(r):
    params=('Sales','Orders','OrderLines')
    result={}
    result['visibility']=r.select("SELECT DB_NAME() AS database_name,HAS_PERMS_BY_NAME(DB_NAME(),'DATABASE','VIEW DEFINITION') AS can_view_definition")
    result['columns']=r.select("SELECT t.name AS table_name,c.column_id,c.name AS column_name,ty.name AS data_type,c.is_nullable,c.is_identity,c.generated_always_type_desc,dc.definition AS default_definition FROM sys.tables t JOIN sys.schemas s ON t.schema_id=s.schema_id JOIN sys.columns c ON t.object_id=c.object_id JOIN sys.types ty ON ty.user_type_id=c.user_type_id LEFT JOIN sys.default_constraints dc ON dc.object_id=c.default_object_id WHERE s.name=? AND t.name IN (?,?) ORDER BY t.name,c.column_id",params)
    result['history_configuration']=r.select("SELECT s.name AS schema_name,t.name AS table_name,t.temporal_type_desc,t.is_tracked_by_cdc,OBJECT_SCHEMA_NAME(t.history_table_id) AS history_schema,OBJECT_NAME(t.history_table_id) AS history_table,CASE WHEN ct.object_id IS NULL THEN 0 ELSE 1 END AS change_tracking FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id LEFT JOIN sys.change_tracking_tables ct ON t.object_id=ct.object_id WHERE s.name=? AND t.name IN (?,?)",params)
    result['triggers']=r.select("SELECT t.name AS table_name,tr.name AS trigger_name,tr.is_disabled,m.definition FROM sys.triggers tr JOIN sys.tables t ON t.object_id=tr.parent_id JOIN sys.schemas s ON s.schema_id=t.schema_id LEFT JOIN sys.sql_modules m ON m.object_id=tr.object_id WHERE s.name=? AND t.name IN (?,?)",params)
    result['candidate_history_tables']=r.select("SELECT s.name AS schema_name,t.name AS table_name,t.temporal_type_desc FROM sys.tables t JOIN sys.schemas s ON t.schema_id=s.schema_id WHERE t.name LIKE ? OR t.name LIKE ? OR t.name LIKE ? OR t.name LIKE ?",('%Order%','%Audit%','%Histor%','%Archiv%'))
    result['candidate_history_views']=r.select("SELECT s.name AS schema_name,v.name AS view_name FROM sys.views v JOIN sys.schemas s ON s.schema_id=v.schema_id WHERE v.name LIKE ? OR v.name LIKE ? OR v.name LIKE ?",('%Order%','%Audit%','%Line%'))
    result['candidate_history_synonyms']=r.select("SELECT s.name AS schema_name,n.name AS synonym_name,n.base_object_name FROM sys.synonyms n JOIN sys.schemas s ON s.schema_id=n.schema_id WHERE n.name LIKE ? OR n.name LIKE ? OR n.name LIKE ?",('%Order%','%Audit%','%Line%'))
    result['descriptions']=r.select("SELECT t.name AS table_name,c.name AS column_name,CAST(ep.value AS nvarchar(4000)) AS description FROM sys.extended_properties ep JOIN sys.tables t ON t.object_id=ep.major_id JOIN sys.schemas s ON s.schema_id=t.schema_id LEFT JOIN sys.columns c ON c.object_id=t.object_id AND c.column_id=ep.minor_id WHERE s.name=? AND t.name IN (?,?) AND ep.name=?",params+('MS_Description',))
    return result


def datos_desarrollo(r):
    dev=Desarrollo(r);b=base_sql();s=line_sql();result={}
    result['fingerprint']=dev.select('SELECT COUNT_BIG(*) AS lines,CHECKSUM_AGG(BINARY_CHECKSUM(OrderID,OrderLineID,OrderDate,StockItemID,CustomerID,pick_at,line_edited,header_edited,Quantity,UnitPrice)) AS checksum FROM ('+b+') q')[0]
    result['identity']=dev.select('SELECT COUNT_BIG(*) AS lines,COUNT_BIG(DISTINCT OrderLineID) AS order_line_ids,COUNT_BIG(DISTINCT OrderID) AS current_order_ids FROM ('+b+') q')[0]
    result['duplicate_current_order_stock']=dev.select('SELECT COUNT_BIG(*) AS groups,COALESCE(SUM(n-1),0) AS excess FROM (SELECT OrderID,StockItemID,COUNT_BIG(*) AS n FROM ('+b+') q GROUP BY OrderID,StockItemID HAVING COUNT_BIG(*)>1) z')[0]
    result['dw_exact_bridge']=dev.select('SELECT COUNT_BIG(*) AS dw_rows,COUNT_BIG(l.OrderLineID) AS matched_lines,COUNT_BIG(DISTINCT l.OrderLineID) AS distinct_lines,SUM(CAST(CASE WHEN l.OrderLineID IS NULL THEN 1 ELSE 0 END AS bigint)) AS no_source,SUM(CAST(CASE WHEN d.[Order Date Key]<>o.OrderDate THEN 1 ELSE 0 END AS bigint)) AS date_discrepancies FROM [WideWorldImportersDW].[Fact].[Order] d JOIN [WideWorldImportersDW].[Dimension].[Stock Item] si ON si.[Stock Item Key]=d.[Stock Item Key] JOIN Sales.Orders o ON o.OrderID=d.[WWI Order ID] LEFT JOIN Sales.OrderLines l ON l.OrderID=o.OrderID AND l.StockItemID=si.[WWI Stock Item ID] WHERE o.OrderDate < ?')[0]
    result['last_edit_by_period']=dev.select("SELECT CASE WHEN OrderDate<'20150101' THEN 'train' ELSE 'validation' END AS partition,COUNT_BIG(*) AS lines,SUM(CAST(CASE WHEN CAST(line_edited AS date)=OrderDate THEN 1 ELSE 0 END AS bigint)) AS last_edit_same_day,SUM(CAST(CASE WHEN CAST(line_edited AS date)>OrderDate THEN 1 ELSE 0 END AS bigint)) AS last_edit_later_day,SUM(CAST(CASE WHEN CAST(line_edited AS date)<OrderDate THEN 1 ELSE 0 END AS bigint)) AS last_edit_earlier_day FROM ("+b+") q GROUP BY CASE WHEN OrderDate<'20150101' THEN 'train' ELSE 'validation' END")
    result['restricted_diagnostic_not_dataset']=dev.select("SELECT CASE WHEN OrderDate<'20150101' THEN 'train' ELSE 'validation' END AS partition,COUNT_BIG(*) AS previously_eligible,SUM(CAST(CASE WHEN CAST(line_edited AS date)=OrderDate THEN 1 ELSE 0 END AS bigint)) AS same_day_edit,SUM(CAST(CASE WHEN CAST(line_edited AS date)=OrderDate AND state='after_two_days' THEN 1 ELSE 0 END AS bigint)) AS positive_with_same_day_edit,SUM(CAST(CASE WHEN CAST(line_edited AS date)>OrderDate THEN 1 ELSE 0 END AS bigint)) AS later_day_edit,SUM(CAST(CASE WHEN state='after_two_days' THEN 1 ELSE 0 END AS bigint)) AS previous_positive FROM ("+s+") q WHERE state IN ('within_two_days','after_two_days') GROUP BY CASE WHEN OrderDate<'20150101' THEN 'train' ELSE 'validation' END")
    result['quantity_price_versions']=dev.select('SELECT COUNT_BIG(*) AS lines,SUM(CAST(CASE WHEN UnitPrice IS NULL THEN 1 ELSE 0 END AS bigint)) AS current_price_null,SUM(CAST(CASE WHEN Quantity<=0 THEN 1 ELSE 0 END AS bigint)) AS current_nonpositive_quantity,SUM(CAST(CASE WHEN UnitPrice<0 THEN 1 ELSE 0 END AS bigint)) AS current_negative_price FROM ('+b+') q')[0]
    result['backorder_lineage']=dev.select('SELECT COUNT_BIG(*) AS edges,SUM(CAST(CASE WHEN o.OrderDate<>c.OrderDate THEN 1 ELSE 0 END AS bigint)) AS different_reference_dates,SUM(CAST(CASE WHEN c.OrderID<=o.OrderID THEN 1 ELSE 0 END AS bigint)) AS non_increasing_ids FROM Sales.Orders o JOIN Sales.Orders c ON c.OrderID=o.BackorderOrderID WHERE o.OrderDate < ?')[0]
    result['metadata_sample_no_picking']=dev.select('SELECT TOP (12) l.OrderLineID,l.OrderID AS current_order_id,o.OrderDate AS reference_date,l.LastEditedWhen AS last_edit_only FROM Sales.OrderLines l JOIN Sales.Orders o ON l.OrderID=o.OrderID WHERE o.OrderDate < ? ORDER BY l.OrderLineID')
    return result


def decision(meta,data):
    # Ninguna de estas fuentes ofrece una alta independiente por fila.
    birth=[c for c in meta['columns'] if c['table_name']=='OrderLines' and c['column_name'].lower() in {'createdat','createdwhen','createdon','creationdate','createddate','insertedat','insertedwhen'}]
    available_history=any(t['temporal_type_desc']!='NON_TEMPORAL_TABLE' or t['is_tracked_by_cdc'] or t['change_tracking'] for t in meta['history_configuration'])
    return {'status':'blocked_pending_independent_creation_evidence','reason':'La existencia en la fecha original no se prueba por una clave actual, una fecha heredada o última edición. No se cuenta con evidencia independiente por línea en las fuentes consultadas.',
        'dedicated_creation_columns_found':birth,'history_configuration_found':bool(available_history),
        'development_candidates':data['identity']['lines'],'included_in_dataset':0,'blocked_from_materialization':data['identity']['lines'],
        'not_a_claim_that_all_lines_were_created_late':True,'matrices_generated':False,'preprocessor_fitted':False,'test_predictors_generated':False,'test_outcomes_read':False,
        'next_step':'Obtener log de inserción/CreatedAt verificable o snapshot de líneas al registro, con historial de traslados y valores originales. Si no existe, acordar explícitamente otro instante/objetivo antes de construir.'}


def snapshot():
    result={}
    for root in ['datos','referencias','resultados','src','tests','informe']:
        for p in (BASE/root).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents and not (BASE/'src/datamarts') in p.parents and p!=BASE/'tests/test_dataset_pedidos.py':result[str(p.relative_to(BASE))]=sha(p)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--server');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',snapshot())
    verified=json.loads((BASE/'resultados/extension_datamarts/auditoria_dw/conexion_verificada.json').read_text(encoding='utf-8'))
    cfg=Configuracion.entorno(args.server or 'lpc:'+verified['connection'][0]['server_name'],'WideWorldImporters')
    reader=Lector(cfg.conectar())
    try:
        meta=metadata(reader);guardar(OUT/'metadatos_creacion.json',meta)
        guardar(OUT/'consultas_metadatos.json',reader.consultas)
        reader.consultas=[]
        pre=Desarrollo(reader).select('SELECT COUNT_BIG(*) AS lines,CHECKSUM_AGG(BINARY_CHECKSUM(OrderID,OrderLineID,OrderDate,StockItemID,CustomerID,pick_at,line_edited,header_edited,Quantity,UnitPrice)) AS checksum FROM ('+base_sql()+') q')[0]
        guardar(OUT/'huella_antes.json',pre)
        data=datos_desarrollo(reader);guardar(OUT/'trazabilidad_desarrollo.json',data)
        if pre!=data['fingerprint']:raise RuntimeError('La huella de desarrollo cambió durante la verificación')
        guardar(OUT/'huella_despues.json',data['fingerprint'])
        verdict=decision(meta,data);guardar(OUT/'estado_construccion.json',verdict)
        from .reportes_dataset import exportar
        exportar(meta,data,verdict)
        old=json.loads((OUT/'integridad_antes.json').read_text(encoding='utf-8'))
        after={name:sha(BASE/name) for name in old}
        if old!=after:raise RuntimeError('Cambió un archivo anterior')
        guardar(OUT/'integridad_despues.json',after)
        print(json.dumps(verdict))
    finally:guardar(OUT/'consultas_desarrollo.json',reader.consultas);reader.close()
    manifiesto()


def manifiesto():
    impl=list((BASE/'src/datamarts/pedidos').glob('*.py'))+[BASE/'src/datamarts/__init__.py',BASE/'tests/test_dataset_pedidos.py']
    guardar(OUT/'manifiesto.json',{'scope':'Puerta temporal del dataset: construcción detenida por evidencia insuficiente de existencia al registro',
        'test_seal':'Ningún outcome de OrderDate>=2016-01-01 consultado','dataset_materialized':False,
        'implementation_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(impl) if p.exists()},
        'artifacts_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'manifiesto.json','validaciones.json'}}})


if __name__=='__main__':main()
