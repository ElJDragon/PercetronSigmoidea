"""Contrato de Pedidos: agregados de desarrollo, sin matrices ni etiquetas test."""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

from .conexion import Configuracion,Lector
from .auditoria_dw import BASE,guardar,sha

OUT=BASE/'resultados/extension_datamarts/pedidos'
AUDIT=BASE/'resultados/extension_datamarts/auditoria_dw'
TEST_START=dt.date(2016,1,1)
TRAIN_CUT=dt.datetime(2015,1,1)
VAL_CUT=dt.datetime(2016,1,1)
STATES=('invalid','insufficient_followup','missing_pick_indeterminate','pick_not_observed_by_cut','row_not_available_by_cut','within_two_days','after_two_days')


def clasificar(order_date,pick,known_at,cut):
    """Especificación pura para comprobar fronteras; no construye dataset."""
    if order_date is None or (pick is not None and pick<dt.datetime.combine(order_date,dt.time())):return 'invalid'
    if dt.datetime.combine(order_date+dt.timedelta(days=3),dt.time())>cut:return 'insufficient_followup'
    if pick is None:return 'missing_pick_indeterminate'
    if pick>=cut:return 'pick_not_observed_by_cut'
    if known_at is None or known_at>=cut:return 'row_not_available_by_cut'
    return 'after_two_days' if pick.date()>order_date+dt.timedelta(days=2) else 'within_two_days'


def estado_documento(states):
    if not states or 'invalid' in states:return 'invalid'
    if 'insufficient_followup' in states:return 'insufficient_followup'
    if 'after_two_days' in states:return 'after_two_days'
    if all(s=='within_two_days' for s in states):return 'within_two_days'
    return 'indeterminate_document'


class Desarrollo:
    """La única vía para datos de outcomes exige corte exclusivo de test.

    Los metadatos se leen con Lector; las plantillas de datos siempre anclan
    Sales.Orders o y enlazan sus líneas por ID. El corte no es configurable.
    """
    def __init__(self,reader):self.reader=reader
    def select(self,sql):
        clean=re.sub(r"N?'(?:''|[^'])*'",' ',sql)
        if clean.count('WHERE o.OrderDate < ?')!=1 or clean.count('?')!=1 or re.search(r'\b(UNION|EXCEPT|INTERSECT)\b',clean,re.I):raise ValueError('Falta barrera única de desarrollo o consulta fuera de plantillas')
        return self.reader.select(sql,(TEST_START,))


def base_sql():
    return """SELECT o.OrderID,l.OrderLineID,o.OrderDate,l.StockItemID,o.CustomerID,
    l.Quantity,l.UnitPrice,l.PickedQuantity,l.PickingCompletedWhen AS pick_at,
    o.PickingCompletedWhen AS header_pick_at,l.LastEditedWhen AS line_edited,
    o.LastEditedWhen AS header_edited,
    CASE WHEN l.LastEditedWhen>o.LastEditedWhen THEN l.LastEditedWhen ELSE o.LastEditedWhen END AS known_at,
    DATEADD(day,3,CAST(o.OrderDate AS datetime2)) AS horizon_end,
    CASE WHEN o.OrderDate<'20150101' THEN CAST('20150101' AS datetime2) ELSE CAST('20160101' AS datetime2) END AS partition_cut
    FROM Sales.OrderLines l JOIN Sales.Orders o ON l.OrderID=o.OrderID WHERE o.OrderDate < ?"""


def state_sql():
    return """CASE WHEN OrderDate IS NULL OR pick_at<CAST(OrderDate AS datetime2) THEN 'invalid'
    WHEN horizon_end>partition_cut THEN 'insufficient_followup'
    WHEN pick_at IS NULL THEN 'missing_pick_indeterminate'
    WHEN pick_at>=partition_cut THEN 'pick_not_observed_by_cut'
    WHEN known_at IS NULL OR known_at>=partition_cut THEN 'row_not_available_by_cut'
    WHEN CAST(pick_at AS date)>DATEADD(day,2,OrderDate) THEN 'after_two_days'
    ELSE 'within_two_days' END"""


def line_sql():return 'SELECT b.*,'+state_sql()+' AS state FROM ('+base_sql()+') b'


def obtener(reader):
    dev=Desarrollo(reader);result={}
    b=base_sql();s=line_sql()
    result['line_states_monthly']=dev.select("SELECT CONVERT(char(7),OrderDate,126) AS month,state,COUNT_BIG(*) AS lines,COUNT(DISTINCT OrderID) AS documents FROM ("+s+") q GROUP BY CONVERT(char(7),OrderDate,126),state ORDER BY month,state")
    result['line_structure_monthly']=dev.select('SELECT CONVERT(char(7),OrderDate,126) AS month,COUNT_BIG(*) AS lines,COUNT(DISTINCT OrderID) AS documents,COUNT(DISTINCT OrderLineID) AS stable_lines,COUNT(DISTINCT StockItemID) AS products,COUNT(DISTINCT CustomerID) AS customers FROM ('+b+') q GROUP BY CONVERT(char(7),OrderDate,126) ORDER BY month')
    grouped="""SELECT OrderID,OrderDate,partition_cut,MAX(horizon_end) AS horizon_end,COUNT_BIG(*) AS n,
    SUM(CASE WHEN state='invalid' THEN 1 ELSE 0 END) AS invalid_n,
    SUM(CASE WHEN state='after_two_days' THEN 1 ELSE 0 END) AS late_n,
    SUM(CASE WHEN state='within_two_days' THEN 1 ELSE 0 END) AS within_n,
    SUM(CASE WHEN pick_at IS NULL THEN 1 ELSE 0 END) AS missing_n,
    COUNT(DISTINCT CAST(pick_at AS date)) AS picked_dates,
    MIN(pick_at) AS first_pick,MAX(pick_at) AS last_pick,MAX(header_pick_at) AS header_pick
    FROM ("""+s+') q GROUP BY OrderID,OrderDate,partition_cut'
    doc_state="""CASE WHEN invalid_n>0 THEN 'invalid' WHEN horizon_end>partition_cut THEN 'insufficient_followup'
    WHEN late_n>0 THEN 'after_two_days' WHEN within_n=n THEN 'within_two_days' ELSE 'indeterminate_document' END"""
    result['document_states_monthly']=dev.select('SELECT CONVERT(char(7),OrderDate,126) AS month,'+doc_state+' AS state,COUNT_BIG(*) AS documents,SUM(n) AS lines FROM ('+grouped+') q GROUP BY CONVERT(char(7),OrderDate,126),'+doc_state+' ORDER BY month,state')
    result['document_composition']=dev.select('SELECT COUNT_BIG(*) AS documents,SUM(n) AS lines,MIN(n) AS min_lines,MAX(n) AS max_lines,SUM(CASE WHEN picked_dates>1 THEN 1 ELSE 0 END) AS documents_multiple_picked_dates,SUM(CASE WHEN missing_n>0 AND missing_n<n THEN 1 ELSE 0 END) AS mixed_missing_observed,SUM(CASE WHEN missing_n=n THEN 1 ELSE 0 END) AS all_missing,SUM(CASE WHEN late_n>0 AND within_n>0 THEN 1 ELSE 0 END) AS mixed_visible_classes,SUM(CASE WHEN header_pick IS NOT NULL AND last_pick IS NOT NULL AND header_pick<>last_pick THEN 1 ELSE 0 END) AS header_vs_last_line_timestamp_different FROM ('+grouped+') q')[0]
    result['line_count_distribution']=dev.select('SELECT n AS lines_per_order,COUNT_BIG(*) AS documents FROM ('+grouped+') q GROUP BY n ORDER BY n')
    result['line_delay_histogram_development']=dev.select("SELECT DATEDIFF(day,OrderDate,CAST(pick_at AS date)) AS delay_days,COUNT_BIG(*) AS lines FROM ("+b+") q GROUP BY DATEDIFF(day,OrderDate,CAST(pick_at AS date)) ORDER BY delay_days")
    result['edited_dates']=dev.select('SELECT COUNT_BIG(*) AS lines,SUM(CASE WHEN CAST(line_edited AS date)=OrderDate THEN 1 ELSE 0 END) AS last_line_edit_same_calendar_day,SUM(CASE WHEN CAST(line_edited AS date)>OrderDate THEN 1 ELSE 0 END) AS last_line_edit_later_day,SUM(CASE WHEN pick_at IS NOT NULL AND line_edited<pick_at THEN 1 ELSE 0 END) AS last_edit_before_pick FROM ('+b+') q')[0]
    result['backorder_links']=dev.select('SELECT COUNT_BIG(*) AS links,SUM(CASE WHEN c.OrderDate<>o.OrderDate THEN 1 ELSE 0 END) AS different_order_date,SUM(CASE WHEN c.OrderID<=o.OrderID THEN 1 ELSE 0 END) AS non_increasing_ids FROM Sales.Orders o JOIN Sales.Orders c ON o.BackorderOrderID=c.OrderID WHERE o.OrderDate < ?')[0]
    result['line_states_by_backorder_document']=dev.select('SELECT state,CASE WHEN p.id IS NULL THEN 0 ELSE 1 END AS is_derived_document,COUNT_BIG(*) AS lines FROM ('+s+') q LEFT JOIN (SELECT DISTINCT BackorderOrderID AS id FROM Sales.Orders WHERE BackorderOrderID IS NOT NULL) p ON p.id=q.OrderID GROUP BY state,CASE WHEN p.id IS NULL THEN 0 ELSE 1 END ORDER BY state,is_derived_document')
    result['development_document_child_status']=dev.select('SELECT COUNT_BIG(*) AS documents,SUM(CASE WHEN p.id IS NOT NULL THEN 1 ELSE 0 END) AS derived_backorder_documents FROM Sales.Orders o LEFT JOIN (SELECT DISTINCT BackorderOrderID AS id FROM Sales.Orders WHERE BackorderOrderID IS NOT NULL) p ON p.id=o.OrderID WHERE o.OrderDate < ?')[0]
    result['unknown_customer_development']=dev.select('SELECT COUNT_BIG(*) AS dw_lines,SUM(CASE WHEN d.[Customer Key]=0 THEN 1 ELSE 0 END) AS zero_customer,SUM(CASE WHEN d.[Customer Key] IS NULL THEN 1 ELSE 0 END) AS null_customer,COUNT(DISTINCT CASE WHEN d.[Customer Key]=0 THEN o.CustomerID END) AS operational_ids_for_unknown,SUM(CASE WHEN d.[Customer Key]=0 AND c.[Customer Key] IS NULL THEN 1 ELSE 0 END) AS missing_business_id_in_dimension FROM [WideWorldImportersDW].[Fact].[Order] d JOIN Sales.Orders o ON d.[WWI Order ID]=o.OrderID LEFT JOIN [WideWorldImportersDW].[Dimension].[Customer] c ON c.[WWI Customer ID]=o.CustomerID WHERE o.OrderDate < ?')[0]
    result['fingerprint_development']=dev.select('SELECT COUNT_BIG(*) AS lines,CHECKSUM_AGG(BINARY_CHECKSUM(OrderID,OrderLineID,OrderDate,StockItemID,CustomerID,pick_at,line_edited,header_edited)) AS checksum FROM ('+b+') q')[0]
    return result


def reconciliar_auditoria():
    """Solo reutiliza agregados ya publicados por auditoría; sin nueva lectura test."""
    a=json.loads((AUDIT/'comprobaciones_adicionales.json').read_text(encoding='utf-8'))['order_line_candidate_2day']
    p=json.loads((AUDIT/'perfiles_calidad.json').read_text(encoding='utf-8'))['Fact.Order']
    rows={'picking_dentro_horizonte_observado_maduro':a['observed_within_2d'],
          'picking_posterior_observado_maduro':a['observed_late'],
          'sin_picking_con_seguimiento_nominal_indeterminado':a['unknown_completion'],
          'sin_seguimiento_nominal_suficiente':p['rows']-a['mature_lines'],
          'inconsistencia_fecha_pick_anterior_a_pedido':json.loads((AUDIT/'granularidad.json').read_text(encoding='utf-8'))['Order_dates']['picked_before_order']}
    if sum(rows.values())!=p['rows']:raise ValueError('Conciliación histórica inconsistente')
    return {'source_scope':'Agregados históricos de toda la auditoría anterior, incluidos meses ahora reservados; solo conciliación, nunca selección o prevalencia test.','states':rows,'total':p['rows'],'observed_mature':a['observed_within_2d']+a['observed_late'],'null_picking_total':p['columns']['Picked Date Key']['nulls'],'null_picking_in_immature':p['columns']['Picked Date Key']['nulls']-a['unknown_completion']}


def resumir(rows,count_key):
    result={}
    for r in rows:
        partition='train' if r['month']<'2015-01' else 'validation'
        for key in [partition,r['month'][:4],r['month']]:
            item=result.setdefault(key,{'states':{},'positive':0,'negative':0,'eligible':0,'total':0})
            n=r[count_key];item['total']+=n;item['states'][r['state']]=item['states'].get(r['state'],0)+n
            if r['state']=='after_two_days':item['positive']+=n
            if r['state']=='within_two_days':item['negative']+=n
    for item in result.values():
        item['eligible']=item['positive']+item['negative'];item['positive_proportion']=item['positive']/item['eligible'] if item['eligible'] else None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--server');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():
        protected={}
        for root in ['datos','referencias','resultados','src','tests','informe']:
            for p in (BASE/root).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts and OUT not in p.parents:protected[str(p.relative_to(BASE))]=sha(p)
        guardar(OUT/'integridad_antes.json',protected)
    verified=json.loads((AUDIT/'conexion_verificada.json').read_text(encoding='utf-8'))
    cfg=Configuracion.entorno(args.server or 'lpc:'+verified['connection'][0]['server_name'],'WideWorldImporters')
    r=Lector(cfg.conectar())
    try:
        pre=Desarrollo(r).select('SELECT COUNT_BIG(*) AS lines,CHECKSUM_AGG(BINARY_CHECKSUM(OrderID,OrderLineID,OrderDate,StockItemID,CustomerID,pick_at,line_edited,header_edited)) AS checksum FROM ('+base_sql()+') q')[0]
        guardar(OUT/'huella_desarrollo_antes.json',pre)
        data=obtener(r);guardar(OUT/'agregados_desarrollo.json',data)
        if pre!=data['fingerprint_development']:raise RuntimeError('Cambió la huella de desarrollo durante la consulta')
        guardar(OUT/'huella_desarrollo_despues.json',data['fingerprint_development'])
        guardar(OUT/'reconciliacion_historica.json',reconciliar_auditoria())
        guardar(OUT/'distribucion_etiquetas_desarrollo.json',{'line':resumir(data['line_states_monthly'],'lines'),'document_diagnostic':resumir(data['document_states_monthly'],'documents'),'test':'No consultado: se excluye OrderDate>=2016-01-01 antes de agregar outcomes.'})
        from .reportes_pedidos import exportar
        exportar()
        before=json.loads((OUT/'integridad_antes.json').read_text(encoding='utf-8'))
        after={p:sha(BASE/p) for p in before}
        if before!=after:raise RuntimeError('Se modificó una evidencia previa')
        guardar(OUT/'integridad_despues.json',after)
        print(json.dumps({'line':resumir(data['line_states_monthly'],'lines')['train'],'validation':resumir(data['line_states_monthly'],'lines')['validation'],'composition':data['document_composition']},default=str))
    finally:guardar(OUT/'consultas_desarrollo.json',r.consultas);r.close()
    manifiesto()


def manifiesto():
    impl=[BASE/'src/db/especificar_pedidos.py',BASE/'src/db/reportes_pedidos.py',BASE/'tests/test_especificacion_pedidos.py']
    guardar(OUT/'manifiesto.json',{'scope':'Especificación de Pedidos, solo desarrollo: sin matrices, entrenamiento, ETL o nueva consulta de outcomes test',
        'test_outcome_boundary_exclusive':str(TEST_START),'historical_audit_counts_reused_only_for_reconciliation':True,
        'implementation_sha256':{str(p.relative_to(BASE)):sha(p) for p in impl if p.exists()},
        'artifacts_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'manifiesto.json','validaciones.json'}}})


if __name__=='__main__':main()
