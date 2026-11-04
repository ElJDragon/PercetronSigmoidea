"""Investigación focalizada de Compras; SELECT, desarrollo cerrado antes de 2016."""
import argparse
import datetime as dt
import json
from collections import defaultdict, Counter
from statistics import median
from pathlib import Path

from src.db.conexion import Configuracion, Lector, validar_select
from src.db.auditoria_dw import BASE, guardar, sha

OUT = BASE / 'resultados/extension_datamarts/compras'
CUT = dt.date(2016, 1, 1)
NEW = {'src/datamarts/compras_viabilidad.py', 'src/datamarts/reportes_compras.py', 'tests/test_compras_viabilidad.py'}


def snapshot():
    return {str(p.relative_to(BASE)): sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
            for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and OUT not in p.parents and str(p.relative_to(BASE)).replace('\\','/') not in NEW}


def reader(database):
    return Lector(Configuracion('lpc:.\\AMBATO', database).conectar())


def discover():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'integridad_antes.json').exists(): guardar(OUT/'integridad_antes.json', snapshot())
    r = reader('WideWorldImporters')
    try:
        guardar(OUT/'metadatos_oltp.json', r.select("SELECT s.name AS schema_name,t.name AS table_name,c.name AS column_name,ty.name AS data_type,c.is_nullable,t.temporal_type_desc,t.is_tracked_by_cdc,t.history_table_id FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON ty.user_type_id=c.user_type_id WHERE (s.name='Purchasing' AND t.name IN ('PurchaseOrders','PurchaseOrderLines','Suppliers')) OR (s.name='Warehouse' AND t.name='StockItems') ORDER BY s.name,t.name,c.column_id"))
        guardar(OUT/'modulos_compras.json', r.select("SELECT s.name AS schema_name,o.name AS module_name,m.definition FROM sys.sql_modules m JOIN sys.objects o ON o.object_id=m.object_id JOIN sys.schemas s ON s.schema_id=o.schema_id WHERE m.definition LIKE '%Purchasing.PurchaseOrders%' OR m.definition LIKE '%Purchasing.PurchaseOrderLines%'"))
        guardar(OUT/'restricciones.json', r.select("SELECT s.name AS schema_name,t.name AS table_name,i.name AS index_name,i.is_unique,c.name AS column_name,ic.key_ordinal FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.indexes i ON i.object_id=t.object_id JOIN sys.index_columns ic ON ic.object_id=i.object_id AND ic.index_id=i.index_id JOIN sys.columns c ON c.object_id=ic.object_id AND c.column_id=ic.column_id WHERE s.name='Purchasing' AND t.name IN ('PurchaseOrders','PurchaseOrderLines') AND i.is_unique=1 ORDER BY t.name,i.name,ic.key_ordinal"))
        guardar(OUT/'tracking.json', r.select("SELECT t.name AS table_name,(SELECT COUNT_BIG(*) FROM sys.change_tracking_tables ct WHERE ct.object_id=t.object_id) AS change_tracking,(SELECT COUNT_BIG(*) FROM sys.triggers tr WHERE tr.parent_id=t.object_id) AS triggers FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id WHERE s.name='Purchasing' AND t.name IN ('PurchaseOrders','PurchaseOrderLines')"))
    finally:
        guardar(OUT/'consultas_metadatos.json', r.consultas);r.close()


def scoped(r, sql):
    """Plantillas de datos siempre con un corte parametrizado único de desarrollo."""
    validar_select(sql)
    markers=['WHERE po.OrderDate < ?','WHERE p.[Date Key] < ?','WHERE si.ValidFrom < ?',
             'WHERE s.ValidFrom < ?','WHERE s.[Valid From] < ?','WHERE i.[Valid From] < ?']
    if sql.count('?') != 1 or sum(sql.count(m) for m in markers)!=1 or any(w in sql.upper().split() for w in ['UNION','EXCEPT','INTERSECT']):
        raise ValueError('Falta frontera fija de desarrollo')
    return r.select(sql, (CUT,))


def extract():
    r=reader('WideWorldImporters')
    try:
        fp="SELECT COUNT_BIG(*) AS lines,CHECKSUM_AGG(BINARY_CHECKSUM(po.PurchaseOrderID,po.SupplierID,po.OrderDate,po.LastEditedWhen,pol.PurchaseOrderLineID,pol.StockItemID,pol.OrderedOuters,pol.ReceivedOuters,pol.LastEditedWhen)) AS checksum FROM Purchasing.PurchaseOrders po JOIN Purchasing.PurchaseOrderLines pol ON po.PurchaseOrderID=pol.PurchaseOrderID WHERE po.OrderDate < ?"
        guardar(OUT/'huella_oltp_antes.json',scoped(r,fp))
        guardar(OUT/'ordenes_desarrollo.json',scoped(r,"SELECT po.PurchaseOrderID,po.SupplierID,po.OrderDate,po.ExpectedDeliveryDate,po.LastEditedWhen,po.IsOrderFinalized FROM Purchasing.PurchaseOrders po WHERE po.OrderDate < ? ORDER BY po.OrderDate,po.PurchaseOrderID"))
        guardar(OUT/'lineas_desarrollo.json',scoped(r,"SELECT pol.PurchaseOrderLineID,pol.PurchaseOrderID,pol.StockItemID,pol.OrderedOuters,pol.ReceivedOuters,pol.PackageTypeID,pol.LastReceiptDate,pol.IsOrderLineFinalized,pol.LastEditedWhen,si.QuantityPerOuter FROM Purchasing.PurchaseOrderLines pol JOIN Purchasing.PurchaseOrders po ON po.PurchaseOrderID=pol.PurchaseOrderID JOIN Warehouse.StockItems si ON si.StockItemID=pol.StockItemID WHERE po.OrderDate < ? ORDER BY pol.PurchaseOrderLineID"))
        guardar(OUT/'unidades_historicas.json',scoped(r,"SELECT si.StockItemID,si.QuantityPerOuter,si.ValidFrom,si.ValidTo FROM Warehouse.StockItems FOR SYSTEM_TIME ALL si WHERE si.ValidFrom < ? AND si.StockItemID IN (SELECT pol.StockItemID FROM Purchasing.PurchaseOrderLines pol JOIN Purchasing.PurchaseOrders po ON po.PurchaseOrderID=pol.PurchaseOrderID WHERE po.OrderDate<'20160101') ORDER BY si.StockItemID,si.ValidFrom"))
        guardar(OUT/'proveedores_historicos.json',scoped(r,"SELECT s.SupplierID,s.ValidFrom,s.ValidTo FROM Purchasing.Suppliers FOR SYSTEM_TIME ALL s WHERE s.ValidFrom < ? AND s.SupplierID IN (SELECT po.SupplierID FROM Purchasing.PurchaseOrders po WHERE po.OrderDate<'20160101') ORDER BY s.SupplierID,s.ValidFrom"))
        guardar(OUT/'huella_oltp_despues.json',scoped(r,fp))
    finally: guardar(OUT/'consultas_oltp.json',r.consultas);r.close()
    r=reader('WideWorldImportersDW')
    try:
        fp="SELECT COUNT_BIG(*) AS lines,CHECKSUM_AGG(BINARY_CHECKSUM(p.[Purchase Key],p.[Date Key],p.[WWI Purchase Order ID],p.[Supplier Key],p.[Stock Item Key],p.[Ordered Quantity],p.[Ordered Outers],p.[Received Outers],p.[Is Order Finalized])) AS checksum FROM Fact.Purchase p WHERE p.[Date Key] < ?"
        guardar(OUT/'huella_dw_antes.json',scoped(r,fp))
        guardar(OUT/'hechos_desarrollo.json',scoped(r,"SELECT p.[Purchase Key] AS purchase_key,p.[Date Key] AS order_date,p.[WWI Purchase Order ID] AS order_id,p.[Supplier Key] AS supplier_key,p.[Stock Item Key] AS item_key,s.[WWI Supplier ID] AS supplier_id,i.[WWI Stock Item ID] AS item_id,s.[Valid From] AS supplier_from,s.[Valid To] AS supplier_to,i.[Valid From] AS item_from,i.[Valid To] AS item_to,p.[Ordered Outers] AS ordered_outers,p.[Ordered Quantity] AS ordered_quantity,p.[Received Outers] AS received_outers,p.[Is Order Finalized] AS finalized,p.Package FROM Fact.Purchase p JOIN Dimension.Supplier s ON s.[Supplier Key]=p.[Supplier Key] JOIN Dimension.[Stock Item] i ON i.[Stock Item Key]=p.[Stock Item Key] WHERE p.[Date Key] < ? ORDER BY p.[Purchase Key]"))
        guardar(OUT/'dimensiones_desarrollo.json',scoped(r,"SELECT 'supplier' AS kind,s.[Supplier Key] AS surrogate,s.[WWI Supplier ID] AS business_id,s.[Valid From] AS valid_from,s.[Valid To] AS valid_to,CAST(NULL AS int) AS quantity_per_outer FROM Dimension.Supplier s WHERE s.[Valid From] < ?"))
        guardar(OUT/'productos_dim_desarrollo.json',scoped(r,"SELECT i.[Stock Item Key] AS surrogate,i.[WWI Stock Item ID] AS business_id,i.[Valid From] AS valid_from,i.[Valid To] AS valid_to,i.[Quantity Per Outer] AS quantity_per_outer FROM Dimension.[Stock Item] i WHERE i.[Valid From] < ?"))
        guardar(OUT/'huella_dw_despues.json',scoped(r,fp))
    finally: guardar(OUT/'consultas_dw.json',r.consultas);r.close()


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def month_add(month, delta):
    y,m=map(int,month[:7].split('-'));n=y*12+m-1+delta
    return f'{n//12:04d}-{n%12+1:02d}'


def month_start(month):
    return month+'-01 00:00:00'


def counts(rows):
    """Resumen sin crear X/y ni ajustar modelos."""
    out={}
    for split in ['train','validation']:
        selected=[r for r in rows if r['split']==split]
        monthly=[]
        for m in sorted({r['month'] for r in selected}):
            v=[r for r in selected if r['month']==m];p=sum(r['label'] for r in v)
            monthly.append({'month':m,'eligible':len(v),'positive':p,'negative':len(v)-p,'positive_proportion':p/len(v)})
        labels=defaultdict(set)
        for r in selected:labels[r['entity']].add(r['label'])
        p=sum(r['label'] for r in selected)
        out[split]={'eligible':len(selected),'positive':p,'negative':len(selected)-p,'positive_proportion':p/len(selected) if selected else None,
                    'entities':len(labels),'entities_switching_class':sum(len(v)>1 for v in labels.values()),
                    'entities_single_class':sum(len(v)==1 for v in labels.values()),
                    'target_zero':sum(r['q']==0 for r in selected),'reference_zero':sum(r['reference']==0 for r in selected),
                    'monthly':monthly}
    return out


def panel(groups, mode='median', zero_policy='observed_only'):
    """Panel diagnóstico de cantidades o conteos. Nunca interpreta ausencia como cero real certificado.

    Grupos observados: (entidad, mes) -> q, known_at. La ausencia solo se admite
    en el escenario explícito de libro completo después de primera observación.
    Se exige además que los tres agregados previos estén disponibles al inicio.
    """
    if mode not in {'median','activity'} or zero_policy not in {'conditional_ledger','observed_only'}:
        raise ValueError('Modo o política de cobertura no reconocido')
    first={}
    for e,m in groups:first[e]=min(first.get(e,m),m)
    scenarios={'retrospective':[],'availability_gated':[]}
    exclusions=Counter()
    for e,first_month in sorted(first.items()):
        m=month_add(first_month,3)
        while m<'2016-01':
            history=[month_add(m,-i) for i in [1,2,3]]
            months=[m]+history
            if zero_policy=='observed_only' and any((e,x) not in groups for x in months):
                exclusions['missing_observed_month']+=1;m=month_add(m,1);continue
            q=groups.get((e,m),{}).get('q',0)
            prev=[groups.get((e,x),{}).get('q',0) for x in history]
            reference=median(prev)
            label=int(q>0) if mode=='activity' else int(q>reference)
            row={'entity':str(e),'month':m,'split':'train' if m<'2015-01' else 'validation','q':q,'reference':reference,'label':label}
            cut='2015-01-01 00:00:00' if m<'2015-01' else '2016-01-01 00:00:00'
            hist_known=all(groups.get((e,x),{}).get('known_at','')<month_start(m) for x in history)
            target_known=groups.get((e,m),{}).get('known_at','')<cut
            closed=month_start(month_add(m,1))<=cut
            # Incluso el diagnóstico de X retrospectiva mantiene sellado el
            # outcome no disponible antes del corte de su partición.
            if target_known and closed:scenarios['retrospective'].append(row)
            if not hist_known:exclusions['historical_version_after_prediction']+=1
            if not target_known:exclusions['target_version_after_split_cut']+=1
            if not closed:exclusions['target_month_not_closed']+=1
            if hist_known and target_known and closed:scenarios['availability_gated'].append(row)
            m=month_add(m,1)
    return {'zero_policy':zero_policy,'scenarios':{k:counts(v) for k,v in scenarios.items()},'excluded_reasons_nonexclusive':dict(exclusions)}


def analyze():
    orders=load('ordenes_desarrollo.json');lines=load('lineas_desarrollo.json');facts=load('hechos_desarrollo.json')
    by_order={o['PurchaseOrderID']:o for o in orders}
    assert len(by_order)==len(orders)
    assert len({l['PurchaseOrderLineID'] for l in lines})==len(lines)
    source=defaultdict(list)
    for l in lines:source[l['PurchaseOrderID'],l['StockItemID']].append(l)
    mapped=[];mismatches=Counter()
    for f in facts:
        matches=source.get((f['order_id'],f['item_id']),[])
        if len(matches)!=1:mismatches['ambiguous_or_missing_line']+=1;continue
        l=matches[0];o=by_order[f['order_id']];mapped.append(l['PurchaseOrderLineID'])
        checks={'date':f['order_date']==o['OrderDate'],'supplier':f['supplier_id']==o['SupplierID'],
                'outers':f['ordered_outers']==l['OrderedOuters'],'units':f['ordered_quantity']==l['OrderedOuters']*l['QuantityPerOuter'],
                'received':f['received_outers']==l['ReceivedOuters'],'finalized':f['finalized']==l['IsOrderLineFinalized']}
        for key,ok in checks.items():
            if not ok:mismatches[key]+=1
    # ID-only joins above; dimension attributes never enter labels/features.
    dimchecks={}
    for name,idkey in [('dimensiones_desarrollo.json','supplier_id'),('productos_dim_desarrollo.json','item_id')]:
        dim=load(name);unknown=sum(f[idkey] in (None,0) for f in facts)
        prefix='supplier' if idkey=='supplier_id' else 'item'
        outside=sum(not(f[prefix+'_from']<=f['order_date']+' 00:00:00'<f[prefix+'_to']) for f in facts)
        overlaps=[]
        for i,a in enumerate(dim):
            for b in dim[i+1:]:
                if a['business_id']==b['business_id'] and a['valid_from']<b['valid_to'] and b['valid_from']<a['valid_to']:
                    overlaps.append({'business_id':a['business_id'],'surrogates':[a['surrogate'],b['surrogate']]})
        dimchecks[prefix]={'unknown_fact_rows':unknown,'fact_dimension_version_outside_order_midnight':outside,'overlapping_pairs':overlaps,
                           'versions':len(dim),'business_ids_including_unknown':len({d['business_id'] for d in dim})}
    units=load('unidades_historicas.json');unit_by=defaultdict(list)
    for u in units:unit_by[u['StockItemID']].append(u)
    unit_issues=Counter()
    for l in lines:
        date=by_order[l['PurchaseOrderID']]['OrderDate']+' 00:00:00'
        versions=[u for u in unit_by[l['StockItemID']] if u['ValidFrom']<=date<u['ValidTo']]
        if len(versions)!=1:unit_issues['nonunique_or_missing_at_order']+=1
        elif versions[0]['QuantityPerOuter']!=l['QuantityPerOuter']:unit_issues['different_at_order']+=1
    groups={k:defaultdict(lambda:{'q':0,'known_at':'','orders':set(),'packages':set()}) for k in ['product','supplier','product_supplier']}
    for l in lines:
        o=by_order[l['PurchaseOrderID']];m=o['OrderDate'][:7]
        for name,e in [('product',l['StockItemID']),('supplier',o['SupplierID']),('product_supplier',(l['StockItemID'],o['SupplierID']))]:
            g=groups[name][e,m];g['q']+=l['OrderedOuters']*l['QuantityPerOuter']
            g['known_at']=max(g['known_at'],o['LastEditedWhen'],l['LastEditedWhen']);g['orders'].add(o['PurchaseOrderID']);g['packages'].add(l['PackageTypeID'])
    freq=defaultdict(lambda:{'q':0,'known_at':''})
    for o in orders:
        g=freq[o['SupplierID'],o['OrderDate'][:7]];g['q']+=1;g['known_at']=max(g['known_at'],o['LastEditedWhen'])
    granularities={}
    for name,gg in groups.items():
        activity=defaultdict(set);packages=defaultdict(set)
        for (e,m),g in gg.items():activity[str(e)].add(m);packages[str(e)].update(g['packages'])
        potential=sum(36-(int(min(v)[:4])-2013)*12-int(min(v)[5:])+1 for v in activity.values())
        granularities[name]={'identities':len(activity),'active_entity_months':len(gg),'ledger_month_slots_after_first_observation':potential,
             'unobserved_month_slots_conditional_not_certified_zero':potential-len(gg),'activity_months_histogram':dict(sorted(Counter(len(v) for v in activity.values()).items())),
             'first_observation_months':dict(Counter(min(v) for v in activity.values())),
             'entities_multiple_package_types':sum(len(v)>1 for v in packages.values()),
             'products_per_supplier':dict(Counter({str(s):len({l['StockItemID'] for l in lines if by_order[l['PurchaseOrderID']]['SupplierID']==s}) for s in {o['SupplierID'] for o in orders}})) if name=='supplier' else None}
    candidates={}
    for name in ['product','product_supplier']:
        candidates['volume_'+name]=panel(groups[name],zero_policy='conditional_ledger')
        candidates['volume_'+name+'_observed_only']=panel(groups[name],zero_policy='observed_only')
    candidates['order_frequency_supplier']=panel(freq,zero_policy='conditional_ledger')
    candidates['order_activity_supplier']=panel(freq,mode='activity',zero_policy='conditional_ledger')
    candidates['order_frequency_supplier_observed_only']=panel(freq,zero_policy='observed_only')
    diagnostics={'orders':len(orders),'lines':len(lines),'facts':len(facts),'date_min':min(o['OrderDate'] for o in orders),'date_max':max(o['OrderDate'] for o in orders),
      'order_months':len({o['OrderDate'][:7] for o in orders}),'monthly_coverage':dict(sorted(Counter(o['OrderDate'][:7] for o in orders).items())),
      'line_to_fact_bijection':len(mapped)==len(lines)==len(facts) and len(set(mapped))==len(lines),'mapping_mismatches':dict(mismatches),
      'max_lines_per_order':max(Counter(l['PurchaseOrderID'] for l in lines).values()),'duplicate_order_item_pairs':sum(len(v)>1 for v in source.values()),
      'current_partial_lines':sum(0<l['ReceivedOuters']<l['OrderedOuters'] for l in lines),'current_no_receipt_lines':sum(l['ReceivedOuters']==0 for l in lines),
      'current_finalized_lines':sum(l['IsOrderLineFinalized'] for l in lines),'receipt_before_order':sum(l['LastReceiptDate'] is not None and l['LastReceiptDate']<by_order[l['PurchaseOrderID']]['OrderDate'] for l in lines),
      'late_expected_receipt_development_observed_before_cut':sum(l['LastReceiptDate'] is not None and l['LastReceiptDate']<'2016-01-01' and by_order[l['PurchaseOrderID']]['ExpectedDeliveryDate'] is not None and l['LastReceiptDate']>by_order[l['PurchaseOrderID']]['ExpectedDeliveryDate'] for l in lines),
      'unit_history_issues':dict(unit_issues),'historical_unit_values':dict(Counter(u['QuantityPerOuter'] for u in units)),
      'product_unit_changes_before_2016':sum(len({u['QuantityPerOuter'] for u in v})>1 for v in unit_by.values()),'dimension_checks':dimchecks,
      'supplier_per_product_histogram':dict(Counter(len({by_order[l['PurchaseOrderID']]['SupplierID'] for l in lines if l['StockItemID']==i}) for i in {l['StockItemID'] for l in lines})),
      'orders_edited_after_order_date':sum(o['LastEditedWhen'][:10]>o['OrderDate'] for o in orders),
      'headers_current_state_available_before_next_month':sum(o['LastEditedWhen']<month_start(month_add(o['OrderDate'][:7],1)) for o in orders)}
    guardar(OUT/'comprobaciones_focalizadas.json',diagnostics);guardar(OUT/'granularidades.json',granularities);guardar(OUT/'etiquetas_candidatas.json',candidates)
    print(json.dumps({'diagnostics':diagnostics,'granularities':granularities,'candidates':{k:{s:{p:{kk:vv for kk,vv in q.items() if kk!='monthly'} for p,q in v.items()} for s,v in c['scenarios'].items()} for k,c in candidates.items()}},ensure_ascii=True))
    return diagnostics,granularities,candidates


def manifest():
    prior=load('integridad_antes.json')
    after={n:sha(BASE/n) for n in prior}
    if prior!=after:raise RuntimeError('Se modificó evidencia anterior a Compras')
    guardar(OUT/'integridad_despues.json',after)
    guardar(OUT/'manifiesto.json',{
        'scope':'Viabilidad de Compras, sin matrices ni entrenamiento; 2016 reservado',
        'implementation_sha256':{n:sha(BASE/n) for n in sorted(NEW) if (BASE/n).exists()},
        'artifacts_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*'))
                            if p.is_file() and p.name not in {'manifiesto.json','validaciones.json'}}})


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase', choices=['discover','extract','analyze','manifest']);args=ap.parse_args()
    {'discover':discover,'extract':extract,'analyze':analyze,'manifest':manifest}[args.phase]()
