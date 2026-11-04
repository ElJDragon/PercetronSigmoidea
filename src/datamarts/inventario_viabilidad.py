"""Investigación focalizada de inventario; solo SELECT de eventos de desarrollo.

No produce matrices X/y, no ajusta modelos y no abre outcomes de 2016.
"""
import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from decimal import Decimal
from statistics import median

from src.db.conexion import Configuracion, Lector, validar_select
from src.db.auditoria_dw import BASE, guardar, sha

OUT=BASE/'resultados/extension_datamarts/inventario'
CUT=dt.date(2016,1,1)
NEW={'src/datamarts/inventario_viabilidad.py','src/datamarts/reportes_inventario.py','tests/test_inventario_viabilidad.py'}

def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def snapshot():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
            for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and OUT not in p.parents and str(p.relative_to(BASE)).replace('\\','/') not in NEW}

def reader(database):return Lector(Configuracion('lpc:.\\AMBATO',database).conectar())

def scoped(r,sql):
    validar_select(sql)
    allowed=['WHERE st.TransactionOccurredWhen < ?', 'WHERE f.[Date Key] < ?',
             'WHERE f.[Invoice Date Key] < ?', 'WHERE d.[Valid From] < ?', 'WHERE d.ValidFrom < ?']
    if sql.count('?')!=1 or sum(sql.count(m) for m in allowed)!=1:
        raise ValueError('Falta frontera fija de desarrollo')
    if any(x in sql.upper().split() for x in ['UNION','EXCEPT','INTERSECT']):
        raise ValueError('Composición fuera del contrato')
    return r.select(sql,(CUT,))

def discover():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',snapshot())
    r=reader('WideWorldImporters')
    try:
        guardar(OUT/'metadatos_focalizados.json',r.select("SELECT s.name AS schema_name,t.name AS table_name,c.name AS column_name,ty.name AS data_type,c.precision,c.scale,c.is_nullable,t.temporal_type_desc FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON ty.user_type_id=c.user_type_id WHERE s.name='Warehouse' AND t.name IN ('StockItemTransactions','StockItems','StockItems_Archive') ORDER BY t.name,c.column_id"))
        guardar(OUT/'tipos_operacionales.json',r.select('SELECT TransactionTypeID,TransactionTypeName FROM Application.TransactionTypes ORDER BY TransactionTypeID'))
        guardar(OUT/'modulos_movimientos.json',r.select("SELECT s.name AS schema_name,o.name AS module_name,m.definition FROM sys.sql_modules m JOIN sys.objects o ON o.object_id=m.object_id JOIN sys.schemas s ON s.schema_id=o.schema_id WHERE m.definition LIKE '%Warehouse.StockItemTransactions%'"))
    finally:guardar(OUT/'consultas_metadatos.json',r.consultas);r.close()

def extract():
    r=reader('WideWorldImporters')
    fp='SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM(st.StockItemTransactionID,st.StockItemID,st.TransactionTypeID,st.TransactionOccurredWhen,st.Quantity,st.InvoiceID,st.PurchaseOrderID,st.LastEditedWhen)) AS checksum FROM Warehouse.StockItemTransactions st WHERE st.TransactionOccurredWhen < ?'
    try:
        guardar(OUT/'huella_oltp_antes.json',scoped(r,fp))
        guardar(OUT/'eventos_desarrollo.json',scoped(r,'SELECT st.StockItemTransactionID AS event_id,st.StockItemID AS item_id,st.TransactionTypeID AS type_id,st.TransactionOccurredWhen AS occurred_at,st.Quantity AS quantity,st.InvoiceID AS invoice_id,st.PurchaseOrderID AS purchase_order_id,st.CustomerID AS customer_id,st.SupplierID AS supplier_id,st.LastEditedWhen AS edited_at FROM Warehouse.StockItemTransactions st WHERE st.TransactionOccurredWhen < ? ORDER BY st.StockItemTransactionID'))
        guardar(OUT/'versiones_articulos_oltp.json',scoped(r,'SELECT d.StockItemID AS item_id,d.UnitPackageID AS unit_package_id,d.OuterPackageID AS outer_package_id,d.QuantityPerOuter AS quantity_per_outer,d.ValidFrom AS valid_from,d.ValidTo AS valid_to FROM Warehouse.StockItems FOR SYSTEM_TIME ALL d WHERE d.ValidFrom < ? ORDER BY d.StockItemID,d.ValidFrom'))
        guardar(OUT/'huella_oltp_despues.json',scoped(r,fp))
    finally:guardar(OUT/'consultas_oltp.json',r.consultas);r.close()
    r=reader('WideWorldImportersDW')
    fp='SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM(f.[Movement Key],f.[Date Key],f.[Stock Item Key],f.[WWI Stock Item Transaction ID],f.[Transaction Type Key],f.Quantity,f.[WWI Invoice ID],f.[WWI Purchase Order ID])) AS checksum FROM Fact.Movement f WHERE f.[Date Key] < ?'
    try:
        guardar(OUT/'huella_dw_antes.json',scoped(r,fp))
        guardar(OUT/'movimientos_dw_desarrollo.json',scoped(r,'SELECT f.[Movement Key] AS fact_id,f.[Date Key] AS event_date,f.[WWI Stock Item Transaction ID] AS event_id,f.[Stock Item Key] AS item_key,d.[WWI Stock Item ID] AS item_id,t.[WWI Transaction Type ID] AS type_id,t.[Transaction Type] AS type_name,f.Quantity AS quantity,f.[WWI Invoice ID] AS invoice_id,f.[WWI Purchase Order ID] AS purchase_order_id FROM Fact.Movement f LEFT JOIN Dimension.[Stock Item] d ON d.[Stock Item Key]=f.[Stock Item Key] LEFT JOIN Dimension.[Transaction Type] t ON t.[Transaction Type Key]=f.[Transaction Type Key] WHERE f.[Date Key] < ? ORDER BY f.[Movement Key]'))
        guardar(OUT/'ventas_agrupadas_desarrollo.json',scoped(r,'SELECT f.[Invoice Date Key] AS event_date,f.[WWI Invoice ID] AS invoice_id,d.[WWI Stock Item ID] AS item_id,COUNT_BIG(*) AS lines,SUM(f.Quantity) AS quantity FROM Fact.Sale f LEFT JOIN Dimension.[Stock Item] d ON d.[Stock Item Key]=f.[Stock Item Key] WHERE f.[Invoice Date Key] < ? GROUP BY f.[Invoice Date Key],f.[WWI Invoice ID],d.[WWI Stock Item ID]'))
        guardar(OUT/'versiones_articulos_dw.json',scoped(r,'SELECT d.[Stock Item Key] AS item_key,d.[WWI Stock Item ID] AS item_id,d.[Quantity Per Outer] AS quantity_per_outer,d.[Selling Package] AS selling_package,d.[Buying Package] AS buying_package,d.[Valid From] AS valid_from,d.[Valid To] AS valid_to FROM Dimension.[Stock Item] d WHERE d.[Valid From] < ? ORDER BY d.[Stock Item Key]'))
        guardar(OUT/'huella_dw_despues.json',scoped(r,fp))
    finally:guardar(OUT/'consultas_dw.json',r.consultas);r.close()

def add_month(month,delta):
    y,m=map(int,month.split('-'));n=y*12+m-1+delta
    return f'{n//12:04d}-{n%12+1:02d}'

def month_range(first,last):
    while first<=last:
        yield first
        first=add_month(first,1)

def aggregate(rows):
    """Estadísticos mensuales de eventos, no matrices de aprendizaje."""
    monthly=defaultdict(lambda:dict(receipts=0,receipt_units=Decimal(0),adjustments=0,net=Decimal(0),issues=Decimal(0),events=0))
    first={};seen=set()
    for e in rows:
        if e['event_id'] in seen:raise ValueError('Evento duplicado')
        seen.add(e['event_id'])
        if not e['item_id'] or e['item_id']<=0:raise ValueError('Identidad desconocida')
        when=dt.datetime.fromisoformat(e['occurred_at']);edited=dt.datetime.fromisoformat(e['edited_at'])
        if when.date()>=CUT:raise ValueError('Outcome fuera de desarrollo')
        # Esta extracción solo puede respaldar disponibilidad si no hay edición posterior.
        if edited!=when:raise ValueError('Disponibilidad histórica requiere investigación adicional')
        if e['type_id'] not in {10,11,12}:raise ValueError('Tipo no interpretado')
        q=Decimal(str(e['quantity']));typ=e['type_id']
        if (typ==10 and q>=0) or (typ==11 and q<=0):raise ValueError('Signo inesperado')
        month=e['occurred_at'][:7];item=e['item_id'];first[item]=min(first.get(item,month),month)
        g=monthly[item,month];g['events']+=1;g['net']+=q
        if typ==11:g['receipts']+=1;g['receipt_units']+=q
        if typ==12:g['adjustments']+=1
        if typ==10:g['issues']-=q
    return dict(monthly),first

def value(monthly,item,month,key):return monthly.get((item,month),{}).get(key,0)

def binary_metrics(truth,pred):
    from src.evaluation.metricas import metricas_binarias
    if not truth:return dict(confusion_matrix=[[0,0],[0,0]],balanced_accuracy=None,accuracy=None)
    return metricas_binarias(truth,pred)

def candidate(monthly,first,key,*,assume_coverage=False):
    """Calcula solo distribuciones y baselines de desarrollo; no exporta X/y."""
    if not assume_coverage:raise ValueError('La ausencia solo es cero bajo supuesto explícito de cobertura')
    if key not in {'receipts','receipt_units','adjustments','net'}:raise ValueError('Agregado candidato no autorizado')
    records=[]
    for item,start in sorted(first.items()):
        for month in month_range(add_month(start,3),'2015-12'):
            history=[value(monthly,item,add_month(month,-k),key) for k in (1,2,3)]
            target=value(monthly,item,month,key);reference=median(history)
            label=int(target>reference)
            historical=int(history[0]>reference)
            issue_history=[value(monthly,item,add_month(month,-k),'issues') for k in (1,2,3)]
            sales_label=int(value(monthly,item,month,'issues')>median(issue_history))
            records.append(dict(item=item,month=month,label=label,historical=historical,sales_label=sales_label,
                                target_zero=target==0,history_zero=all(x==0 for x in history),tie=target==reference))
    summaries={}
    train=[r for r in records if r['month']<'2015-01'];majority=int(sum(r['label'] for r in train)>len(train)/2)
    for split,rs in [('train',train),('val',[r for r in records if r['month']>='2015-01'])]:
        entities=defaultdict(set)
        for r in rs:entities[r['item']].add(r['label'])
        pos=sum(r['label'] for r in rs)
        summaries[split]=dict(entities=len(entities),observations=len(rs),positive=pos,negative=len(rs)-pos,
            prevalence=pos/len(rs) if rs else None,entities_varying=sum(len(v)==2 for v in entities.values()),
            positive_entities=sum(1 in v for v in entities.values()),
            positives_by_item=dict(sorted(Counter(r['item'] for r in rs if r['label']).items())),
            target_zero=sum(r['target_zero'] for r in rs),all_history_zero=sum(r['history_zero'] for r in rs),
            ties=sum(r['tie'] for r in rs),majority_class_from_train=majority,
            majority=binary_metrics([r['label'] for r in rs],[majority]*len(rs)),
            historical=binary_metrics([r['label'] for r in rs],[r['historical'] for r in rs]),
            agreement_with_sales_volume_label=sum(r['label']==r['sales_label'] for r in rs)/len(rs) if rs else None)
    bymonth=[]
    for month in sorted({r['month'] for r in records}):
        rs=[r for r in records if r['month']==month];p=sum(r['label'] for r in rs)
        bymonth.append(dict(month=month,entities=len(rs),observations=len(rs),positive=p,negative=len(rs)-p,prevalence=p/len(rs)))
    variation=defaultdict(set)
    for r in records:variation[r['item']].add(r['label'])
    return dict(statistic=key,history_months=3,academic_coverage_assumption=True,summary=summaries,monthly=bymonth,
                entities_varying_development=sum(len(v)==2 for v in variation.values()),
                observations=len(records),targets_2016=0)

def analyze():
    events=load('eventos_desarrollo.json');dw=load('movimientos_dw_desarrollo.json')
    sales=load('ventas_agrupadas_desarrollo.json');versions=load('versiones_articulos_oltp.json')
    panel,first=aggregate(events)
    types=[]
    names={x['TransactionTypeID']:x['TransactionTypeName'] for x in load('tipos_operacionales.json')}
    for typ in sorted({r['type_id'] for r in events}):
        rs=[r for r in events if r['type_id']==typ];qs=[Decimal(r['quantity']) for r in rs]
        types.append(dict(type_id=typ,name=names[typ],events=len(rs),items=len({r['item_id'] for r in rs}),
            negative=sum(q<0 for q in qs),positive=sum(q>0 for q in qs),zero=sum(q==0 for q in qs),
            fractional=sum(q!=int(q) for q in qs),quantity_sum=str(sum(qs)),
            invoice_links=sum(r['invoice_id'] is not None for r in rs),purchase_links=sum(r['purchase_order_id'] is not None for r in rs),
            customer_links=sum(r['customer_id'] is not None for r in rs),supplier_links=sum(r['supplier_id'] is not None for r in rs),
            monthly=dict(sorted(Counter(r['occurred_at'][:7] for r in rs).items()))))
    op={r['event_id']:r for r in events};matched=Counter({key:0 for key in ['dw_missing_in_oltp','identity_mismatch','type_mismatch','quantity_mismatch','date_mismatch','invoice_mismatch','purchase_mismatch']})
    for r in dw:
        e=op.get(r['event_id'])
        if not e:matched['dw_missing_in_oltp']+=1;continue
        for name,condition in [('identity',r['item_id']==e['item_id']),('type',r['type_id']==e['type_id']),
            ('quantity',Decimal(r['quantity'])==Decimal(e['quantity'])),('date',r['event_date']==e['occurred_at'][:10]),
            ('invoice',r['invoice_id']==e['invoice_id']),('purchase',r['purchase_order_id']==e['purchase_order_id'])]:
            if not condition:matched[name+'_mismatch']+=1
    issues=defaultdict(lambda:dict(events=0,quantity=Decimal(0)))
    for e in events:
        if e['type_id']==10:
            key=(e['occurred_at'][:10],e['invoice_id'],e['item_id'])
            issues[key]['events']+=1;issues[key]['quantity']-=Decimal(e['quantity'])
    sale_groups={(r['event_date'],r['invoice_id'],r['item_id']):r for r in sales}
    overlap=dict(issue_events=sum(v['events'] for v in issues.values()),issue_groups=len(issues),sale_groups=len(sale_groups),
        sales_lines=sum(r['lines'] for r in sales),missing_sale_groups=sum(k not in sale_groups for k in issues),
        sales_groups_without_issue=sum(k not in issues for k in sale_groups),
        quantity_mismatches=sum(k in sale_groups and v['quantity']!=Decimal(sale_groups[k]['quantity']) for k,v in issues.items()),
        line_count_mismatches=sum(k in sale_groups and v['events']!=sale_groups[k]['lines'] for k,v in issues.items()))
    itemversions=defaultdict(list)
    for v in versions:itemversions[v['item_id']].append(v)
    stable_units={i:len({(v['unit_package_id'],v['quantity_per_outer']) for v in vs})==1 for i,vs in itemversions.items()}
    itemmonths=[]
    for item,start in sorted(first.items()):
        months=list(month_range(start,'2015-12'));observed=[m for m in months if (item,m) in panel]
        itemmonths.append(dict(item_id=item,first_event_month=start,observable_months=len(months),months_with_movements=len(observed),
            months_without_movements=len(months)-len(observed),months_with_receipts=sum(value(panel,item,m,'receipts')>0 for m in months),
            months_without_receipts=sum(value(panel,item,m,'receipts')==0 for m in months),
            eligible_train=sum(m<'2015-01' for m in months[3:]),eligible_val=sum(m>='2015-01' for m in months[3:])))
    receipt_keys=Counter((r['item_id'],r['purchase_order_id']) for r in events if r['type_id']==11)
    non_sales_months=[]
    for item,start in first.items():
        for month in month_range(add_month(start,3),'2015-12'):
            window=[add_month(month,-k) for k in (0,1,2,3)]
            clean=all(value(panel,item,m,'receipts')==0 and value(panel,item,m,'adjustments')==0 for m in window)
            if clean:
                q=[value(panel,item,m,'issues') for m in window]
                net_label=int(-q[0]>median([-x for x in q[1:]]))
                lower_sales=int(q[0]<median(q[1:]))
                non_sales_months.append((net_label,lower_sales))
    overlap['net_windows_only_sales']=len(non_sales_months)
    overlap['net_exactly_lower_sales_volume']=sum(a==b for a,b in non_sales_months)
    quality=dict(development_events=len(events),development_dw=len(dw),unique_event_ids=len(op),
        duplicate_dw_event_ids=len(dw)-len({r['event_id'] for r in dw}),oltp_events_missing_in_dw=len(set(op)-{r['event_id'] for r in dw}),
        reconciliation=dict(matched),items=len(first),dw_unknown_items=sum(not r['item_id'] for r in dw),
        dw_keys_per_item=dict(sorted(Counter(r['item_id'] for r in load('versiones_articulos_dw.json')).items())),
        development_stock_versions=len(versions),stable_unit_items=sum(stable_units.values()),
        unstable_unit_items=[i for i,v in stable_units.items() if not v],missing_unit_metadata=sorted(set(first)-set(itemversions)),
        first_event=min(r['occurred_at'] for r in events),last_event=max(r['occurred_at'] for r in events),
        last_edit_differs_from_event=sum(r['edited_at']!=r['occurred_at'] for r in events),
        global_months=dict(sorted(Counter(r['occurred_at'][:7] for r in events).items())),
        first_appearances=dict(sorted(Counter(first.values()).items())),
        item_months=sum(v['observable_months'] for v in itemmonths),
        months_without_movements=sum(v['months_without_movements'] for v in itemmonths),
        months_without_receipts=sum(v['months_without_receipts'] for v in itemmonths),
        receipt_item_order_groups=len(receipt_keys),receipt_repeated_item_order_groups=sum(v>1 for v in receipt_keys.values()),
        receipts_after_initial_month_items=sorted({r['item_id'] for r in events if r['type_id']==11 and r['occurred_at'][:7]>'2013-01'}),
        receipt_item_months_by_year={str(year):len({(r['item_id'],r['occurred_at'][:7]) for r in events if r['type_id']==11 and r['occurred_at'].startswith(str(year))}) for year in (2013,2014,2015)},
        coverage_status='Supuesto académico: extracción retrospectiva conservada, no prueba de cierres ni auditoría de borrados.',
        test_opened=False)
    guardar(OUT/'tipos_y_signos.json',types);guardar(OUT/'calidad_cobertura.json',quality)
    guardar(OUT/'cobertura_articulos.json',itemmonths);guardar(OUT/'solapamiento_ventas.json',overlap)
    candidates={name:candidate(panel,first,key,assume_coverage=True) for name,key in [('A','receipts'),('B','receipt_units'),('C','adjustments'),('D','net')]}
    guardar(OUT/'exploracion_etiquetas.json',candidates)
    before=load('integridad_antes.json');after=snapshot()
    if before!=after:raise AssertionError('Se alteraron archivos protegidos')
    guardar(OUT/'integridad_despues.json',after)
    from src.datamarts.reportes_inventario import generate
    generate()
    print(json.dumps(dict(events=len(events),items=len(first),candidates={k:{s:{f:v[f] for f in ['observations','positive','negative','positive_entities']} for s,v in c['summary'].items()} for k,c in candidates.items()})))

def verify():
    """Pruebas y contraste offline; bloquea operaciones fuera de esta investigación."""
    import importlib
    import io
    import unittest
    from contextlib import ExitStack,redirect_stdout
    from unittest.mock import patch
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    generated=['tipos_y_signos.json','calidad_cobertura.json','cobertura_articulos.json','solapamiento_ventas.json',
        'exploracion_etiquetas.json','catalogo_caracteristicas.json','semantica_movimientos.md','analisis_etiquetas.md',
        'contrato_datos.md','diseno_evaluacion.md']
    before={n:sha(OUT/n) for n in generated}
    guard=[]
    with ExitStack() as stack:
        for mod,cls,methods in [('src.models.perceptron','PerceptronSimple',['fit','predict','decision_function']),
            ('src.models.logistic_regression','LogisticRegressionManual',['fit','predict','predict_proba','decision_function']),
            ('src.visualization.pca','PCA2D',['fit','transform']),
            ('src.datamarts.transacciones.preprocesamiento','PreprocesadorFinanciero',['fit','transform']),
            ('src.db.conexion','Configuracion',['conectar'])]:
            obj=getattr(importlib.import_module(mod),cls)
            for method in methods:
                stack.enter_context(patch.object(obj,method,side_effect=AssertionError('Operación prohibida en viabilidad')))
                guard.append(cls+'.'+method)
        with redirect_stdout(io.StringIO()):analyze()
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_inventario_viabilidad')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (OUT/'pruebas.txt').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    check('Pruebas focalizadas correctas',result.wasSuccessful() and result.testsRun==24)
    check('Reproducción determinista de diez salidas',before=={n:sha(OUT/n) for n in generated})
    protected=load('integridad_antes.json');after=snapshot()
    check('Todos los archivos previos conservados',protected==after)
    for domain in ['oltp','dw']:check('Huella estable '+domain,load('huella_'+domain+'_antes.json')==load('huella_'+domain+'_despues.json'))
    for filename in ['consultas_oltp.json','consultas_dw.json']:
        for index,query in enumerate(load(filename)):
            check(f'{filename} consulta {index+1} SELECT',validar_select(query['sql'])==query['sql'] and query['status']=='successful')
            check(f'{filename} consulta {index+1} acotada',query['parameters']==['2016-01-01'] and query['sql'].count('?')==1)
    for index,query in enumerate(load('consultas_metadatos.json')):
        check(f'Metadatos SELECT {index+1}',validar_select(query['sql'])==query['sql'] and query['status']=='successful')
    events=load('eventos_desarrollo.json');dw=load('movimientos_dw_desarrollo.json');q=load('calidad_cobertura.json');c=load('exploracion_etiquetas.json')
    check('206143 eventos conciliados',len(events)==len(dw)==q['unique_event_ids']==206143)
    check('Cero discrepancias fuente',all(v==0 for v in q['reconciliation'].values()) and q['duplicate_dw_event_ids']==q['oltp_events_missing_in_dw']==q['dw_unknown_items']==0)
    check('Sin outcomes test extraídos',all(e['occurred_at']<'2016-01-01' for e in events) and all(e['event_date']<'2016-01-01' for e in dw))
    check('Unidades sustentadas para B',q['stable_unit_items']==219 and not q['unstable_unit_items'] and not q['missing_unit_metadata'])
    check('Disponibilidad evento respaldada',q['last_edit_differs_from_event']==0)
    check('Cobertura mensual de desarrollo completa',len(q['global_months'])==36 and q['months_without_movements']==0)
    expected={'A':(86,35),'B':(177,98),'C':(22,14),'D':(2223,1306)}
    for name,pair in expected.items():
        for split,pos,n in [('train',pair[0],4599),('val',pair[1],2628)]:
            stats=c[name]['summary'][split]
            check(f'{name} {split} clases verificadas',stats['positive']==pos and stats['negative']==n-pos and stats['observations']==n)
            check(f'{name} {split} suma mensual',sum(m['positive'] for m in c[name]['monthly'] if (m['month']<'2015-01')==(split=='train'))==pos)
        check(f'{name} sin test y con supuesto explícito',c[name]['targets_2016']==0 and c[name]['academic_coverage_assumption'])
    overlap=load('solapamiento_ventas.json')
    check('Duplicidad de salidas cuantificada',overlap['issue_events']==overlap['sales_lines']==198807 and overlap['missing_sale_groups']==overlap['sales_groups_without_issue']==overlap['quantity_mismatches']==overlap['line_count_mismatches']==0)
    check('Neto dominado por las mismas ventas',overlap['net_windows_only_sales']==overlap['net_exactly_lower_sales_volume']==6594)
    check('17 características sin matrices',len(load('catalogo_caracteristicas.json'))==17 and not list(OUT.glob('*.npy')) and not list(OUT.glob('*.npz')))
    guardar(OUT/'validaciones.json',dict(status='successful',checks=checks,checks_passed=len(checks),tests_run=result.testsRun,
        test_failures=len(result.failures),test_errors=len(result.errors),test_report='pruebas.txt',
        preserved_files=len(protected),preserved_sales_transactions_report=True,experimental_models_modified=False,
        SQL_mode='SELECT only',SQL_event_boundary='strictly before 2016-01-01',test_outcomes_opened=False,
        matrices_built=False,models_trained=False,classifier_predictions_computed=False,
        development_baseline_predictions_computed=True,PCA_executed=False,git_operations=0,
        analysis_reproduced=True,guarded_operations=guard,recommended_candidate='A',
        viability='Viable bajo supuestos académicos explícitos',dataset_construction_recommended_for_later=True,
        caveat='Eventos infrecuentes: 86 positivos train y 35 val; no garantiza utilidad ni mejora sobre baseline.',
        reproduce=['python -m src.datamarts.inventario_viabilidad discover','python -m src.datamarts.inventario_viabilidad extract',
                   'python -m src.datamarts.inventario_viabilidad analyze','python -m src.datamarts.inventario_viabilidad verify']))
    guardar(OUT/'integridad_despues.json',after)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifiesto.json']
    paths.extend(BASE/name for name in sorted(NEW))
    guardar(OUT/'manifiesto.json',dict(artifacts={str(p.relative_to(BASE)):sha(p) for p in sorted(paths)},
        audit_reused='resultados/extension_datamarts/auditoria_dw',outcomes_2016_queried=False))
    print(json.dumps(dict(status='successful',checks=len(checks),tests=result.testsRun,preserved_files=len(protected),candidate='A')))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['discover','extract','analyze','verify'])
    action=parser.parse_args().action
    if action=='discover':discover()
    elif action=='extract':extract()
    elif action=='analyze':analyze()
    else:verify()

if __name__=='__main__':main()
