"""Auditoría de metadatos y agregados existentes, sin extracción masiva ni ETL."""
import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from .conexion import Configuracion,Lector,identificador as qi

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'resultados/extension_datamarts/auditoria_dw'
FACTS=['Order','Purchase','Movement','Stock Holding','Transaction']
DIMENSIONS=['Date','Stock Item','Customer','Supplier','Transaction Type','Payment Method','City','Employee']


def serial(value):
    if isinstance(value,(dt.date,dt.datetime,Decimal)):return str(value)
    raise TypeError(type(value).__name__)


def guardar(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    result={}
    for root in ['datos','referencias','resultados','src','tests','informe']:
        for p in (BASE/root).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or OUT in p.parents or BASE/'src/db' in p.parents or p==BASE/'tests/test_auditoria_dw.py':continue
            result[str(p.relative_to(BASE))]=sha(p)
    return result


def metadatos(reader):
    cols=reader.select("""SELECT s.name AS schema_name,t.name AS table_name,c.column_id,c.name AS column_name,ty.name AS data_type,c.max_length,c.precision,c.scale,c.is_nullable,c.is_identity,c.is_computed,dc.definition AS default_definition,t.temporal_type_desc
    FROM sys.tables t JOIN sys.schemas s ON t.schema_id=s.schema_id JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON c.user_type_id=ty.user_type_id LEFT JOIN sys.default_constraints dc ON c.default_object_id=dc.object_id WHERE s.name IN (?,?) ORDER BY s.name,t.name,c.column_id""",('Fact','Dimension'))
    keys=reader.select("""SELECT s.name AS schema_name,t.name AS table_name,i.name AS index_name,i.type_desc,i.is_primary_key,i.is_unique,i.is_unique_constraint,i.has_filter,i.filter_definition,c.name AS column_name,ic.key_ordinal,ic.is_included_column
    FROM sys.tables t JOIN sys.schemas s ON t.schema_id=s.schema_id JOIN sys.indexes i ON i.object_id=t.object_id JOIN sys.index_columns ic ON ic.object_id=i.object_id AND ic.index_id=i.index_id JOIN sys.columns c ON c.object_id=ic.object_id AND c.column_id=ic.column_id WHERE s.name IN (?,?) ORDER BY s.name,t.name,i.index_id,ic.index_column_id""",('Fact','Dimension'))
    fks=reader.select("""SELECT fk.name AS fk_name,ps.name AS parent_schema,pt.name AS parent_table,pc.name AS parent_column,rs.name AS referenced_schema,rt.name AS referenced_table,rc.name AS referenced_column,fk.is_disabled,fk.is_not_trusted,fkc.constraint_column_id
    FROM sys.foreign_keys fk JOIN sys.foreign_key_columns fkc ON fk.object_id=fkc.constraint_object_id JOIN sys.tables pt ON pt.object_id=fkc.parent_object_id JOIN sys.schemas ps ON ps.schema_id=pt.schema_id JOIN sys.columns pc ON pc.object_id=pt.object_id AND pc.column_id=fkc.parent_column_id JOIN sys.tables rt ON rt.object_id=fkc.referenced_object_id JOIN sys.schemas rs ON rs.schema_id=rt.schema_id JOIN sys.columns rc ON rc.object_id=rt.object_id AND rc.column_id=fkc.referenced_column_id WHERE ps.name IN (?,?) ORDER BY ps.name,pt.name,fk.name,fkc.constraint_column_id""",('Fact','Dimension'))
    checks=reader.select("""SELECT s.name AS schema_name,t.name AS table_name,c.name,c.definition,c.is_disabled,c.is_not_trusted FROM sys.check_constraints c JOIN sys.tables t ON t.object_id=c.parent_object_id JOIN sys.schemas s ON s.schema_id=t.schema_id WHERE s.name IN (?,?)""",('Fact','Dimension'))
    return {'columns':cols,'indexes':keys,'foreign_keys':fks,'check_constraints':checks}


NUMERIC={'int','bigint','smallint','tinyint','decimal','numeric','float','real','money','smallmoney','bit'}
DATES={'date','datetime','datetime2','smalldatetime','datetimeoffset'}


def fechas(columnas):return [c['column_name'] for c in columnas if c['data_type'] in DATES]


def cobertura(rows,calendar=True):
    """Calendario entre extremos observados; ausencia de filas no certifica cierre."""
    months={r['month']:int(r['n']) for r in rows};missing=[];changes=[]
    if months and calendar:
        year,month=map(int,min(months).split('-'));last=max(months)
        while f'{year:04}-{month:02}'<=last:
            key=f'{year:04}-{month:02}'
            if key not in months:missing.append(key)
            month+=1
            if month==13:year+=1;month=1
        keys=sorted(months)
        for a,b in zip(keys,keys[1:]):
            ratio=months[b]/months[a]
            if ratio<.5 or ratio>2:changes.append({'previous_observed_month':a,'month':b,'ratio':ratio})
    return {'months_observed':len(months),'missing_months_between_extremes':missing if calendar else None,'volume_changes_outside_half_double':changes,
            'note':'Cambios exploratorios; extremos y meses con filas no prueban completitud ni disponibilidad al cierre.'}


def perfilar(reader,schema,table,meta):
    qualified=qi(schema)+'.'+qi(table)
    cols=[c for c in meta['columns'] if c['schema_name']==schema and c['table_name']==table]
    result={'rows':reader.select(f'SELECT COUNT_BIG(*) AS n FROM {qualified}')[0]['n'],'columns':{},'temporal':{},'foreign_keys':[]}
    for c in cols:
        name=c['column_name'];q=qi(name);kind=c['data_type']
        fields=['COUNT_BIG(*)-COUNT_BIG('+q+') AS nulls']
        if kind not in {'geography','geometry','varbinary','binary','image','xml','text','ntext'}:fields+=['COUNT_BIG(DISTINCT '+q+') AS distinct_nonnull']
        if kind in NUMERIC|DATES:
            v=f'CAST({q} AS int)' if kind=='bit' else q
            fields += [f'MIN({v}) AS min',f'MAX({v}) AS max']
            if kind in NUMERIC:fields += [f'SUM(CAST(CASE WHEN {q}<0 THEN 1 ELSE 0 END AS bigint)) AS negative',f'SUM(CAST(CASE WHEN {q}=0 THEN 1 ELSE 0 END AS bigint)) AS zero']
        result['columns'][name]=reader.select('SELECT '+','.join(fields)+' FROM '+qualified)[0]
    pks=[i['column_name'] for i in meta['indexes'] if i['schema_name']==schema and i['table_name']==table and i['is_primary_key'] and i['key_ordinal']>0]
    if pks:
        result['primary_key_columns']=pks
        result['primary_key_duplicates']=reader.select(f'SELECT COUNT_BIG(*) AS groups,COALESCE(SUM(n-1),0) AS excess FROM (SELECT COUNT_BIG(*) AS n FROM {qualified} GROUP BY '+','.join(map(qi,pks))+' HAVING COUNT_BIG(*)>1) d')[0]
    payload=[c['column_name'] for c in cols if c['column_name'] not in pks+['Lineage Key'] and c['data_type'] not in {'geography','geometry','varbinary','image','xml'}]
    result['payload_duplicates_excluding_pk_lineage']=reader.select(f'SELECT COUNT_BIG(*) AS groups,COALESCE(SUM(n-1),0) AS excess FROM (SELECT COUNT_BIG(*) AS n FROM {qualified} GROUP BY '+','.join(map(qi,payload))+' HAVING COUNT_BIG(*)>1) d')[0]
    result['duplicate_note']='Repetición de contenido sin clave técnica/linaje, no eliminación automática ni prueba de error.'
    for name in fechas(cols):
        q=qi(name)
        monthly=reader.select(f'SELECT CONVERT(char(7),{q},126) AS month,COUNT_BIG(*) AS n FROM {qualified} WHERE {q} IS NOT NULL GROUP BY CONVERT(char(7),{q},126) ORDER BY month')
        yearly=reader.select(f'SELECT YEAR({q}) AS year,COUNT_BIG(*) AS n FROM {qualified} WHERE {q} IS NOT NULL GROUP BY YEAR({q}) ORDER BY year')
        result['temporal'][name]={'monthly':monthly,'yearly':yearly,**cobertura(monthly,calendar=name not in {'Valid From','Valid To'})}
    for f in meta['foreign_keys']:
        if f['parent_schema']!=schema or f['parent_table']!=table:continue
        if f['constraint_column_id']!=1:raise ValueError('FK compuesta: requiere auditoría agrupada explícita')
        p=qi(f['parent_column']);r=qi(f['referenced_column']);target=qi(f['referenced_schema'])+'.'+qi(f['referenced_table'])
        row=reader.select(f'SELECT COUNT_BIG(*) AS nonnull_references,SUM(CAST(CASE WHEN d.{r} IS NULL THEN 1 ELSE 0 END AS bigint)) AS orphans FROM {qualified} f LEFT JOIN {target} d ON f.{p}=d.{r} WHERE f.{p} IS NOT NULL')[0]
        result['foreign_keys'].append({**f,**row})
    result['distributions']={}
    if schema=='Fact':
        for name in ['Quantity','Ordered Quantity','Ordered Outers','Received Outers','Quantity On Hand','Outstanding Balance','Total Including Tax']:
            if name not in result['columns']:continue
            q=qi(name)
            stats=reader.select('SELECT DISTINCT '+','.join(f'PERCENTILE_CONT({p}) WITHIN GROUP(ORDER BY CAST({q} AS float)) OVER() AS {alias}' for p,alias in [(.05,'q05'),(.25,'q25'),(.5,'median'),(.75,'q75'),(.95,'q95')])+f' FROM {qualified} WHERE {q} IS NOT NULL')[0]
            lo=stats['q25']-1.5*(stats['q75']-stats['q25']);hi=stats['q75']+1.5*(stats['q75']-stats['q25'])
            stats.update(reader.select(f'SELECT COUNT_BIG(*) AS outside_1_5_IQR FROM {qualified} WHERE {q}<? OR {q}>?',(lo,hi))[0]);stats['lower_fence']=lo;stats['upper_fence']=hi
            result['distributions'][name]=stats
    return result


def granos(reader):
    results={}
    for table,key in [('Order','WWI Order ID'),('Purchase','WWI Purchase Order ID'),('Movement','WWI Stock Item Transaction ID')]:
        name=qi(key);t='[Fact].'+qi(table)
        results[table]=reader.select(f'SELECT COUNT_BIG(*) AS documents,MIN(n) AS min_lines,MAX(n) AS max_lines,AVG(CAST(n AS float)) AS mean_lines,SUM(CAST(CASE WHEN n>1 THEN 1 ELSE 0 END AS bigint)) AS documents_multiple_lines FROM (SELECT {name},COUNT_BIG(*) AS n FROM {t} WHERE {name} IS NOT NULL GROUP BY {name}) d')[0]
        results[table]['document_date_inconsistencies']=reader.select(f'SELECT COUNT_BIG(*) AS n FROM (SELECT {name} FROM {t} WHERE {name} IS NOT NULL GROUP BY {name} HAVING COUNT(DISTINCT '+qi('Order Date Key' if table=='Order' else 'Date Key')+')>1) d')[0]['n']
    for key in ['WWI Customer Transaction ID','WWI Supplier Transaction ID','WWI Invoice ID','WWI Purchase Order ID']:
        q=qi(key)
        results['Transaction:'+key]=reader.select(f'SELECT COUNT_BIG(*) AS identifiers,MAX(n) AS max_rows_per_identifier,SUM(CAST(CASE WHEN n>1 THEN 1 ELSE 0 END AS bigint)) AS repeated_identifiers FROM (SELECT {q},COUNT_BIG(*) AS n FROM [Fact].[Transaction] WHERE {q} IS NOT NULL GROUP BY {q}) d')[0]
    for key in ['WWI Invoice ID','WWI Purchase Order ID']:
        q=qi(key)
        results['Movement:'+key]=reader.select(f'SELECT COUNT_BIG(*) AS documents,MIN(n) AS min_movements,MAX(n) AS max_movements,AVG(CAST(n AS float)) AS mean_movements FROM (SELECT {q},COUNT_BIG(*) AS n FROM [Fact].[Movement] WHERE {q} IS NOT NULL GROUP BY {q}) d')[0]
    results['Order_dates']=reader.select('SELECT SUM(CAST(CASE WHEN [Picked Date Key]<[Order Date Key] THEN 1 ELSE 0 END AS bigint)) AS picked_before_order,COUNT(DISTINCT [WWI Order ID]) AS documents FROM [Fact].[Order]')[0]
    results['Order_delay_histogram']=reader.select('SELECT DATEDIFF(day,[Order Date Key],[Picked Date Key]) AS days_to_pick,COUNT_BIG(*) AS lines FROM [Fact].[Order] GROUP BY DATEDIFF(day,[Order Date Key],[Picked Date Key]) ORDER BY days_to_pick')
    results['Movement_by_type']=reader.select('SELECT t.[Transaction Type],COUNT_BIG(*) AS rows,MIN(f.[Quantity]) AS min_quantity,MAX(f.[Quantity]) AS max_quantity,SUM(CAST(CASE WHEN f.[Quantity]<0 THEN 1 ELSE 0 END AS bigint)) AS negative,SUM(CAST(CASE WHEN f.[Quantity]>0 THEN 1 ELSE 0 END AS bigint)) AS positive,SUM(CAST(f.[Quantity] AS bigint)) AS net_units FROM [Fact].[Movement] f JOIN [Dimension].[Transaction Type] t ON f.[Transaction Type Key]=t.[Transaction Type Key] GROUP BY t.[Transaction Type] ORDER BY t.[Transaction Type]')
    results['Transaction_by_type']=reader.select('SELECT t.[Transaction Type],COUNT_BIG(*) AS rows,SUM(CAST(CASE WHEN f.[Total Including Tax]<0 THEN 1 ELSE 0 END AS bigint)) AS negative_amount,SUM(CAST(CASE WHEN f.[Outstanding Balance]<>0 THEN 1 ELSE 0 END AS bigint)) AS nonzero_outstanding,SUM(CAST(f.[Is Finalized] AS bigint)) AS finalized FROM [Fact].[Transaction] f JOIN [Dimension].[Transaction Type] t ON f.[Transaction Type Key]=t.[Transaction Type Key] GROUP BY t.[Transaction Type] ORDER BY t.[Transaction Type]')
    results['Purchase_receipts']=reader.select('SELECT [Is Order Finalized] AS finalized,COUNT_BIG(*) AS lines,SUM(CAST(CASE WHEN [Received Outers]>[Ordered Outers] THEN 1 ELSE 0 END AS bigint)) AS received_exceeds_ordered,SUM(CAST(CASE WHEN [Received Outers]=0 THEN 1 ELSE 0 END AS bigint)) AS zero_received FROM [Fact].[Purchase] GROUP BY [Is Order Finalized]')
    results['Stock_holding_history']='No existe columna fecha en el esquema real: instantánea, no serie temporal demostrada.'
    return results


def vigencias(reader,meta):
    result={}
    for table in DIMENSIONS:
        cols=[c['column_name'] for c in meta['columns'] if c['schema_name']=='Dimension' and c['table_name']==table]
        if 'Valid From' not in cols:continue
        business=next(c for c in cols if c.startswith('WWI ') and c.endswith(' ID'))
        t='[Dimension].'+qi(table);k=qi(table+' Key');b=qi(business)
        result[table]={'business_id':business,'versions':reader.select(f'SELECT COUNT_BIG(*) AS business_ids,MAX(n) AS max_versions,SUM(CAST(CASE WHEN n>1 THEN 1 ELSE 0 END AS bigint)) AS ids_multiple_versions FROM (SELECT {b},COUNT_BIG(*) AS n FROM {t} GROUP BY {b}) d')[0],
                       'invalid_intervals':reader.select(f'SELECT COUNT_BIG(*) AS n FROM {t} WHERE [Valid From]>=[Valid To]')[0]['n'],
                       'overlapping_version_pairs':reader.select(f'SELECT COUNT_BIG(*) AS n FROM {t} a JOIN {t} b ON a.{b}=b.{b} AND a.{k}<b.{k} AND a.[Valid From]<b.[Valid To] AND b.[Valid From]<a.[Valid To]')[0]['n']}
    # Revisión de claves dimensionales respecto a fecha del evento, no joins actuales.
    align=[]
    dates={'Order':'Order Date Key','Purchase':'Date Key','Movement':'Date Key','Transaction':'Date Key'}
    for fk in meta['foreign_keys']:
        table=fk['parent_table'];dim=fk['referenced_table']
        if table not in dates or dim not in result:continue
        t='[Fact].'+qi(table);d='[Dimension].'+qi(dim);p=qi(fk['parent_column']);key=qi(fk['referenced_column']);date=qi(dates[table])
        v=reader.select(f'SELECT COUNT_BIG(*) AS outside_event_date_interval FROM {t} f JOIN {d} d ON f.{p}=d.{key} WHERE CAST(f.{date} AS datetime2)<d.[Valid From] OR CAST(f.{date} AS datetime2)>=d.[Valid To]')[0]
        align.append({'fact':table,'dimension':dim,'column':fk['parent_column'],**v})
    return {'dimensions':result,'fact_version_alignment_by_date_midnight':align,'note':'Intervalos semiabiertos [Valid From,Valid To). Las claves fecha carecen de hora: desacuerdos intradía requieren resolver contra hora OLTP; los valores posteriores al corte no se vuelven predictores por tener FK válida.'}


def inspeccionar_oltp(reader):
    """Agregados para resolver cumplimiento, recepciones y pago; no ETL."""
    result={}
    result['columns']=reader.select("SELECT s.name AS schema_name,t.name AS table_name,c.name AS column_name,ty.name AS data_type,c.is_nullable,t.temporal_type_desc FROM sys.tables t JOIN sys.schemas s ON t.schema_id=s.schema_id JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON c.user_type_id=ty.user_type_id WHERE t.name IN (?,?,?,?,?,?,?,?) ORDER BY s.name,t.name,c.column_id",('Orders','OrderLines','PurchaseOrders','PurchaseOrderLines','StockItemTransactions','StockItemHoldings','CustomerTransactions','SupplierTransactions'))
    result['integration_definitions']=reader.select("SELECT s.name AS schema_name,o.name AS module_name,m.definition FROM sys.sql_modules m JOIN sys.objects o ON o.object_id=m.object_id JOIN sys.schemas s ON s.schema_id=o.schema_id WHERE s.name=? AND o.name IN (?,?,?,?,?)",('Integration','GetOrderUpdates','GetPurchaseUpdates','GetMovementUpdates','GetStockHoldingUpdates','GetTransactionUpdates'))
    result['orders']=reader.select('SELECT COUNT_BIG(*) AS orders,MIN(OrderDate) AS min_order_date,MAX(OrderDate) AS max_order_date,SUM(CAST(CASE WHEN PickingCompletedWhen IS NULL THEN 1 ELSE 0 END AS bigint)) AS no_completion,SUM(CAST(CASE WHEN BackorderOrderID IS NOT NULL THEN 1 ELSE 0 END AS bigint)) AS with_backorder,SUM(CAST(CASE WHEN LastEditedWhen>CAST(OrderDate AS datetime2) THEN 1 ELSE 0 END AS bigint)) AS edited_after_order_date FROM Sales.Orders')[0]
    result['order_line_keys']=reader.select('SELECT COUNT_BIG(*) AS repeated_order_stock_pairs,COALESCE(SUM(n-1),0) AS excess FROM (SELECT OrderID,StockItemID,COUNT_BIG(*) AS n FROM Sales.OrderLines GROUP BY OrderID,StockItemID HAVING COUNT_BIG(*)>1) d')[0]
    result['order_line_link']=reader.select('SELECT COUNT_BIG(*) AS matched, SUM(CAST(CASE WHEN d.[Quantity]<>ol.Quantity OR d.[Order Date Key]<>o.OrderDate OR (d.[Picked Date Key]<>CAST(ol.PickingCompletedWhen AS date)) OR (d.[Picked Date Key] IS NULL AND ol.PickingCompletedWhen IS NOT NULL) OR (d.[Picked Date Key] IS NOT NULL AND ol.PickingCompletedWhen IS NULL) THEN 1 ELSE 0 END AS bigint)) AS discrepancies FROM [WideWorldImportersDW].[Fact].[Order] d JOIN [WideWorldImportersDW].[Dimension].[Stock Item] si ON si.[Stock Item Key]=d.[Stock Item Key] JOIN Sales.OrderLines ol ON ol.OrderID=d.[WWI Order ID] AND ol.StockItemID=si.[WWI Stock Item ID] JOIN Sales.Orders o ON o.OrderID=ol.OrderID')[0]
    result['order_candidate_2day']=reader.select('SELECT COUNT_BIG(*) AS eligible_documents,SUM(CAST(CASE WHEN PickingCompletedWhen IS NULL OR CAST(PickingCompletedWhen AS date)>DATEADD(day,2,OrderDate) THEN 1 ELSE 0 END AS bigint)) AS possible_positive,SUM(CAST(CASE WHEN PickingCompletedWhen IS NOT NULL AND CAST(PickingCompletedWhen AS date)<=DATEADD(day,2,OrderDate) THEN 1 ELSE 0 END AS bigint)) AS possible_negative FROM Sales.Orders WHERE OrderDate<=DATEADD(day,-2,(SELECT MAX([Order Date Key]) FROM [WideWorldImportersDW].[Fact].[Order]))')[0]
    result['order_candidate_censoring']=reader.select('SELECT COUNT_BIG(*) AS no_completion_before_2day_cutoff FROM Sales.Orders WHERE PickingCompletedWhen IS NULL AND OrderDate<=DATEADD(day,-2,(SELECT MAX([Order Date Key]) FROM [WideWorldImportersDW].[Fact].[Order]))')[0]
    result['purchase_dates']=reader.select('SELECT COUNT_BIG(*) AS lines,SUM(CAST(CASE WHEN l.LastReceiptDate IS NULL THEN 1 ELSE 0 END AS bigint)) AS missing_receipt,SUM(CAST(CASE WHEN l.LastReceiptDate<p.OrderDate THEN 1 ELSE 0 END AS bigint)) AS receipt_before_order,SUM(CAST(CASE WHEN l.LastReceiptDate>p.ExpectedDeliveryDate THEN 1 ELSE 0 END AS bigint)) AS last_receipt_after_expected,SUM(CAST(CASE WHEN l.LastEditedWhen>CAST(p.OrderDate AS datetime2) THEN 1 ELSE 0 END AS bigint)) AS edited_after_order_date FROM Purchasing.PurchaseOrderLines l JOIN Purchasing.PurchaseOrders p ON l.PurchaseOrderID=p.PurchaseOrderID')[0]
    result['purchase_candidate_late']=reader.select('SELECT COUNT_BIG(*) AS closed_observed_lines,SUM(CAST(CASE WHEN l.LastReceiptDate>p.ExpectedDeliveryDate THEN 1 ELSE 0 END AS bigint)) AS possible_late,COUNT(DISTINCT p.PurchaseOrderID) AS documents FROM Purchasing.PurchaseOrderLines l JOIN Purchasing.PurchaseOrders p ON l.PurchaseOrderID=p.PurchaseOrderID WHERE l.IsOrderLineFinalized=1 AND l.LastReceiptDate IS NOT NULL AND p.ExpectedDeliveryDate<=(SELECT MAX([Date Key]) FROM [WideWorldImportersDW].[Fact].[Purchase])')[0]
    result['movements']=reader.select('SELECT COUNT_BIG(*) AS transactions,SUM(CAST(CASE WHEN Quantity<>FLOOR(Quantity) THEN 1 ELSE 0 END AS bigint)) AS fractional_quantities,MIN(TransactionOccurredWhen) AS min_event,MAX(TransactionOccurredWhen) AS max_event,SUM(CAST(CASE WHEN LastEditedWhen>TransactionOccurredWhen THEN 1 ELSE 0 END AS bigint)) AS edited_after_event FROM Warehouse.StockItemTransactions')[0]
    result['movement_link']=reader.select('SELECT COUNT_BIG(*) AS matched,SUM(CAST(CASE WHEN d.[Date Key]<>CAST(s.TransactionOccurredWhen AS date) OR d.[Quantity]<>s.Quantity THEN 1 ELSE 0 END AS bigint)) AS discrepancies FROM [WideWorldImportersDW].[Fact].[Movement] d JOIN Warehouse.StockItemTransactions s ON s.StockItemTransactionID=d.[WWI Stock Item Transaction ID]')[0]
    for source,table,key in [('customer','Sales.CustomerTransactions','WWI Customer Transaction ID'),('supplier','Purchasing.SupplierTransactions','WWI Supplier Transaction ID')]:
        result[source+'_transactions']=reader.select(f'SELECT COUNT_BIG(*) AS transactions,MIN(TransactionDate) AS min_event,MAX(TransactionDate) AS max_event,MAX(FinalizationDate) AS latest_finalization,SUM(CAST(CASE WHEN FinalizationDate>TransactionDate THEN 1 ELSE 0 END AS bigint)) AS finalized_later,SUM(CAST(CASE WHEN IsFinalized=1 AND OutstandingBalance<>0 THEN 1 ELSE 0 END AS bigint)) AS finalized_nonzero_balance,SUM(CAST(CASE WHEN IsFinalized=0 AND FinalizationDate IS NOT NULL THEN 1 ELSE 0 END AS bigint)) AS unfinalized_with_date FROM {table}')[0]
    result['invoice_payment_candidate_30day']=reader.select('SELECT COUNT_BIG(*) AS eligible_positive_invoices,SUM(CAST(CASE WHEN FinalizationDate IS NULL OR FinalizationDate>DATEADD(day,30,TransactionDate) THEN 1 ELSE 0 END AS bigint)) AS possible_unpaid_after_30d,MAX(FinalizationDate) AS latest_finalization FROM Sales.CustomerTransactions WHERE InvoiceID IS NOT NULL AND TransactionAmount>0 AND TransactionDate<=DATEADD(day,-30,(SELECT MAX([Date Key]) FROM [WideWorldImportersDW].[Fact].[Transaction]))')[0]
    result['purchase_unit_conversion']=reader.select('SELECT COUNT_BIG(*) AS matched,SUM(CAST(CASE WHEN d.[Ordered Quantity]<>l.OrderedOuters*s.QuantityPerOuter OR d.[Is Order Finalized]<>l.IsOrderLineFinalized OR d.[Received Outers]<>l.ReceivedOuters THEN 1 ELSE 0 END AS bigint)) AS discrepancies FROM [WideWorldImportersDW].[Fact].[Purchase] d JOIN [WideWorldImportersDW].[Dimension].[Stock Item] si ON si.[Stock Item Key]=d.[Stock Item Key] JOIN Purchasing.PurchaseOrderLines l ON l.PurchaseOrderID=d.[WWI Purchase Order ID] AND l.StockItemID=si.[WWI Stock Item ID] JOIN Warehouse.StockItems s ON s.StockItemID=l.StockItemID')[0]
    result['scope']='Metadatos y agregados necesarios; las definiciones de procedimientos se leen desde sys.sql_modules, nunca se ejecutan.'
    return result


def complementos(reader,oltp):
    result={}
    for table in ['Order','Purchase','Movement']:
        key='Order Date Key' if table=='Order' else 'Date Key'
        # Solo agregado de grupos observados. No calendario imputado ni dataset.
        result[table+'_observed_product_months']=reader.select('SELECT COUNT_BIG(*) AS groups,MIN(month) AS min_month,MAX(month) AS max_month,SUM(CAST(CASE WHEN net_units>0 THEN 1 ELSE 0 END AS bigint)) AS positive_net_groups,SUM(CAST(CASE WHEN net_units<0 THEN 1 ELSE 0 END AS bigint)) AS negative_net_groups,SUM(CAST(CASE WHEN net_units=0 THEN 1 ELSE 0 END AS bigint)) AS zero_net_groups FROM (SELECT si.[WWI Stock Item ID],CONVERT(char(7),f.'+qi(key)+',126) AS month,SUM(CAST(f.'+qi('Ordered Quantity' if table=='Purchase' else 'Quantity')+' AS bigint)) AS net_units FROM [Fact].'+qi(table)+' f JOIN [Dimension].[Stock Item] si ON f.[Stock Item Key]=si.[Stock Item Key] GROUP BY si.[WWI Stock Item ID],CONVERT(char(7),f.'+qi(key)+',126)) g')[0]
    result['stock_holding_current_version']=reader.select('SELECT COUNT_BIG(*) AS rows,COUNT(DISTINCT s.[WWI Stock Item ID]) AS business_products,MIN(s.[Valid From]) AS min_dimension_valid_from,MAX(s.[Valid From]) AS max_dimension_valid_from FROM [Fact].[Stock Holding] h JOIN [Dimension].[Stock Item] s ON h.[Stock Item Key]=s.[Stock Item Key]')[0]
    result['order_backorder_chain']=oltp.select('SELECT COUNT_BIG(*) AS orders,SUM(CAST(CASE WHEN o.PickingCompletedWhen IS NULL AND o.BackorderOrderID IS NOT NULL THEN 1 ELSE 0 END AS bigint)) AS null_completion_with_backorder,SUM(CAST(CASE WHEN o.PickingCompletedWhen IS NULL AND o.BackorderOrderID IS NULL THEN 1 ELSE 0 END AS bigint)) AS null_completion_without_backorder,SUM(CAST(CASE WHEN o.BackorderOrderID IS NOT NULL AND b.OrderID IS NULL THEN 1 ELSE 0 END AS bigint)) AS orphan_backorders,SUM(CAST(CASE WHEN b.OrderDate<o.OrderDate THEN 1 ELSE 0 END AS bigint)) AS backorder_before_parent,MAX(DATEDIFF(day,o.OrderDate,b.OrderDate)) AS max_backorder_lag_days FROM Sales.Orders o LEFT JOIN Sales.Orders b ON o.BackorderOrderID=b.OrderID')[0]
    result['order_backorder_candidate_2day']=oltp.select('SELECT COUNT_BIG(*) AS mature_orders,SUM(CAST(CASE WHEN b.OrderID IS NOT NULL AND b.OrderDate<=DATEADD(day,2,o.OrderDate) THEN 1 ELSE 0 END AS bigint)) AS backorder_within_2d,SUM(CAST(CASE WHEN b.OrderID IS NULL OR b.OrderDate>DATEADD(day,2,o.OrderDate) THEN 1 ELSE 0 END AS bigint)) AS no_backorder_within_2d FROM Sales.Orders o LEFT JOIN Sales.Orders b ON o.BackorderOrderID=b.OrderID WHERE o.OrderDate<=DATEADD(day,-2,(SELECT MAX([Order Date Key]) FROM [WideWorldImportersDW].[Fact].[Order]))')[0]
    result['payment_delay_distribution']=oltp.select('SELECT DATEDIFF(day,TransactionDate,FinalizationDate) AS days_to_finalize,COUNT_BIG(*) AS invoices FROM Sales.CustomerTransactions WHERE InvoiceID IS NOT NULL AND TransactionAmount>0 GROUP BY DATEDIFF(day,TransactionDate,FinalizationDate) ORDER BY days_to_finalize')
    # Ejemplo exploratorio de horizonte corto, sin fijar el contrato final.
    result['invoice_payment_candidate_7day']=oltp.select('SELECT COUNT_BIG(*) AS mature_invoices,SUM(CAST(CASE WHEN FinalizationDate IS NULL OR FinalizationDate>DATEADD(day,7,TransactionDate) THEN 1 ELSE 0 END AS bigint)) AS possible_positive FROM Sales.CustomerTransactions WHERE InvoiceID IS NOT NULL AND TransactionAmount>0 AND TransactionDate<=DATEADD(day,-7,(SELECT MAX([Date Key]) FROM [WideWorldImportersDW].[Fact].[Transaction]))')[0]
    result['purchase_receipt_delay_histogram']=oltp.select('SELECT DATEDIFF(day,p.OrderDate,l.LastReceiptDate) AS days_to_last_receipt,COUNT_BIG(*) AS lines FROM Purchasing.PurchaseOrderLines l JOIN Purchasing.PurchaseOrders p ON l.PurchaseOrderID=p.PurchaseOrderID GROUP BY DATEDIFF(day,p.OrderDate,l.LastReceiptDate) ORDER BY days_to_last_receipt')
    result['order_line_candidate_2day']=oltp.select('SELECT COUNT_BIG(*) AS mature_lines,SUM(CAST(CASE WHEN l.PickingCompletedWhen IS NULL THEN 1 ELSE 0 END AS bigint)) AS unknown_completion,SUM(CAST(CASE WHEN l.PickingCompletedWhen IS NOT NULL AND CAST(l.PickingCompletedWhen AS date)>DATEADD(day,2,o.OrderDate) THEN 1 ELSE 0 END AS bigint)) AS observed_late,SUM(CAST(CASE WHEN l.PickingCompletedWhen IS NOT NULL AND CAST(l.PickingCompletedWhen AS date)<=DATEADD(day,2,o.OrderDate) THEN 1 ELSE 0 END AS bigint)) AS observed_within_2d FROM Sales.OrderLines l JOIN Sales.Orders o ON o.OrderID=l.OrderID WHERE o.OrderDate<=DATEADD(day,-2,(SELECT MAX([Order Date Key]) FROM [WideWorldImportersDW].[Fact].[Order]))')[0]
    result['order_unknown_customer_source']=oltp.select('SELECT COUNT_BIG(*) AS affected_lines,COUNT(DISTINCT o.CustomerID) AS source_customer_ids,SUM(CAST(CASE WHEN c.[Customer Key] IS NULL THEN 1 ELSE 0 END AS bigint)) AS lines_without_business_id_in_dimension FROM [WideWorldImportersDW].[Fact].[Order] d JOIN Sales.Orders o ON d.[WWI Order ID]=o.OrderID LEFT JOIN [WideWorldImportersDW].[Dimension].[Customer] c ON c.[WWI Customer ID]=o.CustomerID WHERE d.[Customer Key]=0')[0]
    result['unknown_dimension_keys']=[]
    result['business_cardinalities']=[]
    for fk in reader.select("SELECT ps.name AS parent_schema,pt.name AS parent_table,pc.name AS parent_column FROM sys.foreign_key_columns f JOIN sys.tables pt ON pt.object_id=f.parent_object_id JOIN sys.schemas ps ON ps.schema_id=pt.schema_id JOIN sys.columns pc ON pc.object_id=pt.object_id AND pc.column_id=f.parent_column_id JOIN sys.tables rt ON rt.object_id=f.referenced_object_id WHERE ps.name=? AND rt.name<>?",('Fact','Date')):
        if fk['parent_table'] not in FACTS:continue
        n=reader.select('SELECT COUNT_BIG(*) AS zero_key_rows FROM [Fact].'+qi(fk['parent_table'])+' WHERE '+qi(fk['parent_column'])+'=0')[0]['zero_key_rows']
        result['unknown_dimension_keys'].append({**fk,'zero_key_rows':n})
    relations=reader.select("SELECT pt.name AS fact,pc.name AS role,rt.name AS dimension,rc.name AS dimension_key,bc.name AS business_id FROM sys.foreign_key_columns f JOIN sys.tables pt ON pt.object_id=f.parent_object_id JOIN sys.schemas ps ON ps.schema_id=pt.schema_id JOIN sys.columns pc ON pc.object_id=pt.object_id AND pc.column_id=f.parent_column_id JOIN sys.tables rt ON rt.object_id=f.referenced_object_id JOIN sys.columns rc ON rc.object_id=rt.object_id AND rc.column_id=f.referenced_column_id JOIN sys.columns bc ON bc.object_id=rt.object_id WHERE ps.name=? AND bc.name LIKE ?",('Fact','WWI % ID'))
    for fk in relations:
        if fk['fact'] not in FACTS:continue
        b=qi(fk['business_id']);k=qi(fk['dimension_key']);role=qi(fk['role'])
        row=reader.select('SELECT COUNT_BIG(DISTINCT CASE WHEN d.'+b+'<>0 THEN d.'+b+' END) AS real_business_ids FROM [Fact].'+qi(fk['fact'])+' f JOIN [Dimension].'+qi(fk['dimension'])+' d ON f.'+role+'=d.'+k)[0]
        result['business_cardinalities'].append({**fk,**row})
    # Resumen acotado de intervalos superpuestos, sin nombres/contactos.
    for table,key in [('Supplier','Supplier'),('City','City'),('Employee','Employee')]:
        t='[Dimension].'+qi(table);k=qi(key+' Key');b=qi('WWI '+key+' ID')
        result['overlap_'+table]=reader.select(f'SELECT TOP (10) a.{b} AS business_id,a.{k} AS key_a,b.{k} AS key_b,a.[Valid From] AS from_a,a.[Valid To] AS to_a,b.[Valid From] AS from_b,b.[Valid To] AS to_b FROM {t} a JOIN {t} b ON a.{b}=b.{b} AND a.{k}<b.{k} AND a.[Valid From]<b.[Valid To] AND b.[Valid From]<a.[Valid To] ORDER BY a.{b},a.{k},b.{k}')
    result['note']='Conteos de propuestas exploratorias y ejemplos acotados de intervalos, sin materializar etiquetas ni elegir umbrales por rendimiento predictivo.'
    return result


def huella(reader,meta):
    """Huella de control no criptográfica de filas escalares; no prueba concurrencia."""
    result={}
    for schema,tables in [('Fact',FACTS),('Dimension',DIMENSIONS)]:
        for table in tables:
            cols=[qi(c['column_name']) for c in meta['columns'] if c['schema_name']==schema and c['table_name']==table and c['data_type'] not in {'geography','geometry','varbinary','image','xml'}]
            result[schema+'.'+table]=reader.select('SELECT COUNT_BIG(*) AS rows,CHECKSUM_AGG(BINARY_CHECKSUM('+','.join(cols)+')) AS checksum FROM '+qi(schema)+'.'+qi(table))[0]
    return result


def huella_oltp(reader):
    cols=reader.select("SELECT s.name AS schema_name,t.name AS table_name,c.name AS column_name FROM sys.tables t JOIN sys.schemas s ON t.schema_id=s.schema_id JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON c.user_type_id=ty.user_type_id WHERE t.name IN (?,?,?,?,?,?,?,?) AND ty.name NOT IN (?,?,?,?) ORDER BY s.name,t.name,c.column_id",('Orders','OrderLines','PurchaseOrders','PurchaseOrderLines','StockItemTransactions','StockItemHoldings','CustomerTransactions','SupplierTransactions','geography','geometry','image','xml'))
    result={}
    for key in dict.fromkeys((c['schema_name'],c['table_name']) for c in cols):
        names=','.join(qi(c['column_name']) for c in cols if (c['schema_name'],c['table_name'])==key)
        result['.'.join(key)]=reader.select('SELECT COUNT_BIG(*) AS rows,CHECKSUM_AGG(BINARY_CHECKSUM('+names+')) AS checksum FROM '+'.'.join(map(qi,key)))[0]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--server');parser.add_argument('--metadata-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    before_path=OUT/'integridad_antes.json'
    if not before_path.exists():guardar(before_path,snapshot())
    cfg=Configuracion.entorno(args.server)
    reader=Lector(cfg.conectar())
    try:
        info=reader.select("SELECT CAST(SERVERPROPERTY('ServerName') AS nvarchar(128)) AS server_name,DB_NAME() AS database_name,CAST(SERVERPROPERTY('ProductVersion') AS nvarchar(128)) AS version")
        meta=metadatos(reader);guardar(OUT/'diccionario_tablas.json',meta)
        required={('Fact',t) for t in FACTS}|{('Dimension',t) for t in DIMENSIONS}
        actual={(c['schema_name'],c['table_name']) for c in meta['columns']}
        if not required<=actual:raise ValueError('Faltan tablas requeridas: '+str(sorted(required-actual)))
        guardar(OUT/'conexion_verificada.json',{'connection':info,'authentication':'Windows integrada','driver':cfg.driver,'local_readonly':True,'note':'Acceso ODBC readonly e intención de lectura; toda consulta emitida se valida SELECT, no se cambia ningún permiso en el servidor.'})
        if not args.metadata_only:
            initial_fingerprint=huella(reader,meta);guardar(OUT/'huella_bd_antes_comprobacion.json',initial_fingerprint)
            profiles={}
            for schema,tables in [('Fact',FACTS),('Dimension',DIMENSIONS)]:
                for table in tables:
                    profiles[schema+'.'+table]=perfilar(reader,schema,table,meta)
                    print('Inspeccionada',schema+'.'+table,profiles[schema+'.'+table]['rows'],flush=True)
                    guardar(OUT/'perfiles_calidad.json',profiles)
            guardar(OUT/'granularidad.json',granos(reader))
            guardar(OUT/'vigencia_dimensional.json',vigencias(reader,meta))
            oltp=Lector(Configuracion.entorno(cfg.server,'WideWorldImporters').conectar())
            try:
                oltp_before=huella_oltp(oltp);guardar(OUT/'huella_oltp_antes.json',oltp_before)
                guardar(OUT/'comprobaciones_oltp.json',inspeccionar_oltp(oltp))
                guardar(OUT/'comprobaciones_adicionales.json',complementos(reader,oltp))
                oltp_after=huella_oltp(oltp);guardar(OUT/'huella_oltp_despues.json',oltp_after)
                if oltp_before!=oltp_after:raise RuntimeError('Cambió la huella OLTP; revisar concurrencia')
            finally:
                guardar(OUT/'consultas_oltp.json',oltp.consultas);oltp.close()
            final_fingerprint=huella(reader,meta);guardar(OUT/'huella_bd_despues_comprobacion.json',final_fingerprint)
            if initial_fingerprint!=final_fingerprint:raise RuntimeError('Cambió la huella de la base durante la auditoría; revisar concurrencia')
            from .reportes_dw import exportar
            exportar()
            before=json.loads(before_path.read_text(encoding='utf-8'))
            after={path:sha(BASE/path) for path in before}
            if before!=after:raise RuntimeError('Cambió un archivo anterior durante la auditoría')
            guardar(OUT/'integridad_despues.json',after)
        guardar(OUT/'consultas.json',reader.consultas)
        print(json.dumps({'connection':info,'columns':len(meta['columns']),'foreign_key_columns':len(meta['foreign_keys'])}))
    finally:
        guardar(OUT/'consultas.json',reader.consultas);reader.close()
    if not args.metadata_only:manifiesto()


def manifiesto():
    guardar(OUT/'manifiesto.json',{
        'scope':'Auditoría local de lectura, sin etiquetas definitivas, entrenamiento o cambios de Ventas',
        'databases':['WideWorldImportersDW','WideWorldImporters'],'only_select':True,
        'prior_files_preserved':len(json.loads((OUT/'integridad_antes.json').read_text(encoding='utf-8'))),
        'database_fingerprints_match':json.loads((OUT/'huella_bd_antes_comprobacion.json').read_text(encoding='utf-8'))==json.loads((OUT/'huella_bd_despues_comprobacion.json').read_text(encoding='utf-8')),
        'oltp_fingerprints_match':json.loads((OUT/'huella_oltp_antes.json').read_text(encoding='utf-8'))==json.loads((OUT/'huella_oltp_despues.json').read_text(encoding='utf-8')),
        'fingerprint_limit':'COUNT y CHECKSUM_AGG son controles acotados no criptográficos; no prueban ausencia de cambios concurrentes fuera del intervalo o colisiones.',
        'implementation_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(list((BASE/'src/db').glob('*.py'))+[BASE/'tests/test_auditoria_dw.py'])},
        'artifacts_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'manifiesto.json','validaciones.json'}}})


if __name__=='__main__':main()
