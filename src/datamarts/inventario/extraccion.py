"""Recepciones exclusivamente anteriores a 2016; SELECT y snapshot determinista."""
import csv
import datetime as dt
from decimal import Decimal
import gzip
import io

from src.db.conexion import Configuracion,Lector

PARAMS=(11,dt.date(2013,1,1),dt.date(2016,1,1))
SOURCE_SQL="""SELECT st.StockItemTransactionID AS event_id,st.StockItemID AS item_id,
st.TransactionTypeID AS type_id,st.TransactionOccurredWhen AS occurred_at,
st.Quantity AS quantity,st.LastEditedWhen AS edited_at,st.InvoiceID AS invoice_id,
st.PurchaseOrderID AS purchase_order_id,st.CustomerID AS customer_id,st.SupplierID AS supplier_id
FROM Warehouse.StockItemTransactions st
WHERE st.TransactionTypeID=? AND st.TransactionOccurredWhen>=? AND st.TransactionOccurredWhen<?
ORDER BY st.StockItemTransactionID"""
SOURCE_FP="""SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM(st.StockItemTransactionID,
st.StockItemID,st.TransactionTypeID,st.TransactionOccurredWhen,st.Quantity,st.LastEditedWhen,
st.InvoiceID,st.PurchaseOrderID,st.CustomerID,st.SupplierID)) AS checksum
FROM Warehouse.StockItemTransactions st
WHERE st.TransactionTypeID=? AND st.TransactionOccurredWhen>=? AND st.TransactionOccurredWhen<?"""
DW_FROM="""FROM Fact.Movement f JOIN Dimension.[Stock Item] d ON d.[Stock Item Key]=f.[Stock Item Key]
JOIN Dimension.[Transaction Type] tt ON tt.[Transaction Type Key]=f.[Transaction Type Key]
WHERE tt.[WWI Transaction Type ID]=? AND f.[Date Key]>=? AND f.[Date Key]<?"""
DW_SQL="""SELECT f.[WWI Stock Item Transaction ID] AS event_id,d.[WWI Stock Item ID] AS item_id,
tt.[WWI Transaction Type ID] AS type_id,f.[Date Key] AS event_date,f.Quantity AS quantity,
f.[WWI Purchase Order ID] AS purchase_order_id,f.[WWI Invoice ID] AS invoice_id,
f.[Stock Item Key] AS item_key,d.[Valid From] AS valid_from,d.[Valid To] AS valid_to
"""+DW_FROM+' ORDER BY f.[WWI Stock Item Transaction ID]'
DW_FP="""SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM(f.[Movement Key],
f.[WWI Stock Item Transaction ID],d.[WWI Stock Item ID],tt.[WWI Transaction Type ID],
f.[Date Key],f.Quantity,f.[WWI Purchase Order ID],f.[WWI Invoice ID],f.[Stock Item Key])) AS checksum
"""+DW_FROM
FIELDS=['event_id','item_id','type_id','occurred_at','quantity','edited_at','invoice_id','purchase_order_id','customer_id','supplier_id']

def normalize(rows):
    result=[];seen=set()
    for row in rows:
        if set(row)!=set(FIELDS):raise ValueError('Esquema de recepciones inválido')
        r=dict(row)
        for key in ['event_id','item_id','type_id','purchase_order_id','supplier_id']:
            value=r[key]
            if value is None or isinstance(value,bool) or Decimal(str(value))!=int(value) or int(value)<=0:
                raise ValueError('Identidad esencial inválida')
            r[key]=int(value)
        if r['event_id'] in seen:raise ValueError('Evento duplicado')
        seen.add(r['event_id'])
        if r['type_id']!=11 or r['invoice_id'] not in (None,'') or r['customer_id'] not in (None,''):
            raise ValueError('No corresponde a recepción aprobada')
        r['invoice_id']=None;r['customer_id']=None
        when=dt.datetime.fromisoformat(str(r['occurred_at']));edited=dt.datetime.fromisoformat(str(r['edited_at']))
        if when.tzinfo or edited.tzinfo or not dt.datetime(2013,1,1)<=when<dt.datetime(2016,1,1):
            raise ValueError('Solo eventos de desarrollo, sin test')
        if edited!=when:raise ValueError('Edición posterior/incoherente: investigar disponibilidad')
        q=Decimal(str(r['quantity']))
        if not q.is_finite() or q<=0 or q!=q.to_integral_value():raise ValueError('Cantidad no comparable con DW entero')
        r['quantity']=str(int(q));r['occurred_at']=when.isoformat(sep=' ');r['edited_at']=edited.isoformat(sep=' ')
        result.append(r)
    if not result:raise ValueError('Sin recepciones de desarrollo')
    return sorted(result,key=lambda r:r['event_id'])

def csv_bytes(rows):
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n')
    w.writeheader();w.writerows(normalize(rows));return stream.getvalue().encode('utf-8')

def save_snapshot(path,rows):
    with path.open('wb') as f:
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=9) as gz:gz.write(csv_bytes(rows))

def read_snapshot(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:return normalize(list(csv.DictReader(f)))

def reconcile(rows,dw):
    expected=[(r['event_id'],r['item_id'],r['type_id'],r['occurred_at'][:10],int(r['quantity']),r['purchase_order_id'],None) for r in rows]
    actual=[(int(r['event_id']),int(r['item_id']),int(r['type_id']),str(r['event_date']),int(r['quantity']),r['purchase_order_id'],r['invoice_id']) for r in dw]
    if expected!=actual or len({r['event_id'] for r in dw})!=len(dw):raise ValueError('No hay biyección OLTP/DW de recepciones')
    if any(Decimal(str(r['quantity']))!=int(r['quantity']) for r in dw):raise ValueError('Cantidad DW no entera')
    if any(str(r['valid_from'])>rows[j]['occurred_at'] or str(r['valid_to'])<=rows[j]['occurred_at'] for j,r in enumerate(dw)):
        raise ValueError('Versión DW no vigente al evento')
    return sorted({(int(r['item_key']),int(r['item_id']),str(r['valid_from']),str(r['valid_to'])) for r in dw})

def extract(server,log):
    data={};fingerprints={}
    for database,sql,fp in [('WideWorldImporters',SOURCE_SQL,SOURCE_FP),('WideWorldImportersDW',DW_SQL,DW_FP)]:
        reader=Lector(Configuracion(server,database).conectar())
        try:
            before=reader.select(fp,PARAMS);data[database]=reader.select(sql,PARAMS);after=reader.select(fp,PARAMS)
            if before!=after:raise RuntimeError('Cambió la fuente durante lectura')
            fingerprints[database]=dict(before=before,after=after)
        finally:log.extend(dict(database=database,**r) for r in reader.consultas);reader.close()
    rows=normalize(data['WideWorldImporters']);mapping=reconcile(rows,data['WideWorldImportersDW'])
    return rows,fingerprints,mapping
