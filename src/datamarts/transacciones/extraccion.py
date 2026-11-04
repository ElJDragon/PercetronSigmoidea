"""SELECT focalizado y snapshot CSV/gzip determinista de cobros de desarrollo."""
import csv
import datetime as dt
import gzip
import hashlib
import io
import json

from src.db.conexion import Configuracion,Lector

PARAMS=(3,dt.date(2013,1,1),dt.date(2016,1,1))
SOURCE_SQL="""SELECT ct.CustomerTransactionID AS event_id,ct.CustomerID AS customer_id,
ct.TransactionDate AS event_date,ct.TransactionTypeID AS type_id,ct.LastEditedWhen AS edited_at,
CASE WHEN ct.TransactionAmount<0 THEN 1 ELSE 0 END AS negative_payment
FROM Sales.CustomerTransactions ct
WHERE ct.TransactionTypeID=? AND ct.TransactionDate>=? AND ct.TransactionDate<?
ORDER BY ct.CustomerTransactionID"""
SOURCE_FP="""SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM(ct.CustomerTransactionID,
ct.CustomerID,ct.TransactionDate,ct.TransactionTypeID,ct.LastEditedWhen,
CASE WHEN ct.TransactionAmount<0 THEN 1 ELSE 0 END)) AS checksum
FROM Sales.CustomerTransactions ct
WHERE ct.TransactionTypeID=? AND ct.TransactionDate>=? AND ct.TransactionDate<?"""
DW_SQL="""SELECT f.[WWI Customer Transaction ID] AS event_id,f.[Date Key] AS event_date,
t.[WWI Transaction Type ID] AS type_id
FROM Fact.[Transaction] f JOIN Dimension.[Transaction Type] t
ON t.[Transaction Type Key]=f.[Transaction Type Key]
WHERE t.[WWI Transaction Type ID]=? AND f.[Date Key]>=? AND f.[Date Key]<?
AND f.[WWI Customer Transaction ID] IS NOT NULL
ORDER BY f.[WWI Customer Transaction ID]"""
DW_FP="""SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM(f.[WWI Customer Transaction ID],
f.[Date Key],t.[WWI Transaction Type ID])) AS checksum
FROM Fact.[Transaction] f JOIN Dimension.[Transaction Type] t
ON t.[Transaction Type Key]=f.[Transaction Type Key]
WHERE t.[WWI Transaction Type ID]=? AND f.[Date Key]>=? AND f.[Date Key]<?
AND f.[WWI Customer Transaction ID] IS NOT NULL"""
FIELDS=['event_id','customer_id','event_date','type_id','edited_at','negative_payment']


def normalize(rows):
    result=[];seen=set()
    for raw in rows:
        if set(raw)!=set(FIELDS):raise ValueError('Esquema del snapshot no aprobado')
        if any(raw[k] is None for k in FIELDS):raise ValueError('Cobro con dato esencial nulo')
        r={k:str(raw[k]) for k in FIELDS}
        for k in ['event_id','customer_id','type_id','negative_payment']:r[k]=int(r[k])
        date=dt.date.fromisoformat(r['event_date']);edited=dt.datetime.fromisoformat(r['edited_at'])
        if not dt.date(2013,1,1)<=date<dt.date(2016,1,1):raise ValueError('Evento fuera de desarrollo; test sellado')
        if r['type_id']!=3 or r['negative_payment']!=1:raise ValueError('Tipo/signo ajeno al cobro aprobado')
        if r['event_id']<=0 or r['customer_id']<=0 or r['event_id'] in seen:raise ValueError('Identidad nula, inválida o duplicada')
        if edited.date()<date:raise ValueError('Edición anterior al evento')
        seen.add(r['event_id']);result.append(r)
    if not result:raise ValueError('Extracción de cobros vacía')
    return sorted(result,key=lambda r:r['event_id'])


def csv_bytes(rows,fields=FIELDS):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def save_snapshot(path,rows):
    raw=csv_bytes(normalize(rows))
    # mtime y nombre embebido fijos; no contiene fecha de ejecución ni credenciales.
    with path.open('wb') as fh:
        with gzip.GzipFile(filename='',mode='wb',fileobj=fh,mtime=0,compresslevel=9) as gz:gz.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_snapshot(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as fh:return normalize(list(csv.DictReader(fh)))


def extract(server,log):
    """No admite fechas ni SQL suministrados por el llamador: frontera fija <2016."""
    results={};fingerprints={}
    for database,sql,fp in [('WideWorldImporters',SOURCE_SQL,SOURCE_FP),('WideWorldImportersDW',DW_SQL,DW_FP)]:
        reader=Lector(Configuracion(server,database).conectar())
        try:
            if database=='WideWorldImporters':
                kind=reader.select('SELECT TransactionTypeName FROM Application.TransactionTypes WHERE TransactionTypeID=?',(3,))
                if kind!=[{'TransactionTypeName':'Customer Payment Received'}]:raise ValueError('Tipo operacional no confirmado')
            before=reader.select(fp,PARAMS);results[database]=reader.select(sql,PARAMS);after=reader.select(fp,PARAMS)
            if before!=after:raise RuntimeError('La fuente cambió durante la extracción')
            fingerprints[database]={'before':before,'after':after}
        finally:
            log.extend(dict(database=database,**r) for r in reader.consultas);reader.close()
    rows=normalize(results['WideWorldImporters']);dw=results['WideWorldImportersDW']
    source_keys=[(r['event_id'],r['event_date'],r['type_id']) for r in rows]
    dw_keys=[(r['event_id'],str(r['event_date']),r['type_id']) for r in dw]
    if source_keys!=dw_keys or len(set(dw_keys))!=len(dw):raise RuntimeError('Cobros OLTP/DW no forman una biyección')
    return rows,fingerprints
