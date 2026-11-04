"""Semántica y viabilidad financiera local: eventos separados de estados."""
import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from decimal import Decimal
from statistics import median

from src.db.conexion import Configuracion, Lector, validar_select
from src.db.auditoria_dw import BASE, guardar, sha

OUT=BASE/'resultados/extension_datamarts/transacciones'
CUT=dt.date(2016,1,1)
NEW={'src/datamarts/transacciones_viabilidad.py','src/datamarts/reportes_transacciones.py','tests/test_transacciones_viabilidad.py'}


def load(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def snapshot():
    return {str(p.relative_to(BASE)):sha(p) for root in ['datos','referencias','resultados','src','tests','informe']
            for p in (BASE/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and OUT not in p.parents and str(p.relative_to(BASE)).replace('\\','/') not in NEW}


def reader(database):return Lector(Configuracion('lpc:.\\AMBATO',database).conectar())


def scoped(r,sql):
    validar_select(sql)
    allowed=['WHERE ct.TransactionDate < ?','WHERE st.TransactionDate < ?','WHERE f.[Date Key] < ?',
             'WHERE d.[Valid From] < ?']
    if sql.count('?')!=1 or sum(sql.count(m) for m in allowed)!=1 or any(x in sql.upper().split() for x in ['UNION','EXCEPT','INTERSECT']):
        raise ValueError('Consulta sin frontera fija de desarrollo')
    return r.select(sql,(CUT,))


def discover():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'integridad_antes.json').exists():guardar(OUT/'integridad_antes.json',snapshot())
    r=reader('WideWorldImporters')
    try:
        guardar(OUT/'metadatos_oltp.json',r.select("SELECT s.name AS schema_name,t.name AS table_name,c.name AS column_name,ty.name AS data_type,c.is_nullable,t.temporal_type_desc,t.is_tracked_by_cdc,(SELECT COUNT_BIG(*) FROM sys.change_tracking_tables ct WHERE ct.object_id=t.object_id) AS change_tracking,(SELECT COUNT_BIG(*) FROM sys.triggers tr WHERE tr.parent_id=t.object_id) AS triggers FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.columns c ON c.object_id=t.object_id JOIN sys.types ty ON ty.user_type_id=c.user_type_id WHERE (s.name='Sales' AND t.name='CustomerTransactions') OR (s.name='Purchasing' AND t.name='SupplierTransactions') ORDER BY s.name,t.name,c.column_id"))
        guardar(OUT/'tipos_operacionales.json',r.select('SELECT TransactionTypeID,TransactionTypeName FROM Application.TransactionTypes ORDER BY TransactionTypeID'))
        guardar(OUT/'medios_operacionales.json',r.select('SELECT PaymentMethodID,PaymentMethodName FROM Application.PaymentMethods ORDER BY PaymentMethodID'))
        guardar(OUT/'modulos_financieros.json',r.select("SELECT s.name AS schema_name,o.name AS module_name,m.definition FROM sys.sql_modules m JOIN sys.objects o ON o.object_id=m.object_id JOIN sys.schemas s ON s.schema_id=o.schema_id WHERE m.definition LIKE '%Sales.CustomerTransactions%' OR m.definition LIKE '%Purchasing.SupplierTransactions%'"))
        guardar(OUT/'indices_origen.json',r.select("SELECT s.name AS schema_name,t.name AS table_name,i.name AS index_name,i.is_unique,c.name AS column_name,ic.key_ordinal FROM sys.tables t JOIN sys.schemas s ON s.schema_id=t.schema_id JOIN sys.indexes i ON i.object_id=t.object_id JOIN sys.index_columns ic ON ic.object_id=i.object_id AND ic.index_id=i.index_id JOIN sys.columns c ON c.object_id=ic.object_id AND c.column_id=ic.column_id WHERE (s.name='Sales' AND t.name='CustomerTransactions') OR (s.name='Purchasing' AND t.name='SupplierTransactions') ORDER BY s.name,t.name,i.name,ic.key_ordinal"))
    finally:guardar(OUT/'consultas_metadatos.json',r.consultas);r.close()


def extract():
    r=reader('WideWorldImporters')
    try:
        for side,alias,table,pk,entity,doc in [('customer','ct','Sales.CustomerTransactions','CustomerTransactionID','CustomerID','InvoiceID'),('supplier','st','Purchasing.SupplierTransactions','SupplierTransactionID','SupplierID','PurchaseOrderID')]:
            fp=f"SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM({alias}.{pk},{alias}.{entity},{alias}.TransactionTypeID,{alias}.TransactionDate,{alias}.TransactionAmount,{alias}.LastEditedWhen)) AS checksum FROM {table} {alias} WHERE {alias}.TransactionDate < ?"
            guardar(OUT/f'huella_{side}_antes.json',scoped(r,fp))
            sql=f"SELECT {alias}.{pk} AS event_id,{alias}.{entity} AS entity_id,{alias}.{doc} AS document_id,{alias}.TransactionTypeID AS type_id,{alias}.PaymentMethodID AS payment_id,{alias}.TransactionDate AS event_date,{alias}.AmountExcludingTax AS excluding_tax,{alias}.TaxAmount AS tax,{alias}.TransactionAmount AS amount,{alias}.OutstandingBalance AS current_balance,{alias}.FinalizationDate AS current_finalization_date,{alias}.IsFinalized AS current_finalized,{alias}.LastEditedWhen AS edited_at FROM {table} {alias} WHERE {alias}.TransactionDate < ? ORDER BY {alias}.{pk}"
            guardar(OUT/f'eventos_{side}_desarrollo.json',scoped(r,sql))
            guardar(OUT/f'huella_{side}_despues.json',scoped(r,fp))
    finally:guardar(OUT/'consultas_oltp.json',r.consultas);r.close()
    r=reader('WideWorldImportersDW')
    try:
        fp="SELECT COUNT_BIG(*) AS events,CHECKSUM_AGG(BINARY_CHECKSUM(f.[Transaction Key],f.[Date Key],f.[WWI Customer Transaction ID],f.[WWI Supplier Transaction ID],f.[Total Including Tax],f.[Outstanding Balance],f.[Is Finalized])) AS checksum FROM Fact.[Transaction] f WHERE f.[Date Key] < ?"
        guardar(OUT/'huella_dw_antes.json',scoped(r,fp))
        guardar(OUT/'hechos_desarrollo.json',scoped(r,"SELECT f.[Transaction Key] AS fact_id,f.[Date Key] AS event_date,f.[WWI Customer Transaction ID] AS customer_event_id,f.[WWI Supplier Transaction ID] AS supplier_event_id,f.[WWI Invoice ID] AS invoice_id,f.[WWI Purchase Order ID] AS purchase_order_id,f.[Customer Key] AS customer_key,f.[Bill To Customer Key] AS bill_to_key,f.[Supplier Key] AS supplier_key,c.[WWI Customer ID] AS customer_id,b.[WWI Customer ID] AS bill_to_id,s.[WWI Supplier ID] AS supplier_id,t.[WWI Transaction Type ID] AS type_id,t.[Transaction Type] AS type_name,p.[WWI Payment Method ID] AS payment_id,f.[Total Excluding Tax] AS excluding_tax,f.[Tax Amount] AS tax,f.[Total Including Tax] AS amount,f.[Outstanding Balance] AS current_balance,f.[Is Finalized] AS current_finalized FROM Fact.[Transaction] f LEFT JOIN Dimension.Customer c ON c.[Customer Key]=f.[Customer Key] LEFT JOIN Dimension.Customer b ON b.[Customer Key]=f.[Bill To Customer Key] LEFT JOIN Dimension.Supplier s ON s.[Supplier Key]=f.[Supplier Key] JOIN Dimension.[Transaction Type] t ON t.[Transaction Type Key]=f.[Transaction Type Key] LEFT JOIN Dimension.[Payment Method] p ON p.[Payment Method Key]=f.[Payment Method Key] WHERE f.[Date Key] < ? ORDER BY f.[Transaction Key]"))
        for table,business,extra in [('Transaction Type','WWI Transaction Type ID','[Transaction Type]'),('Payment Method','WWI Payment Method ID','[Payment Method]'),('Customer','WWI Customer ID','CAST(NULL AS nvarchar(100))'),('Supplier','WWI Supplier ID','CAST(NULL AS nvarchar(100))')]:
            guardar(OUT/('dimension_'+table.lower().replace(' ','_')+'.json'),scoped(r,f"SELECT d.[{table} Key] AS surrogate,d.[{business}] AS business_id,{extra} AS label,d.[Valid From] AS valid_from,d.[Valid To] AS valid_to FROM Dimension.[{table}] d WHERE d.[Valid From] < ? ORDER BY surrogate"))
        guardar(OUT/'huella_dw_despues.json',scoped(r,fp))
    finally:guardar(OUT/'consultas_dw.json',r.consultas);r.close()


TYPES={1:('customer','invoice'),2:('customer','credit_note'),3:('customer','payment_received'),
       4:('customer','refund'),5:('supplier','invoice'),6:('supplier','credit_note'),
       7:('supplier','payment_issued'),8:('supplier','refund'),13:('customer','contra')}


def classify(side,type_id):
    domain,kind=TYPES.get(type_id,(None,'unclassified'))
    return kind if domain==side else 'unclassified'


def add_month(month,delta):
    y,m=map(int,month.split('-'));n=y*12+m-1+delta
    return f'{n//12:04d}-{n%12+1:02d}'


def start(month):return month+'-01 00:00:00'


def aggregate_payments(rows,side):
    """Cuenta IDs operacionales únicos de un tipo de pago, nunca saldos o facturas."""
    expected=3 if side=='customer' else 7 if side=='supplier' else None
    if expected is None:raise ValueError('Dominio financiero no válido')
    groups=defaultdict(lambda:{'n':0,'known_at':''})
    seen=set();first={}
    for r in rows:
        if r['type_id']!=expected:continue
        if r['event_id'] is None or r['entity_id'] in (None,0) or r['event_date'] is None or not r['edited_at']:
            raise ValueError('Identidad, fecha o disponibilidad indeterminada; no rellenar cero')
        if r['event_date']>='2016-01-01':raise ValueError('Outcome de reserva fuera del análisis')
        if r['event_id'] in seen:raise ValueError('Evento duplicado')
        seen.add(r['event_id']);e=r['entity_id'];m=r['event_date'][:7]
        first[e]=min(first.get(e,m),m)
        g=groups[e,m];g['n']+=1
        g['known_at']=max(g['known_at'],r['edited_at'])
    return dict(groups),first


def summary(rows):
    result={}
    for split in ['train','validation']:
        part=[r for r in rows if r['split']==split];labels=defaultdict(set)
        for r in part:labels[r['entity']].add(r['y'])
        def simple(v):
            p=sum(x['y'] for x in v)
            return {'eligible':len(v),'positive':p,'negative':len(v)-p,'positive_proportion':p/len(v) if v else None,
                    'target_zero':sum(x['n']==0 for x in v),'reference_zero':sum(x['reference']==0 for x in v)}
        result[split]=dict(simple(part),entities=len(labels),entities_switching_class=sum(len(v)>1 for v in labels.values()),
                           entities_single_class=sum(len(v)==1 for v in labels.values()),
                           monthly=[dict(month=m,**simple([r for r in part if r['month']==m])) for m in sorted({r['month'] for r in part})],
                           yearly=[dict(year=y,**simple([r for r in part if r['month'][:4]==y])) for y in sorted({r['month'][:4] for r in part})])
    return result


def candidate(groups,first,coverage='unknown',require_known_version=True):
    """Etiqueta de frecuencia; ceros solo con cobertura de libro empresarial explícita.

    No exige snapshots ni trata edición como alta. La versión sirve aquí como
    control adicional observable de eventos de pago registrados el mismo día.
    """
    if coverage not in {'unknown','business_ledger_academic'}:raise ValueError('Cobertura inválida')
    rows=[];reasons=Counter();entities=[]
    for e,first_month in sorted(first.items()):
        local=[];m=add_month(first_month,3)
        while m<'2016-01':
            prev=[add_month(m,-k) for k in [1,2,3]]
            if coverage=='unknown' and any((e,x) not in groups for x in [m]+prev):
                reasons['unobserved_month_unknown_coverage']+=1;m=add_month(m,1);continue
            cut='2015-01-01 00:00:00' if m<'2015-01' else '2016-01-01 00:00:00'
            if require_known_version and any(groups.get((e,x),{}).get('known_at','')>=start(m) for x in prev):
                reasons['historical_event_version_not_prior']+=1;m=add_month(m,1);continue
            if groups.get((e,m),{}).get('known_at','')>=cut:
                reasons['target_event_version_not_before_cut']+=1;m=add_month(m,1);continue
            n=groups.get((e,m),{}).get('n',0);ref=median(groups.get((e,x),{}).get('n',0) for x in prev)
            r={'entity':e,'month':m,'split':'train' if m<'2015-01' else 'validation','n':n,'reference':ref,'y':int(n>ref)}
            rows.append(r);local.append(r);m=add_month(m,1)
        entities.append({'entity':e,'first_payment_month':first_month,'train':summary(local)['train'],'validation':summary(local)['validation']})
    # Solo resúmenes agregados, no matrices de características/etiquetas.
    return {'coverage':coverage,'require_known_version':require_known_version,'summary':summary(rows),
            'excluded_reasons':dict(reasons),'by_entity':entities}


def analyze():
    data={s:load(f'eventos_{s}_desarrollo.json') for s in ['customer','supplier']};facts=load('hechos_desarrollo.json')
    indexes={s:{r['event_id']:r for r in rows} for s,rows in data.items()}
    if any(len(indexes[s])!=len(data[s]) for s in data):raise ValueError('ID operacional repetido')
    seen=set();mismatch=Counter();dimension_unknown=Counter();identity_mismatch=Counter();profiles=[]
    for f in facts:
        customer=f['customer_event_id'] is not None;supplier=f['supplier_event_id'] is not None
        if customer==supplier:raise ValueError('Origen financiero ambiguo')
        side='customer' if customer else 'supplier';eid=f[side+'_event_id'];key=side,eid
        if key in seen:raise ValueError('Evento multiplicado en DW')
        seen.add(key);r=indexes[side].get(eid)
        if r is None:mismatch['missing_source']+=1;continue
        for k in ['event_date','type_id','amount','excluding_tax','tax','current_balance','current_finalized']:
            a,b=f[k],r[k]
            if k in ['amount','excluding_tax','tax','current_balance']:a,b=Decimal(str(a)),Decimal(str(b))
            if a!=b:mismatch[k]+=1
        if f['invoice_id' if customer else 'purchase_order_id']!=r['document_id']:mismatch['document_id']+=1
        entitykey='bill_to_id' if customer else 'supplier_id'
        if f[entitykey] in (None,0):dimension_unknown[side]+=1
        elif f[entitykey]!=r['entity_id']:identity_mismatch[side]+=1
        expected=r['payment_id'] or 0
        if f['payment_id'] in (None,0) and expected:dimension_unknown['payment_method']+=1
        elif f['payment_id'] not in (None,0) and f['payment_id']!=expected:identity_mismatch['payment_method']+=1
    for side,rows in data.items():
        for typ in sorted({r['type_id'] for r in rows}):
            v=[r for r in rows if r['type_id']==typ]
            profiles.append({'side':side,'type_id':typ,'kind':classify(side,typ),'events':len(v),'entities':len({r['entity_id'] for r in v}),
               'positive_amount':sum(Decimal(str(r['amount']))>0 for r in v),'negative_amount':sum(Decimal(str(r['amount']))<0 for r in v),'zero_amount':sum(Decimal(str(r['amount']))==0 for r in v),
               'null_document':sum(r['document_id'] is None for r in v),'distinct_nonnull_documents':len({r['document_id'] for r in v if r['document_id'] is not None}),
               'same_day_last_edit':sum(r['edited_at'][:10]==r['event_date'] for r in v),'edit_before_event':sum(r['edited_at'][:10]<r['event_date'] for r in v),
               'finalization_before_event':sum(r['current_finalization_date'] is not None and r['current_finalization_date']<r['event_date'] for r in v),
               'null_payment_method':sum(r['payment_id'] is None for r in v),'payment_methods':dict(Counter(str(r['payment_id']) for r in v)),
               'month_counts':dict(sorted(Counter(r['event_date'][:7] for r in v).items()))})
    dimensional={}
    for name in ['customer','supplier','transaction_type','payment_method']:
        versions=load('dimension_'+name+'.json');by_id=defaultdict(list)
        for r in versions:by_id[r['business_id']].append(r)
        overlaps=[]
        for bid,vs in by_id.items():
            for i,a in enumerate(vs):
                for b in vs[i+1:]:
                    if a['valid_from']<b['valid_to'] and b['valid_from']<a['valid_to']:overlaps.append({'business_id':bid,'surrogates':[a['surrogate'],b['surrogate']]})
        dimensional[name]={'business_ids':len(by_id),'versions':len(versions),'max_versions':max(map(len,by_id.values())),'overlaps':overlaps}
    outputs={};grains={};payment_checks={}
    for side,rows in data.items():
        groups,first=aggregate_payments(rows,side)
        pay=[r for r in rows if classify(side,r['type_id']) in ['payment_received','payment_issued']]
        active=Counter(e for e,m in groups)
        slots=sum((2016*12)-(int(m[:4])*12+int(m[5:])-1) for m in first.values())
        grains[side]={'identities':len(first),'observed_payment_months':len(groups),'slots_after_first_payment':slots,'absences_after_first_payment':slots-len(groups),
                      'months_with_activity_histogram':dict(sorted(Counter(active.values()).items())),
                      'first_payment_month_histogram':dict(sorted(Counter(first.values()).items()))}
        payment_checks[side]={'events':len(pay),'all_edited_on_event_day':all(r['event_date']==r['edited_at'][:10] for r in pay),
            'all_negative_amount':all(Decimal(str(r['amount']))<0 for r in pay),'all_null_document':all(r['document_id'] is None for r in pay),
            'all_finalized_same_day':all(r['current_finalization_date']==r['event_date'] for r in pay),
            'duplicate_entity_date_groups':sum(n>1 for n in Counter((r['entity_id'],r['event_date']) for r in pay).values()),
            'event_dates_min':min(r['event_date'] for r in pay),'event_dates_max':max(r['event_date'] for r in pay)}
        for cov in ['business_ledger_academic','unknown']:
            outputs[side+'_'+cov]=candidate(groups,first,cov)
        outputs[side+'_event_semantics_without_edit_gate']=candidate(groups,first,'business_ledger_academic',False)
    diagnostics={'source_counts':{s:len(r) for s,r in data.items()},'dw_count':len(facts),'side_scoped_identity_bijection':len(seen)==sum(len(r) for r in data.values()) and not mismatch,
                 'mapping_mismatches':dict(mismatch),'dimension_unknown_rows':dict(dimension_unknown),'dimension_known_identity_mismatches':dict(identity_mismatch),
                 'source_id_numeric_overlap_between_sides':len(set(indexes['customer'])&set(indexes['supplier'])),
                 'payment_checks':payment_checks,'dimensions':dimensional,
                 'invoice_amount_identity':all(Decimal(str(r['amount']))==Decimal(str(r['excluding_tax']))+Decimal(str(r['tax'])) for rows in data.values() for r in rows if r['type_id'] in [1,5]),
                 'all_dates_before_2016':all(r['event_date']<'2016-01-01' for rows in data.values() for r in rows)}
    guardar(OUT/'comprobaciones_focalizadas.json',diagnostics);guardar(OUT/'perfiles_tipos.json',profiles)
    guardar(OUT/'granularidades.json',grains);guardar(OUT/'etiquetas_candidatas.json',outputs)
    print(json.dumps({'diagnostics':diagnostics,'grains':grains,'candidates':{n:{s:{k:v for k,v in a.items() if k not in ['monthly','yearly']} for s,a in o['summary'].items()} for n,o in outputs.items()}},ensure_ascii=True))
    return diagnostics,grains,outputs


def manifest():
    prior=load('integridad_antes.json');after={n:sha(BASE/n) for n in prior}
    if prior!=after:raise RuntimeError('Se modificó evidencia anterior')
    guardar(OUT/'integridad_despues.json',after)
    guardar(OUT/'manifiesto.json',{'scope':'Viabilidad financiera de desarrollo, eventos distintos de estados; sin matrices ni entrenamiento',
      'implementation_sha256':{n:sha(BASE/n) for n in sorted(NEW) if (BASE/n).exists()},
      'artifacts_sha256':{str(p.relative_to(BASE)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['manifiesto.json','validaciones.json']}})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['discover','extract','analyze','manifest']);a=ap.parse_args()
    {'discover':discover,'extract':extract,'analyze':analyze,'manifest':manifest}[a.phase]()
