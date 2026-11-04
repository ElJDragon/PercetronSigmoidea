"""Calendario cubierto, etiqueta estricta y las 17 entradas aprobadas."""
import calendar
import datetime as dt
from collections import defaultdict
from statistics import median
import math

import numpy as np
import pandas as pd

from .extraccion import normalize
from src.datamarts.inventario_viabilidad import add_month,month_range

COVERAGE='event_ledger_academic'
FEATURES=['recepciones_lag_1','recepciones_lag_2','recepciones_lag_3',
    'unidades_recibidas_lag_1','unidades_recibidas_lag_2','unidades_recibidas_lag_3',
    'recepciones_media_3','recepciones_mediana_3','recepciones_std_3','recepciones_cambio_1',
    'recepciones_tendencia_3','meses_con_recepciones_3','mes_anterior_sin_recepciones',
    'antiguedad_observada_meses','mes_sin','mes_cos','dias_mes_objetivo']

def ordinal(month):y,m=map(int,month.split('-'));return y*12+m-1

def features(history,month,first_month):
    n=np.array([r['receipt_count'] for r in history],dtype=np.float64)
    q=[r['receipt_units'] for r in history];m=int(month[5:]);y=int(month[:4])
    values=list(n)+q+[float(n.mean()),float(np.median(n)),float(n.std(ddof=0)),
        float(n[0]-n[1]),float((n[0]-n[2])/2),int((n>0).sum()),int(n[0]==0),
        ordinal(month)-ordinal(first_month),math.sin(2*math.pi*(m-1)/12),
        math.cos(2*math.pi*(m-1)/12),calendar.monthrange(y,m)[1]]
    return dict(zip(FEATURES,values))

def build(rows,first_appearances,*,coverage=None,month_status=None):
    if coverage!=COVERAGE:raise ValueError('Los ceros requieren cobertura académica explícita')
    rows=normalize(rows);month_status=month_status or {};first={}
    for r in first_appearances:
        item=int(r['item_id']);when=dt.datetime.fromisoformat(r['first_event_at'])
        if item<=0 or item in first or not dt.datetime(2013,1,1)<=when<dt.datetime(2016,1,1):raise ValueError('Primera aparición inválida')
        first[item]=when.isoformat(sep=' ')
    events=defaultdict(list)
    for r in rows:
        if r['item_id'] not in first or r['occurred_at']<first[r['item_id']]:raise ValueError('Recepción anterior a identidad conocida')
        events[r['item_id'],r['occurred_at'][:7]].append(r)
    monthly=[];lookup={};meta=[];xs=[];labels=[];excluded=[]
    for item,when in sorted(first.items()):
        for month in month_range('2013-01','2015-12'):
            rs=events[item,month];status=month_status.get((item,month),'covered')
            if status not in {'covered','unknown','incomplete'}:raise ValueError('Estado de cobertura inválido')
            if month<when[:7]:status='before_first_appearance'
            row=dict(item_id=item,month=month,coverage_status=status,observed_receipts=len(rs),
                receipt_count=len(rs) if status=='covered' else None,
                receipt_units=sum(int(r['quantity']) for r in rs) if status=='covered' else None,
                zero_assumed_under_coverage=bool(not rs and status=='covered'),
                latest_event_at=max((r['occurred_at'] for r in rs),default=''),
                latest_edited_at=max((r['edited_at'] for r in rs),default=''))
            monthly.append(row);lookup[item,month]=row
        for month in month_range('2013-01','2015-12'):
            target=lookup[item,month];start=month+'-01 00:00:00'
            hmonths=[add_month(month,-k) for k in (1,2,3)]
            history=[lookup.get((item,m)) for m in hmonths]
            reason=None
            if target['coverage_status']!='covered':reason='target_'+target['coverage_status']
            elif any(h is None or h['coverage_status']=='before_first_appearance' for h in history):reason='insufficient_history'
            elif any(h['coverage_status']!='covered' for h in history):reason='history_unknown_or_incomplete'
            if reason:
                excluded.append(dict(item_id=item,month=month,reason=reason));continue
            if when>=start or any(h['latest_edited_at']>=start for h in history):raise ValueError('Información no disponible antes de t')
            oid=f'{item}:{month}';part='train' if month<'2015-01' else 'val'
            reference=median([h['receipt_count'] for h in history]);label=int(target['receipt_count']>reference)
            meta.append(dict(observation_id=oid,item_id=item,month=month,partition=part,prediction_at=start,
                first_event_at=when,history_month_1=hmonths[0],history_month_2=hmonths[1],history_month_3=hmonths[2],
                latest_history_event_at=max(h['latest_event_at'] for h in history),
                latest_history_edited_at=max(h['latest_edited_at'] for h in history),coverage_mode=coverage))
            xs.append(dict(observation_id=oid,**features(history,month,when[:7])))
            labels.append(dict(observation_id=oid,target_count=target['receipt_count'],historical_reference=reference,y=label,
                historical_baseline=int(history[0]['receipt_count']>reference)))
    return (pd.DataFrame(meta),pd.DataFrame(xs,columns=['observation_id']+FEATURES),pd.DataFrame(labels),
            pd.DataFrame(monthly),pd.DataFrame(excluded,columns=['item_id','month','reason']))
