"""Población y características causales; etiquetas separadas de X."""
import calendar
import math
from collections import defaultdict
from statistics import median,pstdev

import pandas as pd

from src.datamarts.transacciones_viabilidad import add_month,start

FEATURES=['cobros_lag_1','cobros_lag_2','cobros_lag_3','cobros_media_3','cobros_mediana_3',
          'cobros_std_3','cobros_cambio_1','cobros_tendencia_3','meses_con_cobros_3','mes_anterior_sin_cobros',
          'variacion_relativa_cobros','base_variacion_cero','antiguedad_observada_meses','mes_sin','mes_cos','dias_mes_objetivo']
OPTIONAL={'variacion_relativa_cobros'}
COVERAGE='business_ledger_academic'


def ordinal(m):return int(m[:4])*12+int(m[5:])-1


def feature_row(previous,month,first_month):
    n1,n2,n3=previous;y,m=map(int,month.split('-'))
    values=[n1,n2,n3,(n1+n2+n3)/3,median(previous),pstdev(previous),n1-n2,(n1-n3)/2,
            sum(n>0 for n in previous),int(n1==0),(n1-n2)/n2 if n2>0 else float('nan'),
            int(n2==0),ordinal(month)-ordinal(first_month),math.sin(2*math.pi*(m-1)/12),math.cos(2*math.pi*(m-1)/12),calendar.monthrange(y,m)[1]]
    return dict(zip(FEATURES,values))


def build(rows,*,coverage):
    if coverage!=COVERAGE:raise ValueError('Construcción bloqueada: declarar cobertura académica aprobada explícitamente')
    groups=defaultdict(lambda:{'n':0,'known_at':''});first={};seen=set()
    for r in rows:
        if r['type_id']!=3:raise ValueError('La entrada debe contener exclusivamente cobros tipo 3')
        if r['event_date']>='2016-01-01':raise ValueError('Test reservado fuera del constructor de desarrollo')
        if r['event_id'] in seen:raise ValueError('Evento duplicado')
        if r['customer_id'] in (None,0) or not r['event_date'] or not r['edited_at']:raise ValueError('Evento no identificable')
        seen.add(r['event_id']);c=r['customer_id'];month=r['event_date'][:7]
        first[c]=min(first.get(c,month),month);g=groups[c,month];g['n']+=1;g['known_at']=max(g['known_at'],r['edited_at'])
    metadata=[];features=[];labels=[];eligibility=[];monthly=[]
    for c,fm in sorted(first.items()):
        m=fm
        while m<'2016-01':
            oid=f'customer:{c}:{m}';current=groups.get((c,m),{'n':0,'known_at':''})
            monthly.append(dict(observation_id=oid,customer_id=c,month=m,n=current['n'],known_at=current['known_at'],state='observed_payments' if current['n'] else 'covered_zero'))
            hist_months=[add_month(m,-k) for k in [1,2,3]]
            reason='insufficient_history' if ordinal(m)-ordinal(fm)<3 else 'eligible'
            history=[groups.get((c,h),{'n':0,'known_at':''}) for h in hist_months]
            cutoff='2015-01-01 00:00:00' if m<'2015-01' else '2016-01-01 00:00:00'
            if reason=='eligible' and any(h['known_at']>=start(m) for h in history):reason='history_version_not_available'
            if reason=='eligible' and current['known_at']>=cutoff:reason='target_version_not_available_by_cut'
            eligibility.append(dict(observation_id=oid,customer_id=c,month=m,state=reason,coverage=coverage))
            if reason=='eligible':
                n=[h['n'] for h in history];ref=median(n)
                metadata.append(dict(observation_id=oid,customer_id=c,month=m,partition='train' if m<'2015-01' else 'val',
                    first_observed_month=fm,prediction_time=start(m),history_months='|'.join(hist_months),
                    history_known_at=max(h['known_at'] for h in history),target_known_at=current['known_at'],label_cut_exclusive=cutoff,coverage=coverage))
                features.append(dict(observation_id=oid,**feature_row(n,m,fm)))
                labels.append(dict(observation_id=oid,target_count=current['n'],historical_reference=ref,y=int(current['n']>ref)))
            m=add_month(m,1)
    # Orden único: mes objetivo, CustomerID; todos los artefactos lo comparten.
    order=sorted(range(len(metadata)),key=lambda i:(metadata[i]['month'],metadata[i]['customer_id']))
    frames=tuple(pd.DataFrame([records[i] for i in order]) for records in [metadata,features,labels])
    if not metadata:raise ValueError('No hay observaciones elegibles')
    return (*frames,pd.DataFrame(eligibility).sort_values(['month','customer_id']).reset_index(drop=True),
            pd.DataFrame(monthly).sort_values(['month','customer_id']).reset_index(drop=True))
