"""Adaptador numérico financiero: catálogo propio, mediana y z-score train-only.

Conserva las reglas de PreprocesadorComun de Ventas sin importar su FEATURES,
modificar su clase congelada ni parchear variables globales.
"""
import copy
import json

import numpy as np

from .caracteristicas import FEATURES,OPTIONAL


class PreprocesadorFinanciero:
    def __init__(self):self.state=None

    @staticmethod
    def validate(frame):
        if list(frame.columns)!=FEATURES:raise ValueError('Catálogo/orden financiero incompatible')
        a=frame.to_numpy(dtype=np.float64,copy=True)
        if np.isinf(a).any():raise ValueError('Características infinitas')
        for j,name in enumerate(FEATURES):
            if name not in OPTIONAL and np.isnan(a[:,j]).any():raise ValueError('Historia esencial faltante; no imputable')
        return a

    def fit(self,frame,*,partition):
        if partition!='train' or self.state is not None:raise ValueError('Fit único y exclusivo de train')
        a=self.validate(frame)
        if not len(a):raise ValueError('Train vacío')
        kept=[];discarded={};params={}
        for j,name in enumerate(FEATURES):
            col=a[:,j]
            if np.isnan(col).all():discarded[name]='all_null_train';continue
            med=float(np.nanmedian(col));filled=np.where(np.isnan(col),med,col)
            mean=float(filled.mean());std=float(filled.std(ddof=0))
            if not np.isfinite([med,mean,std]).all():raise ValueError('Estadísticos inválidos')
            if std==0:discarded[name]='constant_train_after_imputation';continue
            params[name]={'median':med,'mean':mean,'std_population':std,'impute':'median_train' if name in OPTIONAL else 'not_needed_required_finite'};kept.append(name)
        if not kept:raise ValueError('Sin características no constantes')
        self.state={'version':1,'fit_partition':'train','train_rows':len(a),'input_features':FEATURES.copy(),
                    'optional':sorted(OPTIONAL),'feature_names':kept,'discarded':discarded,'parameters':params,'dtype':'float64',
                    'ddof':0,'formula':'(x_filled_with_train_median - train_mean) / train_population_std',
                    'redundancy_policy':'keep_all_nonconstant_authorized_features; no selection using validation'}
        return self

    def transform(self,frame):
        if self.state is None:raise ValueError('Preprocesador no ajustado')
        a=self.validate(frame);columns=[]
        for name in self.state['feature_names']:
            p=self.state['parameters'][name];col=a[:,FEATURES.index(name)]
            columns.append((np.where(np.isnan(col),p['median'],col)-p['mean'])/p['std_population'])
        result=np.column_stack(columns).astype(np.float64)
        if not np.isfinite(result).all():raise ValueError('Matriz transformada no finita')
        return result

    def to_dict(self):
        if self.state is None:raise ValueError('Sin estado')
        return copy.deepcopy(self.state)

    @classmethod
    def from_dict(cls,state):
        if state.get('version')!=1 or state.get('input_features')!=FEATURES or state.get('fit_partition')!='train':raise ValueError('Estado incompatible')
        names=state.get('feature_names',[])
        if not names or names!=[n for n in FEATURES if n in names] or len(set(names))!=len(names):raise ValueError('Orden de salida inválido')
        if set(names)&set(state['discarded']) or set(names)|set(state['discarded'])!=set(FEATURES):raise ValueError('Catálogo descartado inconsistente')
        for name in names:
            p=state['parameters'][name]
            if not np.isfinite([p['median'],p['mean'],p['std_population']]).all() or p['std_population']<=0:raise ValueError('Escala inválida')
        obj=cls();obj.state=copy.deepcopy(state);return obj
