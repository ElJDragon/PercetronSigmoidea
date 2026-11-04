"""Adaptador del esquema train-only existente al catálogo propio de Inventario.

No modifica ni parchea los catálogos globales de Ventas/Transacciones.
Todas las entradas son esenciales, no requieren imputación.
"""
import copy
import numpy as np
from pandas.api.types import is_numeric_dtype,is_bool_dtype
from .caracteristicas import FEATURES

class PreprocesadorInventario:
    def __init__(self):self.state=None

    @staticmethod
    def validate(frame):
        if list(frame.columns)!=FEATURES or frame.columns.duplicated().any():raise ValueError('Orden/catálogo inválido')
        if any(not is_numeric_dtype(dtype) or is_bool_dtype(dtype) for dtype in frame.dtypes):raise ValueError('Solo tipos numéricos')
        a=frame.to_numpy(dtype=np.float64,copy=True)
        if not np.isfinite(a).all():raise ValueError('Historia esencial ausente o infinita; no imputar')
        return a

    def fit(self,frame,*,partition):
        if partition!='train' or self.state is not None:raise ValueError('Fit único y exclusivo de train')
        a=self.validate(frame)
        if not len(a):raise ValueError('Train vacío')
        kept=[];discarded={};params={}
        for j,name in enumerate(FEATURES):
            col=a[:,j];mean=float(col.mean());std=float(col.std(ddof=0))
            if not np.isfinite([mean,std]).all():raise ValueError('Escala no finita')
            if std==0:discarded[name]='constant_train';continue
            kept.append(name);params[name]=dict(mean=mean,std_population=std,imputation='not_needed_required_finite')
        if not kept:raise ValueError('Sin entradas no constantes')
        self.state=dict(version=1,fit_partition='train',input_features=FEATURES.copy(),feature_names=kept,discarded=discarded,
            parameters=params,train_rows=len(a),dtype='float64',ddof=0,imputation='none; all features required finite',
            formula='(x-train_mean)/train_population_std',feature_selection='only constants in train; no validation selection')
        return self

    def transform(self,frame):
        if self.state is None:raise ValueError('Sin ajuste')
        a=self.validate(frame)
        x=np.column_stack([(a[:,FEATURES.index(n)]-self.state['parameters'][n]['mean'])/self.state['parameters'][n]['std_population'] for n in self.state['feature_names']]).astype(np.float64)
        if not np.isfinite(x).all():raise ValueError('Transformación no finita')
        return x

    def to_dict(self):
        if self.state is None:raise ValueError('Sin ajuste')
        return copy.deepcopy(self.state)

    @classmethod
    def from_dict(cls,state):
        state=copy.deepcopy(state)
        if state.get('version')!=1 or state.get('input_features')!=FEATURES or state.get('fit_partition')!='train' or state.get('ddof')!=0:raise ValueError('Estado incompatible')
        names=state['feature_names'];discarded=state['discarded']
        if not names or names!=[n for n in FEATURES if n not in discarded] or set(names)&set(discarded) or set(names)|set(discarded)!=set(FEATURES):raise ValueError('Máscara incompatible')
        if set(state['parameters'])!=set(names):raise ValueError('Parámetros incompatibles')
        for n in names:
            p=state['parameters'][n]
            if not np.isfinite([p['mean'],p['std_population']]).all() or p['std_population']<=0:raise ValueError('Escala inválida')
        obj=cls();obj.state=state;return obj
