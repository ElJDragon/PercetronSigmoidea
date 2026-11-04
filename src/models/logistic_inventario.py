"""Variante ponderada aislada y autorizada para Inventario; clases previas intactas."""
import json
from pathlib import Path
import numpy as np
from src.models.logistic_regression import LogisticRegressionManual,sigmoid,binary_cross_entropy_logits
from src.evaluation.metricas import validar_etiquetas,metricas_binarias,clave_seleccion

def balanced_weights(y):
    y=validar_etiquetas(y);n0=int((y==0).sum());n1=int(y.sum())
    if not n0 or not n1:raise ValueError('Se requieren ambas clases en train')
    return {0:len(y)/(2*n0),1:len(y)/(2*n1)}

def weighted_loss_gradients(X,y,w,b,class_weights):
    x=LogisticRegressionManual._matrix(X);y=validar_etiquetas(y,n=len(x));w=np.asarray(w,dtype=np.float64)
    if w.shape!=(x.shape[1],) or not np.isfinite(w).all() or not np.isscalar(b) or not np.isfinite(b):raise ValueError('Parámetros inválidos')
    if set(class_weights)!={0,1} or any(not np.isfinite(v) or v<=0 for v in class_weights.values()):raise ValueError('Pesos de clase inválidos')
    weights=np.where(y==1,class_weights[1],class_weights[0]);weights=weights/weights.sum()
    z=x@w+b
    if not np.isfinite(z).all():raise ValueError('Logits no finitos')
    loss=float(np.sum(weights*(np.logaddexp(0,z)-y*z)));residual=weights*(sigmoid(z)-y)
    gw=x.T@residual;gb=float(residual.sum())
    if not np.isfinite(loss) or not np.isfinite(gw).all() or not np.isfinite(gb):raise ValueError('Gradientes no finitos')
    return loss,gw,gb

class LogisticRegressionInventarioPonderada(LogisticRegressionManual):
    def __init__(self,*,class_weights=None,**kwargs):
        super().__init__(**kwargs)
        self.class_weights_=None if class_weights is None else {int(k):float(v) for k,v in class_weights.items()}

    def fit(self,X,y,X_val=None,y_val=None,*,feature_names=None):
        x=self._matrix(X);y=validar_etiquetas(y,n=len(x));cw=balanced_weights(y)
        if self.class_weights_ is not None and self.class_weights_!=cw:raise ValueError('Pesos solo de train; no reutilizar otra población')
        self.class_weights_=cw
        if X_val is None or y_val is None:raise ValueError('Validación requerida para este contrato')
        val=self._matrix(X_val);yv=validar_etiquetas(y_val,n=len(val));d=x.shape[1]
        if val.shape[1]!=d or len(np.unique(yv))!=2:raise ValueError('Validación incompatible')
        if self.initial_weights is not None and len(self.initial_weights)!=d:raise ValueError('Dimensión inicial inválida')
        rng=np.random.default_rng(self.seed)
        self.weights_=(self.initial_weights.copy() if self.initial_weights is not None else np.zeros(d) if self.initialization=='zeros' else rng.normal(0,.01,d))
        self.bias_=self.initial_bias;self.feature_names_=self._names(feature_names,d)
        self.history_=[];self.history_threshold_=self.threshold;self.converged_=False
        loss,gw,gb=weighted_loss_gradients(x,y,self.weights_,self.bias_,cw)
        self.initial_train_loss_=loss;self.initial_val_loss_=binary_cross_entropy_logits(yv,self.decision_function(val))
        best=None
        for epoch in range(1,self.max_epochs+1):
            used=float(np.linalg.norm(np.append(gw,gb)))
            self.weights_-=self.learning_rate*gw;self.bias_-=self.learning_rate*gb
            loss,gw,gb=weighted_loss_gradients(x,y,self.weights_,self.bias_,cw);norm=float(np.linalg.norm(np.append(gw,gb)))
            tm=metricas_binarias(y,self.predict(x));vm=metricas_binarias(yv,self.predict(val))
            self.history_.append(dict(epoch=epoch,train_loss=loss,val_loss=binary_cross_entropy_logits(yv,self.decision_function(val)),
                train_accuracy=tm['accuracy'],val_accuracy=vm['accuracy'],val_balanced_accuracy=vm['balanced_accuracy'],val_f1=vm['f1'],
                gradient_norm=norm,gradient_used_norm=used,weights_norm=float(np.linalg.norm(self.weights_)),threshold=self.threshold,
                loss_definition='weighted_train_BCE_and_unweighted_validation_BCE'))
            key=clave_seleccion(vm,epoch)
            if best is None or key>best:
                best=key;bw=self.weights_.copy();bb=float(self.bias_);self.best_epoch_=epoch
            if self.tolerance is not None and norm<=self.tolerance:self.converged_=True;break
        self.last_weights_=self.weights_.copy();self.last_bias_=float(self.bias_);self.weights_=bw;self.bias_=bb
        self.selection_='validation_unweighted_BA_F1_earliest_epoch'
        return self

    def get_params(self):return dict(super().get_params(),class_weights=self.class_weights_)

    def save(self,path):
        if self.weights_ is None:raise ValueError('Sin ajuste')
        Path(path).write_text(json.dumps(dict(format_version=1,algorithm='inventory_manual_weighted_binary_logistic',
            params=self.get_params(),n_features=len(self.weights_),feature_names=self.feature_names_,weights=self.weights_.tolist(),
            bias=self.bias_,selected_epoch=self.best_epoch_,selection=self.selection_,history_threshold=self.threshold,
            scores_calibrated=False,loss='sum(weight_i*BCE_logits)/sum(weight_i)'),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        p=json.loads(Path(path).read_text(encoding='utf-8'))
        if p.get('algorithm')!='inventory_manual_weighted_binary_logistic' or p.get('format_version')!=1:raise ValueError('Formato ponderado incompatible')
        obj=cls(**p['params']);w=np.asarray(p['weights'],dtype=np.float64)
        if w.shape!=(p['n_features'],) or not len(w) or not np.isfinite(w).all() or not np.isfinite(p['bias']):raise ValueError('Pesos inválidos')
        if not obj.class_weights_ or set(obj.class_weights_)!={0,1} or any(not np.isfinite(v) or v<=0 for v in obj.class_weights_.values()):raise ValueError('Pesos de clase inválidos')
        if not isinstance(p['selected_epoch'],int) or not 1<=p['selected_epoch']<=obj.max_epochs:raise ValueError('Checkpoint inválido')
        obj.feature_names_=obj._names(p['feature_names'],len(w));obj.weights_=w;obj.bias_=float(p['bias']);obj.best_epoch_=p['selected_epoch']
        obj.selection_=p['selection'];obj.history_threshold_=p['history_threshold'];return obj
