"""Regresión logística binaria NumPy, gradientes analíticos y lote completo."""
import json
from numbers import Real
from pathlib import Path

import numpy as np

from src.evaluation.metricas import clave_seleccion, metricas_binarias, validar_etiquetas
from src.models.perceptron import PerceptronSimple


def sigmoid(z):
    raw = np.asarray(z)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("Logits reales finitos requeridos")
    values = raw.astype(np.float64, copy=False)
    result = np.empty(values.shape, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1/(1+np.exp(-values[positive]))
    exp_negative = np.exp(values[~positive])
    result[~positive] = exp_negative/(1+exp_negative)
    return result


def binary_cross_entropy_logits(y, z):
    values = np.asarray(z)
    if values.ndim != 1 or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("Logits deben ser un vector real finito")
    labels = validar_etiquetas(y, n=len(values))
    # logaddexp evita exp(z) y log(0); dividir antes de sumar evita overflow de la suma.
    values = values.astype(np.float64, copy=False)
    loss = float(np.sum((np.logaddexp(0, values)-labels*values)/len(values)))
    if not np.isfinite(loss):
        raise ValueError("Entropía cruzada no finita")
    return loss


def loss_and_gradients(X, y, weights, bias):
    # Solo se comparten validadores con el perceptrón; no su aprendizaje/predicción.
    x = PerceptronSimple._matrix(X)
    labels = validar_etiquetas(y, n=len(x))
    w = np.asarray(weights)
    if w.shape != (x.shape[1],) or w.dtype.kind not in "iuf" or not np.isfinite(w).all():
        raise ValueError("Pesos incompatibles o no finitos")
    if isinstance(bias, bool) or not isinstance(bias, Real) or not np.isfinite(bias):
        raise ValueError("Sesgo real escalar finito requerido")
    with np.errstate(over="ignore", invalid="ignore"):
        z = x @ w + bias
    if not np.isfinite(z).all():
        raise ValueError("Combinación lineal no finita")
    residual = sigmoid(z)-labels
    # X.T @ ((p-y)/m) equivale a (X.T @ (p-y))/m, con menor riesgo de overflow.
    gw = x.T @ (residual/len(x))
    gb = float(np.mean(residual))
    if not np.isfinite(gw).all() or not np.isfinite(gb):
        raise ValueError("Gradientes no finitos")
    return binary_cross_entropy_logits(labels, z), gw, gb


def validar_umbral(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Umbral real finito entre 0 y 1 requerido")
    return float(value)


def gradient_check(X, y, weights, bias, *, h=1e-6, atol=1e-8, rtol=1e-5):
    """Diferencias centrales por parámetro; h_j=h*max(1,abs(parametro))."""
    if not np.isfinite(h) or h <= 0 or atol < 0 or rtol < 0:
        raise ValueError("Paso/tolerancias inválidas")
    w = np.asarray(weights, dtype=np.float64)
    _, gw, gb = loss_and_gradients(X, y, w, bias)
    analytical = np.append(gw, gb)
    params = np.append(w, bias)
    numerical = np.empty_like(params)
    for j, value in enumerate(params):
        step = h*max(1, abs(value))
        plus, minus = params.copy(), params.copy()
        plus[j] += step
        minus[j] -= step
        lp = loss_and_gradients(X, y, plus[:-1], float(plus[-1]))[0]
        lm = loss_and_gradients(X, y, minus[:-1], float(minus[-1]))[0]
        numerical[j] = (lp-lm)/(2*step)
    absolute = np.abs(analytical-numerical)
    relative = absolute/np.maximum(1e-12, np.maximum(np.abs(analytical), np.abs(numerical)))
    return {"passed":bool(np.allclose(analytical, numerical, atol=atol, rtol=rtol)),
            "h":h, "step_rule":"h*max(1,abs(parameter))", "atol":atol, "rtol":rtol,
            "analytic_weights":gw.tolist(), "analytic_bias":gb,
            "numerical_weights":numerical[:-1].tolist(), "numerical_bias":float(numerical[-1]),
            "max_absolute_error":float(absolute.max()), "max_relative_error":float(relative.max())}


def verificar_gradientes():
    cases = {
        "moderado":gradient_check([[1,-2,.5],[-.3,.2,1],[2,1,-1],[-1,.5,2]], [0,1,1,0], [.2,-.4,.1], -.1),
        "logits_extremos":gradient_check([[1000,-2000],[-1500,500],[2000,1000],[-900,-600]], [0,1,0,0], [.8,-.4], .3,
                                         h=1e-5, atol=2e-6, rtol=1e-5)}
    if not all(c["passed"] for c in cases.values()):
        raise AssertionError("Los gradientes no superan diferencias finitas; WWI bloqueado")
    return cases


class LogisticRegressionManual:
    _matrix = staticmethod(PerceptronSimple._matrix)
    _names = staticmethod(PerceptronSimple._names)
    _predict_matrix = PerceptronSimple._predict_matrix

    def __init__(self, learning_rate=0.05, max_epochs=200, seed=42, initialization="zeros",
                 initial_weights=None, initial_bias=0., tolerance=None, threshold=0.5):
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real) or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate positivo finito requerido")
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, (int, np.integer)) or max_epochs < 1:
            raise ValueError("max_epochs debe ser entero positivo")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Semilla entera no negativa requerida")
        if initialization not in {"zeros", "normal"}:
            raise ValueError("Inicialización zeros o normal requerida")
        if isinstance(initial_bias, bool) or not isinstance(initial_bias, Real) or not np.isfinite(initial_bias):
            raise ValueError("Sesgo inicial real finito requerido")
        if tolerance is not None and (isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("Tolerancia no negativa o None requerida")
        w = None
        if initial_weights is not None:
            w = np.asarray(initial_weights)
            if w.ndim != 1 or not w.size or w.dtype.kind not in "iuf" or not np.isfinite(w).all():
                raise ValueError("Pesos iniciales inválidos")
            w = w.astype(np.float64, copy=True)
        self.learning_rate, self.max_epochs, self.seed = float(learning_rate), int(max_epochs), int(seed)
        self.initialization, self.initial_weights, self.initial_bias = initialization, w, float(initial_bias)
        self.tolerance = None if tolerance is None else float(tolerance)
        self.threshold = validar_umbral(threshold)
        self.weights_, self.bias_, self.feature_names_ = None, None, None
        self.history_ = []

    def decision_function(self, X, *, feature_names=None):
        x = self._predict_matrix(X, feature_names)
        with np.errstate(over="ignore", invalid="ignore"):
            z = x @ self.weights_ + self.bias_
        if not np.isfinite(z).all():
            raise ValueError("Combinación lineal no finita")
        return z

    def predict_proba(self, X, *, feature_names=None):
        """Vector (m,) de P(y=1|X); no implica calibración perfecta."""
        return sigmoid(self.decision_function(X, feature_names=feature_names))

    def predict(self, X, threshold=None, *, feature_names=None):
        cutoff = self.threshold if threshold is None else validar_umbral(threshold)
        return (self.predict_proba(X, feature_names=feature_names) >= cutoff).astype(np.int8)

    def fit(self, X, y, X_val=None, y_val=None, *, feature_names=None):
        x = self._matrix(X)
        labels = validar_etiquetas(y, n=len(x))
        d = x.shape[1]
        if (X_val is None) != (y_val is None):
            raise ValueError("Validación requiere X_val e y_val juntos")
        val = None
        if X_val is not None:
            val = self._matrix(X_val)
            y_val = validar_etiquetas(y_val, n=len(val))
            if val.shape[1] != d or len(np.unique(y_val)) != 2:
                raise ValueError("Validación requiere misma dimensión y ambas clases")
        if self.initial_weights is not None and len(self.initial_weights) != d:
            raise ValueError("Pesos iniciales incompatibles")
        rng = np.random.default_rng(self.seed)
        self.weights_ = (self.initial_weights.copy() if self.initial_weights is not None else
                         np.zeros(d, dtype=np.float64) if self.initialization == "zeros" else rng.normal(0, .01, d))
        self.bias_ = self.initial_bias
        self.feature_names_ = self._names(feature_names, d)
        self.history_, self.history_threshold_, self.converged_ = [], self.threshold, False
        initial, gw, gb = loss_and_gradients(x, labels, self.weights_, self.bias_)
        self.initial_train_loss_ = initial
        self.initial_val_loss_ = binary_cross_entropy_logits(y_val, self.decision_function(val)) if val is not None else None
        best_key, best_w, best_b = None, None, None
        for epoch in range(1, self.max_epochs+1):
            used_norm = float(np.linalg.norm(np.append(gw, gb)))
            self.weights_ -= self.learning_rate*gw
            self.bias_ -= self.learning_rate*gb
            loss, gw, gb = loss_and_gradients(x, labels, self.weights_, self.bias_)
            gradient_norm = float(np.linalg.norm(np.append(gw, gb)))
            tm = metricas_binarias(labels, self.predict(x))
            vm = metricas_binarias(y_val, self.predict(val)) if val is not None else None
            val_loss = binary_cross_entropy_logits(y_val, self.decision_function(val)) if val is not None else None
            self.history_.append({"epoch":epoch, "train_loss":loss, "val_loss":val_loss,
                                  "train_accuracy":tm["accuracy"], "val_accuracy":vm["accuracy"] if vm else None,
                                  "val_balanced_accuracy":vm["balanced_accuracy"] if vm else None,
                                  "val_f1":vm["f1"] if vm else None, "gradient_norm":gradient_norm,
                                  "gradient_used_norm":used_norm, "weights_norm":float(np.linalg.norm(self.weights_)),
                                  "threshold":self.history_threshold_})
            key = clave_seleccion(vm, epoch) if vm else (epoch,)
            if best_key is None or key > best_key:
                best_key, best_w, best_b = key, self.weights_.copy(), float(self.bias_)
                self.best_epoch_ = epoch
            if self.tolerance is not None and gradient_norm <= self.tolerance:
                self.converged_ = True
                break
        self.last_weights_, self.last_bias_ = self.weights_.copy(), float(self.bias_)
        self.weights_, self.bias_ = best_w, best_b
        self.selection_ = "validation_BA_F1_earliest_epoch" if val is not None else "last_epoch_without_validation"
        return self

    def get_params(self):
        return {"learning_rate":self.learning_rate, "max_epochs":self.max_epochs, "seed":self.seed,
                "initialization":self.initialization, "initial_weights":self.initial_weights.tolist() if self.initial_weights is not None else None,
                "initial_bias":self.initial_bias, "tolerance":self.tolerance, "threshold":self.threshold}

    def save(self, path):
        if self.weights_ is None:
            raise ValueError("Modelo sin ajustar")
        value = {"format_version":1, "algorithm":"manual_numpy_binary_logistic_regression",
                 "params":self.get_params(), "label_values":[0,1], "n_features":len(self.weights_),
                 "feature_names":self.feature_names_, "weights":self.weights_.tolist(), "bias":self.bias_,
                 "selected_epoch":self.best_epoch_, "selection":self.selection_, "history_threshold":self.history_threshold_}
        Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if p.get("format_version") != 1 or p.get("algorithm") != "manual_numpy_binary_logistic_regression" or p.get("label_values") != [0,1]:
            raise ValueError("Formato de modelo inválido")
        obj = cls(**p["params"])
        w = np.asarray(p["weights"])
        if w.shape != (p["n_features"],) or w.size == 0 or w.dtype.kind not in "iuf" or not np.isfinite(w).all() or not isinstance(p["bias"], Real) or not np.isfinite(p["bias"]):
            raise ValueError("Pesos/sesgo inválidos")
        if isinstance(p["selected_epoch"], bool) or not isinstance(p["selected_epoch"], int) or not 1 <= p["selected_epoch"] <= obj.max_epochs:
            raise ValueError("Época inválida")
        obj.weights_, obj.bias_ = w.astype(np.float64), float(p["bias"])
        obj.feature_names_ = obj._names(p["feature_names"], len(w))
        obj.best_epoch_, obj.selection_, obj.history_threshold_ = p["selected_epoch"], p["selection"], validar_umbral(p["history_threshold"])
        return obj
