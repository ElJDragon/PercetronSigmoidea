"""Una neurona multidimensional: escalón 0/1 y actualización online manual.

No optimiza entropía cruzada ni produce probabilidades. Con validación conserva
la mejor época según BA/F1; la validación nunca interviene en las actualizaciones.
"""
import json
from numbers import Real
from pathlib import Path

import numpy as np

from src.evaluation.metricas import clave_seleccion, metricas_binarias, validar_etiquetas


class PerceptronSimple:
    def __init__(self, learning_rate=0.01, max_epochs=100, seed=42, shuffle=True,
                 initial_weights=None, initial_bias=0.0):
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real) or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Tasa de aprendizaje positiva y finita requerida")
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, (int, np.integer)) or max_epochs < 1:
            raise ValueError("max_epochs debe ser entero positivo")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Semilla entera no negativa requerida")
        if not isinstance(shuffle, (bool, np.bool_)):
            raise ValueError("shuffle debe ser booleano")
        if isinstance(initial_bias, bool) or not isinstance(initial_bias, Real) or not np.isfinite(initial_bias):
            raise ValueError("Sesgo inicial real finito requerido")
        w0 = None
        if initial_weights is not None:
            w0 = np.asarray(initial_weights)
            if w0.ndim != 1 or not len(w0) or w0.dtype.kind not in "iuf" or not np.isfinite(w0).all():
                raise ValueError("Pesos iniciales: vector real finito no vacío")
            w0 = w0.astype(np.float64, copy=True)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.initial_weights = w0
        self.initial_bias = float(initial_bias)
        self.weights_ = None
        self.bias_ = None
        self.feature_names_ = None
        self.history_ = []

    @staticmethod
    def _matrix(X):
        x = np.asarray(X)
        if x.ndim != 2 or 0 in x.shape or x.dtype.kind not in "iuf" or not np.isfinite(x).all():
            raise ValueError("X debe ser una matriz real finita no vacía de dos dimensiones")
        return x.astype(np.float64, copy=False)

    @staticmethod
    def _names(names, d):
        names = list(names) if names is not None else [f"x_{i}" for i in range(d)]
        if len(names) != d or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != d:
            raise ValueError("Nombres de características incompatibles o duplicados")
        return names

    def _predict_matrix(self, X, feature_names=None):
        if self.weights_ is None:
            raise ValueError("Modelo sin ajustar")
        x = self._matrix(X)
        if x.shape[1] != len(self.weights_):
            raise ValueError("Número de características incompatible con pesos")
        if feature_names is not None and list(feature_names) != self.feature_names_:
            raise ValueError("Orden/nombres de características incompatible con modelo")
        return x

    def decision_function(self, X, *, feature_names=None):
        x = self._predict_matrix(X, feature_names)
        scores = x @ self.weights_ + self.bias_
        if not np.isfinite(scores).all():
            raise ValueError("Combinación lineal no finita")
        return scores

    def predict(self, X, *, feature_names=None):
        return (self.decision_function(X, feature_names=feature_names) >= 0).astype(np.int8)

    def fit(self, X, y, *, X_val=None, y_val=None, feature_names=None):
        x = self._matrix(X)
        y = validar_etiquetas(y, n=len(x))
        d = x.shape[1]
        names = self._names(feature_names, d)
        if (X_val is None) != (y_val is None):
            raise ValueError("Validación requiere X_val e y_val juntos")
        val = None
        if X_val is not None:
            val = self._matrix(X_val)
            if val.shape[1] != d:
                raise ValueError("Validación con dimensión incompatible")
            y_val = validar_etiquetas(y_val, n=len(val))
            if len(np.unique(y_val)) != 2:
                raise ValueError("Validación debe contener ambas clases para seleccionar por balanced accuracy")
        if self.initial_weights is not None and len(self.initial_weights) != d:
            raise ValueError("Pesos iniciales incompatibles con X.shape[1]")
        # Cada fit es una ejecución nueva; no warm-start ni actualización con val/test.
        self.weights_ = np.zeros(d, dtype=np.float64) if self.initial_weights is None else self.initial_weights.copy()
        self.bias_ = self.initial_bias
        self.feature_names_ = names
        self.history_ = []
        rng = np.random.default_rng(self.seed)
        best_key, best_w, best_b = None, None, None
        for epoch in range(1, self.max_epochs+1):
            order = rng.permutation(len(x)) if self.shuffle else range(len(x))
            mistakes = 0
            for i in order:
                z = float(np.dot(self.weights_, x[i]) + self.bias_)
                if not np.isfinite(z):
                    raise ValueError("Combinación lineal no finita durante entrenamiento")
                predicted = int(z >= 0)
                error = int(y[i]) - predicted
                if error != 0:
                    self.weights_ += self.learning_rate * error * x[i]
                    self.bias_ += self.learning_rate * error
                    mistakes += 1
            if not np.isfinite(self.weights_).all() or not np.isfinite(self.bias_):
                raise ValueError("Actualización no finita; entrenamiento detenido")
            train_metrics = metricas_binarias(y, self.predict(x))
            val_metrics = metricas_binarias(y_val, self.predict(val)) if val is not None else None
            self.history_.append({"epoch": epoch, "online_errors": mistakes, "updates": mistakes,
                                  "train_errors_end": train_metrics["confusion_matrix"][0][1]+train_metrics["confusion_matrix"][1][0],
                                  "train_accuracy": train_metrics["accuracy"],
                                  "val_accuracy": val_metrics["accuracy"] if val_metrics else None,
                                  "val_balanced_accuracy": val_metrics["balanced_accuracy"] if val_metrics else None,
                                  "val_f1": val_metrics["f1"] if val_metrics else None,
                                  "weights_norm": float(np.linalg.norm(self.weights_))})
            key = clave_seleccion(val_metrics, epoch) if val_metrics else (epoch,)
            if best_key is None or key > best_key:
                best_key, best_w, best_b = key, self.weights_.copy(), float(self.bias_)
                self.best_epoch_ = epoch
        self.last_weights_, self.last_bias_ = self.weights_.copy(), float(self.bias_)
        self.weights_, self.bias_ = best_w, best_b
        self.selection_ = "validation_balanced_accuracy_then_f1_then_earliest_epoch" if val is not None else "last_epoch_without_validation"
        return self

    def get_params(self):
        return {"learning_rate": self.learning_rate, "max_epochs": self.max_epochs, "seed": self.seed,
                "shuffle": self.shuffle, "initial_weights": self.initial_weights.tolist() if self.initial_weights is not None else None,
                "initial_bias": self.initial_bias}

    def save(self, path):
        if self.weights_ is None:
            raise ValueError("No hay modelo ajustado para exportar")
        payload = {"format_version": 1, "algorithm": "manual_binary_perceptron", "label_values": [0, 1],
                   "activation": "1 if z >= 0 else 0", "params": self.get_params(),
                   "n_features": len(self.weights_), "feature_names": self.feature_names_,
                   "weights": self.weights_.tolist(), "bias": self.bias_,
                   "selected_epoch": self.best_epoch_, "selection": self.selection_}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if p.get("format_version") != 1 or p.get("algorithm") != "manual_binary_perceptron" or p.get("activation") != "1 if z >= 0 else 0" or p.get("label_values") != [0, 1]:
            raise ValueError("Modelo/esquema no compatible")
        obj = cls(**p["params"])
        w = np.asarray(p["weights"])
        if w.ndim != 1 or len(w) != p["n_features"] or w.dtype.kind not in "iuf" or not np.isfinite(w).all() or not np.isfinite(p["bias"]):
            raise ValueError("Pesos/sesgo serializados inválidos")
        obj.feature_names_ = obj._names(p["feature_names"], len(w))
        obj.weights_, obj.bias_ = w.astype(np.float64), float(p["bias"])
        if not isinstance(p["selected_epoch"], int) or not 1 <= p["selected_epoch"] <= obj.max_epochs:
            raise ValueError("Época serializada inválida")
        obj.best_epoch_, obj.selection_ = p["selected_epoch"], p["selection"]
        return obj
