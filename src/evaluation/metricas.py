"""Métricas 0/1 manuales, convenciones del contrato de fase 2."""
import numpy as np


def validar_etiquetas(values, *, n=None):
    y = np.asarray(values)
    if y.ndim != 1 or not y.size or y.dtype.kind not in "biuf" or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Se requiere un vector no vacío de etiquetas numéricas 0/1 finitas")
    if n is not None and len(y) != n:
        raise ValueError("Número de etiquetas incompatible con observaciones")
    return y.astype(np.int8, copy=False)


def metricas_binarias(y_true, y_pred):
    real = validar_etiquetas(y_true)
    predicted = validar_etiquetas(y_pred, n=len(real))
    tp = int(((real == 1) & (predicted == 1)).sum())
    tn = int(((real == 0) & (predicted == 0)).sum())
    fp = int(((real == 0) & (predicted == 1)).sum())
    fn = int(((real == 1) & (predicted == 0)).sum())
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    return {"n": len(real), "accuracy": (tp+tn)/len(real),
            "precision": tp/(tp+fp) if tp+fp else 0.0,
            "recall": recall, "specificity": specificity,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            "balanced_accuracy": (recall+specificity)/2 if recall is not None and specificity is not None else None,
            "confusion_matrix": [[tn, fp], [fn, tp]], "confusion_order": [0, 1],
            "support": {"0": tn+fp, "1": tp+fn},
            "class_proportions": {"0": (tn+fp)/len(real), "1": (tp+fn)/len(real)},
            "denominator_zero": {"precision": tp+fp == 0, "recall": tp+fn == 0,
                                 "specificity": tn+fp == 0, "f1": 2*tp+fp+fn == 0}}


def clave_seleccion(metrics, epoch, experiment_index=0):
    if metrics["balanced_accuracy"] is None:
        raise ValueError("Selección bloqueada: balanced accuracy de validación no estimable")
    return (metrics["balanced_accuracy"], metrics["f1"], -epoch, -experiment_index)
