"""Imputación y z-score numéricos train-only, estado JSON sin pickle."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .caracteristicas import FEATURES, OPTIONAL


class PreprocesadorComun:
    def __init__(self):
        self.estado = None

    @staticmethod
    def _validar(frame):
        if frame.columns.duplicated().any() or set(frame.columns) != set(FEATURES):
            raise ValueError("X debe contener exactamente el catálogo numérico base autorizado")
        data = frame.loc[:, FEATURES].astype(np.float64)
        if np.isinf(data.to_numpy()).any():
            raise ValueError("Características infinitas: matrices bloqueadas")
        if data[[f for f in FEATURES if f not in OPTIONAL]].isna().any().any():
            raise ValueError("Falta historia esencial; no se imputa para fabricar elegibilidad")
        return data

    def fit(self, frame, *, particion):
        if particion != "train":
            raise ValueError("fit solo autorizado con train")
        if self.estado is not None:
            raise ValueError("Instancia ya ajustada; usar transform sin reajustar")
        data = self._validar(frame)
        if data.empty:
            raise ValueError("Sin train conocido: matrices bloqueadas por cobertura")
        columns, discarded, params = [], {}, {}
        for name in FEATURES:
            values = data[name]
            if values.isna().all():
                discarded[name] = "completamente_nula_en_train"
                continue
            median = float(values.median())
            filled = values.fillna(median).to_numpy(np.float64)
            mean, std = float(np.mean(filled)), float(np.std(filled, ddof=0))
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ValueError("Parámetros de escala no finitos: matrices bloqueadas")
            if std == 0:
                discarded[name] = "constante_en_train_tras_imputacion"
                continue
            columns.append(name)
            params[name] = {"imputacion": "mediana_train", "mediana": median,
                            "media": mean, "desviacion_poblacional": std,
                            "dtype": "float64", "formula": "(x_imputado-media_train)/std_train"}
        if not columns:
            raise ValueError("Todas las columnas son constantes o nulas: matrices bloqueadas")
        self.estado = {"version": 1, "ajuste": "train_exclusivamente", "input_features": FEATURES.copy(),
                       "numeric_features": FEATURES.copy(), "categorical_features": [],
                       "category_vocabulary": {}, "feature_names": columns,
                       "discarded": discarded, "parameters": params,
                       "exact_redundancy_removal": "ninguna; se conservan todas las columnas no constantes autorizadas",
                       "train_rows": len(data), "ddof": 0}
        return self

    def transform(self, frame):
        if self.estado is None:
            raise ValueError("El preprocesador todavía no fue ajustado")
        data = self._validar(frame)
        result = np.empty((len(data), len(self.estado["feature_names"])), dtype=np.float64)
        for j, name in enumerate(self.estado["feature_names"]):
            p = self.estado["parameters"][name]
            result[:, j] = (data[name].fillna(p["mediana"]).to_numpy(np.float64) - p["media"]) / p["desviacion_poblacional"]
        if not np.isfinite(result).all():
            raise ValueError("Resultado no finito: matrices bloqueadas")
        return result

    def to_dict(self):
        if self.estado is None:
            raise ValueError("No hay estado ajustado")
        return copy.deepcopy(self.estado)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        obj = cls()
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != 1 or state.get("input_features") != FEATURES or state.get("categorical_features") != []:
            raise ValueError("Versión/esquema del preprocesador no compatible")
        if not state["feature_names"] or len(set(state["feature_names"])) != len(state["feature_names"]) or not set(state["feature_names"]) <= set(FEATURES):
            raise ValueError("Nombres de salida inválidos")
        for name in state["feature_names"]:
            p = state["parameters"][name]
            if not all(np.isfinite(p[k]) for k in ["mediana", "media", "desviacion_poblacional"]) or p["desviacion_poblacional"] <= 0:
                raise ValueError("Parámetros de escala inválidos")
        obj.estado = state
        return obj
