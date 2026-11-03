"""PCA bidimensional por SVD NumPy, ajustado exclusivamente con entrenamiento."""
import json
from pathlib import Path

import numpy as np


class PCA2D:
    def __init__(self):
        self.mean_ = None

    @staticmethod
    def _matrix(X):
        x = np.asarray(X)
        if x.ndim != 2 or 0 in x.shape or x.dtype.kind not in "iuf" or not np.isfinite(x).all():
            raise ValueError("PCA requiere matriz real finita no vacía")
        return x.astype(np.float64, copy=False)

    def fit(self, X, *, feature_names=None):
        x = self._matrix(X)
        m, d = x.shape
        if m < 3 or d < 2:
            raise ValueError("Dos componentes requieren al menos tres filas y dos características")
        names = list(feature_names) if feature_names is not None else [f"x_{j}" for j in range(d)]
        if len(names) != d or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != d:
            raise ValueError("Nombres de características incompatibles")
        mean = x.mean(axis=0)
        centered = x-mean
        if not np.isfinite(centered).all():
            raise ValueError("Centrado no finito")
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        tolerance = float(np.finfo(np.float64).eps*max(m,d)*singular[0])
        rank = int((singular > tolerance).sum())
        if rank < 2:
            raise ValueError("Rango insuficiente: no se fabrican dos componentes informativos")
        components = vt[:2].T.copy()  # W2 tiene dimensión d×2.
        for j in range(2):
            pivot = int(np.argmax(np.abs(components[:,j])))
            if components[pivot,j] < 0:
                components[:,j] *= -1
        variances = singular**2/(m-1)
        total = float(variances.sum())
        if not np.isfinite(total) or total <= 0 or not np.isfinite(components).all():
            raise ValueError("Varianza/SVD no finita o nula")
        self.mean_, self.components_, self.singular_values_ = mean, components, singular
        self.explained_variance_all_, self.total_variance_ = variances, total
        self.explained_variance_ratio_all_ = variances/total
        self.explained_variance_ = variances[:2]
        self.explained_variance_ratio_ = self.explained_variance_ratio_all_[:2]
        self.feature_names_, self.n_samples_, self.n_features_ = names, m, d
        self.rank_, self.rank_tolerance_ = rank, tolerance
        return self

    def transform(self, X, *, feature_names=None):
        if self.mean_ is None:
            raise ValueError("PCA no ajustado")
        x = self._matrix(X)
        if x.shape[1] != self.n_features_ or feature_names is not None and list(feature_names) != self.feature_names_:
            raise ValueError("Dimensión/orden de características incompatible con PCA")
        z = (x-self.mean_) @ self.components_
        if not np.isfinite(z).all():
            raise ValueError("Proyección no finita")
        return z

    def to_dict(self):
        if self.mean_ is None:
            raise ValueError("PCA sin ajustar")
        return {"format_version":1,"algorithm":"numpy_SVD_PCA_2D","fit_partition":"train_only",
                "n_samples":self.n_samples_,"n_features":self.n_features_,"n_components":2,
                "feature_names":self.feature_names_.copy(),"mean_train":self.mean_.tolist(),
                "components_d_by_2":self.components_.tolist(),"singular_values_all":self.singular_values_.tolist(),
                "explained_variance_all":self.explained_variance_all_.tolist(),
                "explained_variance_ratio_all":self.explained_variance_ratio_all_.tolist(),
                "total_variance":self.total_variance_,"variance_ddof":1,"rank":self.rank_,"rank_tolerance":self.rank_tolerance_,
                "sign_convention":"largest absolute coefficient positive; first input index on ties",
                "retained_ratio_2d":float(self.explained_variance_ratio_.sum()),
                "discarded_ratio":float(max(0.,1-self.explained_variance_ratio_.sum()))}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")

    @classmethod
    def load(cls, path):
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if p.get("format_version") != 1 or p.get("algorithm") != "numpy_SVD_PCA_2D" or p.get("n_components") != 2:
            raise ValueError("Formato PCA incompatible")
        obj = cls()
        obj.mean_ = np.asarray(p["mean_train"],dtype=np.float64)
        obj.components_ = np.asarray(p["components_d_by_2"],dtype=np.float64)
        obj.singular_values_ = np.asarray(p["singular_values_all"],dtype=np.float64)
        obj.explained_variance_all_ = np.asarray(p["explained_variance_all"],dtype=np.float64)
        obj.explained_variance_ratio_all_ = np.asarray(p["explained_variance_ratio_all"],dtype=np.float64)
        obj.total_variance_,obj.n_samples_,obj.n_features_ = p["total_variance"],p["n_samples"],p["n_features"]
        obj.feature_names_,obj.rank_,obj.rank_tolerance_ = p["feature_names"],p["rank"],p["rank_tolerance"]
        arrays=[obj.mean_,obj.components_,obj.singular_values_,obj.explained_variance_all_,obj.explained_variance_ratio_all_]
        if (obj.mean_.shape != (obj.n_features_,) or obj.components_.shape != (obj.n_features_,2)
                or not all(np.isfinite(a).all() for a in arrays)
                or len(obj.feature_names_) != obj.n_features_ or len(set(obj.feature_names_)) != obj.n_features_
                or obj.rank_ < 2 or not np.isfinite(obj.total_variance_) or obj.total_variance_ <= 0
                or not np.allclose(obj.components_.T@obj.components_,np.eye(2),atol=1e-12,rtol=0)):
            raise ValueError("Parámetros PCA inválidos")
        obj.explained_variance_ = obj.explained_variance_all_[:2]
        obj.explained_variance_ratio_ = obj.explained_variance_ratio_all_[:2]
        return obj
