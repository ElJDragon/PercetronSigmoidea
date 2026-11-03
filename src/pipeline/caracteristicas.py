"""Calendario, estados de cobertura, etiquetas y características causales."""
from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np
import pandas as pd

from .agregacion import MEASURES
from .particiones import particion

FEATURE_RULES = {
    "unidades_lag_1": "Q(p,t-1)", "unidades_lag_2": "Q(p,t-2)", "unidades_lag_3": "Q(p,t-3)",
    "unidades_media_3": "(q1+q2+q3)/3", "unidades_mediana_3": "mediana(q1,q2,q3)",
    "unidades_std_3": "std(q1,q2,q3), ddof=0", "cambio_unidades_1": "q1-q2",
    "variacion_relativa_1": "(q1-q2)/q2 si q2>0; nulo en otro caso",
    "base_variacion_cero": "1[q2=0]", "tendencia_unidades_3": "(q1-q3)/2",
    "facturas_lag_1": "COUNT DISTINCT InvoiceID(p,t-1)",
    "facturas_media_3": "media de facturas distintas en t-1,t-2,t-3",
    "ingresos_lag_1": "SUM VentaSinImpuesto(p,t-1)",
    "ingresos_media_3": "media de ingresos de t-1,t-2,t-3",
    "precio_unitario_ponderado_lag_1": "ingresos_lag_1/q1 si q1>0; nulo si q1=0",
    "mes_anterior_sin_ventas": "1[q1=0]; indicador también para precio sin base",
    "clientes_distintos_lag_1": "COUNT DISTINCT NombreCliente(p,t-1)",
    "unidades_por_factura_lag_1": "q1/facturas_lag_1 si facturas_lag_1>0; nulo si no",
    "facturas_lag_1_cero": "1[facturas_lag_1=0]; indicador de unidades_por_factura sin base",
    "antiguedad_observada_meses": "ordinal(t)-ordinal(primer mes observado antes del corte)",
    "mes_objetivo_sin": "sin(2*pi*(mes(t)-1)/12)",
    "mes_objetivo_cos": "cos(2*pi*(mes(t)-1)/12)",
    "dias_mes_objetivo": "días calendario de t, con bisiestos",
}
FEATURES = list(FEATURE_RULES)
OPTIONAL = {"variacion_relativa_1", "precio_unitario_ponderado_lag_1", "unidades_por_factura_lag_1"}


@dataclass(frozen=True)
class Cobertura:
    """Desconocida por defecto. Los cierres explícitos prevalecen sobre supuesto académico."""
    modo: str = "desconocida"
    inicio: str = "2013-01"
    fin: str = "2016-05"
    meses: dict = field(default_factory=dict)
    evidencia: str = "Sin certificación de cierres en el CSV"

    def __post_init__(self):
        if self.modo not in {"academica", "desconocida", "explicita"}:
            raise ValueError("Modo de cobertura inválido")
        if self.inicio > self.fin:
            raise ValueError("Rango de cobertura inválido")
        for month, state in self.meses.items():
            if str(pd.Period(month, freq="M")) != month or state not in {"completa", "desconocida"}:
                raise ValueError("Cierre mensual inválido")
        if self.modo == "explicita" and not self.evidencia.strip():
            raise ValueError("La cobertura explícita requiere procedencia")

    def estado(self, month):
        if month in self.meses:
            return self.meses[month]
        return "completa" if self.modo == "academica" and self.inicio <= month <= self.fin else "desconocida"

    def origen(self, month):
        if month in self.meses:
            return "configuracion_explicita"
        return "supuesto_academico_no_certificado" if self.estado(month) == "completa" else "sin_confirmar"


def estado_mes(month, first, observed, coverage):
    if month < first:
        return "producto_todavia_no_observado"
    if coverage.estado(month) != "completa":
        return "cobertura_desconocida_con_ventas" if observed else "cobertura_desconocida_sin_ventas"
    return "ventas_observadas" if observed else "mes_completo_sin_ventas"


def observation_id(product, month):
    value = json.dumps([product, month], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def construir_observaciones(aggregated, coverage, inicio="2013-01", fin="2016-05"):
    """No fabrica predicciones para el catálogo futuro ni etiquetas sin cierre.

    El calendario comercial incluye primera aparición; las observaciones de
    predicción comienzan en el mes siguiente a su reconocimiento.
    """
    calendar, observations = [], []
    for product, group in aggregated.groupby(level=0, sort=True):
        actual = group.droplevel(0)
        first = actual.index.min()
        if first > fin:
            continue
        months = pd.period_range(first, fin, freq="M")
        known = {}
        states = {}
        for period in months:
            month = str(period)
            observed = month in actual.index
            values = actual.loc[month].to_dict() if observed else {m: 0 for m in MEASURES}
            complete = coverage.estado(month) == "completa"
            state = estado_mes(month, first, observed, coverage)
            # Hechos observados parciales se conservan como evidencia, nunca como historia completa.
            states[month] = state
            known[month] = values if complete else None
            calendar.append({"producto_provisional": product, "mes": month,
                             **{k: values[k] if observed or complete else None for k in MEASURES},
                             "cobertura": coverage.estado(month), "origen_cobertura": coverage.origen(month),
                             "estado_mes": state})
        for period in months:
            month = str(period)
            if month < inicio or month == first:
                continue  # Todavía no se reconocía el producto al comenzar su primer mes.
            prior = [str(period - j) for j in [1, 2, 3]]
            history = [known.get(m) for m in prior]
            age = period.ordinal - pd.Period(first, freq="M").ordinal
            eligible = age >= 3 and all(v is not None for v in history)
            status = "elegible" if eligible else "historial_insuficiente" if age < 3 else "historial_cobertura_desconocida"
            response = known[month] if coverage.estado(month) == "completa" else None
            row = {"observacion_id": observation_id(product, month), "producto_provisional": product,
                   "mes_objetivo": month, "corte_exclusivo": f"{month}-01T00:00:00Z",
                   "primer_mes_observado_conocido": first, "particion": particion(month),
                   "cobertura_objetivo": coverage.estado(month), "origen_cobertura": coverage.origen(month),
                   "estado_mes_objetivo": states[month], "estado_elegibilidad": status,
                   "elegible_para_prediccion": eligible, "respuesta_disponible": response is not None,
                   "estado_respuesta": "disponible" if response is not None else "pendiente_cobertura",
                   "historia_meses": "|".join(prior), "historia_estados": "|".join(states.get(m, "producto_todavia_no_observado") for m in prior),
                   "Q_objetivo": response["unidades_observadas"] if response else None,
                   "R_referencia": None, "y": None, **{f: None for f in FEATURES}}
            if eligible:
                q = np.array([v["unidades_observadas"] for v in history], dtype=np.float64)
                q1, q2, q3 = q
                h1 = history[0]
                n1, v1 = h1["facturas_distintas"], h1["ingresos_sin_impuesto"]
                ref = float(np.median(q))
                features = [q1, q2, q3, float(q.mean()), ref, float(q.std(ddof=0)), q1-q2,
                            (q1-q2)/q2 if q2 > 0 else None, int(q2 == 0), (q1-q3)/2,
                            n1, sum(v["facturas_distintas"] for v in history)/3,
                            v1, sum(v["ingresos_sin_impuesto"] for v in history)/3,
                            v1/q1 if q1 > 0 else None, int(q1 == 0), h1["clientes_distintos"],
                            q1/n1 if n1 > 0 else None, int(n1 == 0), age,
                            math.sin(2*math.pi*(period.month-1)/12), math.cos(2*math.pi*(period.month-1)/12),
                            period.days_in_month]
                row.update(dict(zip(FEATURES, features, strict=True)))
                row["R_referencia"] = ref
                if response is not None:
                    row["y"] = int(response["unidades_observadas"] > ref)
            observations.append(row)
    obs = pd.DataFrame(observations)
    if obs.empty:
        raise ValueError("No hay productos reconocidos antes de los objetivos indicados")
    obs = obs.sort_values(["mes_objetivo", "producto_provisional"], kind="stable").reset_index(drop=True)
    obs["y"] = pd.array(obs.y, dtype="Int8")
    if obs.observacion_id.duplicated().any():
        raise AssertionError("Clave producto-mes no única")
    cal = pd.DataFrame(calendar).sort_values(["mes", "producto_provisional"], kind="stable").reset_index(drop=True)
    return cal, obs
