"""Lectura validada y agregación determinista sin mutar la fuente."""
from decimal import Decimal

import numpy as np
import pandas as pd

from src.auditar_datos import read_csv, sha256

MEASURES = ["unidades_observadas", "facturas_distintas", "ingresos_sin_impuesto",
            "numero_lineas", "clientes_distintos"]


def cargar_fuente(path, expected_hash, expected_columns):
    if sha256(path) != expected_hash:
        raise ValueError("Hash distinto del contrato: no se cambia la fuente ni la regla")
    raw, fmt = read_csv(path)
    if fmt["delimiter"] != ";" or list(raw.columns) != list(expected_columns):
        raise ValueError("Esquema/separador distinto del contrato")
    return raw, fmt


def agregar_producto_mes(raw):
    needed = ["NombreProducto", "Fecha", "CantidadVendida", "InvoiceID", "InvoiceLineID",
              "VentaSinImpuesto", "NombreCliente"]
    if not set(needed) <= set(raw.columns):
        raise ValueError("Faltan columnas para agregar producto-mes")
    d = raw[needed].copy(deep=True)
    if d[needed].isna().any().any() or d[["NombreProducto", "NombreCliente", "InvoiceID", "InvoiceLineID"]].eq("").any().any():
        raise ValueError("Identificadores/clientes ausentes: no se pueden contar sin ambigüedad")
    if d.InvoiceLineID.duplicated().any():
        raise ValueError("InvoiceLineID duplicado: revisar fuente, no eliminar filas")
    d["mes"] = pd.to_datetime(d.Fecha, utc=True, format="ISO8601", errors="raise").dt.strftime("%Y-%m")
    d["unidades"] = pd.to_numeric(d.CantidadVendida, errors="raise")
    if not np.isfinite(d.unidades).all() or not ((d.unidades >= 0) & (d.unidades % 1 == 0)).all():
        raise ValueError("CantidadVendida debe ser entera, finita y no negativa")
    d["unidades"] = d.unidades.astype("int64")
    # Suma monetaria en centavos exactos: independiente del orden de entrada.
    def cents(v):
        value = Decimal(str(v)) * 100
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError("VentaSinImpuesto no representa centavos exactos")
        return int(value)
    money = {v: cents(v) for v in d.VentaSinImpuesto.unique()}
    d["centavos"] = d.VentaSinImpuesto.map(money).astype("int64")
    g = d.groupby(["NombreProducto", "mes"], sort=True).agg(
        unidades_observadas=("unidades", "sum"), facturas_distintas=("InvoiceID", "nunique"),
        centavos=("centavos", "sum"), numero_lineas=("InvoiceLineID", "size"),
        clientes_distintos=("NombreCliente", "nunique"))
    if int(g.unidades_observadas.sum()) != int(d.unidades.sum()) or int(g.centavos.sum()) != int(d.centavos.sum()):
        raise AssertionError("Las sumas de agregación no se conservan")
    g["ingresos_sin_impuesto"] = g.pop("centavos") / 100.0
    return g[MEASURES].rename_axis(index=["producto_provisional", "mes"])
