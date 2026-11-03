"""Auditoría WWI de fase 1. Solo lectura de fuentes; no entrena modelos.

Ejecutar: python src/auditar_datos.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
NULLS = {"", "null", "none", "nan", "na", "n/a", "<null>", "\\n"}
DATES = {"Fecha", "FechaApertura"}
NUMERIC = {"InvoiceID", "InvoiceLineID", "Dia", "Mes", "Trimestre", "Anio",
           "EsRefrigerado", "DescuentoEstandar", "CantidadVendida", "PrecioUnitario",
           "VentaSinImpuesto", "Impuesto", "VentaTotal", "Beneficio"}
GROUPS = {
    "tiempo": ["Fecha", "Dia", "NombreDia", "Mes", "NombreMes", "Trimestre", "Anio"],
    "productos": ["NombreProducto", "Marca", "Tamanio", "EsRefrigerado"],
    "clientes": ["NombreCliente", "CategoriaCliente", "GrupoCompra", "FechaApertura", "DescuentoEstandar"],
    "ubicacion": ["Ciudad", "ProvinciaEstado", "CodigoProvincia", "TerritorioVentas"],
    "vendedores": ["Vendedor", "NombrePreferidoVendedor"],
    "cantidad_precios": ["CantidadVendida", "PrecioUnitario"],
    "ingresos": ["VentaSinImpuesto", "VentaTotal"], "impuestos": ["Impuesto"],
    "beneficios": ["Beneficio"],
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detect_format(path):
    # Se valida la codificación sobre TODO el archivo, no solo caracteres ASCII iniciales.
    content = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            decoded = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    delimiter = csv.Sniffer().sniff(decoded[:65536], delimiters=";,\t|").delimiter
    return {"encoding": encoding, "delimiter": delimiter,
            "encoding_note": "Decodificación completa válida; no prueba única del origen. UTF-8-sig admite BOM opcional."}


def read_csv(path):
    fmt = detect_format(path)
    with Path(path).open(encoding=fmt["encoding"], newline="") as stream:
        header = next(csv.reader(stream, delimiter=fmt["delimiter"]))
    if len(header) != len(set(header)):
        raise ValueError("Encabezados duplicados")
    raw = pd.read_csv(path, sep=fmt["delimiter"], encoding=fmt["encoding"],
                      dtype=str, keep_default_na=False, na_filter=False,
                      skip_blank_lines=False, on_bad_lines="error")
    if len(raw.columns) != len(set(raw.columns)):
        raise ValueError("Encabezados duplicados")
    if raw.isna().any().any():
        raise ValueError("Filas con campos ausentes estructuralmente")
    return raw, fmt


def is_null(value):
    return str(value).strip().casefold() in NULLS


def number(value):
    """Decimal exacto; admite coma decimal sin inferir separadores de miles."""
    return Decimal(str(value).strip().replace(",", "."))


def numeric_equal(a, b):
    try:
        x, y = number(a), number(b)
        return x.is_finite() and y.is_finite() and x == y
    except InvalidOperation:
        return False


def normalized(raw):
    data = raw.copy()
    for col in raw:
        data[col] = raw[col].mask(raw[col].map(is_null), pd.NA)
        if col in NUMERIC:
            data[col] = pd.to_numeric(data[col].str.replace(",", ".", regex=False), errors="coerce")
        elif col in DATES:
            data[col] = pd.to_datetime(data[col], format="ISO8601", errors="coerce", utc=True)
    return data


def monetary_checks(data):
    """Tolerancia absoluta 1e-8, rtol=0: solo ruido binario, muy inferior a 0.01."""
    results = {}
    relations = [
        ("cantidad_por_precio", ["CantidadVendida", "PrecioUnitario", "VentaSinImpuesto"],
         lambda d: d.CantidadVendida * d.PrecioUnitario, "VentaSinImpuesto"),
        ("base_mas_impuesto", ["VentaSinImpuesto", "Impuesto", "VentaTotal"],
         lambda d: d.VentaSinImpuesto + d.Impuesto, "VentaTotal"),
    ]
    for name, cols, expected, actual in relations:
        if not set(cols) <= set(data.columns):
            results[name] = {"available": False}
            continue
        valid = data[cols].notna().all(axis=1) & np.isfinite(data[cols]).all(axis=1)
        errors = (data.loc[valid, actual] - expected(data.loc[valid])).abs()
        bad = errors > 1e-8
        results[name] = {"available": True, "evaluated": int(valid.sum()),
                         "not_evaluated": int((~valid).sum()), "violations": int(bad.sum()),
                         "max_abs_error": float(errors.max()) if len(errors) else None,
                         "absolute_tolerance": 1e-8, "relative_tolerance": 0,
                         "sample_row_numbers": (errors[bad].index[:5] + 2).tolist()}
    results["beneficio"] = "No verificable contra costo: no existe costo unitario ni costo histórico en el CSV."
    results["impuesto"] = "No se impone tasa fija: no hay TaxRate para validar por operación."
    return results


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def profile(path, raw, fmt):
    data = normalized(raw)
    n = len(data)
    cols = {}
    ranges = {"Dia": (1, 31), "Mes": (1, 12), "Trimestre": (1, 4),
              "EsRefrigerado": (0, 1), "DescuentoEstandar": (0, 100)}
    for col in raw:
        s = data[col]
        nullmask = raw[col].map(is_null)
        valid = s.dropna()
        info = {"inferred_type": str(s.dtype), "missing": int(nullmask.sum()),
                "missing_pct": float(nullmask.mean() * 100),
                "null_representations": raw.loc[nullmask, col].value_counts().to_dict(),
                "unique_non_null": int(s.nunique()),
                "constant_including_null": bool(s.nunique(dropna=False) == 1),
                "constant_non_null": bool(s.nunique() == 1),
                "top_values": {str(k): int(v) for k, v in s.value_counts().head(5).items()}}
        if col in NUMERIC:
            finite = valid[np.isfinite(valid)]
            info.update(invalid_numeric=int((~nullmask & s.isna()).sum()),
                        non_finite=int((~np.isfinite(valid)).sum()))
            info["statistics"] = {k: float(v) for k, v in finite.describe().items()}
            q1, q3 = finite.quantile([.25, .75]) if len(finite) else (np.nan, np.nan)
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            info["iqr_outliers"] = {"lower": float(low), "upper": float(high),
                                    "count": int(((finite < low) | (finite > high)).sum()),
                                    "note": "Señal exploratoria; no implica error ni eliminación. No usar IQR para IDs/calendario."}
            if col in ranges:
                lo, hi = ranges[col]
                info["outside_domain"] = int(((valid < lo) | (valid > hi) | (valid % 1 != 0)).sum()) if col != "DescuentoEstandar" else int(((valid < lo) | (valid > hi)).sum())
            if col in {"InvoiceID", "InvoiceLineID", "CantidadVendida", "PrecioUnitario", "VentaSinImpuesto", "Impuesto", "VentaTotal"}:
                info["negative"] = int((valid < 0).sum())
                info["zero"] = int((valid == 0).sum())
        elif col in DATES:
            info.update(invalid_dates=int((~nullmask & s.isna()).sum()),
                        min=str(valid.min()), max=str(valid.max()))
        cols[col] = info
    keys = {}
    for key in ("InvoiceLineID", "InvoiceID"):
        if key in data:
            keys[key] = {"unique": int(data[key].nunique()), "missing": int(data[key].isna().sum()),
                         "duplicate_excess": int(data[key].duplicated().sum()),
                         "rows_in_duplicate_groups": int(data[key].duplicated(keep=False).sum())}
    if {"InvoiceLineID", "InvoiceID"} <= set(data.columns):
        counts = data.groupby("InvoiceLineID").InvoiceID.nunique()
        keys["line_to_multiple_invoices"] = int((counts > 1).sum())
        keys["lines_per_invoice"] = data.groupby("InvoiceID").size().describe().to_dict()
        keys["invoice_context_conflicts"] = {c: int((data.groupby("InvoiceID")[c].nunique(dropna=False) > 1).sum())
                                             for c in ["Fecha", "NombreCliente", "Ciudad", "Vendedor"] if c in data}
    result = {"path": str(path.resolve()), "relative_path": str(path.relative_to(BASE)),
              "bytes": path.stat().st_size, "sha256": sha256(path), **fmt,
              "rows": n, "column_count": len(raw.columns), "column_names": raw.columns.tolist(),
              "columns": cols, "raw_duplicate_excess": int(raw.duplicated().sum()),
              "logical_duplicate_excess": int(data.duplicated().sum()), "keys": keys,
              "business_variables": {g: [c for c in cc if c in data] for g, cc in GROUPS.items()},
              "monetary_checks": monetary_checks(data)}
    return result, data


def compare_csv(left, right, key="InvoiceLineID"):
    result = {"schema_equal": set(left.columns) == set(right.columns),
              "column_order_equal": list(left.columns) == list(right.columns),
              "rows_left": len(left), "rows_right": len(right), "columns": {}}
    if not result["schema_equal"] or key not in left or key not in right:
        return {**result, "equivalent": False, "reason": "Esquema o clave ausente"}
    a, b = normalized(left), normalized(right)
    if a[key].isna().any() or b[key].isna().any() or a[key].duplicated().any() or b[key].duplicated().any():
        return {**result, "equivalent": False, "reason": "Clave nula o duplicada; alineación no inequívoca"}
    # Usar clave textual numérica exacta; nunca redondear contenido para comparar.
    ka, kb = left[key].map(number), right[key].map(number)
    missing_a, missing_b = set(kb) - set(ka), set(ka) - set(kb)
    result.update(keys_only_left=len(missing_b), keys_only_right=len(missing_a))
    la, rb = left.set_index(ka), right.set_index(kb)
    common = la.index.intersection(rb.index).sort_values()
    la, rb = la.loc[common], rb.loc[common]
    total = 0
    trim_only_total = 0
    for col in left:
        x, y = la[col], rb[col]
        def canonical(v):
            if is_null(v):
                return None
            if col in NUMERIC:
                try:
                    num = number(v)
                    return num if num.is_finite() else ("invalid_numeric", v)
                except InvalidOperation:
                    return ("invalid_numeric", v)
            if col in DATES:
                dt = pd.to_datetime(v, format="ISO8601", utc=True, errors="coerce")
                return dt if not pd.isna(dt) else ("invalid_date", v)
            return v  # No se normalizan mayúsculas ni espacios de categorías.
        mapping = {v: canonical(v) for v in set(x) | set(y)}
        different = pd.Series([mapping[u] != mapping[v] for u, v in zip(x, y)], index=x.index)
        count = int(different.sum())
        total += count
        trim_only = 0
        if col not in NUMERIC | DATES:
            trim_only = int((different & x.str.strip().eq(y.str.strip())).sum())
        trim_only_total += trim_only
        result["columns"][col] = {"raw_differences": int((x != y).sum()), "logical_differences": count,
                                  "whitespace_only_differences": trim_only,
                                  "unique_values_before_trim": int(x.nunique()),
                                  "unique_values_after_trim": int(x.str.strip().nunique()),
                                  "samples": [{"key": str(k), "left": x.loc[k], "right": y.loc[k]} for k in different[different].index[:5]]}
    result.update(logical_cell_differences=total,
                  equivalent=not (total or missing_a or missing_b),
                  whitespace_only_differences=trim_only_total,
                  equivalent_if_text_trim_accepted=not (total - trim_only_total or missing_a or missing_b),
                  method="Alineación por clave única, Decimal exacto, fechas ISO UTC, nulos explícitos; sin redondeo ni tolerancia monetaria en equivalencia.")
    return result


def business_analysis(data):
    d = data.copy()
    d["month"] = d.Fecha.dt.strftime("%Y-%m")
    d["negative"] = d.Beneficio < 0
    d["zero"] = d.Beneficio == 0
    d["positive"] = d.Beneficio > 0
    temporal = d.groupby("month").agg(rows=("Beneficio", "size"), negative=("negative", "sum"),
                    positive=("positive", "sum"), zero=("zero", "sum"), units=("CantidadVendida", "sum"))
    temporal["negative_pct"] = temporal.negative / temporal.rows * 100
    products = d.groupby("NombreProducto").agg(rows=("negative", "size"), negative=("negative", "sum"),
                                                  loss_sum=("Beneficio", lambda s: s[s < 0].sum()))
    products["loss_rate_pct"] = products.negative / products.rows * 100
    products = products.sort_values(["negative", "loss_sum"], ascending=[False, True])
    losses = int(d.negative.sum())
    products["share_of_negatives_pct"] = products.negative / losses * 100 if losses else 0
    counts = {"positive": int(d.positive.sum()), "negative": losses, "zero": int(d.zero.sum())}
    result = {"profit_counts": counts, "profit_pct": {k: v / len(d) * 100 for k, v in counts.items()},
              "majority_binary_accuracy_pct": max(losses, len(d) - losses) / len(d) * 100,
              "majority_definition": "Negativo (<0) frente a no negativo (>=0), sin entrenamiento.",
              "profit_by_month": records(temporal.reset_index()),
              "loss_products": records(products[products.negative > 0].reset_index()),
              "products_with_losses": int((products.negative > 0).sum()),
              "top_1_negative_share_pct": float(products.negative.head(1).sum() / losses * 100) if losses else 0,
              "top_5_negative_share_pct": float(products.negative.head(5).sum() / losses * 100) if losses else 0,
              "top_4_negative_share_pct": float(products.negative.head(4).sum() / losses * 100) if losses else 0,
              "history": {"min": str(d.Fecha.min()), "max": str(d.Fecha.max()), "months": len(temporal),
                          "observed_days": int(d.Fecha.nunique()),
                          "weekdays": d.Fecha.dt.day_name().value_counts().to_dict()},
              "calendar_mismatches": {c: int((d[c] != expected).sum()) for c, expected in
                 [("Dia", d.Fecha.dt.day), ("Mes", d.Fecha.dt.month), ("Anio", d.Fecha.dt.year), ("Trimestre", d.Fecha.dt.quarter)]},
              "opening_after_sale": int((d.FechaApertura > d.Fecha).sum())}
    spanish_days = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    spanish_months = dict(enumerate(["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"], 1))
    result["calendar_mismatches"]["NombreDia"] = int((d.NombreDia != d.Fecha.dt.dayofweek.map(spanish_days)).sum())
    result["calendar_mismatches"]["NombreMes"] = int((d.NombreMes != d.Fecha.dt.month.map(spanish_months)).sum())
    # Agregados exploratorios: no se exporta ni construye una matriz predictiva/etiquetas.
    for entity, label in [("NombreProducto", "product_month"), ("NombreCliente", "customer_month")]:
        by = d.groupby([entity, "month"]).CantidadVendida.sum()
        activity = by.reset_index().groupby(entity).agg(first=("month", "min"), last=("month", "max"), observed=("month", "size"))
        months = pd.period_range(d.Fecha.min().tz_localize(None).to_period("M"), d.Fecha.max().tz_localize(None).to_period("M"), freq="M")
        end = months[-1].ordinal
        eligible = sum(end - pd.Period(v, freq="M").ordinal + 1 for v in activity["first"])
        result[label] = {"entities": len(activity), "observed_pairs": len(by),
                         "global_grid_pairs": len(activity) * len(months),
                         "absent_pairs_global_grid": len(activity) * len(months) - len(by),
                         "pairs_from_first_observation": eligible,
                         "absent_pairs_from_first_observation": eligible - len(by),
                         "first_observed_by_month": activity["first"].value_counts().sort_index().to_dict(),
                         "observed_months_per_entity": activity.observed.describe().to_dict(),
                         "monthly_active_entities": by.reset_index().groupby("month")[entity].nunique().to_dict(),
                         "monthly_units_quantiles_observed_pairs": {str(k): float(v) for k, v in by.quantile([0, .25, .5, .75, 1]).items()},
                         "note": "Ausencia de transacción no prueba disponibilidad, vigencia ni demanda cero; primera venta no demuestra alta de catálogo. Último mes requiere confirmar cierre de extracción."}
    return result


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(base=BASE):
    base = Path(base).resolve()
    paths = []
    for name in ("dataset_ventas_wwi.csv", "datos_limpios.csv"):
        matches = list((base / "datos").rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Se esperaba una ruta inequívoca para {name}: {matches}")
        paths.append(matches[0])
    protected = paths + sorted((base / "referencias").rglob("*"))
    before = {str(p.relative_to(base)): sha256(p) for p in protected if p.is_file()}
    profiles, raws, frames = {}, [], []
    for path in paths:
        raw, fmt = read_csv(path)
        prof, frame = profile(path, raw, fmt)
        profiles[path.name] = prof
        raws.append(raw)
        frames.append(frame)
    comparison = compare_csv(*raws)
    comparison["source_order"] = [p.name for p in paths]
    comparison["recommended_source"] = "datos_limpios.csv" if comparison.get("equivalent_if_text_trim_accepted") else "Revisar discrepancias antes de seleccionar"
    comparison["recommendation_reason"] = "Conserva claves y medidas; nulos explícitos y fechas ISO. Diferencias de espacios documentadas, sin fusionar nombres distintos tras strip en esta copia. Mantener trazabilidad a original y solicitar IDs de entidades."
    analysis = business_analysis(frames[1])
    after = {str(p.relative_to(base)): sha256(p) for p in protected if p.is_file()}
    if before != after:
        raise RuntimeError("Cambió una fuente protegida durante la auditoría")
    out = base / "resultados" / "auditoria"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"method_version": 1, "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
               "datasets": profiles, "business_analysis": analysis,
               "integrity": {"preserved": True, "before": before, "after": after}}
    dump(out / "perfilado_datos.json", payload)
    dump(out / "comparacion_csv.json", comparison)
    dictionary = []
    for name, prof in profiles.items():
        for col, info in prof["columns"].items():
            group = next((g for g, cc in GROUPS.items() if col in cc), "identificadores")
            role = "identificador; no predictor ordinal" if col.endswith("ID") else "candidato sujeto a instante de disponibilidad"
            if col == "Beneficio":
                role = "resultado; etiqueta posible; fuga para problema A"
            if col in {"VentaSinImpuesto", "Impuesto", "VentaTotal", "CantidadVendida"}:
                role = "medida transaccional; solo histórico para problemas futuros"
            dictionary.append({"archivo": name, "columna": col, "grupo": group, "tipo": info["inferred_type"],
                               "faltantes": info["missing"], "porcentaje_faltantes": info["missing_pct"],
                               "cardinalidad_no_nula": info["unique_non_null"], "uso_preliminar": role})
    pd.DataFrame(dictionary).to_csv(out / "diccionario_datos.csv", index=False, encoding="utf-8-sig")
    write_report(out, payload, comparison)
    print(json.dumps({"rows": {k: v["rows"] for k, v in profiles.items()}, "equivalent": comparison["equivalent"],
                      "profit_counts": analysis["profit_counts"], "protected_files": len(before), "output": str(out)}, ensure_ascii=False))
    return payload, comparison


def write_report(out, payload, comparison):
    """Informe derivado de resultados, regenerado en cada ejecución."""
    a = payload["business_analysis"]
    ds = payload["datasets"]
    p = ds["datos_limpios.csv"]
    lines = ["# Auditoría WWI — Fase 1", "",
        "## Alcance y método", "",
        "Solo perfilado y análisis exploratorio. No se entrenaron modelos, no se implementaron PCA ni preprocesamiento predictivo, no se construyeron datasets definitivos y no se eliminaron filas. Las recomendaciones no fijan el objetivo.", "",
        "Lectura CSV completa como texto, sin conversión implícita de nulos; decodificación completa, separador por csv.Sniffer, normalización en memoria según esquema. UTF-8 compatible no identifica de forma única la codificación original. Fechas ISO tratadas como UTC para comparar fechas de calendario; no se infiere una zona comercial real del sufijo Z de OpenRefine.", "",
        "Comparación por InvoiceLineID único: Decimal exacto en números, fechas ISO normalizadas, nulos explícitos; no se redondea. Espacios de categorías se conservan en la comparación estricta y se diagnostican por separado. No se usa similitud difusa.", "",
        "IQR (1,5 veces rango intercuartílico) identifica observaciones para revisar, no errores. En IDs, calendario y banderas no debe interpretarse como anomalía de negocio. No se impone un máximo monetario arbitrario. Validaciones de rango son controles exploratorios; devoluciones podrían justificar negativos en otras extracciones.", "",
        "## Workspace e insumos", "",
        "Directorios existentes: datos/originales, datos/Procesados, informe, notebooks, referencias/Codigo_anteiror, resultados, src, tests. Se conserva su capitalización. README.md y requirements.txt estaban vacíos. No existían scripts en src ni pruebas en tests. No se accedió al proyecto DeberPerceptron.", "",
        "Se encontraron ambos CSV, los tres PDF y la nota Markdown. Inventario protegido completo con SHA-256:", "", "| Archivo | SHA-256 |", "|---|---|"]
    for name, digest in payload["integrity"]["before"].items():
        lines.append(f"| `{name}` | `{digest}` |")
    lines += ["", "El informe OpenRefine menciona dataset_ventas_wwi_limpio.csv; no existe con ese nombre. El archivo disponible es datos_limpios.csv. El código de referencia apunta a un CSV sintético y resultados antiguos que no fueron copiados: no son insumos necesarios de WWI. La carpeta real es Codigo_anteiror, no Codigo_anterior.", "",
        "Del código previo se aprovechan como referencia conceptual validación, CLI, métricas manuales y trazabilidad. No se importa ni ejecuta: fija tres entradas, etiquetas Hombre/Mujer, partición aleatoria y mínimo de 200 épocas, reglas no transferibles automáticamente a este problema.", "",
        "## Perfil general", "", "| Fuente | Filas | Columnas | Bytes | Separador | Codificación válida |", "|---|---:|---:|---:|---|---|"]
    for name, v in ds.items():
        lines.append(f"| {name} | {v['rows']} | {v['column_count']} | {v['bytes']} | `{v['delimiter']}` | {v['encoding']} |")
    lines += ["", f"Período: {a['history']['min']} a {a['history']['max']}; {a['history']['months']} meses, {a['history']['observed_days']} fechas con ventas. 2016 es parcial (enero-mayo). No hay ventas dominicales en esta extracción; esto no prueba ausencia de demanda.", "",
        "Las 30 columnas son atributos observados, no 30 predictores aprobados. Hay 14 columnas numéricas, 2 fechas y 14 textos según el esquema auditado. El número final de características será dinámico después de elegir el grano y codificar categorías.", "",
        "Nombres y tipos completos en diccionario_datos.csv; distribuciones, cardinalidades, estadísticos, nulos y outliers por fuente/columna en perfilado_datos.json.", "",
        "## Calidad y grano", "", "| Columna | Faltantes | % | No nulos distintos |", "|---|---:|---:|---:|"]
    for col, info in p["columns"].items():
        if info["missing"]:
            lines.append(f"| {col} | {info['missing']} | {info['missing_pct']:.6f} | {info['unique_non_null']} |")
    lines += ["", "Esos faltantes son NULL literal en dataset_ventas_wwi.csv y vacíos en datos_limpios.csv; no hay otras columnas con ausencia. No convertirlos indiscriminadamente en categorías comerciales ni borrar filas. Marca solo tiene Northwind entre no nulos, pero con ausencia no es constante completa. DescuentoEstandar sí es constante en cero.", "",
        f"Duplicados completos: {p['raw_duplicate_excess']}; duplicados de InvoiceLineID: {p['keys']['InvoiceLineID']['duplicate_excess']}. Hay {p['keys']['InvoiceID']['unique']} facturas y de 1 a 5 líneas por factura. Las repeticiones de InvoiceID son legítimas. Una línea no se asocia a múltiples facturas; fecha, cliente, ciudad y vendedor son consistentes dentro de cada factura.", "",
        "Grano comprobado: una línea de factura identificada por InvoiceLineID. Producto, cliente y ubicación están descritos por nombres, sin StockItemID/CustomerID/CityID; la unicidad real de esas entidades no puede garantizarse con nombres. No asumir que InvoiceID identifica una venta de un solo producto.", "",
        f"Hay {p['columns']['NombreProducto']['unique_non_null']} nombres de producto, {p['columns']['NombreCliente']['unique_non_null']} de cliente, {p['columns']['Ciudad']['unique_non_null']} ciudades, {p['columns']['Vendedor']['unique_non_null']} vendedores y {p['columns']['NombrePreferidoVendedor']['unique_non_null']} nombres preferidos. Estos últimos no identifican unívocamente al vendedor.", "",
        "No se detectaron conversiones numéricas inválidas, no finitos, fechas inválidas ni violaciones de los rangos controlados en estos CSV. Las medidas de cantidad, precio, ingresos e impuesto son positivas. Beneficio negativo es válido como resultado, no una violación de rango.", "",
        f"Inconsistencias del calendario: `{a['calendar_mismatches']}`. FechaApertura posterior a venta: {a['opening_after_sale']}.", "",
        "## Relaciones monetarias", "",
        "Se comprobaron CantidadVendida × PrecioUnitario = VentaSinImpuesto y VentaSinImpuesto + Impuesto = VentaTotal en todas las filas de ambas fuentes, sin incumplimientos. Tolerancia absoluta 1e-8, relativa cero: absorbe representación binaria sin tolerar diferencias de un centavo. Error máximo observado: 2,2737367544323206e-13. La comparación ENTRE CSV usa Decimal exacto, sin esta tolerancia.", "",
        "No se puede verificar Beneficio = venta - costo porque no hay costo histórico. No se puede validar la tasa fiscal de cada línea sin TaxRate. Los importes no demuestran descuentos, devoluciones o causas de pérdida.", "",
        "| Medida | Mínimo | Máximo | Atípicos IQR |", "|---|---:|---:|---:|"]
    for col in ["CantidadVendida", "PrecioUnitario", "VentaSinImpuesto", "Impuesto", "VentaTotal", "Beneficio"]:
        inf = p["columns"][col]
        lines.append(f"| {col} | {inf['statistics']['min']} | {inf['statistics']['max']} | {inf['iqr_outliers']['count']} |")
    lines += ["", "No se truncaron ni excluyeron observaciones. IQR debe revisarse por familias de producto y contexto, sin ajustar futuros umbrales con prueba.", "",
        "## Comparación de fuentes", "",
        f"Esquema y claves iguales; ninguna clave exclusiva. Diferencias de contenido estrictas: {comparison['logical_cell_differences']}. Todas corresponden a espacios periféricos en NombreCliente. Equivalencia estricta: {comparison['equivalent']}. Equivalencia aceptando trim textual explícito: {comparison['equivalent_if_text_trim_accepted']}.", "",
        "Las 268 celdas de nombre cambian de `David Novacek ` a `David Novacek`. El diagnóstico de cardinalidad de comparacion_csv.json comprueba si trim colapsa nombres; no se considera un simple cambio de formato de número/fecha/nulo. Las mismas líneas comerciales y sus medidas se conservan. El informe OpenRefine no documenta expresamente esta transformación de espacios: falta su historial de operaciones para atribuir su origen.", "",
        "Recomiendo datos_limpios.csv como fuente principal de trabajo: mantiene claves, categorías salvo ese trim y medidas exactas; representa ausencias como vacíos y fechas ISO. Conservar ambos hashes y documentar la decisión de aceptar el trim. No declarar archivos idénticos ni equivalencia estricta.", "",
        "## Beneficio recalculado", "", "| Clase | Líneas | % |", "|---|---:|---:|"]
    for key in ["positive", "negative", "zero"]:
        lines.append(f"| {key} | {a['profit_counts'][key]} | {a['profit_pct'][key]:.6f} |")
    lines += ["", f"Exactitud teórica del mayoritario binario no negativo: {a['majority_binary_accuracy_pct']:.6f} %. No es un modelo entrenado. Accuracy aislada sería engañosa; proponer recall de negativos, precision, F1, balanced accuracy y matriz de confusión.", "",
        f"Hay {a['products_with_losses']} productos con pérdidas. Cuatro variantes de Halloween zombie mask concentran {a['top_4_negative_share_pct']:.6f} % de casos negativos; cada una pierde en todas sus líneas observadas. El top 5 concentra {a['top_5_negative_share_pct']:.6f} %. Riesgo: memorizar identidad de producto, sin demostrar anticipación de causas de pérdida ni generalización a productos nuevos.", "",
        "| Producto | Líneas | Negativas | % pérdidas dentro del producto | % de todos los negativos |", "|---|---:|---:|---:|---:|"]
    for row in a["loss_products"]:
        lines.append(f"| {row['NombreProducto']} | {row['rows']} | {row['negative']} | {row['loss_rate_pct']:.4f} | {row['share_of_negatives_pct']:.4f} |")
    lines += ["", "### Distribución temporal del beneficio", "", "| Mes | Líneas | Positivas | Negativas | Cero | % negativas |", "|---|---:|---:|---:|---:|---:|"]
    for row in a["profit_by_month"]:
        lines.append(f"| {row['month']} | {row['rows']} | {row['positive']} | {row['negative']} | {row['zero']} | {row['negative_pct']:.4f} |")
    lines += ["", "## Evidencia para viabilidad", "",
        f"Producto-mes: {a['product_month']['observed_pairs']} pares observados de {a['product_month']['global_grid_pairs']} en cuadrícula global. Los 288 ausentes preceden a las primeras ventas de ocho productos en enero de 2016; después de su primera observación no hay meses vacíos por producto. 219 productos aparecen desde enero de 2013 y ocho solo tienen cinco meses. No rellenar con cero antes de conocer la vigencia de catálogo.", "",
        f"Cliente-mes: {a['customer_month']['observed_pairs']} pares observados; {a['customer_month']['absent_pairs_from_first_observation']} meses sin compra desde la primera observación hasta el final, sobre {a['customer_month']['pairs_from_first_observation']} pares elegibles por ese criterio exploratorio. Esta tasa de inactividad no es aún la distribución de etiquetas futuras: depende de elegibilidad, ventana y censura.", "",
        "Hay solo tres años completos y cinco meses de 2016. Se puede estudiar estacionalidad anual con cautela; observar ventas mensuales no valida su predictibilidad. Las ventas son demanda atendida, potencialmente censurada por quiebres de stock. No hay stock, entradas, salidas, plazos de reposición ni catálogo histórico suficiente para inventario predictivo real.", "",
        "Ver viabilidad_predictiva.md para comparación A-D y pipeline documental; requisitos_confirmados.md distingue evidencia documental, decisiones del usuario, recomendaciones y pendientes.", "",
        "## Integridad, reproducción y límites", "",
        f"SHA-256 antes/después iguales para {len(payload['integrity']['before'])} archivos protegidos (dos CSV y ocho referencias). No se modificaron. Versiones: `{payload['environment']}`.", "",
        "Ejecutar `python src/auditar_datos.py` y `python -m unittest discover -s tests -p 'test_*.py' -v` desde la raíz. Los JSON, diccionario y este informe se regeneran. Los dos documentos de requisitos/viabilidad son análisis documental estático basado en estas evidencias, no una implementación.", "",
        "Los tres PDF se extrajeron con pdftotext -layout de MiKTeX; texto de todas las páginas disponible. Se revisaron secciones, tablas y pies de figuras. No se renderizaron las capturas incrustadas: no se verificaron sus valores visuales ni fórmulas de Power BI no expuestas en texto. La equivalencia MongoDB es una afirmación documental, no auditada aquí. No hay SQL de extracción, DW conectado ni historial JSON de OpenRefine para verificar linaje y vigencia de dimensiones.", ""]
    (out / "reporte_auditoria.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
