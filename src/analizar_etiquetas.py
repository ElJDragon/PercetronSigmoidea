"""Fase 2: exploración focalizada de etiquetas; no exporta filas predictivas.

python -m src.analizar_etiquetas
Se excluyen cantidades de 2016 antes de convertirlas o agregarlas.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.auditar_datos import BASE, sha256


def load_development(path):
    parts = []
    used = ["NombreProducto", "Fecha", "CantidadVendida"]
    for chunk in pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=used,
                             dtype=str, keep_default_na=False, chunksize=25000):
        # El parser debe recorrer el CSV, pero no interpreta ni agrega Q de test.
        chunk = chunk.loc[chunk.Fecha.str[:7] < "2016-01"].copy()
        if chunk.empty:
            continue
        chunk["month"] = pd.to_datetime(chunk.Fecha, utc=True, format="ISO8601").dt.strftime("%Y-%m")
        chunk["units"] = pd.to_numeric(chunk.CantidadVendida, errors="raise")
        if not chunk.units.ge(0).all():
            raise ValueError("Revisar cantidades negativas antes de formalizar Q")
        parts.append(chunk[["NombreProducto", "month", "units"]])
    if not parts:
        raise ValueError("Sin observaciones de desarrollo")
    return pd.concat(parts).groupby(["NombreProducto", "month"]).units.sum().sort_index()


def exploratory_cases(monthly, history=3):
    """Agregados temporales en memoria; no es el pipeline definitivo de fase 3."""
    rows = []
    for product, values in monthly.groupby(level=0):
        s = values.droplevel(0)
        periods = pd.period_range(s.index.min(), s.index.max(), freq="M").astype(str)
        # Hueco posterior a primera venta = desconocido, nunca cero implícito.
        s = s.reindex(periods)
        past = s.shift(1)
        ref = past.rolling(history, min_periods=history).median()
        valid = ref.notna() & s.notna()
        for month in s.index[valid]:
            if month >= "2016-01":
                raise ValueError("Test excluido del estudio de etiquetas")
            rows.append({"product": product, "month": month, "q": int(s.loc[month]),
                         "reference": float(ref.loc[month]),
                         "split": "train" if month <= "2014-12" else "validation"})
    return pd.DataFrame(rows)


def summarize(cases, label):
    result = {}
    for split in ["train", "validation"]:
        d = cases[cases.split == split].copy()
        y = label.loc[d.index].astype(int)
        d["positive"] = y
        products = d.groupby("product").positive.agg(["sum", "count", "mean"])
        months = d.groupby("month").positive.agg(["sum", "count", "mean"])
        result[split] = {
            "n": len(d), "products": len(products), "positive": int(y.sum()),
            "negative": int(len(y) - y.sum()), "positive_pct": float(y.mean() * 100),
            "all_positive_products": int(products["mean"].eq(1).sum()),
            "all_negative_products": int(products["mean"].eq(0).sum()),
            "product_positive_pct_quantiles": {str(k): float(v * 100) for k, v in products["mean"].quantile([0, .25, .5, .75, 1]).items()},
            "monthly": {str(m): {"n": int(v["count"]), "positive": int(v["sum"]), "positive_pct": float(v["mean"] * 100)} for m, v in months.iterrows()},
            "per_product": {str(p): {"n": int(v["count"]), "positive": int(v["sum"]), "positive_pct": float(v["mean"] * 100)} for p, v in products.iterrows()},
        }
    result["validation_minus_train_pp"] = result["validation"]["positive_pct"] - result["train"]["positive_pct"]
    return result


def eligibility_from_phase1(a, history):
    cohorts = a["product_month"]["first_observed_by_month"]
    counts = {"train": 0, "validation": 0, "test": 0}
    monthly = {}
    for period in pd.period_range("2013-01", "2016-05", freq="M"):
        n = sum(int(size) for first, size in cohorts.items() if period.ordinal - pd.Period(first, freq="M").ordinal >= history)
        month = str(period)
        split = "train" if month <= "2014-12" else "validation" if month <= "2015-12" else "test"
        counts[split] += n
        monthly[month] = n
    return {"minimum_history": history, "counts": counts, "monthly": monthly,
            "assumption": "Meses completos desde primera venta, sin huecos posteriores, según evidencia fase 1; no implica vigencia comercial probada. Test se cuenta con metadatos, sin consultar cantidades."}


def run():
    audit_path = BASE / "resultados/auditoria/perfilado_datos.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    paths = sorted((BASE / "datos").rglob("datos_limpios.csv"))
    if len(paths) != 1:
        raise ValueError("Fuente principal ambigua o ausente")
    source = paths[0]
    # También se preservan entregables fase 1 y referencias; no se reperfilan CSV.
    protected = [p for folder in ["datos/originales", "referencias", "resultados/auditoria"] for p in (BASE / folder).rglob("*") if p.is_file()]
    before = {str(p.relative_to(BASE)): sha256(p) for p in protected}
    relative = str(source.relative_to(BASE))
    if before[relative] != audit["integrity"]["after"][relative]:
        raise ValueError("Fuente distinta de la auditada en fase 1")
    if audit["business_analysis"]["product_month"]["absent_pairs_from_first_observation"] != 0:
        raise ValueError("Revisar conteos de elegibilidad: hay huecos no contemplados")
    monthly = load_development(source)
    cases = exploratory_cases(monthly, 3)
    train = cases[cases.split == "train"]
    threshold = float(train.q.median())
    a = summarize(cases, cases.q > threshold)
    b = summarize(cases, cases.q > cases.reference)
    a["threshold"] = threshold
    a["ties"] = {s: int(cases.loc[cases.split == s, "q"].eq(threshold).sum()) for s in ["train", "validation"]}
    b["ties"] = {s: int((cases.loc[cases.split == s, "q"] == cases.loc[cases.split == s, "reference"]).sum()) for s in ["train", "validation"]}
    history_options = {str(k): eligibility_from_phase1(audit["business_analysis"], k) for k in [1, 3, 6, 12]}
    for split in ["train", "validation"]:
        if len(cases[cases.split == split]) != history_options["3"]["counts"][split]:
            raise AssertionError("Conteos focalizados no concuerdan con cobertura fase 1")
    after = {str(p.relative_to(BASE)): sha256(p) for p in protected}
    if before != after:
        raise RuntimeError("Cambió un archivo protegido")
    result = {"purpose": "Solo estadística de etiquetas de desarrollo; sin predicciones, modelos ni matrices finales.",
              "source": relative, "source_sha256": before[relative],
              "development_month_range": ["2013-01", "2015-12"],
              "test_quantities_used": False,
              "development_monthly_pairs": len(monthly),
              "eligibility": history_options, "A_global_median": a, "B_trailing_3_median": b,
              "integrity": {"preserved": True, "before": before, "after": after},
              "environment": {"pandas": pd.__version__, "numpy": np.__version__}}
    out = BASE / "resultados/fase_02"
    out.mkdir(parents=True, exist_ok=True)
    (out / "exploracion_etiquetas.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"global_threshold": threshold,
                      "A": {s: {k: a[s][k] for k in ["n", "positive", "negative", "positive_pct", "all_positive_products", "all_negative_products"]} for s in ["train", "validation"]},
                      "B": {s: {k: b[s][k] for k in ["n", "positive", "negative", "positive_pct", "all_positive_products", "all_negative_products"]} for s in ["train", "validation"]},
                      "eligibility": {k: v["counts"] for k, v in history_options.items()},
                      "ties": {"A": a["ties"], "B": b["ties"]}, "protected": len(before)}, ensure_ascii=False))


if __name__ == "__main__":
    # Invocación documentada como módulo para resolver imports del proyecto.
    run()
