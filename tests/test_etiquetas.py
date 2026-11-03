"""Controles temporales del análisis focalizado; no entrenan clasificadores."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.analizar_etiquetas import (eligibility_from_phase1, exploratory_cases,
                                   load_development)


def series(values, start="2013-01"):
    months = pd.period_range(start, periods=len(values), freq="M").astype(str)
    index = pd.MultiIndex.from_product([["Producto"], months], names=["NombreProducto", "month"])
    return pd.Series(values, index=index)


class EtiquetasTests(unittest.TestCase):
    def test_reference_excludes_target_and_future(self):
        original = exploratory_cases(series([10, 20, 30, 40, 50, 60]))
        changed = exploratory_cases(series([10, 20, 30, 999, 888, 777]))
        first = original.iloc[0]
        self.assertEqual(first.month, "2013-04")
        self.assertEqual(first.reference, 20)
        self.assertEqual(changed.iloc[0].reference, first.reference)
        # A partir del siguiente corte, una observación anterior sí puede influir.
        self.assertEqual(changed.iloc[1].reference, 30)

    def test_missing_history_not_silently_zero(self):
        monthly = series([10, 20, 30, 40, 50, 60]).drop(("Producto", "2013-02"))
        cases = exploratory_cases(monthly)
        self.assertEqual(cases.month.tolist(), ["2013-06"])

    def test_development_loader_does_not_convert_test_quantities(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            path.write_text("NombreProducto;Fecha;CantidadVendida\nA;2015-12-01T00:00:00Z;10\nA;2016-01-01T00:00:00Z;TEST_NO_INTERPRETAR\n", encoding="utf-8")
            result = load_development(path)
            self.assertEqual(result.to_dict(), {("A", "2015-12"): 10})

    def test_target_month_sets_partition_and_prior_split_history_is_allowed(self):
        cases = exploratory_cases(series([10, 20, 30, 40, 50], "2014-09"))
        self.assertEqual(cases.month.tolist(), ["2014-12", "2015-01"])
        self.assertEqual(cases.split.tolist(), ["train", "validation"])
        self.assertEqual(cases.iloc[1].reference, 30)

    def test_three_months_needed_for_new_cohort(self):
        audit = {"product_month": {"first_observed_by_month": {"2013-01": 219, "2016-01": 8}}}
        e = eligibility_from_phase1(audit, 3)
        self.assertEqual(e["counts"], {"train": 4599, "validation": 2628, "test": 1111})
        self.assertEqual(e["monthly"]["2016-03"], 219)
        self.assertEqual(e["monthly"]["2016-04"], 227)


if __name__ == "__main__":
    unittest.main()
