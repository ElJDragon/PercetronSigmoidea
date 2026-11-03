"""Fixtures pequeñas: validan auditoría, nunca sustituyen datos WWI."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.auditar_datos import (compare_csv, detect_format, is_null, monetary_checks,
                               normalized, numeric_equal, profile, read_csv, sha256, BASE)


class AuditoriaTests(unittest.TestCase):
    def test_delimiters_read_count_columns_and_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            for delimiter in [";", ",", "\t", "|"]:
                path = Path(folder) / "fixture.csv"
                path.write_text(f"InvoiceLineID{delimiter}Marca\n1{delimiter}NULL\n2{delimiter}\n", encoding="utf-8-sig")
                original = sha256(path)
                raw, fmt = read_csv(path)
                self.assertEqual(fmt["delimiter"], delimiter)
                self.assertEqual(raw.shape, (2, 2))
                self.assertEqual(list(raw), ["InvoiceLineID", "Marca"])
                self.assertEqual(raw.Marca.tolist(), ["NULL", ""])
                self.assertEqual(original, sha256(path))

    def test_nulls(self):
        for value in ["NULL", " null ", "", "NA", "N/A", "None", "nan", "<null>", "\\N"]:
            self.assertTrue(is_null(value), value)
        self.assertFalse(is_null("Northwind"))

    def test_numeric_exactness_no_rounding(self):
        self.assertTrue(numeric_equal("13.00", "13"))
        self.assertTrue(numeric_equal("13,50", "13.5"))
        self.assertFalse(numeric_equal("13.500000000001", "13.5"))
        self.assertFalse(numeric_equal("10000000000000001", "10000000000000000"))
        self.assertFalse(numeric_equal("NaN", "NaN"))

    def test_duplicates_constants_invalid_dates(self):
        # Guardar dentro del workspace para comprobar rutas relativas del perfilador.
        with tempfile.TemporaryDirectory(dir=BASE / "tests") as folder:
            path = Path(folder) / "fixture.csv"
            path.write_text("InvoiceID;InvoiceLineID;Marca;Fecha;CantidadVendida\n1;1;A;2013-01-01;2\n1;1;A;2013-01-01;2\n2;2;A;imposible;-1\n", encoding="utf-8")
            raw, fmt = read_csv(path)
            result, _ = profile(path, raw, fmt)
            self.assertEqual(result["rows"], 3)
            self.assertEqual(result["raw_duplicate_excess"], 1)
            self.assertEqual(result["keys"]["InvoiceLineID"]["duplicate_excess"], 1)
            self.assertTrue(result["columns"]["Marca"]["constant_including_null"])
            self.assertEqual(result["columns"]["Fecha"]["invalid_dates"], 1)
            self.assertEqual(result["columns"]["CantidadVendida"]["negative"], 1)

    def test_comparison_alignment_formats_and_material_difference(self):
        a = pd.DataFrame({"InvoiceLineID": ["1", "2"], "Fecha": ["2013-01-01", "2013-02-01"],
                          "Marca": ["NULL", "A"], "PrecioUnitario": ["13.00", "15.5"]})
        b = pd.DataFrame({"InvoiceLineID": ["2", "1"], "Fecha": ["2013-02-01T00:00:00Z", "2013-01-01T00:00:00Z"],
                          "Marca": ["A", ""], "PrecioUnitario": ["15,5", "13"]})
        self.assertTrue(compare_csv(a, b)["equivalent"])
        b.loc[1, "PrecioUnitario"] = "13.000000000000001"
        self.assertFalse(compare_csv(a, b)["equivalent"])

    def test_trim_is_separate_from_strict_equivalence(self):
        a = pd.DataFrame({"InvoiceLineID": ["1", "2"], "NombreCliente": ["David ", "Otro"]})
        b = a.assign(NombreCliente=["David", "Otro"])
        result = compare_csv(a, b)
        self.assertFalse(result["equivalent"])
        self.assertTrue(result["equivalent_if_text_trim_accepted"])
        self.assertEqual(result["whitespace_only_differences"], 1)

    def test_duplicate_headers_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.csv"
            path.write_text("id;id\n1;2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Encabezados duplicados"):
                read_csv(path)

    def test_protected_wwi_sources_match_saved_hashes(self):
        # Integración ligera contra los archivos reales: no usa fixtures para conclusiones.
        import json
        report = BASE / "resultados" / "auditoria" / "perfilado_datos.json"
        if not report.exists():
            self.skipTest("Ejecute primero la auditoría para obtener el manifiesto de hashes")
        integrity = json.loads(report.read_text(encoding="utf-8"))["integrity"]
        self.assertEqual(integrity["before"], integrity["after"])
        for relative, digest in integrity["before"].items():
            self.assertEqual(sha256(BASE / relative), digest, relative)

    def test_comparison_rejects_missing_duplicate_keys_and_category_changes(self):
        a = pd.DataFrame({"InvoiceLineID": ["1", "2"], "Marca": ["A", "B"]})
        for b in [a.iloc[:1], a.assign(InvoiceLineID=["1", "1"]),
                  a.assign(InvoiceLineID=["1", "NULL"]), a.assign(Marca=["a", "B"])]:
            self.assertFalse(compare_csv(a, b)["equivalent"])

    def test_money_tolerance_and_missing(self):
        d = pd.DataFrame({"CantidadVendida": [3., 2., 1., None], "PrecioUnitario": [.1, 5., 1., 2.],
                          "VentaSinImpuesto": [.3, 10.01, 1., 2.], "Impuesto": [.05, 1.5, .1, .3],
                          "VentaTotal": [.35, 11.51, 1.11, 2.3]})
        check = monetary_checks(d)
        self.assertEqual(check["cantidad_por_precio"]["violations"], 1)
        self.assertEqual(check["cantidad_por_precio"]["not_evaluated"], 1)
        self.assertEqual(check["base_mas_impuesto"]["violations"], 1)
        self.assertFalse(monetary_checks(pd.DataFrame())["base_mas_impuesto"]["available"])

    def test_null_normalization_does_not_change_input(self):
        raw = pd.DataFrame({"Marca": ["NULL", "", "A"], "Beneficio": ["-1,5", "0", "2.00"]})
        original = raw.copy(deep=True)
        d = normalized(raw)
        self.assertEqual(int(d.Marca.isna().sum()), 2)
        self.assertEqual(d.Beneficio.tolist(), [-1.5, 0., 2.])
        pd.testing.assert_frame_equal(raw, original)


if __name__ == "__main__":
    unittest.main()
