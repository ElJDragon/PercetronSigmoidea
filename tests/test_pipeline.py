"""Casos límite sintéticos y controles de integración de la preparación WWI."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.auditar_datos import BASE, sha256
from src.pipeline.agregacion import agregar_producto_mes, cargar_fuente
from src.pipeline.caracteristicas import Cobertura, FEATURES, construir_observaciones, estado_mes
from src.pipeline.particiones import separar
from src.pipeline.preprocesamiento import PreprocesadorComun


def fixture(values, start="2013-01", product="A"):
    rows = []
    for i, (month, units) in enumerate(zip(pd.period_range(start, periods=len(values), freq="M"), values)):
        rows.append({"InvoiceID": str(i+1), "InvoiceLineID": str(i+1), "Fecha": f"{month}-15T00:00:00Z",
                     "NombreProducto": product, "NombreCliente": "Cliente", "CantidadVendida": str(units),
                     "VentaSinImpuesto": str(units*2)})
    return pd.DataFrame(rows)


def build(raw, coverage=None, end="2013-06"):
    return construir_observaciones(agregar_producto_mes(raw), coverage or Cobertura(modo="academica"), fin=end)


class AgregacionTests(unittest.TestCase):
    def test_month_products_distinct_invoices_and_totals(self):
        raw = fixture([10, 20])
        extra = raw.iloc[[0]].copy()
        extra["InvoiceLineID"] = "99"
        extra["CantidadVendida"] = "5"
        extra["VentaSinImpuesto"] = "10"
        b = raw.iloc[[0]].copy()
        b["NombreProducto"] = "B"
        b["InvoiceLineID"] = "100"
        d = pd.concat([raw, extra, b], ignore_index=True)
        original = d.copy(deep=True)
        g = agregar_producto_mes(d)
        self.assertEqual(g.loc[("A", "2013-01"), "unidades_observadas"], 15)
        self.assertEqual(g.loc[("A", "2013-01"), "facturas_distintas"], 1)
        self.assertEqual(g.loc[("A", "2013-01"), "numero_lineas"], 2)
        self.assertEqual(g.loc[("B", "2013-01"), "unidades_observadas"], 10)
        self.assertEqual(g.loc[("A", "2013-02"), "ingresos_sin_impuesto"], 40)
        self.assertEqual(g.unidades_observadas.sum(), 45)
        pd.testing.assert_frame_equal(original, d)

    def test_duplicate_lines_and_invalid_dates_fail_without_dropping(self):
        raw = fixture([10, 20])
        with self.assertRaises(ValueError):
            agregar_producto_mes(pd.concat([raw, raw.iloc[[0]]]))
        raw.loc[0, "Fecha"] = "no-fecha"
        with self.assertRaises(ValueError):
            agregar_producto_mes(raw)

    def test_hash_and_schema_mismatch_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"source.csv"
            fixture([10]).to_csv(path, index=False, sep=";")
            with self.assertRaisesRegex(ValueError, "Hash"):
                cargar_fuente(path, "incorrecto", [])
            with self.assertRaisesRegex(ValueError, "Esquema"):
                cargar_fuente(path, sha256(path), ["otro_esquema"])


class TemporalTests(unittest.TestCase):
    def test_median_tie_and_all_feature_formulas(self):
        _, obs = build(fixture([10, 30, 20, 20, 50, 60]))
        r = obs.set_index("mes_objetivo").loc["2013-04"]
        self.assertEqual(r.R_referencia, 20)
        self.assertEqual(r.y, 0)
        self.assertEqual([r.unidades_lag_1, r.unidades_lag_2, r.unidades_lag_3], [20, 30, 10])
        self.assertEqual(r.unidades_media_3, 20)
        self.assertAlmostEqual(r.unidades_std_3, np.std([20, 30, 10], ddof=0))
        self.assertEqual(r.cambio_unidades_1, -10)
        self.assertAlmostEqual(r.variacion_relativa_1, -1/3)
        self.assertEqual(r.tendencia_unidades_3, 5)
        self.assertEqual(r.facturas_lag_1, 1)
        self.assertEqual(r.facturas_media_3, 1)
        self.assertEqual(r.ingresos_lag_1, 40)
        self.assertEqual(r.ingresos_media_3, 40)
        self.assertEqual(r.precio_unitario_ponderado_lag_1, 2)
        self.assertEqual(r.clientes_distintos_lag_1, 1)
        self.assertEqual(r.unidades_por_factura_lag_1, 20)
        self.assertEqual(r.antiguedad_observada_meses, 3)
        self.assertEqual(r.dias_mes_objetivo, 30)
        self.assertAlmostEqual(r.mes_objetivo_sin, 1)
        self.assertAlmostEqual(r.mes_objetivo_cos, 0)

    def test_first_sale_not_a_prediction_and_three_consecutive_months(self):
        _, obs = build(fixture([10, 20, 30, 40, 50, 60]))
        self.assertNotIn("2013-01", obs.mes_objetivo.tolist())
        self.assertFalse(obs.iloc[0].elegible_para_prediccion)
        self.assertEqual(obs.loc[obs.elegible_para_prediccion, "mes_objetivo"].tolist(), ["2013-04", "2013-05", "2013-06"])
        self.assertEqual(estado_mes("2012-12", "2013-01", False, Cobertura()), "producto_todavia_no_observado")

    def test_complete_zero_month_is_not_dropped(self):
        raw = fixture([10, 20, 30, 40, 50, 60]).drop(index=[1, 3])
        cal, obs = build(raw)
        self.assertEqual(cal.set_index("mes").loc["2013-02", "estado_mes"], "mes_completo_sin_ventas")
        april = obs.set_index("mes_objetivo").loc["2013-04"]
        self.assertTrue(april.elegible_para_prediccion)
        self.assertEqual(april.Q_objetivo, 0)
        self.assertEqual(april.y, 0)
        self.assertTrue(pd.isna(april.variacion_relativa_1))
        self.assertEqual(april.base_variacion_cero, 1)
        may = obs.set_index("mes_objetivo").loc["2013-05"]
        self.assertTrue(pd.isna(may.precio_unitario_ponderado_lag_1))
        self.assertTrue(pd.isna(may.unidades_por_factura_lag_1))
        self.assertEqual(may.facturas_lag_1_cero, 1)

    def test_unknown_history_never_imputed_as_zero(self):
        cov = Cobertura(modo="academica", meses={"2013-02": "desconocida"})
        cal, obs = build(fixture([10, 20, 30, 40, 50, 60]), cov)
        r = obs.set_index("mes_objetivo").loc["2013-04"]
        self.assertFalse(r.elegible_para_prediccion)
        self.assertTrue(pd.isna(r.y))
        self.assertTrue(pd.isna(r.unidades_lag_1))
        self.assertEqual(cal.set_index("mes").loc["2013-02", "estado_mes"], "cobertura_desconocida_con_ventas")

    def test_unknown_target_retains_eligible_prediction_with_no_label(self):
        cov = Cobertura(modo="academica", meses={"2013-04": "desconocida"})
        _, obs = build(fixture([10, 20, 30, 40, 50, 60]), cov)
        r = obs.set_index("mes_objetivo").loc["2013-04"]
        self.assertTrue(r.elegible_para_prediccion)
        self.assertFalse(r.respuesta_disponible)
        self.assertTrue(pd.isna(r.y))
        self.assertTrue(pd.isna(r.Q_objetivo))
        self.assertEqual(r.R_referencia, 20)

    def test_future_unobserved_target_not_turned_into_zero(self):
        cov = Cobertura(modo="academica", inicio="2013-01", fin="2013-05")
        _, obs = build(fixture([10, 20, 30, 40, 50]), cov)
        r = obs.set_index("mes_objetivo").loc["2013-06"]
        self.assertTrue(r.elegible_para_prediccion)
        self.assertTrue(pd.isna(r.y))
        self.assertTrue(pd.isna(r.Q_objetivo))

    def test_explicit_coverage_does_not_certify_unspecified_months(self):
        cov = Cobertura(modo="explicita", meses={"2013-01": "completa"}, evidencia="fixture de cierre")
        self.assertEqual(cov.estado("2013-02"), "desconocida")
        _, obs = build(fixture([10, 20, 30, 40, 50, 60]), cov)
        self.assertFalse(obs.elegible_para_prediccion.any())

    def test_target_changes_only_response_at_same_cut(self):
        raw = fixture([10, 20, 30, 40, 50, 60])
        _, before = build(raw)
        modified = raw.copy()
        modified.loc[3, "CantidadVendida"] = "1"
        modified.loc[3, "VentaSinImpuesto"] = "2"
        _, after = build(modified)
        keys = FEATURES + ["R_referencia", "elegible_para_prediccion", "estado_elegibilidad", "observacion_id"]
        a = before.set_index("mes_objetivo").loc["2013-04"]
        b = after.set_index("mes_objetivo").loc["2013-04"]
        pd.testing.assert_series_equal(a[keys], b[keys])
        self.assertNotEqual(a.y, b.y)

    def test_future_changes_and_new_future_product_leave_past_unchanged(self):
        raw = fixture([10, 20, 30, 40, 50, 60])
        _, before = build(raw)
        raw.loc[5, "CantidadVendida"] = "99999"
        new = fixture([88], start="2013-06", product="Nuevo")
        new["InvoiceLineID"] = "999"
        _, after = build(pd.concat([raw, new], ignore_index=True))
        pd.testing.assert_frame_equal(before[before.mes_objetivo < "2013-06"], after[after.mes_objetivo < "2013-06"])

    def test_stable_permuted_input_and_aggregation_money(self):
        raw = fixture([10, 20, 30, 40, 50, 60])
        cal, obs = build(raw)
        cal2, obs2 = build(raw.sample(frac=1, random_state=57))
        pd.testing.assert_frame_equal(cal, cal2)
        pd.testing.assert_frame_equal(obs, obs2)

    def test_partition_boundaries_and_new_products(self):
        old = fixture(list(range(10, 27)), "2014-01")
        new = fixture([10, 20, 30, 40, 50], "2016-01", product="Nuevo")
        new.InvoiceLineID = [f"N{i}" for i in range(5)]
        _, obs = build(pd.concat([old, new]), end="2016-05")
        sets = separar(obs)
        self.assertTrue(all(set(sets[a].observacion_id).isdisjoint(sets[b].observacion_id) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]))
        recent = obs[(obs.producto_provisional == "Nuevo") & obs.elegible_para_prediccion]
        self.assertEqual(recent.mes_objetivo.tolist(), ["2016-04", "2016-05"])
        self.assertTrue((recent.particion == "test").all())
        january = obs[(obs.producto_provisional == "A") & (obs.mes_objetivo == "2015-01")].iloc[0]
        self.assertEqual(january.particion, "val")
        self.assertEqual(january.historia_meses, "2014-12|2014-11|2014-10")


class PreprocesamientoTests(unittest.TestCase):
    def frame(self):
        # Fixture numérica de transformación, no conclusiones WWI.
        return pd.DataFrame({f: [1., 3., 5., 9.] for f in FEATURES})

    def test_fit_only_train_no_labels_or_other_sets_and_zscore(self):
        x = self.frame()
        with self.assertRaisesRegex(ValueError, "train"):
            PreprocesadorComun().fit(x, particion="test")
        pre = PreprocesadorComun().fit(x, particion="train")
        z = pre.transform(x)
        np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-14)
        state = pre.to_dict()
        pre.transform(x*10000)
        self.assertEqual(state, pre.to_dict())
        with self.assertRaises(ValueError):
            pre.fit(x, particion="train")

    def test_imputation_constants_and_all_missing_only_train(self):
        x = self.frame()
        x["base_variacion_cero"] = 0.
        x["variacion_relativa_1"] = [1., np.nan, 3., 9.]
        x["unidades_por_factura_lag_1"] = np.nan
        pre = PreprocesadorComun().fit(x, particion="train")
        self.assertIn("base_variacion_cero", pre.estado["discarded"])
        self.assertIn("unidades_por_factura_lag_1", pre.estado["discarded"])
        self.assertEqual(pre.estado["parameters"]["variacion_relativa_1"]["mediana"], 3.)
        self.assertTrue(np.isfinite(pre.transform(x)).all())
        # Constante descartada no reaparece aunque varíe en transform/test.
        x["base_variacion_cero"] = [0, 1, 0, 1]
        self.assertEqual(pre.transform(x).shape[1], len(FEATURES)-2)

    def test_essential_nan_infinite_or_unauthorized_categories_block(self):
        x = self.frame()
        for invalid in [x.assign(unidades_lag_1=np.nan), x.assign(unidades_lag_1=np.inf), x.assign(NombreProducto="A")]:
            with self.assertRaises(ValueError):
                PreprocesadorComun().fit(invalid, particion="train")
        pre = PreprocesadorComun().fit(x, particion="train")
        self.assertEqual(pre.estado["categorical_features"], [])
        self.assertEqual(pre.estado["category_vocabulary"], {})

    def test_serialization_transform_only_and_column_order(self):
        x = self.frame()
        pre = PreprocesadorComun().fit(x, particion="train")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"state.json"
            pre.save(path)
            loaded = PreprocesadorComun.load(path)
            np.testing.assert_array_equal(pre.transform(x), loaded.transform(x[FEATURES[::-1]]))
            self.assertEqual(pre.to_dict(), loaded.to_dict())

    def test_altered_test_distribution_cannot_change_train_parameters(self):
        raw = fixture(list(range(10, 51)), "2013-01")
        _, original = build(raw, end="2016-05")
        raw.loc[raw.Fecha.str[:7] >= "2016-01", "CantidadVendida"] = "999999"
        _, changed = build(raw, end="2016-05")
        train = separar(original)["train"]
        train2 = separar(changed)["train"]
        pre = PreprocesadorComun().fit(train[FEATURES], particion="train")
        pre2 = PreprocesadorComun().fit(train2[FEATURES], particion="train")
        self.assertEqual(pre.to_dict(), pre2.to_dict())
        np.testing.assert_array_equal(pre.transform(train[FEATURES]), pre2.transform(train2[FEATURES]))


class IntegracionWWITests(unittest.TestCase):
    def test_repeated_exports_identical_and_unknown_coverage_preserves_previous(self):
        from src.pipeline.construir_dataset import run
        if not (BASE / "resultados/fase_03/manifiesto.json").exists():
            self.skipTest("Ejecutar primero el pipeline WWI")
        # Reproducción completa en destinos temporales: no sobrescribe la entrega.
        with tempfile.TemporaryDirectory(dir=BASE/"tests") as tmp:
            outputs = {"output_dir": Path(tmp)/"datos", "report_dir": Path(tmp)/"resultados"}
            with contextlib.redirect_stdout(io.StringIO()):
                first = run(Cobertura(modo="academica"), **outputs)
                manifest_path = outputs["report_dir"] / "manifiesto.json"
                digest = sha256(manifest_path)
                second = run(Cobertura(modo="academica"), **outputs)
            self.assertEqual(first, second)
            self.assertEqual(digest, sha256(manifest_path))
            for path, expected in first["artifact_sha256"].items():
                self.assertEqual(sha256(BASE/path), expected)
            with self.assertRaisesRegex(ValueError, "Sin train conocido"):
                run(Cobertura(), **outputs)
            self.assertEqual(digest, sha256(manifest_path))

    def test_saved_arrays_alignment_counts_finitude_and_integrity(self):
        manifest_path = BASE / "resultados/fase_03/manifiesto.json"
        if not manifest_path.exists():
            self.skipTest("Ejecutar primero el pipeline WWI")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        folder = BASE / "datos/Procesados/fase_03"
        obs = pd.read_csv(folder/"observaciones.csv", float_precision="round_trip")
        all_keys = set()
        for split, n in {"train":4599, "val":2628, "test":1111}.items():
            x = np.load(folder/f"X_{split}.npy", allow_pickle=False)
            y = np.load(folder/f"y_{split}.npy", allow_pickle=False)
            meta = pd.read_csv(folder/f"metadatos_{split}.csv")
            self.assertEqual(x.shape, (n, manifest["features_after"]))
            self.assertEqual(x.dtype, np.dtype("float64"))
            self.assertEqual(y.dtype, np.dtype("int8"))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(np.isin(y, [0, 1]).all())
            self.assertTrue(all_keys.isdisjoint(meta.observacion_id))
            all_keys.update(meta.observacion_id)
            expected = obs[obs.particion.eq(split) & obs.elegible_para_prediccion & obs.respuesta_disponible]
            self.assertEqual(meta.observacion_id.tolist(), expected.observacion_id.tolist())
            np.testing.assert_array_equal(y, expected.y.to_numpy(np.int8))
            np.testing.assert_array_equal(x, PreprocesadorComun.load(folder/"preprocesador.json").transform(expected[FEATURES]))
        for relative, digest in manifest["integrity"]["before"].items():
            self.assertEqual(sha256(BASE/relative), digest)
        for relative, digest in manifest["artifact_sha256"].items():
            self.assertEqual(sha256(BASE/relative), digest)
        pre = PreprocesadorComun.load(folder/"preprocesador.json")
        # Auditoría de ajuste sin consultar etiquetas de test.
        train = obs[obs.particion.eq("train") & obs.elegible_para_prediccion & obs.respuesta_disponible]
        np.testing.assert_allclose(pre.transform(train[FEATURES]).mean(axis=0), 0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
