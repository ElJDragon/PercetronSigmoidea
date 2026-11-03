"""Métricas derivadas e identidades de agregación, con fixtures solo para pruebas."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.auditar_datos import BASE,sha256
from src.evaluation.metricas import metricas_binarias
from src.experiments.evaluacion_final import METHODS
from src.experiments.analisis_estadistico import (derivar_metricas,descomponer_ba,sensibilidad_temporal,verificar_entradas)


class MetricasDerivadasTests(unittest.TestCase):
    def test_rates_specificity_and_positive_predictions(self):
        m=derivar_metricas(metricas_binarias([0,0,0,1,1],[0,1,0,1,0]))
        self.assertEqual([m[k] for k in ["TN","FP","FN","TP"]],[2,1,1,1])
        self.assertAlmostEqual(m["specificity"],2/3)
        self.assertAlmostEqual(m["false_positive_rate"],1/3)
        self.assertEqual(m["false_negative_rate"],.5)
        self.assertEqual(m["predicted_positive_count"],2)
        self.assertEqual(m["predicted_positive_proportion"],.4)
        self.assertAlmostEqual(m["false_positive_rate"]+m["specificity"],1)
        self.assertAlmostEqual(m["false_negative_rate"]+m["recall"],1)

    def test_absent_class_preserves_unestimable_rates(self):
        m=derivar_metricas(metricas_binarias([1,1],[0,1]))
        self.assertIsNone(m["specificity"])
        self.assertIsNone(m["false_positive_rate"])
        self.assertIsNone(m["balanced_accuracy"])
        m=derivar_metricas(metricas_binarias([0,0],[0,1]))
        self.assertIsNone(m["false_negative_rate"])

    def test_matrix_N_mismatch_rejected(self):
        m=metricas_binarias([0,1],[0,1]); m["n"]=3
        with self.assertRaises(ValueError): derivar_metricas(m)

    def test_pooled_BA_class_weight_identity_and_macro_difference(self):
        # Mes A: 1 positivo, 3 negativos, predice siempre positivo.
        # Mes B: 3 positivos, 1 negativo, predice siempre negativo.
        a=metricas_binarias([1,0,0,0],[1,1,1,1]); b=metricas_binarias([1,1,1,0],[0,0,0,0])
        pooled=metricas_binarias([1,0,0,0,1,1,1,0],[1,1,1,1,0,0,0,0])
        monthly={"a":{n:a for n in METHODS},"b":{n:b for n in METHODS}}
        d=descomponer_ba(monthly,{n:pooled for n in METHODS})["logistica"]
        self.assertEqual(d["pooled_BA"],.25)
        self.assertEqual(d["unweighted_monthly_BA"],.5)
        self.assertEqual(d["gap_pp"],-25)
        self.assertEqual(sum(t["gap_contribution_pp"] for t in d["terms"]),-25)

    def test_missing_month_class_does_not_fabricate_decomposition(self):
        m=metricas_binarias([1,1],[0,1])
        d=descomponer_ba({"a":{n:m for n in METHODS}},{n:m for n in METHODS})
        self.assertFalse(d["perceptron"]["estimable"])

    def test_monthly_omissions_are_grouped_and_do_not_modify_data(self):
        import pandas as pd
        frame=pd.DataFrame({"mes_objetivo":["a","a","b","b","c","c"],"y_real":[0,1,0,1,0,1]})
        for name in METHODS: frame[f"pred_{name}"]=[0,1,1,0,0,1]
        original=frame.copy(deep=True)
        result=sensibilidad_temporal(frame)
        self.assertEqual(set(result["blocks"]),{"a","b","c"})
        for month,value in result["blocks"].items():
            self.assertEqual(value["n"],4)
            self.assertNotIn(month,value["retained_months"])
        pd.testing.assert_frame_equal(frame,original)
        self.assertTrue(result["official_full_test_unchanged"])

    def test_month_omission_single_class_is_not_an_estimated_BA(self):
        import pandas as pd
        data=pd.DataFrame({"mes_objetivo":["a","a","b","b"],"y_real":[0,0,1,1]})
        for name in METHODS: data[f"pred_{name}"]=[0,0,1,1]
        s=sensibilidad_temporal(data)
        self.assertIsNone(s["BA_ranges"]["logistica"]["min"])
        self.assertEqual(s["BA_ranges"]["logistica"]["estimable_blocks"],0)


class AnalisisIntegracionTests(unittest.TestCase):
    def test_analysis_reads_predictions_without_model_calls(self):
        from src.models.perceptron import PerceptronSimple
        from src.models.logistic_regression import LogisticRegressionManual
        from src.pipeline.preprocesamiento import PreprocesadorComun
        from src.visualization.pca import PCA2D
        with patch.object(PerceptronSimple,"fit",side_effect=AssertionError("No fit")), patch.object(PerceptronSimple,"predict",side_effect=AssertionError("No nuevas predicciones")), patch.object(LogisticRegressionManual,"fit",side_effect=AssertionError("No fit")), patch.object(LogisticRegressionManual,"predict",side_effect=AssertionError("No nuevas predicciones")), patch.object(PreprocesadorComun,"fit",side_effect=AssertionError("No escalado")), patch.object(PCA2D,"fit",side_effect=AssertionError("No PCA")):
            _,pred,_,_=verificar_entradas()
        self.assertEqual(len(pred),1111)

    def test_inputs_reproduce_saved_metrics_and_agreements(self):
        frozen,pred,evaluation,agreements=verificar_entradas()
        self.assertEqual(len(pred),1111)
        self.assertEqual(agreements["global"]["both_correct"],554)
        self.assertEqual(agreements["global"]["both_wrong"],374)
        self.assertEqual(agreements["global"]["only_perceptron_correct"],45)
        self.assertEqual(agreements["global"]["only_logistic_correct"],138)
        for name in METHODS:
            total=np.sum([np.asarray(rows[name]["confusion_matrix"]) for rows in evaluation["monthly"].values()],axis=0)
            np.testing.assert_array_equal(total,evaluation["global"][name]["confusion_matrix"])

    def test_wrong_hash_blocks_analysis(self):
        with patch("src.experiments.analisis_estadistico.sha256",return_value="incorrecto"):
            with self.assertRaises(ValueError): verificar_entradas()

    def test_saved_analysis_matches_predictions_and_integrity(self):
        folder=BASE/"resultados/fase_08"
        if not (folder/"manifiesto.json").exists(): self.skipTest("Ejecutar análisis primero")
        manifest=json.loads((folder/"manifiesto.json").read_text(encoding="utf-8"))
        _,_,e,_=verificar_entradas()
        saved=json.loads((folder/"metricas_derivadas.json").read_text(encoding="utf-8"))
        self.assertEqual(saved,{n:derivar_metricas(m) for n,m in e["global"].items()})
        decomp=json.loads((folder/"analisis_temporal.json").read_text(encoding="utf-8"))["BA_decomposition"]
        self.assertEqual(decomp,descomponer_ba(e["monthly"],e["global"]))
        for path,digest in manifest["integrity"]["before"].items(): self.assertEqual(sha256(BASE/path),digest,path)
        for path,digest in manifest["artifacts_sha256"].items(): self.assertEqual(sha256(BASE/path),digest,path)
        self.assertFalse(manifest["new_predictions_generated"])


if __name__=="__main__": unittest.main()
