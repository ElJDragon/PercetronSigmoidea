"""Pruebas de evaluación congelada; fixtures sintéticos solo en pruebas unitarias."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.auditar_datos import BASE,sha256
from src.experiments.evaluacion_final import (METHODS,preflight,run,predecir,evaluar_comun,analizar_errores,
                                             metricas_con_conteos,verificar_congelacion)
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual
from src.pipeline.preprocesamiento import PreprocesadorComun
from src.visualization.pca import PCA2D


def fixtures():
    y=np.array([0,1,0,1],dtype=np.int8)
    predictions={"perceptron":np.array([0,1,1,0]),"logistica":np.array([0,1,0,1]),
                 "mayoritario":np.ones(4,dtype=np.int8),"historico":np.array([0,0,1,1])}
    meta=pd.DataFrame({"observacion_id":["a","b","c","d"],"producto_provisional":["known"]*2+["new"]*2,
                       "mes_objetivo":["2016-01"]*2+["2016-02"]*2})
    return y,predictions,meta


class EvaluacionUnitariaTests(unittest.TestCase):
    def test_common_keys_confusion_sum_and_counts(self):
        y,pred,meta=fixtures(); e=evaluar_comun(y,pred,meta,["known"])
        for name in METHODS:
            m=e["global"][name]
            self.assertEqual(m["n"],4)
            self.assertEqual(m["support"],{"0":2,"1":2})
            self.assertEqual(m["TN"]+m["FP"]+m["FN"]+m["TP"],4)
        self.assertEqual(e["global"]["perceptron"]["confusion_matrix"],[[1,1],[1,1]])
        self.assertEqual(e["global"]["logistica"]["confusion_matrix"],[[2,0],[0,2]])
        self.assertEqual(e["global"]["mayoritario"]["balanced_accuracy"],.5)

    def test_months_cohorts_and_empty_intersections(self):
        y,pred,meta=fixtures(); e=evaluar_comun(y,pred,meta,["known"])
        self.assertEqual(list(e["monthly"]),["2016-01","2016-02"])
        self.assertEqual(e["monthly"]["2016-01"]["perceptron"]["accuracy"],1)
        self.assertEqual(e["monthly"]["2016-02"]["perceptron"]["accuracy"],0)
        self.assertEqual(e["cohorts"]["nuevos"]["rows"],2)
        self.assertEqual(e["cohorts"]["nuevos"]["products"],1)
        self.assertIsNone(e["cohort_month"]["nuevos"]["2016-01"]["logistica"]["metrics"])

    def test_global_is_not_unweighted_monthly_mean_and_unestimable_ba(self):
        y=np.array([0,0,1,1,1]); p=np.array([0,0,1,0,0])
        meta=pd.DataFrame({"observacion_id":list("abcde"),"producto_provisional":["p"]*5,"mes_objetivo":["2016-01"]*3+["2016-02"]*2})
        e=evaluar_comun(y,{n:p.copy() for n in METHODS},meta,["p"])
        self.assertEqual(e["global"]["perceptron"]["accuracy"],.6)
        self.assertEqual(e["monthly_unweighted_means"]["perceptron"]["accuracy"]["mean"],.5)
        self.assertIsNone(e["monthly"]["2016-02"]["perceptron"]["balanced_accuracy"])
        self.assertEqual(e["monthly_unweighted_means"]["perceptron"]["balanced_accuracy"]["months_estimable"],1)

    def test_error_agreement_categories_are_exclusive(self):
        y,pred,meta=fixtures(); a=analizar_errores(y,pred,meta,["known"])
        g=a["global"]
        self.assertEqual(g["agreement"],2); self.assertEqual(g["disagreement"],2)
        self.assertEqual(g["both_correct"],2); self.assertEqual(g["both_wrong"],0)
        self.assertEqual(g["only_logistic_correct"],2)
        self.assertEqual(sum(g[k] for k in ["both_correct","both_wrong","only_perceptron_correct","only_logistic_correct"]),4)
        self.assertEqual(a["errors_by_method"]["perceptron"]["total"],2)

    def test_thresholds_and_history_baseline_use_no_target(self):
        # Estado de fixture asignado explícitamente, sin entrenar.
        p=PerceptronSimple(); p.weights_=np.array([1.,0]); p.bias_=0.; p.feature_names_=["a","b"]
        l=LogisticRegressionManual(); l.weights_=np.array([1.,0]); l.bias_=0.; l.feature_names_=["a","b"]
        x=np.array([[0.,0],[-1,0],[1,0]])
        history=pd.DataFrame({"unidades_lag_1":[10,2,5],"R_referencia":[10,1,6]})
        pred,score,prob=predecir({"perceptron":p,"logistica":l},x,history,1,["a","b"])
        np.testing.assert_array_equal(pred["perceptron"],[1,0,1])
        np.testing.assert_array_equal(pred["logistica"],[1,0,1])
        self.assertEqual(prob[0],.5)
        np.testing.assert_array_equal(pred["mayoritario"],[1,1,1])
        np.testing.assert_array_equal(pred["historico"],[0,1,0])

    def test_invalid_prediction_lengths_and_duplicate_keys_block(self):
        y,pred,meta=fixtures(); pred["logistica"]=np.array([0])
        with self.assertRaises(ValueError): evaluar_comun(y,pred,meta,["known"])
        y,pred,meta=fixtures(); meta.loc[1,"observacion_id"]="a"
        with self.assertRaises(ValueError): evaluar_comun(y,pred,meta,["known"])
        self.assertIsNone(metricas_con_conteos(np.array([]),np.array([]))["metrics"])


class IntegridadEvaluacionTests(unittest.TestCase):
    def test_preflight_cannot_open_test_responses_and_reproduces_validation(self):
        original=np.load; loaded=[]
        def guard(path,*args,**kwargs):
            loaded.append(Path(path).name)
            if Path(path).name=="y_test.npy": raise AssertionError("Se abrió y antes del gate")
            return original(path,*args,**kwargs)
        with patch("numpy.load",side_effect=guard), patch.object(PerceptronSimple,"fit",side_effect=AssertionError("No entrenar")), patch.object(LogisticRegressionManual,"fit",side_effect=AssertionError("No entrenar")), patch.object(PreprocesadorComun,"fit",side_effect=AssertionError("No reajustar")), patch.object(PCA2D,"fit",side_effect=AssertionError("No recalcular PCA")):
            p=preflight()
        self.assertIn("X_test.npy",loaded); self.assertNotIn("y_test.npy",loaded)
        self.assertEqual(p["X"].shape,(1111,20))
        self.assertFalse(p["verification"]["test_responses_loaded"])
        self.assertEqual(p["verification"]["validation_reproduced"]["perceptron"]["val"]["balanced_accuracy"],p["frozen"]["models"]["perceptron"]["development_metrics"]["val"]["balanced_accuracy"])

    def test_bad_hash_blocks_before_test_labels(self):
        with patch("src.experiments.evaluacion_final.sha256",return_value="incorrecto"):
            with self.assertRaises(ValueError): verificar_congelacion()

    def test_failed_attempt_logged_without_opening_labels(self):
        with tempfile.TemporaryDirectory(dir=BASE/"tests") as tmp:
            out=Path(tmp)/"evaluacion"
            with patch("src.experiments.evaluacion_final.preflight",side_effect=ValueError("fixture: integridad incorrecta")), patch("src.experiments.evaluacion_final.cargar_respuestas",side_effect=AssertionError("No abrir")) as load:
                with self.assertRaises(ValueError): run(output_dir=out)
                load.assert_not_called()
            events=[json.loads(line) for line in (out/"registro_ejecuciones.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["stage"],"intento_fallido")
            self.assertFalse(events[-1]["test_labels_opened"])
            self.assertFalse((out/"registro_apertura_test.json").exists())

    def test_opening_record_is_persisted_before_loading_answers(self):
        prepared={"freeze_hash":"fixture_hash","frozen":{},"verification":{"models_sha256":{},"matrix_sha256":{}}}
        with tempfile.TemporaryDirectory(dir=BASE/"tests") as tmp:
            out=Path(tmp)/"evaluacion"
            def inspect_record(*args,**kwargs):
                record=json.loads((out/"registro_apertura_test.json").read_text(encoding="utf-8"))
                self.assertEqual(record["integrity"],"approved_before_test_labels")
                stages=[json.loads(s)["stage"] for s in (out/"registro_ejecuciones.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertIn("integridad_y_validacion_aprobadas_antes_de_y_test",stages)
                raise ValueError("fixture: detener antes de respuestas reales")
            with patch("src.experiments.evaluacion_final.preflight",return_value=prepared), patch("src.experiments.evaluacion_final.cargar_respuestas",side_effect=inspect_record):
                with self.assertRaisesRegex(ValueError,"fixture"):
                    run(output_dir=out)

    def test_saved_results_alignment_and_frozen_integrity(self):
        folder=BASE/"resultados/fase_07"
        if not (folder/"manifiesto.json").exists(): self.skipTest("Ejecutar evaluación autorizada primero")
        manifest=json.loads((folder/"manifiesto.json").read_text(encoding="utf-8"))
        pred=pd.read_csv(folder/"predicciones_alineadas.csv",float_precision="round_trip")
        global_metrics=json.loads((folder/"metricas_globales.json").read_text(encoding="utf-8"))
        self.assertEqual(len(pred),1111)
        self.assertFalse(pred.observacion_id.duplicated().any())
        for name in METHODS:
            self.assertEqual(metricas_con_conteos(pred.y_real.to_numpy(),pred[f"pred_{name}"].to_numpy()),global_metrics[name])
        self.assertEqual(pred.loc[pred.cohorte.eq("nuevos"),"producto_provisional"].nunique(),8)
        for path,digest in manifest["integrity"]["before"].items(): self.assertEqual(sha256(BASE/path),digest,path)
        for path,digest in manifest["artifacts_sha256"].items(): self.assertEqual(sha256(BASE/path),digest,path)


if __name__=="__main__": unittest.main()
