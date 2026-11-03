"""Control matemático, selección y aislamiento de test. Fixtures no empresariales."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.auditar_datos import BASE, sha256
from src.evaluation.metricas import clave_seleccion, metricas_binarias
from src.experiments.entrenar_perceptron import load_development, select_experiments
from src.models.perceptron import PerceptronSimple


class PerceptronTests(unittest.TestCase):
    def test_zero_initialization_tie_and_no_update_when_correct(self):
        model = PerceptronSimple(max_epochs=1).fit([[2., 3., 4.]], [1])
        np.testing.assert_array_equal(model.weights_, [0, 0, 0])
        self.assertEqual(model.bias_, 0)
        np.testing.assert_array_equal(model.decision_function([[99, 88, 77]]), [0])
        np.testing.assert_array_equal(model.predict([[99, 88, 77]]), [1])
        self.assertEqual(model.history_[0]["updates"], 0)

    def test_exact_manual_update_known_weights_and_bias(self):
        initial = np.array([.5, -.5])
        model = PerceptronSimple(learning_rate=.1, max_epochs=1, initial_weights=initial, initial_bias=0).fit([[2., 1.]], [0])
        # z=.5 -> pred=1, e=-1; w=[.5,-.5]-.1*[2,1], b=-.1.
        np.testing.assert_allclose(model.weights_, [.3, -.6], atol=0, rtol=1e-15)
        self.assertEqual(model.bias_, -.1)
        self.assertEqual(model.history_[0]["online_errors"], 1)
        self.assertEqual(model.history_[0]["updates"], 1)
        np.testing.assert_array_equal(initial, [.5, -.5])

    def test_separable_fixture_learns(self):
        x = np.array([[-2,-1], [-1,-2], [1,1], [2,1]], dtype=float)
        y = np.array([0,0,1,1])
        model = PerceptronSimple(max_epochs=25, seed=9).fit(x,y)
        np.testing.assert_array_equal(model.predict(x), y)
        self.assertTrue(any(h["online_errors"] == 0 for h in model.history_))

    def test_non_separable_xor_no_perfect_convergence_required(self):
        x = [[0,0], [0,1], [1,0], [1,1]]
        y = [0,1,1,0]
        model = PerceptronSimple(max_epochs=20, shuffle=False).fit(x,y)
        self.assertLess(metricas_binarias(y,model.predict(x))["accuracy"],1)
        self.assertEqual(len(model.history_),20)
        self.assertTrue(np.isfinite(model.weights_).all())

    def test_dynamic_dimensions_and_float64(self):
        for d in [1,2,7,23]:
            x = np.array([[-1]*d, [1]*d])
            model = PerceptronSimple(max_epochs=10).fit(x,[0,1])
            self.assertEqual(model.weights_.shape,(d,))
            self.assertEqual(model.weights_.dtype,np.dtype("float64"))
            np.testing.assert_array_equal(model.predict(x),[0,1])

    def test_reproducible_seed_reset_and_inputs_preserved(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(15,4))
        y = rng.integers(0,2,15)
        original = x.copy()
        a = PerceptronSimple(max_epochs=12,seed=3).fit(x,y)
        b = PerceptronSimple(max_epochs=12,seed=3).fit(x,y)
        np.testing.assert_array_equal(a.weights_,b.weights_)
        self.assertEqual(a.history_,b.history_)
        first = a.weights_.copy()
        a.fit(x,y)
        np.testing.assert_array_equal(a.weights_,first)
        np.testing.assert_array_equal(x,original)

    def test_validation_never_changes_training_update_trajectory(self):
        x = [[0,0],[0,1],[1,0],[1,1]]
        y = [0,1,1,0]
        a = PerceptronSimple(max_epochs=12,seed=5).fit(x,y,X_val=x,y_val=y)
        b = PerceptronSimple(max_epochs=12,seed=5).fit(x,y,X_val=x,y_val=[1,0,0,1])
        np.testing.assert_array_equal(a.last_weights_,b.last_weights_)
        self.assertEqual(a.last_bias_,b.last_bias_)
        self.assertEqual([h["online_errors"] for h in a.history_],[h["online_errors"] for h in b.history_])

    def test_restores_best_epoch_instead_of_last_and_history_is_real(self):
        x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
        y = np.array([0,1,1,0])
        model = PerceptronSimple(max_epochs=12,seed=5).fit(x,y,X_val=x,y_val=y)
        best = max(model.history_, key=lambda h:(h["val_balanced_accuracy"], h["val_f1"], -h["epoch"]))
        self.assertEqual(model.best_epoch_,best["epoch"])
        self.assertLess(model.best_epoch_,12)
        expected = PerceptronSimple(max_epochs=model.best_epoch_,seed=5).fit(x,y)
        np.testing.assert_array_equal(model.weights_,expected.weights_)
        self.assertEqual(model.bias_,expected.bias_)
        self.assertAlmostEqual(metricas_binarias(y,model.predict(x))["balanced_accuracy"],best["val_balanced_accuracy"])
        for i,h in enumerate(model.history_,1):
            self.assertEqual(h["epoch"],i)
            self.assertEqual(h["online_errors"],h["updates"])
            self.assertEqual(h["train_errors_end"],round((1-h["train_accuracy"])*len(y)))

    def test_invalid_configuration_dimensions_values_and_labels(self):
        for params in [{"learning_rate":0}, {"learning_rate":float("nan")}, {"learning_rate":1j},
                       {"max_epochs":0}, {"max_epochs":1.2}, {"seed":-1}, {"shuffle":"yes"}, {"initial_bias":float("inf")}]:
            with self.assertRaises(ValueError):
                PerceptronSimple(**params)
        for x,y in [([[1,2]],[2]), ([[np.nan,1]],[0]), ([[1+2j,1]],[0]), ([[1,2]],[0,1]), ([],[])]:
            with self.assertRaises(ValueError):
                PerceptronSimple().fit(x,y)
        with self.assertRaises(ValueError):
            PerceptronSimple(initial_weights=[0]).fit([[1,2]],[0])
        model = PerceptronSimple(max_epochs=1).fit([[1,2]],[1])
        with self.assertRaises(ValueError):
            model.predict([[1]])
        with self.assertRaises(ValueError):
            PerceptronSimple().predict([[1]])
        with self.assertRaises(ValueError):
            model.fit([[1,2]],[1],X_val=[[1,2]],y_val=[1])

    def test_persistence_exact_precision_and_feature_order(self):
        model = PerceptronSimple(max_epochs=4).fit([[1,2],[-2,-1]],[1,0],feature_names=["a","b"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.json"
            model.save(path)
            recovered = PerceptronSimple.load(path)
            np.testing.assert_array_equal(recovered.weights_,model.weights_)
            self.assertEqual(recovered.bias_,model.bias_)
            np.testing.assert_array_equal(recovered.decision_function([[3,4]],feature_names=["a","b"]),model.decision_function([[3,4]]))
            with self.assertRaises(ValueError):
                recovered.predict([[3,4]],feature_names=["b","a"])


class MetricasTests(unittest.TestCase):
    def test_manual_confusion_and_metrics(self):
        m = metricas_binarias([0,0,1,1],[0,1,0,1])
        self.assertEqual(m["confusion_matrix"],[[1,1],[1,1]])
        for name in ["accuracy","precision","recall","f1","balanced_accuracy"]:
            self.assertEqual(m[name],.5)

    def test_denominator_conventions_and_unestimable_balanced_accuracy(self):
        m = metricas_binarias([0,0],[0,0])
        self.assertEqual(m["precision"],0)
        self.assertEqual(m["f1"],0)
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["balanced_accuracy"])
        self.assertTrue(m["denominator_zero"]["precision"])
        self.assertTrue(m["denominator_zero"]["f1"])
        m = metricas_binarias([1,1],[0,0])
        self.assertEqual(m["recall"],0)
        self.assertIsNone(m["specificity"])
        with self.assertRaises(ValueError):
            clave_seleccion(m,1)

    def test_invalid_metrics_input_and_tiebreaks(self):
        for real,pred in [([],[]), ([0,1],[1]), ([0,2],[1,0]), ([0,1],[float("nan"),0])]:
            with self.assertRaises(ValueError):
                metricas_binarias(real,pred)
        m = metricas_binarias([0,1],[0,1])
        self.assertGreater(clave_seleccion(m,1,0),clave_seleccion(m,2,0))
        self.assertGreater(clave_seleccion(m,1,0),clave_seleccion(m,1,1))


class ExperimentTests(unittest.TestCase):
    def test_loader_never_loads_test_arrays_and_keeps_feature_order(self):
        original_load = np.load
        called = []
        def restricted(path,*args,**kwargs):
            called.append(Path(path).name)
            if "test" in Path(path).name:
                raise AssertionError("Entrenamiento abrió test")
            return original_load(path,*args,**kwargs)
        with patch("src.experiments.entrenar_perceptron.np.load",side_effect=restricted):
            dev = load_development()
        self.assertEqual(set(called),{"X_train.npy","y_train.npy","X_val.npy","y_val.npy"})
        names = json.loads((BASE/"datos/Procesados/fase_03/caracteristicas.json").read_text(encoding="utf-8"))["feature_names"]
        self.assertEqual(dev["feature_names"],names)
        self.assertEqual(dev["train"]["X"].shape[1],len(names))
        self.assertNotIn("test",dev)

    def test_selection_ignores_any_extra_test_data_and_is_reproducible(self):
        x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
        dev = {"feature_names":["a","b"], "train":{"X":x,"y":np.array([0,1,1,0])},
               "val":{"X":x,"y":np.array([0,1,1,0])}}
        plan = {"initial_bias":0.,"experiments":[{"id":"fixed","learning_rate":.01,"max_epochs":8,"seed":42,"shuffle":False},
                                                   {"id":"shuffled","learning_rate":.01,"max_epochs":8,"seed":42,"shuffle":True}]}
        a,aid,aruns = select_experiments(dev,plan)
        extra = copy.deepcopy(dev)
        extra["test"] = object()  # No operación válida: cualquier consulta semántica fallaría.
        b,bid,bruns = select_experiments(extra,plan)
        self.assertEqual(aid,bid)
        self.assertEqual(aruns,bruns)
        np.testing.assert_array_equal(a.weights_,b.weights_)

    def test_exported_WWI_model_reproduces_development_metrics_and_integrity(self):
        folder = BASE/"resultados/fase_04"
        if not (folder/"manifiesto.json").exists():
            self.skipTest("Ejecutar WWI después de superar primero las pruebas controladas")
        dev = load_development()
        model = PerceptronSimple.load(folder/"modelo_seleccionado.json")
        metrics = json.loads((folder/"metricas.json").read_text(encoding="utf-8"))
        for split in ["train","val"]:
            self.assertEqual(metricas_binarias(dev[split]["y"],model.predict(dev[split]["X"],feature_names=dev["feature_names"])),metrics[split])
        manifest = json.loads((folder/"manifiesto.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["test_evaluated"])
        for path,digest in manifest["integrity"]["before"].items():
            self.assertEqual(sha256(BASE/path),digest,path)
        for path,digest in manifest["artifacts_sha256"].items():
            self.assertEqual(sha256(BASE/path),digest,path)
        self.assertEqual(model.feature_names_,dev["feature_names"])
        self.assertEqual(len(model.weights_),20)


if __name__ == "__main__":
    unittest.main()
