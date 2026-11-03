"""Verificación analítica y numérica previa a WWI, sin clasificación en test."""
import json
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.auditar_datos import BASE, sha256
from src.evaluation.metricas import metricas_binarias
from src.models.logistic_regression import (LogisticRegressionManual, sigmoid, binary_cross_entropy_logits,
                                             loss_and_gradients, gradient_check, verificar_gradientes)


class MatematicaLogisticaTests(unittest.TestCase):
    def test_sigmoid_multidimensional_extreme_and_zero(self):
        z = np.array([[-1e300,-1000,0],[1e-9,1000,1e300]])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            p = sigmoid(z)
        self.assertEqual(p.shape,z.shape)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertEqual(float(sigmoid(0)),.5)
        np.testing.assert_allclose(sigmoid([-2,0,2]),[1/(1+np.exp(2)),.5,1/(1+np.exp(-2))])

    def test_cross_entropy_equivalence_and_extremes(self):
        y = np.array([0,1,1,0])
        z = np.array([-2.,.4,1.,-.6])
        p = sigmoid(z)
        expected = -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
        self.assertAlmostEqual(binary_cross_entropy_logits(y,z),expected,places=14)
        self.assertEqual(binary_cross_entropy_logits([0,1],[0,0]),np.log(2))
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            loss = binary_cross_entropy_logits([1,0],[-1e300,1e300])
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss/1e300,1)

    def test_central_differences_weights_and_bias(self):
        for d in [1,3,8]:
            rng = np.random.default_rng(d)
            report = gradient_check(rng.normal(size=(9,d)),np.array([0,1,0,1,1,0,0,1,0]),rng.normal(size=d),.25)
            self.assertTrue(report["passed"],report)
        cases = verificar_gradientes()
        self.assertTrue(all(c["passed"] for c in cases.values()))

    def test_known_gradient_and_exact_descent_step(self):
        x = np.array([[1.,2.],[-1.,1.]])
        y = np.array([1,0])
        loss, gw, gb = loss_and_gradients(x,y,[0.,0.],0.)
        np.testing.assert_array_equal(gw,[-.5,-.25])
        self.assertEqual(gb,0)
        self.assertEqual(loss,np.log(2))
        model = LogisticRegressionManual(learning_rate=.1,max_epochs=1).fit(x,y)
        np.testing.assert_allclose(model.weights_,[.05,.025],rtol=0,atol=1e-16)
        self.assertEqual(model.bias_,0)

    def test_known_nonzero_gradient_bias_update(self):
        x = np.array([[1.,-2.],[2.,1.],[1.,1.]])
        w = np.array([.1,-.2]); b=.3; y=np.array([1,1,0])
        _,gw,gb = loss_and_gradients(x,y,w,b)
        model = LogisticRegressionManual(learning_rate=.05,max_epochs=1,initial_weights=w,initial_bias=b).fit(x,y)
        np.testing.assert_array_equal(model.weights_,w-.05*gw)
        self.assertEqual(model.bias_,b-.05*gb)


class ModeloLogisticaTests(unittest.TestCase):
    def test_dynamic_dimensions_loss_decreases_and_probability(self):
        for d in [1,4,23]:
            x = np.array([[-2]*d,[-1]*d,[1]*d,[2]*d],dtype=float); y=[0,0,1,1]
            model = LogisticRegressionManual(learning_rate=.05,max_epochs=30).fit(x,y)
            self.assertEqual(model.weights_.shape,(d,))
            self.assertLess(model.history_[-1]["train_loss"],model.initial_train_loss_)
            self.assertTrue(all(b["train_loss"]<=a["train_loss"] for a,b in zip(model.history_,model.history_[1:])))
            np.testing.assert_array_equal(model.predict(x),y)
            p = model.predict_proba(x)
            self.assertEqual(p.shape,(4,))
            self.assertTrue(np.isfinite(p).all())

    def test_threshold_changes_labels_and_half_tie_positive(self):
        model = LogisticRegressionManual(max_epochs=1).fit([[0.],[0.]],[0,1])
        self.assertEqual(float(model.predict_proba([[0.]])[0]),.5)
        self.assertEqual(int(model.predict([[0.]],threshold=.5)[0]),1)
        self.assertEqual(int(model.predict([[0.]],threshold=.6)[0]),0)
        for cutoff in [-.1,1.1,float("nan")]:
            with self.assertRaises(ValueError): model.predict([[0.]],threshold=cutoff)

    def test_seed_reproducibility_and_initialization(self):
        x=np.array([[-1,1],[2,-1],[.5,.3]]); y=[0,1,1]
        a=LogisticRegressionManual(max_epochs=10,initialization="normal",seed=7).fit(x,y)
        b=LogisticRegressionManual(max_epochs=10,initialization="normal",seed=7).fit(x,y)
        np.testing.assert_array_equal(a.weights_,b.weights_)
        self.assertEqual(a.history_,b.history_)
        snapshot=a.weights_.copy(); a.fit(x,y)
        np.testing.assert_array_equal(snapshot,a.weights_)

    def test_restore_checkpoint_history_and_validation_does_not_update_weights(self):
        x=np.array([[-2.,1.],[-1,1],[1,-1],[2,-1]]); y=np.array([0,0,1,1])
        a=LogisticRegressionManual(max_epochs=20).fit(x,y,x,y)
        b=LogisticRegressionManual(max_epochs=20).fit(x,y,x,1-y)
        np.testing.assert_array_equal(a.last_weights_,b.last_weights_)
        self.assertEqual(a.last_bias_,b.last_bias_)
        best=max(a.history_,key=lambda h:(h["val_balanced_accuracy"],h["val_f1"],-h["epoch"]))
        self.assertEqual(a.best_epoch_,best["epoch"])
        expected=LogisticRegressionManual(max_epochs=a.best_epoch_).fit(x,y)
        np.testing.assert_array_equal(a.weights_,expected.weights_)
        self.assertEqual(len(a.history_),20)
        self.assertTrue(all(np.isfinite(h["gradient_norm"]) for h in a.history_))

    def test_tolerance_is_train_gradient_only(self):
        model=LogisticRegressionManual(max_epochs=20,tolerance=0).fit([[0.],[0.]],[0,1])
        self.assertTrue(model.converged_)
        self.assertEqual(len(model.history_),1)

    def test_invalid_input_and_dimensions(self):
        for params in [{"learning_rate":0},{"max_epochs":0},{"seed":-1},{"initialization":"unknown"},{"tolerance":-1},{"threshold":2}]:
            with self.assertRaises(ValueError): LogisticRegressionManual(**params)
        for x,y in [([[np.nan]],[1]),([[1]],[2]),([[1],[2]],[1]),([[1j]],[0])]:
            with self.assertRaises(ValueError): LogisticRegressionManual().fit(x,y)
        with self.assertRaises(ValueError): LogisticRegressionManual(initial_weights=[1,2]).fit([[1.]],[1])
        with self.assertRaises(ValueError): LogisticRegressionManual().fit([[1.]],[1],[[1,2]],[0])
        with self.assertRaises(ValueError): sigmoid([np.inf])
        with self.assertRaises(ValueError): binary_cross_entropy_logits([1],[float("nan")])

    def test_persistence_exact_and_feature_alignment(self):
        model=LogisticRegressionManual(max_epochs=10).fit([[-1,2],[2,-1]],[0,1],feature_names=["a","b"])
        model.threshold=.55
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.json"; model.save(path)
            recovered=LogisticRegressionManual.load(path)
            self.assertEqual(recovered.threshold,.55)
            np.testing.assert_array_equal(recovered.weights_,model.weights_)
            np.testing.assert_array_equal(recovered.predict_proba([[1,3]]),model.predict_proba([[1,3]]))
            with self.assertRaises(ValueError): recovered.predict([[1,3]],feature_names=["b","a"])
            with self.assertRaises(ValueError): recovered.predict([[1]])


class ExperimentosLogisticaTests(unittest.TestCase):
    def test_threshold_grid_known_optimum_and_deterministic_ties(self):
        from src.experiments.entrenar_logistica import seleccionar_umbral
        chosen, records=seleccionar_umbral([0,0,1,1],np.array([.1,.55,.6,.9]),[.5,.6])
        self.assertEqual(chosen,.6)
        self.assertEqual(len(records),2)
        # Distancias decimales iguales: desempate por menor umbral, sin ruido binario.
        chosen,_=seleccionar_umbral([0,1],np.array([0.,1.]),[.95,.05])
        self.assertEqual(chosen,.05)
        chosen,_=seleccionar_umbral([0,1],np.array([0.,1.]),[.95,.05,.5])
        self.assertEqual(chosen,.5)
        with self.assertRaises(ValueError): seleccionar_umbral([0,1],np.array([np.nan,.5]),[.5])

    def test_selected_epoch_and_rate_use_development_only(self):
        from src.experiments.entrenar_logistica import seleccionar_experimentos
        x=np.array([[-2.,1],[-1,1],[1,-1],[2,-1]]); y=np.array([0,0,1,1])
        dev={"feature_names":["a","b"],"train":{"X":x,"y":y},"val":{"X":x,"y":y}}
        plan={"experiments":[{"id":"a","learning_rate":.01,"max_epochs":8,"seed":42},
                             {"id":"b","learning_rate":.1,"max_epochs":8,"seed":42}],
              "initialization":"zeros","initial_bias":0.,"tolerance":None,"history_threshold":.5,"threshold_grid":[.4,.5,.6]}
        a,aid,runs,grid=seleccionar_experimentos(dev,plan)
        changed=copy.deepcopy(dev); changed["test"]=object()
        b,bid,runs2,grid2=seleccionar_experimentos(changed,plan)
        self.assertEqual(runs,runs2); self.assertEqual(aid,bid); self.assertEqual(grid,grid2)
        np.testing.assert_array_equal(a.weights_,b.weights_)
        # Ambos ya clasifican perfecto en época 1: plan/época más tempranos ganan.
        self.assertEqual(aid,"a"); self.assertEqual(a.best_epoch_,1)
        selected=next(r for r in runs if r["id"]==aid)
        np.testing.assert_array_equal(a.weights_,selected["selected_weights"])

    def test_real_loader_never_opens_test_and_perceptron_is_reused(self):
        from src.experiments.entrenar_logistica import load_development, verificar_perceptron
        original=np.load; called=[]
        def guard(path,*args,**kwargs):
            called.append(Path(path).name)
            if "test" in Path(path).name: raise AssertionError("Test no autorizado")
            return original(path,*args,**kwargs)
        with patch("numpy.load",side_effect=guard):
            dev=load_development()
            metrics=verificar_perceptron(dev)
        self.assertEqual(set(called),{"X_train.npy","y_train.npy","X_val.npy","y_val.npy"})
        self.assertEqual(dev["train"]["X"].shape,(4599,20))
        self.assertEqual(dev["val"]["X"].shape,(2628,20))
        self.assertEqual(metrics, json.loads((BASE/"resultados/fase_04/metricas.json").read_text(encoding="utf-8")))

    def test_baselines_read_only_allowed_history_columns(self):
        import pandas as pd
        from src.experiments.entrenar_logistica import load_development, evaluar_baselines
        dev=load_development()
        original=pd.read_csv; calls=[]
        def guard(path,*args,**kwargs):
            if Path(path).name=="observaciones.csv":
                columns=kwargs.get("usecols",[])
                self.assertEqual(set(columns),{"observacion_id","mes_objetivo","unidades_lag_1","R_referencia"})
                self.assertNotIn("y",columns); self.assertNotIn("Q_objetivo",columns)
                calls.append(str(path))
            return original(path,*args,**kwargs)
        with patch("pandas.read_csv",side_effect=guard):
            values=evaluar_baselines(dev)
        self.assertTrue(calls)
        self.assertEqual(values["mayoritario"]["val"]["balanced_accuracy"],.5)
        self.assertEqual(values["historico_ultimo_mes"]["val"]["n"],2628)

    def test_exported_model_metrics_predictions_and_integrity(self):
        from src.experiments.entrenar_logistica import load_development, evaluar_logistica, read_json
        folder=BASE/"resultados/fase_05"
        if not (folder/"manifiesto.json").exists(): self.skipTest("Ejecutar WWI tras verificar gradientes")
        dev=load_development(); model=LogisticRegressionManual.load(folder/"modelo_seleccionado.json")
        self.assertEqual(evaluar_logistica(model,dev),read_json(folder/"metricas.json"))
        self.assertEqual(model.feature_names_,dev["feature_names"])
        manifest=read_json(folder/"manifiesto.json")
        self.assertFalse(manifest["test_labels_consulted"])
        self.assertFalse(manifest["test_predictions_generated"])
        for field in ["before","after"]:
            for path,digest in manifest["integrity"][field].items(): self.assertEqual(sha256(BASE/path),digest,path)
        for path,digest in manifest["artifacts_sha256"].items(): self.assertEqual(sha256(BASE/path),digest,path)


if __name__ == "__main__":
    unittest.main()
