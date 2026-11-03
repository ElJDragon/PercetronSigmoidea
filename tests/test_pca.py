"""PCA matemático y congelación; sin entrenamiento empresarial ni test predictivo."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.auditar_datos import BASE, sha256
from src.visualization.pca import PCA2D


class MatematicaPCATests(unittest.TestCase):
    def data(self,d=5):
        rng=np.random.default_rng(42)
        return rng.normal(size=(25,d))*np.arange(1,d+1)+np.arange(d)*3

    def test_train_center_output_dimensions_and_input_preserved(self):
        x=self.data(); original=x.copy()
        p=PCA2D().fit(x)
        np.testing.assert_array_equal(p.mean_,x.mean(axis=0))
        z=p.transform(x)
        self.assertEqual(z.shape,(25,2))
        np.testing.assert_allclose(z.mean(axis=0),0,atol=1e-13)
        np.testing.assert_array_equal(x,original)
        for d in [2,4,23]:
            self.assertEqual(PCA2D().fit(self.data(d)).components_.shape,(d,2))

    def test_orthogonality_singular_order_and_variance_identity(self):
        x=self.data(); p=PCA2D().fit(x)
        np.testing.assert_allclose(p.components_.T@p.components_,np.eye(2),atol=1e-14)
        self.assertTrue(np.all(np.diff(p.singular_values_)<=0))
        self.assertTrue(np.all(p.explained_variance_ratio_all_>=0))
        self.assertAlmostEqual(float(p.explained_variance_ratio_all_.sum()),1,places=14)
        self.assertAlmostEqual(p.total_variance_,float(x.var(axis=0,ddof=1).sum()),places=12)
        np.testing.assert_allclose(p.transform(x).var(axis=0,ddof=1),p.explained_variance_,atol=1e-13)

    def test_validation_uses_training_center_and_does_not_fit(self):
        x=self.data(); p=PCA2D().fit(x)
        state=p.to_dict(); val=x[:4]+100
        z=p.transform(val)
        np.testing.assert_array_equal(z,(val-x.mean(axis=0))@p.components_)
        self.assertGreater(np.linalg.norm(z.mean(axis=0)),1)
        self.assertEqual(state,p.to_dict())

    def test_label_and_future_data_independence(self):
        x=self.data(); labels=np.arange(len(x))%2
        a=PCA2D().fit(x); labels=1-labels
        b=PCA2D().fit(x)
        self.assertEqual(a.to_dict(),b.to_dict())
        with self.assertRaises(TypeError): PCA2D().fit(x,labels)
        before=a.to_dict()
        a.transform(self.data()*999)
        self.assertEqual(before,a.to_dict())

    def test_deterministic_sign_and_reproducibility(self):
        x=self.data(); a=PCA2D().fit(x); b=PCA2D().fit(x)
        np.testing.assert_array_equal(a.components_,b.components_)
        np.testing.assert_array_equal(a.transform(x),b.transform(x))
        for j in range(2):
            pivot=np.argmax(np.abs(a.components_[:,j]))
            self.assertGreater(a.components_[pivot,j],0)

    def test_constant_columns_supported_without_changing_number_of_inputs(self):
        x=np.column_stack([self.data(3),np.ones(25)*7])
        p=PCA2D().fit(x)
        self.assertEqual(p.components_.shape,(4,2))
        np.testing.assert_allclose(p.components_[-1],0,atol=1e-14)
        self.assertTrue(np.isfinite(p.transform(x)).all())

    def test_insufficient_rank_and_invalid_shapes_block(self):
        for x in [np.ones((10,4)),np.column_stack([np.arange(10),2*np.arange(10),3*np.arange(10)]),
                  [[1,2],[2,3]],np.ones((10,1)),[[1,np.nan],[2,3],[4,5]],[[1,np.inf],[2,3],[4,5]]]:
            with self.assertRaises(ValueError): PCA2D().fit(x)
        with self.assertRaises(ValueError): PCA2D().transform([[1,2]])

    def test_correlated_columns_match_covariance_eigenvalues(self):
        x=self.data(3)
        x=np.column_stack([x,x[:,0]+x[:,1],2*x[:,0]])
        p=PCA2D().fit(x)
        eig=np.linalg.eigvalsh(np.cov(x,rowvar=False))[::-1]
        np.testing.assert_allclose(p.explained_variance_,eig[:2],atol=1e-12)
        self.assertEqual(p.rank_,3)

    def test_safe_persistence_exact_projection_and_feature_order(self):
        x=self.data(3); p=PCA2D().fit(x,feature_names=["a","b","c"])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"pca.json"; p.save(path); q=PCA2D.load(path)
            self.assertEqual(p.to_dict(),q.to_dict())
            np.testing.assert_array_equal(p.transform(x),q.transform(x))
            with self.assertRaises(ValueError): q.transform(x,feature_names=["c","b","a"])
            with self.assertRaises(ValueError): q.transform(x[:,:2])


class CongelacionTests(unittest.TestCase):
    def test_no_classifier_fit_or_test_load_and_predictions_unchanged(self):
        from src.experiments.visualizar_pca import load_development, preparar_congelacion
        from src.models.perceptron import PerceptronSimple
        from src.models.logistic_regression import LogisticRegressionManual
        original=np.load; called=[]
        def guard(path,*args,**kwargs):
            called.append(Path(path).name)
            if "test" in Path(path).name: raise AssertionError("Test no autorizado")
            return original(path,*args,**kwargs)
        with patch("numpy.load",side_effect=guard), patch.object(PerceptronSimple,"fit",side_effect=AssertionError("No reentrenar")), patch.object(LogisticRegressionManual,"fit",side_effect=AssertionError("No reentrenar")):
            dev=load_development(); models,frozen=preparar_congelacion(dev)
            predictions={n:m.predict(dev["val"]["X"]) for n,m in models.items()}
            p=PCA2D().fit(dev["train"]["X"],feature_names=dev["feature_names"])
            p.transform(dev["val"]["X"])
            for n,m in models.items(): np.testing.assert_array_equal(predictions[n],m.predict(dev["val"]["X"]))
        self.assertEqual(set(called),{"X_train.npy","y_train.npy","X_val.npy","y_val.npy"})
        self.assertFalse(frozen["evaluation"]["PCA_used_by_classifiers"])
        self.assertEqual(frozen["models"]["logistica"]["state"]["params"]["threshold"],.5)
        self.assertEqual(frozen["baseline_rules"]["majority_class"],1)

    def test_manifest_contradiction_blocks_freeze(self):
        from src.experiments.visualizar_pca import verificar_manifiestos
        with patch("src.experiments.visualizar_pca.sha256",return_value="incorrecto"):
            with self.assertRaisesRegex(ValueError,"Contradicción"):
                verificar_manifiestos()

    def test_exported_projections_frozen_models_and_integrity(self):
        folder=BASE/"resultados/fase_06"
        if not (folder/"manifiesto.json").exists(): self.skipTest("Ejecutar visualización PCA primero")
        from src.experiments.visualizar_pca import load_development
        manifest=json.loads((folder/"manifiesto.json").read_text(encoding="utf-8"))
        frozen=json.loads((folder/"congelacion_experimental.json").read_text(encoding="utf-8"))
        pca=PCA2D.load(folder/"parametros_pca.json"); dev=load_development()
        self.assertFalse(manifest["test_arrays_loaded_by_phase06"])
        for split in ["train","val"]:
            z=np.load(folder/f"Z_{split}.npy",allow_pickle=False)
            self.assertEqual(z.shape,(len(dev[split]["X"]),2))
            np.testing.assert_array_equal(z,pca.transform(dev[split]["X"]))
        for name,entry in frozen["models"].items():
            self.assertEqual(sha256(BASE/entry["file"]),entry["sha256"])
            self.assertEqual(entry["state"]["n_features"],20)
        for relative,digest in manifest["integrity"]["before"].items(): self.assertEqual(sha256(BASE/relative),digest)
        for relative,digest in manifest["artifacts_sha256"].items(): self.assertEqual(sha256(BASE/relative),digest)
        self.assertEqual(sha256(folder/"congelacion_experimental.json"),manifest["freeze_sha256"])


if __name__=="__main__":
    unittest.main()
