"""Verificación del entrenamiento propio y de la variante ponderada aislada."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual,loss_and_gradients
from src.models.logistic_inventario import balanced_weights,weighted_loss_gradients,LogisticRegressionInventarioPonderada
from src.experiments import entrenar_inventario as inv

def fixture():
    x=np.zeros((8,17));x[:,0]=[-2,-1,0,1,2,3,4,5];x[:,1]=[1,-1,1,-1,1,-1,1,-1]
    return x,np.array([0,0,0,0,0,0,1,1],dtype=np.int8)

def check_gradient(x,y,w,b,cw,h=1e-6):
    _,gw,gb=weighted_loss_gradients(x,y,w,b,cw);params=np.r_[w,b];numeric=[]
    for j,v in enumerate(params):
        step=h*max(1,abs(v));a=params.copy();c=params.copy();a[j]+=step;c[j]-=step
        numeric.append((weighted_loss_gradients(x,y,a[:-1],a[-1],cw)[0]-weighted_loss_gradients(x,y,c[:-1],c[-1],cw)[0])/(2*step))
    return np.r_[gw,gb],np.array(numeric)

class InventoryTrainingTests(unittest.TestCase):
    def test_balanced_weights_train_only(self):
        x,y=fixture();cw=balanced_weights(y);self.assertEqual(cw,{0:8/12,1:2.})
        self.assertAlmostEqual(sum(cw[int(v)] for v in y),len(y))

    def test_weighted_gradient_central_differences(self):
        x,y=fixture();a,n=check_gradient(x,y,np.linspace(-.2,.2,17),.1,balanced_weights(y))
        np.testing.assert_allclose(a,n,atol=1e-8,rtol=1e-5)

    def test_weighted_gradient_extreme_logits(self):
        x=np.array([[1000.,-2000],[-1500,500],[2000,1000],[-900,-600]]);y=np.array([0,1,0,0])
        a,n=check_gradient(x,y,np.array([.8,-.4]),.3,balanced_weights(y),h=1e-5)
        np.testing.assert_allclose(a,n,atol=2e-6,rtol=1e-5)

    def test_unit_weights_match_standard_gradient(self):
        x,y=fixture();w=np.arange(17)/100
        standard=loss_and_gradients(x,y,w,.2);weighted=weighted_loss_gradients(x,y,w,.2,{0:1.,1:1.})
        for a,b in zip(standard,weighted):np.testing.assert_allclose(a,b,atol=1e-15)

    def test_invalid_weights_and_single_class_block(self):
        x,y=fixture()
        with self.assertRaises(ValueError):balanced_weights(np.zeros(8,dtype=np.int8))
        for weights in [{0:0,1:2},{0:1},{0:1,1:float('inf')}]:
            with self.subTest(weights=weights),self.assertRaises(ValueError):weighted_loss_gradients(x,y,np.zeros(17),0,weights)

    def test_traced_perceptron_reuses_exact_base_updates(self):
        x,y=fixture();a=PerceptronSimple(max_epochs=5,seed=7).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        b=inv.TracedPerceptron(max_epochs=5,seed=7).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        np.testing.assert_array_equal(a.weights_,b.weights_);self.assertEqual(a.bias_,b.bias_);self.assertEqual(a.best_epoch_,b.best_epoch_)
        self.assertEqual(len(b.trace_),5)

    def test_traced_standard_logistic_reuses_base_math(self):
        x,y=fixture();a=LogisticRegressionManual(max_epochs=5).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        b=inv.TracedLogistic(max_epochs=5).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        np.testing.assert_array_equal(a.weights_,b.weights_);self.assertEqual(a.history_,[{k:v for k,v in h.items() if k in a.history_[j]} for j,h in enumerate(b.history_)])

    def test_weighted_initialization_and_repeat_training_deterministic(self):
        x,y=fixture();a=inv.TracedWeighted(max_epochs=7,seed=42).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        b=inv.TracedWeighted(max_epochs=7,seed=42).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        np.testing.assert_array_equal(a.weights_,b.weights_);self.assertEqual(a.trace_,b.trace_);self.assertAlmostEqual(a.initial_train_loss_,np.log(2))

    def test_class_weights_do_not_follow_validation(self):
        x,y=fixture();yv=1-y
        m=LogisticRegressionInventarioPonderada(max_epochs=3).fit(x,y,X_val=x,y_val=yv,feature_names=inv.FEATURES)
        self.assertEqual(m.class_weights_,balanced_weights(y));self.assertNotEqual(m.class_weights_,balanced_weights(yv))

    def test_selected_checkpoint_is_trace_not_last(self):
        x,y=fixture();m=inv.TracedPerceptron(max_epochs=8).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        selected=max(range(len(m.trace_)),key=lambda j:inv.clave_seleccion(m.trace_[j]['metrics']['val'],j+1))
        self.assertEqual(m.best_epoch_,selected+1);np.testing.assert_array_equal(m.weights_,m.trace_[selected]['weights'])

    def test_weighted_persistence_exact_predictions(self):
        x,y=fixture();m=inv.TracedWeighted(max_epochs=6).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';m.save(path);loaded=inv.load_model(path)
            np.testing.assert_array_equal(m.predict_proba(x),loaded.predict_proba(x));self.assertEqual(loaded.class_weights_,m.class_weights_)

    def test_model_rejects_feature_order_or_dimension(self):
        x,y=fixture();m=inv.TracedWeighted(max_epochs=3).fit(x,y,X_val=x,y_val=y,feature_names=inv.FEATURES)
        with self.assertRaises(ValueError):m.predict(x[:,:16])
        with self.assertRaises(ValueError):m.predict(x,feature_names=inv.FEATURES[::-1])

    def test_majority_metrics_never_claim_good_minority_detection(self):
        m=inv.metrics([0,0,0,1],[0,0,0,0]);self.assertEqual(m['balanced_accuracy'],.5)
        self.assertEqual(m['recall'],0);self.assertEqual(m['precision'],0);self.assertEqual(m['f1'],0);self.assertTrue(m['only_majority_predicted'])

    def test_monoclass_month_BA_remains_undefined(self):
        self.assertIsNone(inv.metrics([0,0],[0,0])['balanced_accuracy'])

    def test_average_precision_ties_and_distinct_precision(self):
        self.assertEqual(inv.average_precision([1,0,0,0],[.4,.4,.4,.4]),.25)
        self.assertEqual(inv.average_precision([0,1],[.2,.3]),1.)

    def test_plan_budget_thresholds_and_three_month_baseline(self):
        self.assertEqual(len(inv.PLAN['experiments']),6);self.assertEqual(inv.PLAN['thresholds']['candidates'],[.5]);self.assertFalse(inv.PLAN['budget']['adaptive_expansion'])
        self.assertEqual(inv.PLAN['baselines']['history_months'],3);self.assertNotIn('t-4',inv.PLAN['baselines']['historical'])

    def test_preflight_is_offline_and_does_not_load_2016(self):
        with patch.object(inv.Configuracion,'conectar',side_effect=AssertionError('SQL forbidden')):
            dev,state=inv.preflight()
        self.assertEqual(dev['train']['X'].shape,(4599,17));self.assertEqual(dev['val']['X'].shape,(2628,17));self.assertTrue((dev['val']['meta'].month<'2016-01').all())

    def test_preflight_validates_frozen_hashes(self):
        actual=inv.sha
        def altered(path):return 'tampered' if str(path).endswith('X_train.npy') else actual(path)
        with patch.object(inv,'sha',side_effect=altered),self.assertRaises(RuntimeError):inv.preflight()

    def test_wilson_interval_is_descriptive_and_finite(self):
        interval=inv.wilson(7,35);self.assertTrue(0<=interval['low']<.2<interval['high']<=1);self.assertIn('independence not established',interval['caveat'])

if __name__=='__main__':unittest.main()
