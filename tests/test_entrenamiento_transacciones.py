"""Entrenamiento financiero: checkpoints, baselines, persistencia e integridad."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments import entrenar_transacciones as et
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual
from src.evaluation.metricas import clave_seleccion


def tiny_dev():
    rng=np.random.default_rng(23);x=rng.normal(size=(16,16));y=np.array([0,1]*8,dtype=np.int8)
    return {'train':{'X':x,'y':y},'val':{'X':x[::-1].copy(),'y':y[::-1].copy()}}


def tiny_plan():
    p=copy.deepcopy(et.PLAN)
    for config in p['experiments']:config['params']['max_epochs']=3
    return p


class ReglasEntrenamientoFinancieroTests(unittest.TestCase):
    def test_existing_classes_are_reused_and_budget_has_four_fixed_runs(self):
        self.assertIs(et.PerceptronSimple,PerceptronSimple)
        self.assertIs(et.LogisticRegressionManual,LogisticRegressionManual)
        self.assertEqual(len(et.PLAN['experiments']),4)
        self.assertEqual(et.PLAN['threshold_policy']['threshold_candidates'],[.5])
        self.assertFalse(et.PLAN['budget']['adaptive_expansion'])
        self.assertEqual(len(et.PLAN['input_features']),16)

    def test_historical_baseline_uses_approved_four_month_reference(self):
        meta=pd.DataFrame([dict(customer_id=10,month='2013-04',first_observed_month='2013-01',prediction_time='2013-04-01 00:00:00'),
                           dict(customer_id=10,month='2013-05',first_observed_month='2013-01',prediction_time='2013-05-01 00:00:00')])
        monthly=pd.DataFrame([dict(customer_id=10,month=m,n=n,known_at=m+'-02 07:00:00') for m,n in [('2013-01',1),('2013-02',4),('2013-03',2),('2013-04',3),('2013-05',999)]])
        mask,p=et.baseline_history(meta,monthly)
        np.testing.assert_array_equal(mask,[False,True]);np.testing.assert_array_equal(p,[-1,1])
        # 3>median(2,4,1), mientras 3>median(3,2,4) sería falso.
        monthly.loc[monthly.month=='2013-05','n']=0
        np.testing.assert_array_equal(et.baseline_history(meta,monthly)[1],p)

    def test_historical_baseline_rejects_late_fourth_month_and_unknown_history(self):
        meta=pd.DataFrame([dict(customer_id=1,month='2013-05',first_observed_month='2013-01',prediction_time='2013-05-01 00:00:00')])
        monthly=pd.DataFrame([dict(customer_id=1,month=f'2013-{m:02d}',n=1,known_at='2013-05-01 00:00:00' if m==1 else '') for m in range(1,5)])
        self.assertFalse(et.baseline_history(meta,monthly)[0][0])
        self.assertFalse(et.baseline_history(meta,monthly.iloc[1:])[0][0])

    def test_metric_conventions_and_positive_prediction_proportion(self):
        m=et.metrics([0,1],[0,0])
        self.assertEqual(m['balanced_accuracy'],.5);self.assertEqual(m['specificity'],1)
        self.assertEqual(m['positive_prediction_proportion'],0);self.assertTrue(m['denominator_zero']['precision'])

    def test_average_precision_handles_ties_and_no_positives(self):
        self.assertAlmostEqual(et.average_precision([1,0,1],[.9,.8,.8]),5/6)
        self.assertAlmostEqual(et.average_precision([0,1,1],[.5,.5,.5]),2/3)
        self.assertIsNone(et.average_precision([0,0],[.1,.2]))
        with self.assertRaises(ValueError):et.average_precision([0,1],[np.nan,1])

    def test_small_fixtures_reproduce_histories_models_and_checkpoints(self):
        dev=tiny_dev();plan=tiny_plan()
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            runs,selected,models=et.fit_experiments(dev,plan,Path(a))
            again,chosen,reloaded=et.fit_experiments(dev,plan,Path(b))
            self.assertEqual(runs,again)
            self.assertEqual({k:r['id'] for k,r in selected.items()},{k:r['id'] for k,r in chosen.items()})
            for name,m in models.items():
                np.testing.assert_array_equal(m.weights_,reloaded[name].weights_)
                self.assertEqual(m.weights_.shape,(16,))
                cls=PerceptronSimple if name=='perceptron' else LogisticRegressionManual
                restored=cls.load(Path(a)/'runs'/selected[name]['id']/'modelo.json')
                np.testing.assert_array_equal(m.predict(dev['val']['X']),restored.predict(dev['val']['X']))
                best=max(m.history_,key=lambda h:(h['val_balanced_accuracy'],h['val_f1'],-h['epoch']))
                self.assertEqual(m.best_epoch_,best['epoch'])

    def test_validation_cannot_change_training_updates(self):
        dev=tiny_dev()
        for cls in [PerceptronSimple,LogisticRegressionManual]:
            a=cls(max_epochs=3).fit(dev['train']['X'],dev['train']['y'],X_val=dev['val']['X'],y_val=dev['val']['y'],feature_names=et.FEATURES)
            b=cls(max_epochs=3).fit(dev['train']['X'],dev['train']['y'],X_val=dev['val']['X'],y_val=1-dev['val']['y'],feature_names=et.FEATURES)
            np.testing.assert_array_equal(a.last_weights_,b.last_weights_);self.assertEqual(a.last_bias_,b.last_bias_)
            self.assertEqual([h['weights_norm'] for h in a.history_],[h['weights_norm'] for h in b.history_])

    def test_selection_tie_breaking_is_fixed(self):
        m={'balanced_accuracy':.7,'f1':.6}
        self.assertGreater(clave_seleccion(m,2,1),clave_seleccion(m,3,0))
        self.assertGreater(clave_seleccion(m,2,0),clave_seleccion(m,2,1))


class ArtefactosEntrenamientoFinancieroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (et.OUT/'congelacion.json').exists():raise unittest.SkipTest('Entrenar Transacciones primero')
        cls.runs=et.read(et.OUT/'experimentos.json');cls.frozen=et.read(et.OUT/'congelacion.json')
        cls.comparison=et.read(et.OUT/'metricas.json')

    def test_preflight_loads_only_development_and_never_queries_or_rebuilds(self):
        original=np.load;loaded=[]
        def spy(path,*args,**kwargs):loaded.append(Path(path).name);return original(path,*args,**kwargs)
        with patch.object(et.np,'load',side_effect=spy),patch('src.db.conexion.Configuracion.conectar',side_effect=AssertionError('SQL prohibido')),patch.object(et.np,'save',side_effect=AssertionError('No reconstruir matrices')):
            dev,state=et.preflight()
        self.assertEqual(loaded,['X_train.npy','y_train.npy','X_val.npy','y_val.npy'])
        self.assertEqual(dev['train']['X'].shape,(4612,16));self.assertEqual(dev['val']['X'].shape,(2943,16))
        self.assertEqual(state['fit_partition'],'train')

    def test_plan_matches_predeclared_budget_and_artifacts(self):
        self.assertEqual(et.read(et.OUT/'configuracion_experimental.json'),et.PLAN)
        self.assertEqual(len(self.runs),4)
        for run,plan in zip(self.runs,et.PLAN['experiments']):
            self.assertEqual(run['id'],plan['id'])
            for k,v in plan['params'].items():self.assertEqual(run['params'][k],v)
            self.assertLessEqual(run['epochs_run'],plan['params']['max_epochs'])

    def test_selected_checkpoint_and_run_are_best_by_fixed_rule(self):
        for name,frozen in self.frozen['models'].items():
            candidates=[r for r in self.runs if r['algorithm']==name]
            best=max(candidates,key=lambda r:clave_seleccion(r['metrics']['val'],r['selected_epoch'],r['index']))
            self.assertEqual(frozen['selected_run'],best['id'])
            for r in candidates:
                best_h=max(r['history'],key=lambda h:(h['val_balanced_accuracy'],h['val_f1'],-h['epoch']))
                self.assertEqual(r['selected_epoch'],best_h['epoch'])
                self.assertEqual(r['metrics']['val']['balanced_accuracy'],best_h['val_balanced_accuracy'])

    def test_frozen_model_recovery_matches_saved_predictions_and_metrics(self):
        predictions=pd.read_csv(et.OUT/'predicciones_desarrollo.csv')
        for name,record in self.frozen['models'].items():
            cls=PerceptronSimple if name=='perceptron' else LogisticRegressionManual
            model=cls.load(et.BASE/record['path'])
            self.assertEqual(model.feature_names_,et.FEATURES);self.assertEqual(len(model.weights_),16)
            for split in ['train','val']:
                x=np.load(et.DATA/f'X_{split}.npy',allow_pickle=False);y=np.load(et.DATA/f'y_{split}.npy',allow_pickle=False)
                pred=model.predict(x,feature_names=et.FEATURES)
                np.testing.assert_array_equal(pred,predictions[predictions.partition==split][name].to_numpy(dtype=np.int8))
                self.assertEqual(et.metrics(y,pred),self.comparison['full_support'][split][name])
            if name=='logistic':self.assertEqual(model.threshold,.5)

    def test_common_support_is_identical_for_all_four_methods(self):
        frame=pd.read_csv(et.OUT/'predicciones_desarrollo.csv')
        for split,n,excluded in [('train',4376,236),('val',2927,16)]:
            common=frame[(frame.partition==split)&frame.historical_available]
            self.assertEqual(len(common),n);self.assertTrue(common.observation_id.is_unique)
            self.assertEqual(self.comparison['common_four_month_support'][split]['excluded_without_fourth_month'],excluded)
            for method in ['perceptron','logistic','majority','historical']:
                self.assertEqual(et.metrics(common.y.to_numpy(),common[method].to_numpy(dtype=np.int8)),self.comparison['common_four_month_support'][split]['metrics'][method])
            self.assertTrue(frame[(frame.partition==split)&~frame.historical_available].historical.isna().all())

    def test_majority_is_based_only_on_train_and_zero_denominators_are_preserved(self):
        y=np.load(et.DATA/'y_train.npy',allow_pickle=False)
        self.assertGreater(np.sum(y==0),np.sum(y==1));self.assertEqual(self.comparison['majority_train_class'],0)
        self.assertEqual(self.comparison['full_support']['val']['majority']['balanced_accuracy'],.5)
        self.assertTrue(self.comparison['full_support']['val']['majority']['denominator_zero']['precision'])

    def test_actual_convergence_claims_follow_histories(self):
        diagnostics=et.read(et.OUT/'convergencia.json')
        for run in self.runs:
            d=diagnostics[run['id']];self.assertEqual(run['termination'],'max_epochs')
            self.assertTrue(all(np.isfinite(v) for h in run['history'] for v in h.values() if isinstance(v,(float,int))))
            if run['algorithm']=='perceptron':
                self.assertEqual(d['zero_error_epochs'],[]);self.assertGreater(d['last_online_errors'],0)
                self.assertTrue(all(h['online_errors']==h['updates'] for h in run['history']))
            else:
                self.assertFalse(d['gradient_tolerance_reached']);self.assertGreater(d['last_gradient_norm'],1e-6)
                self.assertEqual(d['loss_increases'],0)

    def test_average_precision_reproduces_frozen_validation_scores(self):
        frame=pd.read_csv(et.OUT/'predicciones_desarrollo.csv',float_precision='round_trip');v=frame[frame.partition=='val']
        self.assertEqual(et.average_precision(v.y.to_numpy(),v.logistic_probability.to_numpy()),self.comparison['validation_logistic_average_precision'])

    def test_twelve_real_figures_and_monthly_metrics_exist(self):
        figures=list((et.OUT/'graficos').rglob('*.png'));self.assertEqual(len(figures),12)
        for p in figures:self.assertEqual(p.read_bytes()[:8],b'\x89PNG\r\n\x1a\n');self.assertGreater(p.stat().st_size,10000)
        monthly=et.read(et.OUT/'metricas_mensuales.json')
        self.assertTrue(all(r['month']<'2016-01' for r in monthly))
        for split in ['train','val']:
            rows=[r for r in monthly if r['split']==split and r['support']=='common_four_months' and r['method']=='historical']
            self.assertEqual(sum(r['metrics']['n'] for r in rows),self.comparison['common_four_month_support'][split]['n'])

    def test_frozen_hashes_dataset_preprocessor_and_prior_files(self):
        et.verify_prior()
        self.assertEqual(et.sha(et.DATA/'preprocesador.json'),self.frozen['preprocessor_sha256'])
        for n,h in self.frozen['source_artifacts_sha256'].items():self.assertEqual(et.sha(et.BASE/n),h,n)
        for r in self.frozen['models'].values():self.assertEqual(et.sha(et.BASE/r['path']),r['sha256'])
        manifest=et.read(et.OUT/'manifiesto.json')
        for field in ['implementation_sha256','artifacts_sha256','classifier_implementation_sha256']:
            for n,h in manifest[field].items():self.assertEqual(et.sha(et.BASE/n),h,n)

    def test_test_remains_sealed_and_no_refit_with_validation(self):
        self.assertFalse(self.frozen['test_2016_loaded'])
        self.assertFalse(self.frozen['training_with_validation'])
        self.assertFalse(self.frozen['retraining_train_plus_validation'])
        self.assertFalse(list(et.OUT.rglob('*test*.npy')))
        self.assertTrue((et.OUT/'protocolo_test.md').exists())


if __name__=='__main__':unittest.main()
