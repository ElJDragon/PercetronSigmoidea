"""Conciliación y PCA financiero sin fit de clasificadores ni acceso a test."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments import pca_transacciones as pc
from src.visualization.pca import PCA2D
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual


class ReglasPCAFinancieroTests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(83)
        return rng.normal(size=(64,16)),rng.normal(size=(20,16))+4

    def test_existing_pca_class_and_label_free_interface(self):
        self.assertIs(pc.PCA2D,PCA2D)
        self.assertEqual(list(inspect.signature(pc.fit_pca).parameters),['train_x','val_x'])

    def test_fit_is_only_train_and_validation_only_transformed(self):
        train,val=self.sample();original=PCA2D.fit;calls=[]
        def spy(obj,x,**kwargs):calls.append(x.copy());return original(obj,x,**kwargs)
        with patch.object(PCA2D,'fit',autospec=True,side_effect=spy):p,z,v=pc.fit_pca(train,val)
        self.assertEqual(len(calls),1);np.testing.assert_array_equal(calls[0],train)
        np.testing.assert_array_equal(p.mean_,train.mean(axis=0))
        np.testing.assert_array_equal(v,(val-train.mean(axis=0))@p.components_)
        self.assertEqual(z.shape,(64,2));self.assertEqual(v.shape,(20,2))

    def test_validation_changes_do_not_change_components_or_train_projection(self):
        train,val=self.sample();a,az,_=pc.fit_pca(train,val);b,bz,_=pc.fit_pca(train,val*100-300)
        self.assertEqual(a.to_dict(),b.to_dict());np.testing.assert_array_equal(az,bz)

    def test_label_permutation_cannot_change_pca(self):
        train,val=self.sample();y=np.arange(len(train))%2
        source={'X':train,'y':y};changed={'X':train,'y':y[::-1]}
        a,az,_=pc.fit_pca(source['X'],val);b,bz,_=pc.fit_pca(changed['X'],val)
        self.assertEqual(a.to_dict(),b.to_dict());np.testing.assert_array_equal(az,bz)

    def test_orthogonality_variance_eigenspectrum_and_reconstruction(self):
        train,val=self.sample();p,z,_=pc.fit_pca(train,val)
        result=pc.mathematical_checks(p,train,z)
        self.assertEqual(result['status'],'successful')
        self.assertLess(result['orthogonality_max_error'],1e-12)
        self.assertAlmostEqual(sum(p.explained_variance_ratio_all_),1,places=12)

    def test_serialization_and_sign_convention_reproduce_projection(self):
        train,val=self.sample();p,_,z=pc.fit_pca(train,val)
        for col in p.components_.T:self.assertGreater(col[np.argmax(np.abs(col))],0)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'pca.json';p.save(path);q=PCA2D.load(path)
            np.testing.assert_array_equal(q.transform(val),z)


class EvidenciaPCAFinancieroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (pc.OUT/'dictamen.json').exists():raise unittest.SkipTest('Ejecutar PCA financiero primero')
        cls.dev,_=pc.training.preflight()

    def test_exact_sixteen_excluded_stable_ids_and_months(self):
        x=pd.read_csv(pc.OUT/'observaciones_excluidas.csv',keep_default_na=False)
        expected={1036:'2015-02',1037:'2015-02',1035:'2015-03',1038:'2015-03',1039:'2015-04',1040:'2015-04',1041:'2015-04',1042:'2015-04',
                  1043:'2015-05',1044:'2015-05',1045:'2015-05',1046:'2015-09',1047:'2015-11',1049:'2015-12',1050:'2015-12',1052:'2015-12'}
        self.assertEqual(dict(zip(x.customer_id,x.month)),expected)
        self.assertEqual(set(x.observation_id),{f'customer:{c}:{m}' for c,m in expected.items()})
        self.assertEqual(int(x.y.sum()),7);self.assertEqual(len(x),16)

    def test_all_excluded_rows_are_primary_eligible_labeled_and_predicted(self):
        x=pd.read_csv(pc.OUT/'observaciones_excluidas.csv',keep_default_na=False)
        self.assertEqual(set(x.state),{'eligible'});self.assertTrue((x.age_months==3).all())
        self.assertTrue(x.label_valid.all());self.assertFalse(x.historical_defined.any());self.assertFalse(x.fourth_month_in_population.any())
        self.assertTrue((x.fourth_month<x.first_observed_month).all())
        for name in ['perceptron','logistic','majority']:self.assertTrue(x[name+'_prediction_valid'].all())

    def test_reconciliation_reconstructs_identical_exclusion_set(self):
        report,x=pc.reconcile(self.dev)
        self.assertEqual(report,pc.read(pc.OUT/'conciliacion.json'))
        saved=pd.read_csv(pc.OUT/'observaciones_excluidas.csv')
        self.assertEqual(x.observation_id.tolist(),saved.observation_id.tolist())
        self.assertFalse(report['selection_by_model_errors']);self.assertFalse(report['alignment_discrepancy_detected'])

    def test_label_discrepancy_blocks_readiness_instead_of_hiding_it(self):
        original=pd.read_csv
        def corrupt(path,*args,**kwargs):
            frame=original(path,*args,**kwargs)
            if Path(path).name=='etiquetas_val.csv':
                row=frame.observation_id=='customer:1036:2015-02';frame.loc[row,'y']=1-frame.loc[row,'y']
            return frame
        with patch.object(pc.pd,'read_csv',side_effect=corrupt):report,_=pc.reconcile(self.dev)
        self.assertTrue(report['alignment_discrepancy_detected']);self.assertEqual(report['status'],'blocked_discrepancy')

    def test_key_misalignment_is_rejected(self):
        bad=copy.deepcopy(self.dev);bad['val']['meta']=bad['val']['meta'].iloc[::-1].reset_index(drop=True)
        with self.assertRaises(RuntimeError):pc.reconcile(bad)

    def test_models_scores_metrics_and_hashes_reproduce_without_classifier_fit(self):
        with patch.object(PerceptronSimple,'fit',side_effect=AssertionError('No entrenar')),patch.object(LogisticRegressionManual,'fit',side_effect=AssertionError('No entrenar')):
            models,metrics,verification=pc.verify_models(self.dev)
        self.assertEqual(verification,pc.read(pc.OUT/'verificacion_modelos.json'))
        self.assertEqual(models['perceptron'].best_epoch_,3);self.assertEqual(models['logistic'].best_epoch_,399)
        self.assertEqual(models['logistic'].threshold,.5)
        self.assertEqual(metrics['full_support']['val']['perceptron']['n'],2943)
        self.assertEqual(metrics['common_four_month_support']['val']['metrics']['historical']['n'],2927)

    def test_convergence_is_reconstructed_from_approved_histories(self):
        c=pc.verify_convergence();self.assertEqual(c,pc.read(pc.OUT/'convergencia_verificada.json'))
        self.assertGreater(c['perceptron']['last_online_errors'],0);self.assertEqual(c['perceptron']['zero_error_epochs'],[])
        self.assertEqual(c['logistic']['termination'],'max_epochs');self.assertFalse(c['logistic']['gradient_tolerance_reached'])
        self.assertAlmostEqual(c['logistic']['last_train_loss'],.616972884,places=8)

    def test_real_pca_fits_only_approved_train_and_reproduces_both_projections(self):
        p=pc.PCA2D.load(pc.OUT/'parametros_pca.json')
        self.assertEqual(p.n_samples_,4612);self.assertEqual(p.feature_names_,pc.FEATURES)
        np.testing.assert_array_equal(p.mean_,self.dev['train']['X'].mean(axis=0))
        for split,n in [('train',4612),('val',2943)]:
            z=pc.load_projection(split)
            self.assertEqual(z.shape,(n,2));self.assertTrue(np.isfinite(z).all())
            np.testing.assert_array_equal(z,p.transform(self.dev[split]['X'],feature_names=pc.FEATURES))
            frame=pd.read_csv(pc.OUT/f'proyeccion_{split}.csv')
            self.assertEqual(frame.observation_id.tolist(),self.dev[split]['meta'].observation_id.tolist())
            np.testing.assert_array_equal(frame.y.to_numpy(),self.dev[split]['y'])

    def test_real_variance_conservation_and_expected_ratios(self):
        p=pc.PCA2D.load(pc.OUT/'parametros_pca.json');z=pc.load_projection('train')
        checks=pc.mathematical_checks(p,self.dev['train']['X'],z)
        self.assertEqual(checks,pc.read(pc.OUT/'comprobaciones_matematicas.json'))
        variance=pc.read(pc.OUT/'varianza_explicada.json')
        self.assertAlmostEqual(variance['PC1_percent'],27.89633478995603,places=10)
        self.assertAlmostEqual(variance['PC2_percent'],17.1250590770698,places=10)
        self.assertAlmostEqual(variance['retained_percent']+variance['discarded_percent'],100,places=12)
        self.assertEqual(p.rank_,13)

    def test_three_png_figures_and_component_coefficients(self):
        figures=list((pc.OUT/'graficos').glob('*.png'));self.assertEqual(len(figures),3)
        for path in figures:self.assertEqual(path.read_bytes()[:8],b'\x89PNG\r\n\x1a\n');self.assertGreater(path.stat().st_size,20000)
        coefficients=pd.read_csv(pc.OUT/'coeficientes_componentes.csv',float_precision='round_trip')
        self.assertEqual(coefficients.feature.tolist(),pc.FEATURES)
        p=PCA2D.load(pc.OUT/'parametros_pca.json')
        np.testing.assert_allclose(coefficients[['PC1','PC2']],p.components_,rtol=0,atol=1e-16)

    def test_pipeline_reads_no_test_or_sql_and_never_fits_classifiers(self):
        original=np.load;loaded=[]
        def spy(path,*args,**kwargs):loaded.append(Path(path).name);return original(path,*args,**kwargs)
        with (patch.object(pc.np,'load',side_effect=spy),patch('src.db.conexion.Configuracion.conectar',side_effect=AssertionError('SQL prohibido')),
              patch.object(PerceptronSimple,'fit',side_effect=AssertionError('No entrenar')),patch.object(LogisticRegressionManual,'fit',side_effect=AssertionError('No entrenar'))):
            dev,_=pc.training.preflight();pc.verify_models(dev);pc.reconcile(dev);pc.fit_pca(dev['train']['X'],dev['val']['X'])
        self.assertEqual(loaded,['X_train.npy','y_train.npy','X_val.npy','y_val.npy'])

    def test_final_protocol_keeps_primary_and_secondary_populations_separate(self):
        p=pc.read(pc.OUT/'protocolo_test_definitivo.json')
        self.assertFalse(p['test_executed']);self.assertEqual(p['primary_methods'],['perceptron','logistic','majority'])
        self.assertEqual(p['secondary_methods'],['perceptron','logistic','majority','historical'])
        self.assertIn('Keep row in primary',p['historical_undefined'])
        self.assertEqual(p['planned_months'],['2016-01','2016-02','2016-03','2016-04','2016-05'])
        verdict=pc.read(pc.OUT/'dictamen.json')
        self.assertTrue(verdict['first_test_evaluation_can_be_authorized_in_subsequent_task'])
        self.assertFalse(verdict['test_opened']);self.assertFalse(verdict['final_evaluation_executed'])
        self.assertTrue(verdict['classifier_inputs_remain_16D'])

    def test_manifest_and_all_prior_artifacts_remain_intact(self):
        pc.verify_prior();m=pc.read(pc.OUT/'manifiesto.json')
        self.assertEqual(pc.sha(pc.BASE/'src/visualization/pca.py'),m['pca_implementation_sha256'])
        for section in ['implementation_sha256','artifacts_sha256']:
            for name,digest in m[section].items():self.assertEqual(pc.sha(pc.BASE/name),digest,name)


if __name__=='__main__':unittest.main()
