"""Auditoría sin fit de clasificadores, perturbaciones y PCA train-only."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from src.experiments import pca_inventario as pc
from src.datamarts.inventario.caracteristicas import build,COVERAGE,FEATURES
from src.visualization.pca import PCA2D
from tests.test_inventario_dataset import fixture,event,FIRST

def synthetic():
    rng=np.random.default_rng(10)
    return rng.normal(size=(25,17)),rng.normal(size=(11,17))

class InventoryPCAAuditTests(unittest.TestCase):
    def test_frozen_models_reproduce_predictions_without_fit(self):
        with patch.object(pc.PerceptronSimple,'fit',side_effect=AssertionError('no fit')),patch.object(pc.LogisticRegressionManual,'fit',side_effect=AssertionError('no fit')):
            dev,_=pc.training.preflight();models,audit,baseline,saved,details=pc.verify_models(dev)
        self.assertEqual(audit['val']['perceptron']['confusion_matrix'],[[2528,65],[0,35]])
        self.assertEqual(audit['val']['logistic']['confusion_matrix'],[[2523,70],[0,35]])
        self.assertEqual(models['perceptron'].best_epoch_,30);self.assertEqual(models['logistic'].best_epoch_,156)

    def test_hash_discrepancy_blocks_before_PCA(self):
        dev,_=pc.training.preflight();actual=pc.sha
        def mismatch(path):return 'changed' if str(path).endswith('modelos\\perceptron.json') or str(path).endswith('modelos/perceptron.json') else actual(path)
        with patch.object(pc,'sha',side_effect=mismatch),self.assertRaises(RuntimeError):pc.verify_models(dev)

    def test_confusion_metrics_FPR_and_BA(self):
        y=np.array([0]*2593+[1]*35);p=np.array([0]*2528+[1]*100)
        m=pc.metric_audit(y,p)
        self.assertEqual(m['FP'],65);self.assertEqual(m['TP'],35);self.assertEqual(m['FN'],0)
        self.assertEqual(m['false_positive_rate'],65/2593);self.assertEqual(m['balanced_accuracy'],(1+2528/2593)/2)

    def test_target_receipt_changes_y_not_same_month_X(self):
        a=build(fixture(),FIRST,coverage=COVERAGE);b=build(fixture()+[event(999,'2013-04')],FIRST,coverage=COVERAGE)
        for result in [a,b]:self.assertTrue(result[0].observation_id.is_unique)
        xa=a[1].set_index('observation_id');xb=b[1].set_index('observation_id')
        np.testing.assert_array_equal(xa.loc['10:2013-04'],xb.loc['10:2013-04'])
        self.assertNotEqual(a[2].set_index('observation_id').loc['10:2013-04','y'],b[2].set_index('observation_id').loc['10:2013-04','y'])

    def test_future_changes_do_not_modify_past_X_or_eligibility(self):
        a=build(fixture(),FIRST,coverage=COVERAGE);b=build(fixture()+[event(999,'2015-12')],FIRST,coverage=COVERAGE)
        self.assertEqual(a[0].observation_id.tolist(),b[0].observation_id.tolist())
        mask=a[0].month<'2015-12';np.testing.assert_array_equal(a[1].loc[mask,FEATURES],b[1].loc[mask,FEATURES])

    def test_no_later_activity_needed_to_retain_eligible_rows(self):
        a=build(fixture(),FIRST,coverage=COVERAGE)
        early=[r for r in fixture() if r['occurred_at']<'2013-04']
        b=build(early,FIRST,coverage=COVERAGE)
        self.assertEqual(a[0].observation_id.tolist(),b[0].observation_id.tolist())

    def test_zero_mode_required_and_unknown_month_excluded(self):
        with self.assertRaises(ValueError):build(fixture(),FIRST)
        r=build(fixture(),FIRST,coverage=COVERAGE,month_status={(10,'2013-04'):'unknown'})
        self.assertNotIn('10:2013-04',r[0].observation_id.tolist());self.assertNotIn('10:2013-05',r[0].observation_id.tolist())

    def test_PCA_fit_sees_train_only_once(self):
        x,v=synthetic();original=PCA2D.fit;seen=[]
        def tracking(obj,data,**kwargs):seen.append(data);return original(obj,data,**kwargs)
        with patch.object(pc.PCA2D,'fit',new=tracking):p,z=pc.fit_pca(x,v)
        self.assertEqual(len(seen),1);self.assertIs(seen[0],x);np.testing.assert_array_equal(p.mean_,x.mean(axis=0))
        np.testing.assert_array_equal(z['val'],(v-p.mean_)@p.components_)

    def test_changed_validation_does_not_change_PCA_parameters(self):
        x,v=synthetic();p,_=pc.fit_pca(x,v);q,_=pc.fit_pca(x,v+10000)
        self.assertEqual(p.to_dict(),q.to_dict())

    def test_changed_labels_do_not_change_PCA(self):
        x,v=synthetic();labels=np.zeros(len(x),dtype=np.int8);before=labels.copy();p,z=pc.fit_pca(x,v)
        labels[:]=1;self.assertFalse(np.array_equal(labels,before));q,zz=pc.fit_pca(x,v)
        self.assertEqual(p.to_dict(),q.to_dict());np.testing.assert_array_equal(z['train'],zz['train'])

    def test_orthogonality_variance_and_residual(self):
        x,v=synthetic();p,z=pc.fit_pca(x,v);checks=pc.math_checks(p,x,z['train'])
        self.assertLess(checks['orthogonality_error'],1e-12);self.assertAlmostEqual(checks['explained_ratio_sum'],1)

    def test_PCA_persistence_and_feature_order(self):
        x,v=synthetic();p,z=pc.fit_pca(x,v)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pca.json';p.save(path);q=PCA2D.load(path)
            np.testing.assert_array_equal(q.transform(v,feature_names=FEATURES),z['val'])
        with self.assertRaises(ValueError):p.transform(v,feature_names=FEATURES[::-1])

    def test_primary_common_and_baselines_exact_support(self):
        b=pc.read(pc.OUT/'conciliacion_baselines.json')
        self.assertTrue(b['common_equals_primary']);self.assertEqual(b['history_months'],3)
        for split,n in [('train',4599),('val',2628)]:
            self.assertEqual(b[split]['primary'],n);self.assertEqual(b[split]['common'],n);self.assertEqual(b[split]['missing_historical'],0)

    def test_positive_ids_and_shared_errors_reconcile(self):
        c=pc.read(pc.OUT/'concentracion_positivos.json')
        self.assertEqual(c['positive_item_ids'],[77,78,80,86,95,98,184,193,204]);self.assertEqual(c['positive_observations'],35)
        self.assertEqual(sum(i['val_positive'] for i in c['by_item']),35)
        self.assertEqual(c['shared_false_positives']+c['perceptron_only_false_positives'],65)
        self.assertEqual(c['shared_false_positives']+c['logistic_only_false_positives'],70)
        self.assertEqual(c['shared_true_positives'],35);self.assertEqual(c['both_correct']+c['both_wrong']+c['disagreements'],2628)

    def test_activity_diagnostic_joins_by_key_not_partition_positions(self):
        c=pc.read(pc.OUT/'concentracion_positivos.json')['historical_activity_diagnostic']
        self.assertEqual(c['metrics']['TP'],35);self.assertEqual(c['metrics']['FP'],73);self.assertEqual(c['metrics']['TN'],2520)
        self.assertTrue(c['no_search']);self.assertIn('not_new_baseline',c['status'])

    def test_all_17_formula_checks_no_target_feature(self):
        a=pc.read(pc.OUT/'revision_causal.json');self.assertEqual([f['name'] for f in a['features']],FEATURES)
        self.assertTrue(all(f['max_absolute_formula_error']<1e-12 and not f['target_events_used'] for f in a['features']))
        self.assertEqual(a['observations'],7227)

    def test_original_convergence_is_read_only(self):
        c=pc.convergence_audit();self.assertEqual(c['perceptron']['selected_epoch'],30);self.assertEqual(c['logistic']['selected_epoch'],156)
        self.assertGreater(c['perceptron']['last_online_errors'],0);self.assertFalse(c['logistic']['gradient_tolerance_reached'])

    def test_protocol_is_closed_and_does_not_query_outcomes(self):
        with patch.object(pc.Configuracion,'conectar',side_effect=AssertionError('no SQL')):p=pc.protocol()
        self.assertFalse(p['test_executed']);self.assertEqual(p['receipt_parameters'],[11,'2016-01-01','2016-06-01'])
        self.assertEqual(p['models']['perceptron']['selected_epoch'],30);self.assertEqual(p['models']['logistic']['selected_epoch'],156)
        self.assertTrue(p['no_selection_on_test'])

    def test_all_previous_files_and_frozen_models_remain_unchanged(self):
        old=pc.verify_prior();self.assertTrue(all(pc.sha(pc.BASE/n)==h for n,h in old.items()))

    def test_three_figures_and_all_projection_rows_preserved(self):
        self.assertEqual(len(list((pc.OUT/'graficos').glob('*.png'))),3)
        for split,n in [('train',4599),('val',2628)]:self.assertEqual(np.load(pc.OUT/f'proyeccion_{split}.npy',allow_pickle=False).shape,(n,2))

if __name__=='__main__':unittest.main()
