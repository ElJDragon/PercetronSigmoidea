"""Evaluación financiera congelada: temporalidad, soportes, métricas e integridad."""
import copy
import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.experiments import evaluar_transacciones as ev
from src.datamarts.transacciones.preprocesamiento import PreprocesadorFinanciero
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual
from src.db.conexion import validar_select


def event(i,month,c=10):
    return dict(event_id=i,customer_id=c,event_date=month+'-02',type_id=3,edited_at=month+'-02 07:00:00',negative_payment=1)


def fixture():
    dev=[event(i+1,m) for i,m in enumerate(['2015-09']+['2015-10']*2+['2015-11']*4+['2015-12']*3)]
    test=[event(100+i,'2016-01') for i in range(3)]
    coverage={m:{'covered_under_academic_contract':True} for m in ev.MONTHS}
    return dev,test,coverage


class ReglasEvaluacionFinancieraTests(unittest.TestCase):
    def test_frozen_baseline_is_A_and_not_three_month_variant_B(self):
        d,t,c=fixture();m,x,y,a=ev.build_test(d,t,c)
        jan=y[m.month=='2016-01'].iloc[0]
        # N1=3,N2=4,N3=2,N4=1 => 3>median(4,2,1), pero no 3>median(3,4,2).
        self.assertEqual(jan.historical_prediction,1);self.assertEqual(jan.y,0)
        self.assertEqual(ev.HISTORICAL,ev.train.PLAN['baselines']['history'])

    def test_current_target_change_changes_y_not_its_X(self):
        d,t,c=fixture();before=ev.build_test(d,t,c);after=ev.build_test(d,t+[event(200,'2016-01')],c)
        pd.testing.assert_frame_equal(before[1][before[0].month=='2016-01'],after[1][after[0].month=='2016-01'])
        self.assertEqual(before[2].y.iloc[0],0);self.assertEqual(after[2].y.iloc[0],1)

    def test_future_events_do_not_change_earlier_features_labels_or_population(self):
        d,t,c=fixture();a=ev.build_test(d,t,c);b=ev.build_test(d,t+[event(200,'2016-05'),event(201,'2016-05',90)],c)
        ids=a[0].loc[a[0].month<'2016-05','observation_id']
        for i in [0,1,2]:pd.testing.assert_frame_equal(a[i][a[i].observation_id.isin(ids)].reset_index(drop=True),b[i][b[i].observation_id.isin(ids)].reset_index(drop=True))
        self.assertFalse(((b[3].customer_id==90)&(b[3].month<'2016-05')).any())

    def test_later_test_month_uses_previous_test_events_without_retraining(self):
        d,t,c=fixture();m,x,y,a=ev.build_test(d,t,c)
        feb=x[m.month=='2016-02'].iloc[0]
        self.assertEqual(feb.cobros_lag_1,3);self.assertEqual(feb.cobros_lag_2,3);self.assertEqual(feb.cobros_lag_3,4)

    def test_missing_fourth_month_does_not_remove_primary_observation(self):
        d,t,c=fixture();d=[r for r in d if r['event_date']>='2015-10-01']
        m,x,y,a=ev.build_test(d,t,c);j=m[m.month=='2016-01'].iloc[0]
        self.assertTrue(j.primary_evaluation_eligible);self.assertFalse(j.historical_available)
        self.assertTrue(pd.isna(y.historical_prediction.iloc[0]))

    def test_unknown_coverage_and_late_versions_are_not_filled_with_zero(self):
        d,t,c=fixture();c['2016-01']['covered_under_academic_contract']=False
        m,x,y,a=ev.build_test(d,t,c)
        self.assertNotIn('2016-01',set(m.month));self.assertNotIn('2016-02',set(m.month))
        self.assertEqual(a.loc[a.month=='2016-01','primary_reason'].iloc[0],'unknown_target_month_coverage')
        d,t,c=fixture();d[-1]['edited_at']='2016-01-01 00:00:00'
        m,x,y,a=ev.build_test(d,t,c)
        self.assertNotIn('2016-01',set(m.month))

    def test_label_known_at_cut_is_excluded_and_reason_preserved(self):
        d,t,c=fixture();t[0]['edited_at']=ev.CUT
        m,x,y,a=ev.build_test(d,t,c)
        self.assertNotIn('2016-01',set(m.month))
        self.assertEqual(a.loc[a.month=='2016-01','primary_reason'].iloc[0],'target_not_known_before_cut')

    def test_snapshot_scope_semantics_duplicates_and_determinism(self):
        _,t,_=fixture()
        with tempfile.TemporaryDirectory() as root:
            a=Path(root)/'a.gz';b=Path(root)/'b.gz';ev.save_test(a,t);ev.save_test(b,list(reversed(t)))
            self.assertEqual(a.read_bytes(),b.read_bytes());self.assertEqual(ev.load_test(a),ev.normalize_test(t))
        for key,value in [('event_date','2015-12-31'),('event_date','2016-06-01'),('type_id',1),('negative_payment',0),('customer_id',None)]:
            bad=copy.deepcopy(t);bad[0][key]=value
            with self.assertRaises((ValueError,TypeError)):ev.normalize_test(bad)
        with self.assertRaises(ValueError):ev.normalize_test(t+[t[0]])

    def test_duplicate_between_development_and_test_rejected(self):
        d,t,c=fixture();t[0]['event_id']=d[0]['event_id']
        with self.assertRaises(ValueError):ev.build_test(d,t,c)

    def test_confusion_false_alarm_rate_and_no_data_conventions(self):
        m=ev.metrics([0,0,1,1],[0,1,0,1])
        self.assertEqual([m[k] for k in ['TN','FP','FN','TP']],[1,1,1,1]);self.assertEqual(m['false_alarm_rate'],.5)
        self.assertEqual(ev.metrics([],[])['n'],0)
        self.assertIsNone(ev.metrics([1,1],[0,1])['balanced_accuracy'])

    def test_primary_and_common_metrics_do_not_impute_historical_predictions(self):
        f=pd.DataFrame(dict(y=[0,1,1],perceptron=[0,1,0],logistic=[0,0,1],majority=[0,0,0],historical=[0,1,np.nan],historical_available=[True,True,False],logistic_probability=[.1,.4,.7]))
        m=ev.evaluate_frame(f)
        self.assertEqual(m['primary_n'],3);self.assertEqual(m['common_n'],2)
        self.assertEqual(m['common']['historical']['n'],2);self.assertEqual(m['primary']['perceptron']['n'],3)

    def test_query_templates_are_minimal_select_with_fixed_test_bounds(self):
        self.assertEqual(ev.PARAMS,(3,dt.date(2016,1,1),dt.date(2016,6,1)))
        for sql in [ev.SOURCE_SQL,ev.DW_SQL,ev.SOURCE_FP,ev.DW_FP]:
            validar_select(sql);self.assertEqual(sql.count('?'),3)
            self.assertNotIn('OutstandingBalance',sql);self.assertNotIn('IsFinalized',sql)


class ArtefactosEvaluacionFinancieraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ev.OUT/'metricas_globales.json').exists():raise unittest.SkipTest('Evaluar test financiero primero')

    def test_opening_record_precedes_first_SQL_and_freezes_baseline(self):
        r=ev.read(ev.OUT/'apertura_previa.json');start=ev.read(ev.OUT/'inicio_consulta_test.json')
        self.assertLess(r['recorded_at_utc'],start['started_at_utc'])
        self.assertEqual(start['opening_record_sha256'],ev.sha(ev.OUT/'apertura_previa.json'))
        self.assertFalse(r['test_outcomes_accessed_when_record_written']);self.assertEqual(r['baseline'],ev.HISTORICAL)
        self.assertEqual(r['models_sha256'],ev.review.EXPECTED_MODELS)

    def test_preflight_models_and_preprocessor_remain_frozen(self):
        with patch.object(PerceptronSimple,'fit',side_effect=AssertionError('No fit')),patch.object(LogisticRegressionManual,'fit',side_effect=AssertionError('No fit')),patch.object(PreprocesadorFinanciero,'fit',side_effect=AssertionError('No fit')):
            _,state,models,checks=ev.preflight()
        self.assertEqual(state['fit_partition'],'train');self.assertEqual(models['perceptron'].best_epoch_,3)
        self.assertEqual(models['logistic'].best_epoch_,399);self.assertEqual(models['logistic'].threshold,.5)
        self.assertTrue(checks['requires_four_prior_months'])

    def test_source_snapshot_bijection_and_readonly_queries(self):
        p=ev.read(ev.OUT/'procedencia_test.json');rows=ev.load_test(ev.OUT/'snapshot_test.csv.gz')
        self.assertEqual(len(rows),p['events']);self.assertEqual(len({r['event_id'] for r in rows}),len(rows))
        self.assertEqual(ev.sha(ev.OUT/'snapshot_test.csv.gz'),p['snapshot_sha256']);self.assertTrue(p['source_dw_bijection'])
        for fp in p['fingerprints'].values():self.assertEqual(fp['before'],fp['after'])
        for q in ev.read(ev.OUT/'consultas_test.json'):validar_select(q['sql']);self.assertEqual(q['status'],'successful')

    def test_test_X_is_frozen_transform_of_causal_features(self):
        state=ev.read(ev.train.DATA/'preprocesador.json');p=PreprocesadorFinanciero.from_dict(state)
        raw=pd.read_csv(ev.OUT/'caracteristicas_crudas_test.csv',float_precision='round_trip');x=pd.read_csv(ev.OUT/'X_test.csv',float_precision='round_trip')
        self.assertEqual(list(x.columns),['observation_id']+ev.FEATURES)
        np.testing.assert_array_equal(p.transform(raw[ev.FEATURES]),x[ev.FEATURES].to_numpy(dtype=np.float64))
        self.assertTrue(np.isfinite(x[ev.FEATURES].to_numpy()).all())

    def test_population_alignment_and_causal_availability(self):
        m=pd.read_csv(ev.OUT/'metadatos_test.csv',keep_default_na=False);x=pd.read_csv(ev.OUT/'X_test.csv');y=pd.read_csv(ev.OUT/'y_test.csv');p=pd.read_csv(ev.OUT/'predicciones_test.csv')
        self.assertTrue(m.observation_id.is_unique)
        for f in [x,y,p]:self.assertEqual(m.observation_id.tolist(),f.observation_id.tolist())
        self.assertTrue((m.history_known_at<m.prediction_time).all());self.assertTrue((m.target_known_at<ev.CUT).all())
        np.testing.assert_array_equal(y.y,(y.target_count>y.historical_reference).astype(np.int8))

    def test_metrics_reproduce_predictions_and_global_monthly_counts(self):
        f=pd.read_csv(ev.OUT/'predicciones_test.csv',float_precision='round_trip')
        self.assertEqual(ev.evaluate_frame(f),ev.read(ev.OUT/'metricas_globales.json'))
        monthly=ev.read(ev.OUT/'metricas_mensuales.json')
        self.assertEqual(list(monthly),ev.MONTHS)
        self.assertEqual(sum(v['primary_n'] for v in monthly.values()),len(f))
        self.assertEqual(sum(v['common_n'] for v in monthly.values()),int(f.historical_available.sum()))
        for method in ['perceptron','logistic','majority']:
            globalm=ev.read(ev.OUT/'metricas_globales.json')['primary'][method]
            for key in ['TN','FP','FN','TP']:self.assertEqual(sum(v['primary'][method].get(key,0) for v in monthly.values()),globalm[key])

    def test_error_partitions_and_cohorts_account_for_all_primary_rows(self):
        f=pd.read_csv(ev.OUT/'predicciones_test.csv');e=ev.read(ev.OUT/'analisis_errores.json')['global_counts']
        self.assertEqual(e['shared_correct']+e['shared_error']+e['disagreement'],len(f))
        cohorts=ev.read(ev.OUT/'cohortes.json')
        self.assertEqual(sum(c['metrics']['primary_n'] for c in cohorts.values()),len(f))

    def test_seven_real_graphs_and_prior_integrity(self):
        files=list((ev.OUT/'graficos').glob('*.png'));self.assertEqual(len(files),7)
        for p in files:self.assertEqual(p.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        ev.verify_prior()
        m=ev.read(ev.OUT/'manifiesto.json')
        for field in ['implementation_sha256','artifacts_sha256']:
            for name,digest in m[field].items():self.assertEqual(ev.sha(ev.BASE/name),digest,name)


if __name__=='__main__':unittest.main()
