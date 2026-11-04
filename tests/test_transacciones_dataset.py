"""Dataset financiero: causalidad, snapshots, alineación y fit train-only."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.datamarts.transacciones.caracteristicas import build,feature_row,FEATURES,COVERAGE
from src.datamarts.transacciones.extraccion import normalize,save_snapshot,read_snapshot,extract,PARAMS,SOURCE_SQL,DW_SQL
from src.datamarts.transacciones.preprocesamiento import PreprocesadorFinanciero
from src.datamarts.transacciones import construir as pipeline
from src.db.auditoria_dw import sha
from src.db.conexion import validar_select


def event(eid,month,customer=10):
    return dict(event_id=eid,customer_id=customer,event_date=month+'-02',type_id=3,edited_at=month+'-02 07:00:00',negative_payment=1)


def fixture():
    return [event(i+1,m) for i,m in enumerate(['2013-01']*2+['2013-02']*4+['2013-03']*3+['2013-04']*3)]


def observation(frames,month):
    m,x,y=frames[:3];mask=m.month==month
    return x.loc[mask,FEATURES].reset_index(drop=True),y.loc[mask].reset_index(drop=True)


class ReglasDatasetFinancieroTests(unittest.TestCase):
    def test_coverage_must_be_explicit_not_default_zeros(self):
        for cov in ['unknown','academica',None]:
            with self.assertRaises(ValueError):build(fixture(),coverage=cov)

    def test_current_month_change_updates_y_without_changing_its_X(self):
        before=build(fixture(),coverage=COVERAGE)
        after=build(fixture()+[event(99,'2013-04')],coverage=COVERAGE)
        xa,ya=observation(before,'2013-04');xb,yb=observation(after,'2013-04')
        pd.testing.assert_frame_equal(xa,xb)
        self.assertEqual(int(ya.y.iloc[0]),0);self.assertEqual(int(yb.y.iloc[0]),1)
        self.assertEqual(float(ya.historical_reference.iloc[0]),3)

    def test_future_events_do_not_change_earlier_X_y_or_eligibility(self):
        before=build(fixture(),coverage=COVERAGE)
        after=build(fixture()+[event(99,'2015-12'),event(100,'2015-12',customer=90)],coverage=COVERAGE)
        for i in [0,1,2]:
            a=before[i];b=after[i]
            # Compare by IDs of all target months before the changed future month.
            ids=before[0].loc[before[0].month<'2015-12','observation_id']
            pd.testing.assert_frame_equal(a[a.observation_id.isin(ids)].reset_index(drop=True),b[b.observation_id.isin(ids)].reset_index(drop=True))
        pd.testing.assert_frame_equal(before[3][before[3].month<'2015-12'].reset_index(drop=True),after[3][after[3].month<'2015-12'].reset_index(drop=True))

    def test_historical_statistics_only_use_three_previous_months(self):
        x,y=observation(build(fixture(),coverage=COVERAGE),'2013-04')
        r=x.iloc[0]
        self.assertEqual(r.cobros_lag_1,3);self.assertEqual(r.cobros_lag_2,4);self.assertEqual(r.cobros_lag_3,2)
        self.assertEqual(r.cobros_media_3,3);self.assertAlmostEqual(r.cobros_std_3,np.std([3,4,2],ddof=0))
        self.assertEqual(r.cobros_tendencia_3,.5);self.assertEqual(r.cobros_cambio_1,-1)
        self.assertEqual(r.cobros_mediana_3,y.historical_reference.iloc[0])

    def test_first_seen_prevents_preappearance_and_insufficient_history(self):
        frames=build(fixture()+[event(99,'2015-10',customer=90)],coverage=COVERAGE)
        self.assertFalse((frames[0].customer_id==90).any())
        e=frames[3][frames[3].customer_id==90]
        self.assertEqual(e.month.tolist(),['2015-10','2015-11','2015-12'])
        self.assertEqual(set(e.state),{'insufficient_history'})

    def test_three_calendar_months_not_last_three_active_months(self):
        frames=build([event(1,'2013-01'),event(2,'2013-03')],coverage=COVERAGE)
        x,_=observation(frames,'2013-04')
        self.assertEqual(x.iloc[0].cobros_lag_2,0)
        self.assertTrue(np.isnan(x.iloc[0].variacion_relativa_cobros))
        self.assertEqual(x.iloc[0].base_variacion_cero,1)

    def test_whole_aggregate_with_late_version_is_not_partially_counted(self):
        rows=fixture();rows[6]['edited_at']='2013-04-01 00:00:00'
        m,_,_,e,_=build(rows,coverage=COVERAGE)
        self.assertFalse((m.month=='2013-04').any())
        self.assertEqual(e.loc[e.month=='2013-04','state'].iloc[0],'history_version_not_available')

    def test_target_cut_is_exclusive(self):
        rows=[event(i+1,f'2014-{month:02d}') for i,month in enumerate([9,10,11,12])]
        rows[-1]['edited_at']='2015-01-01 00:00:00'
        frames=build(rows,coverage=COVERAGE)
        self.assertFalse((frames[0].month=='2014-12').any())
        self.assertEqual(frames[3].loc[frames[3].month=='2014-12','state'].iloc[0],'target_version_not_available_by_cut')

    def test_identity_uniqueness_and_closed_catalog(self):
        m,x,y,_,_=build(fixture(),coverage=COVERAGE)
        self.assertTrue(m.observation_id.is_unique)
        self.assertEqual(list(x.columns),['observation_id']+FEATURES)
        self.assertFalse(set(FEATURES)&{'customer_id','y','target_count','historical_reference','amount','current_balance','type_id','edited_at'})
        self.assertEqual(m.observation_id.tolist(),y.observation_id.tolist())

    def test_duplicate_events_or_nonpayment_type_rejected(self):
        with self.assertRaises(ValueError):build(fixture()+[fixture()[0]],coverage=COVERAGE)
        bad=fixture();bad[0]['type_id']=1
        with self.assertRaises(ValueError):build(bad,coverage=COVERAGE)

    def test_snapshot_deterministic_and_canonical_order(self):
        with tempfile.TemporaryDirectory() as root:
            a=Path(root)/'a.csv.gz';b=Path(root)/'b.csv.gz'
            self.assertEqual(save_snapshot(a,fixture()),save_snapshot(b,list(reversed(fixture()))))
            self.assertEqual(a.read_bytes(),b.read_bytes())
            self.assertEqual(read_snapshot(a),normalize(fixture()))

    def test_snapshot_refuses_2016_nulls_wrong_sign_and_duplicate(self):
        variants=[fixture()+[fixture()[0]]]
        for field,value in [('event_date','2016-01-01'),('customer_id',None),('type_id',1),('negative_payment',0)]:
            rows=fixture();rows[0][field]=value;variants.append(rows)
        for rows in variants:
            with self.assertRaises((ValueError,TypeError)):normalize(rows)

    def test_extraction_is_minimal_readonly_and_fixed_development(self):
        calls=[];source=normalize(fixture())
        class Reader:
            consultas=[]
            def select(self,sql,params):
                validar_select(sql);calls.append((sql,params))
                if 'SELECT TransactionTypeName' in sql:return [{'TransactionTypeName':'Customer Payment Received'}]
                if sql==SOURCE_SQL:return source
                if sql==DW_SQL:return [dict(event_id=r['event_id'],event_date=r['event_date'],type_id=r['type_id']) for r in source]
                return [{'events':len(source),'checksum':5}]
            def close(self):pass
        class Config:
            def __init__(self,*args):pass
            def conectar(self):return object()
        with patch('src.datamarts.transacciones.extraccion.Configuracion',Config),patch('src.datamarts.transacciones.extraccion.Lector',return_value=Reader()):
            rows,_=extract('local',[])
        self.assertEqual(rows,source);self.assertEqual(len(calls),7)
        for sql,params in calls:
            self.assertFalse(any(k in sql for k in ['OutstandingBalance','IsFinalized','SupplierTransactions','Fact.Sale','Fact.Movement']))
            if 'SELECT TransactionTypeName' not in sql:self.assertEqual(params,PARAMS)

    def test_extraction_rejects_dw_join_duplication(self):
        source=normalize(fixture())
        class Reader:
            consultas=[]
            def select(self,sql,params):
                if 'SELECT TransactionTypeName' in sql:return [{'TransactionTypeName':'Customer Payment Received'}]
                if sql==SOURCE_SQL:return source
                if sql==DW_SQL:
                    records=[dict(event_id=r['event_id'],event_date=r['event_date'],type_id=r['type_id']) for r in source]
                    return records+[records[0]]
                return [{'events':len(source),'checksum':5}]
            def close(self):pass
        class Config:
            def __init__(self,*args):pass
            def conectar(self):return object()
        with patch('src.datamarts.transacciones.extraccion.Configuracion',Config),patch('src.datamarts.transacciones.extraccion.Lector',return_value=Reader()):
            with self.assertRaises(RuntimeError):extract('local',[])


class PreprocesamientoFinancieroTests(unittest.TestCase):
    def frames(self):
        return pd.DataFrame([feature_row([1,2,3],'2013-04','2013-01'),feature_row([4,0,1],'2013-06','2013-01'),feature_row([3,4,5],'2013-09','2013-01')],columns=FEATURES,dtype=np.float64)

    def test_fit_train_only_and_no_validation_dependency(self):
        train=self.frames();p=PreprocesadorFinanciero().fit(train,partition='train');before=p.to_dict()
        val=self.frames()*1000
        p.transform(val)
        self.assertEqual(before,p.to_dict())
        with self.assertRaises(ValueError):PreprocesadorFinanciero().fit(val,partition='val')
        with self.assertRaises(ValueError):p.fit(train,partition='train')

    def test_optional_imputation_is_train_median_and_standardization_train_only(self):
        train=self.frames();p=PreprocesadorFinanciero().fit(train,partition='train')
        stats=p.state['parameters']['variacion_relativa_cobros']
        self.assertEqual(stats['median'],np.nanmedian(train.variacion_relativa_cobros))
        transformed=p.transform(train)
        self.assertEqual(transformed.dtype,np.float64);self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_allclose(transformed.mean(axis=0),0,atol=1e-14)
        np.testing.assert_allclose(transformed.std(axis=0),1,atol=1e-14)

    def test_constant_elimination_uses_only_train_even_if_validation_changes(self):
        train=self.frames();train['cobros_lag_1']=2
        p=PreprocesadorFinanciero().fit(train,partition='train')
        self.assertNotIn('cobros_lag_1',p.state['feature_names'])
        val=self.frames();val['cobros_lag_1']=[10,20,30]
        self.assertEqual(p.transform(val).shape[1],len(p.state['feature_names']))

    def test_all_null_optional_discarded_required_missing_or_inf_rejected(self):
        a=self.frames();a['variacion_relativa_cobros']=np.nan
        self.assertEqual(PreprocesadorFinanciero().fit(a,partition='train').state['discarded']['variacion_relativa_cobros'],'all_null_train')
        for value in [np.nan,np.inf]:
            bad=self.frames();bad.loc[0,'cobros_lag_1']=value
            with self.assertRaises(ValueError):PreprocesadorFinanciero().fit(bad,partition='train')

    def test_reload_order_and_numeric_state_validation(self):
        frame=self.frames();p=PreprocesadorFinanciero().fit(frame,partition='train')
        q=PreprocesadorFinanciero.from_dict(p.to_dict())
        np.testing.assert_array_equal(p.transform(frame),q.transform(frame))
        with self.assertRaises(ValueError):p.transform(frame[FEATURES[::-1]])
        state=p.to_dict();state['parameters'][state['feature_names'][0]]['std_population']=0
        with self.assertRaises(ValueError):PreprocesadorFinanciero.from_dict(state)


class ArtefactosDatasetFinancieroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (pipeline.OUT/'matrices.json').exists():raise unittest.SkipTest('Construir dataset financiero primero')

    def test_shapes_dtypes_and_class_counts_match_contract(self):
        for split,n,pos in [('train',4612,1798),('val',2943,1198)]:
            x=np.load(pipeline.DATA/f'X_{split}.npy',allow_pickle=False);y=np.load(pipeline.DATA/f'y_{split}.npy',allow_pickle=False)
            self.assertEqual(x.shape,(n,16));self.assertEqual(y.shape,(n,))
            self.assertEqual(x.dtype,np.float64);self.assertEqual(y.dtype,np.int8)
            self.assertTrue(np.isfinite(x).all());self.assertEqual(int(y.sum()),pos);self.assertTrue(np.isin(y,[0,1]).all())

    def test_metadata_raw_features_labels_and_arrays_are_aligned(self):
        p=PreprocesadorFinanciero.from_dict(pipeline.read(pipeline.DATA/'preprocesador.json'))
        for split in ['train','val']:
            m=pd.read_csv(pipeline.DATA/f'metadatos_{split}.csv',keep_default_na=False)
            f=pd.read_csv(pipeline.DATA/f'caracteristicas_crudas_{split}.csv',float_precision='round_trip')
            y=pd.read_csv(pipeline.DATA/f'etiquetas_{split}.csv')
            pipeline.validate_alignment(m,f,y)
            self.assertEqual(m[['month','customer_id']].values.tolist(),m.sort_values(['month','customer_id'])[['month','customer_id']].values.tolist())
            np.testing.assert_array_equal(y.y.to_numpy(dtype=np.int8),np.load(pipeline.DATA/f'y_{split}.npy',allow_pickle=False))
            np.testing.assert_allclose(p.transform(f[FEATURES]),np.load(pipeline.DATA/f'X_{split}.npy',allow_pickle=False),rtol=0,atol=1e-14)

    def test_numeric_parameters_recomputed_exclusively_from_raw_train(self):
        f=pd.read_csv(pipeline.DATA/'caracteristicas_crudas_train.csv',float_precision='round_trip')
        state=PreprocesadorFinanciero().fit(f[FEATURES],partition='train').to_dict()
        self.assertEqual(state,pipeline.read(pipeline.DATA/'preprocesador.json'))

    def test_snapshot_minimal_and_matches_approved_audit(self):
        rows=read_snapshot(pipeline.DATA/pipeline.SNAPSHOT)
        self.assertEqual(len(rows),23018)
        self.assertEqual(rows,pipeline.approved_projection())
        self.assertEqual(set(rows[0]),{'event_id','customer_id','event_date','type_id','edited_at','negative_payment'})

    def test_monthly_population_and_zero_provenance(self):
        e=pd.read_csv(pipeline.DATA/'elegibilidad.csv.gz');m=pd.read_csv(pipeline.DATA/'cuentas_meses.csv.gz',keep_default_na=False)
        self.assertEqual(len(e),8320);self.assertEqual(int((e.state=='eligible').sum()),7555)
        self.assertEqual(int((e.state=='insufficient_history').sum()),765)
        self.assertTrue(e.observation_id.is_unique)
        self.assertEqual(int((m.state=='covered_zero').sum()),611)
        self.assertTrue((m.loc[m.state=='covered_zero','n']==0).all())
        self.assertEqual(set(e.coverage),{COVERAGE})

    def test_monthly_reconciliation_and_disjoint_split_keys(self):
        summary=pipeline.read(pipeline.OUT/'conteos.json')
        for v in summary.values():self.assertEqual(v['eligible'],sum(m['eligible'] for m in v['monthly']))
        a=pd.read_csv(pipeline.DATA/'metadatos_train.csv');b=pd.read_csv(pipeline.DATA/'metadatos_val.csv')
        self.assertFalse(set(a.observation_id)&set(b.observation_id))
        self.assertLess(a.month.max(),'2015-01');self.assertEqual(b.month.min(),'2015-01');self.assertEqual(b.month.max(),'2015-12')

    def test_temporal_availability_and_test_sealed(self):
        for split in ['train','val']:
            m=pd.read_csv(pipeline.DATA/f'metadatos_{split}.csv',keep_default_na=False)
            self.assertTrue((m.history_known_at<m.prediction_time).all())
            self.assertTrue((m.target_known_at<m.label_cut_exclusive).all())
        spec=pipeline.read(pipeline.OUT/'especificacion_test.json')
        for key in ['queries_executed','X_test_created','y_test_created']:self.assertFalse(spec[key])
        self.assertEqual(spec['planned_target_months'],['2016-01','2016-02','2016-03','2016-04','2016-05'])
        self.assertEqual(spec['preprocessor_sha256'],sha(pipeline.DATA/'preprocesador.json'))
        self.assertFalse(list(pipeline.DATA.glob('*test*')))

    def test_source_projection_and_queries_are_read_only(self):
        log=pipeline.read(pipeline.OUT/'consultas_sql.json')
        self.assertEqual(len(log),7)
        for r in log:validar_select(r['sql']);self.assertEqual(r['status'],'successful')
        provenance=pipeline.read(pipeline.OUT/'procedencia.json')
        self.assertEqual(provenance['events'],23018);self.assertTrue(provenance['approved_audit_projection_matches'])
        for fp in provenance['fingerprints'].values():self.assertEqual(fp['before'],fp['after'])

    def test_artifact_and_prior_integrity_hashes(self):
        pipeline.verify_prior()
        m=pipeline.read(pipeline.OUT/'manifiesto.json')
        for field in ['implementation_sha256','artifacts_sha256']:
            for name,digest in m[field].items():self.assertEqual(sha(pipeline.BASE/name),digest,name)


if __name__=='__main__':unittest.main()
