"""Fixtures sintéticos para invariancia temporal, cobertura, extracción y escala."""
import copy
import datetime as dt
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

import numpy as np
import pandas as pd

from src.datamarts.inventario import extraccion as ex
from src.datamarts.inventario.caracteristicas import build,features,FEATURES,COVERAGE
from src.datamarts.inventario.preprocesamiento import PreprocesadorInventario
from src.datamarts.inventario.construir import validate_alignment,test_spec
from src.db.conexion import validar_select

FIRST=[dict(item_id=10,first_event_at='2013-01-01 07:00:00')]

def event(eid=1,month='2013-01',q='10',item=10,typ=11):
    return dict(event_id=eid,item_id=item,type_id=typ,occurred_at=month+'-01 07:00:00',edited_at=month+'-01 07:00:00',
        quantity=q,invoice_id=None,customer_id=None,purchase_order_id=eid,supplier_id=1)

def fixture():
    rows=[]
    for month,n in [('2013-01',1),('2013-02',3),('2013-03',2),('2013-04',2),('2013-05',4),('2013-06',1)]:
        start=len(rows);rows.extend(event(start+i+1,month) for i in range(n))
    return rows

def built(rows=None,first=None,**kw):return build(fixture() if rows is None else rows,FIRST if first is None else first,coverage=COVERAGE,**kw)

def xrow(result,oid='10:2013-04'):
    x=result[1];return x.loc[x.observation_id==oid,FEATURES].to_numpy(np.float64)


class InventoryDatasetTests(unittest.TestCase):
    def test_target_change_changes_label_not_own_features(self):
        a=built();rows=fixture()+[event(999,'2013-04')];b=built(rows)
        np.testing.assert_array_equal(xrow(a),xrow(b))
        self.assertEqual(a[2].set_index('observation_id').loc['10:2013-04','y'],0)
        self.assertEqual(b[2].set_index('observation_id').loc['10:2013-04','y'],1)

    def test_future_events_do_not_modify_past_features(self):
        a=built();b=built(fixture()+[event(999,'2015-12',q='99999')])
        mask=a[0].month<'2015-12'
        np.testing.assert_array_equal(a[1].loc[mask,FEATURES],b[1].loc[mask,FEATURES])

    def test_changing_target_units_does_not_change_own_x(self):
        rows=fixture();a=built(rows)
        for r in rows:
            if r['occurred_at'].startswith('2013-04'):r['quantity']='999999'
        np.testing.assert_array_equal(xrow(a),xrow(built(rows)))

    def test_exact_median_of_three_closed_months_and_tie_zero(self):
        m,x,y,_,_=built();label=y.set_index('observation_id').loc['10:2013-04']
        self.assertEqual(label.historical_reference,2);self.assertEqual(label.target_count,2);self.assertEqual(label.y,0)
        np.testing.assert_array_equal(xrow((m,x,y))[:,:3],[[2,3,1]])

    def test_all_seventeen_formulas(self):
        history=[dict(receipt_count=2,receipt_units=20),dict(receipt_count=3,receipt_units=30),dict(receipt_count=1,receipt_units=10)]
        f=features(history,'2013-04','2013-01');self.assertEqual(list(f),FEATURES)
        expected=[2,3,1,20,30,10,2,2,np.sqrt(2/3),-1,.5,3,0,3,1,0,30]
        np.testing.assert_allclose(list(f.values()),expected,atol=1e-15)

    def test_unique_identity_and_exact_alignment(self):
        meta,x,y,_,_=built();validate_alignment(meta,x,y)
        self.assertEqual(len(meta),33);self.assertFalse(meta[['item_id','month']].duplicated().any())
        with self.assertRaises(ValueError):validate_alignment(meta,x.iloc[::-1],y)

    def test_current_target_never_in_feature_catalog(self):
        _,x,_,_,_=built()
        self.assertEqual(list(x.columns),['observation_id']+FEATURES)
        self.assertFalse({'target_count','y','historical_reference','item_id','month'}&set(FEATURES))

    def test_duplicate_id_rejected_not_silently_deduplicated(self):
        with self.assertRaises(ValueError):built(fixture()+[copy.deepcopy(fixture()[0])])

    def test_sales_and_adjustments_rejected_by_receipt_boundary(self):
        for typ,q in [(10,'-10'),(12,'10'),(9,'10')]:
            with self.subTest(typ=typ),self.assertRaises(ValueError):ex.normalize([event(typ=typ,q=q)])

    def test_positive_sign_and_unit_integrality_required(self):
        for q in ['0','-1','1.5','NaN','Infinity']:
            with self.subTest(q=q),self.assertRaises(ValueError):ex.normalize([event(q=q)])

    def test_edit_at_future_cutoff_blocks(self):
        e=event();e['edited_at']='2016-01-01 00:00:00'
        with self.assertRaises(ValueError):ex.normalize([e])

    def test_zero_requires_explicit_academic_mode(self):
        for mode in [None,'assume','none']:
            with self.subTest(mode=mode),self.assertRaises(ValueError):build(fixture(),FIRST,coverage=mode)

    def test_covered_absence_zero_unknown_not_zero(self):
        result=built(month_status={(10,'2013-07'):'unknown'})
        monthly=result[3].set_index(['item_id','month'])
        self.assertEqual(monthly.loc[(10,'2013-08'),'receipt_count'],0)
        self.assertTrue(pd.isna(monthly.loc[(10,'2013-07'),'receipt_count']))
        excluded=result[4].set_index(['item_id','month'])
        self.assertEqual(excluded.loc[(10,'2013-07'),'reason'],'target_unknown')
        self.assertEqual(excluded.loc[(10,'2013-08'),'reason'],'history_unknown_or_incomplete')
        self.assertNotIn('10:2013-08',result[0].observation_id.tolist())

    def test_incomplete_month_and_its_following_history_excluded(self):
        result=built(month_status={(10,'2013-07'):'incomplete'})
        ids=result[0].observation_id.tolist()
        for month in ['07','08','09','10']:self.assertNotIn('10:2013-'+month,ids)
        self.assertIn('10:2013-11',ids)

    def test_before_first_appearance_not_padded(self):
        r=built([event(month='2014-12')],[dict(item_id=10,first_event_at='2014-12-01 07:00:00')])
        self.assertEqual(r[0].month.min(),'2015-03')
        self.assertEqual(r[3].coverage_status.eq('before_first_appearance').sum(),23)

    def test_item_without_recent_receipt_is_not_removed(self):
        first=FIRST+[dict(item_id=20,first_event_at='2013-01-02 07:00:00')]
        r=built(first=first)
        self.assertEqual(r[0].item_id.value_counts().to_dict(),{10:33,20:33})

    def test_first_appearance_is_any_movement_not_first_receipt(self):
        r=built([event(month='2013-06')])
        self.assertIn('10:2013-04',r[0].observation_id.tolist())
        self.assertEqual(xrow(r)[0,13],3)

    def test_validation_uses_concluded_validation_history(self):
        rows=fixture()+[event(999,'2015-01'),event(1000,'2015-02')]
        r=built(rows)
        self.assertEqual(xrow(r,'10:2015-03')[0,0],1)
        self.assertEqual(xrow(r,'10:2015-03')[0,1],1)

    def test_outcomes_2016_rejected(self):
        with self.assertRaises(ValueError):built(fixture()+[event(999,'2016-01')])
        m,*_=built();self.assertEqual(m.month.max(),'2015-12')

    def test_historical_baseline_uses_same_three_month_reference(self):
        r=built();y=r[2].set_index('observation_id')
        self.assertEqual(y.loc['10:2013-04','historical_baseline'],0)
        self.assertEqual(y.loc['10:2013-05','historical_baseline'],0)

    def test_fit_only_train_and_once(self):
        meta,x,*_=built();train=x.loc[meta.partition=='train',FEATURES]
        p=PreprocesadorInventario()
        with self.assertRaises(ValueError):p.fit(train,partition='val')
        p.fit(train,partition='train')
        with self.assertRaises(ValueError):p.fit(train,partition='train')

    def test_params_equal_train_statistics_not_validation(self):
        meta,x,*_=built();train=x.loc[meta.partition=='train',FEATURES];val=x.loc[meta.partition=='val',FEATURES].copy()
        p=PreprocesadorInventario().fit(train,partition='train');before=p.to_dict();val[:]=999999
        p.transform(val);self.assertEqual(p.to_dict(),before)
        for n,params in before['parameters'].items():
            self.assertEqual(params['mean'],float(train[n].to_numpy(np.float64).mean()))
            self.assertEqual(params['std_population'],float(train[n].to_numpy(np.float64).std(ddof=0)))

    def test_constant_mask_only_from_train(self):
        meta,x,*_=built();train=x.loc[meta.partition=='train',FEATURES].copy();val=x.loc[meta.partition=='val',FEATURES].copy()
        train[FEATURES[0]]=7;val[FEATURES[0]]=999
        p=PreprocesadorInventario().fit(train,partition='train')
        self.assertIn(FEATURES[0],p.to_dict()['discarded']);self.assertEqual(p.transform(val).shape[1],len(p.to_dict()['feature_names']))

    def test_state_roundtrip_and_float64(self):
        meta,x,*_=built();p=PreprocesadorInventario().fit(x.loc[meta.partition=='train',FEATURES],partition='train')
        q=PreprocesadorInventario.from_dict(p.to_dict())
        np.testing.assert_array_equal(p.transform(x[FEATURES]),q.transform(x[FEATURES]));self.assertEqual(q.transform(x[FEATURES]).dtype,np.float64)

    def test_preprocess_rejects_missing_inf_text_and_bad_order(self):
        _,x,*_=built();base=x[FEATURES]
        for kind in ['missing','inf','text','order']:
            bad=base.copy()
            if kind=='missing':bad.loc[0,FEATURES[0]]=np.nan
            elif kind=='inf':bad.loc[0,FEATURES[0]]=np.inf
            elif kind=='text':bad[FEATURES[0]]=bad[FEATURES[0]].astype(str)
            else:bad=bad[FEATURES[::-1]]
            with self.subTest(kind=kind),self.assertRaises(ValueError):PreprocesadorInventario().fit(bad,partition='train')

    def test_snapshot_reproducible_with_fixed_gzip_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.gz';b=Path(tmp)/'b.gz';ex.save_snapshot(a,fixture());ex.save_snapshot(b,fixture()[::-1])
            self.assertEqual(a.read_bytes(),b.read_bytes());self.assertEqual(ex.read_snapshot(a),ex.normalize(fixture()))

    def test_receipts_are_not_collapsed_by_day_or_item(self):
        r=built();monthly=r[3].set_index(['item_id','month'])
        self.assertEqual(monthly.loc[(10,'2013-02'),'receipt_count'],3)

    def test_sql_is_select_fixed_receipt_type_and_development(self):
        for sql in [ex.SOURCE_SQL,ex.SOURCE_FP,ex.DW_SQL,ex.DW_FP]:
            self.assertEqual(validar_select(sql),sql);self.assertEqual(sql.count('?'),3)
        self.assertEqual(ex.PARAMS,(11,dt.date(2013,1,1),dt.date(2016,1,1)))

    def test_reconciliation_detects_duplicate_or_changed_identity(self):
        rows=ex.normalize([event()]);dw=[dict(event_id=1,item_id=10,type_id=11,event_date='2013-01-01',quantity=10,purchase_order_id=1,invoice_id=None,item_key=2,valid_from='2013-01-01 00:00:00',valid_to='9999-12-31 23:59:59')]
        self.assertEqual(len(ex.reconcile(rows,dw)),1)
        with self.assertRaises(ValueError):ex.reconcile(rows,dw+dw)
        dw[0]['item_id']=20
        with self.assertRaises(ValueError):ex.reconcile(rows,dw)

    def test_reserved_test_spec_does_not_query(self):
        with patch.object(ex.Configuracion,'conectar',side_effect=AssertionError('test closed')):
            spec=test_spec(dict(feature_names=FEATURES))
        self.assertFalse(spec['outcomes_queried']);self.assertFalse(spec['X_test_built']);self.assertFalse(spec['y_test_built'])

if __name__=='__main__':unittest.main()
