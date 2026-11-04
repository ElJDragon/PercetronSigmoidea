"""Causalidad, semántica y protección del análisis focalizado de Inventario."""
import copy
import datetime as dt
import unittest
from unittest.mock import patch

from src.datamarts import inventario_viabilidad as iv
from src.datamarts.reportes_inventario import feature_catalog


def event(eid=1,item=10,typ=11,when='2013-01-02 07:00:00',q='5.000'):
    return dict(event_id=eid,item_id=item,type_id=typ,occurred_at=when,edited_at=when,
                quantity=q,invoice_id=None,purchase_order_id=eid,customer_id=None,supplier_id=1)

def example_panel():
    return {(10,'2013-01'):{'receipts':1},(10,'2013-02'):{'receipts':3},(10,'2013-03'):{'receipts':2},
            (10,'2013-04'):{'receipts':2}}

def candidate(panel=None,first=None):
    return iv.candidate(panel if panel is not None else example_panel(),first or {10:'2013-01'},'receipts',assume_coverage=True)


class InventoryRulesTests(unittest.TestCase):
    def test_only_select_development_boundary_is_bound(self):
        r=unittest.mock.Mock();sql='SELECT st.Quantity FROM Warehouse.StockItemTransactions st WHERE st.TransactionOccurredWhen < ?'
        iv.scoped(r,sql);r.select.assert_called_once_with(sql,(dt.date(2016,1,1),))

    def test_scope_rejects_unbounded_and_mutation(self):
        for sql in ['SELECT * FROM Fact.Movement','DELETE FROM Fact.Movement',
                    'SELECT st.Quantity FROM Warehouse.StockItemTransactions st WHERE st.TransactionOccurredWhen < ?; SELECT 1',
                    'SELECT st.Quantity FROM Warehouse.StockItemTransactions st WHERE st.TransactionOccurredWhen < ? UNION SELECT 1']:
            with self.subTest(sql=sql),self.assertRaises(ValueError):iv.scoped(unittest.mock.Mock(),sql)

    def test_future_outcomes_blocked_at_aggregation(self):
        for date in ['2016-01-01 00:00:00','2016-05-01 07:00:00']:
            with self.subTest(date=date),self.assertRaises(ValueError):iv.aggregate([event(when=date)])

    def test_duplicate_event_is_not_double_counted(self):
        with self.assertRaises(ValueError):iv.aggregate([event(),event()])

    def test_two_ids_same_day_count_as_two_receipts(self):
        panel,_=iv.aggregate([event(1),event(2)])
        self.assertEqual(panel[10,'2013-01']['receipts'],2)
        self.assertEqual(panel[10,'2013-01']['receipt_units'],10)

    def test_negative_adjustment_is_not_sale(self):
        panel,_=iv.aggregate([event(typ=12,q='-3')])
        self.assertEqual(panel[10,'2013-01']['adjustments'],1)
        self.assertEqual(panel[10,'2013-01']['issues'],0)

    def test_positive_adjustment_is_not_receipt(self):
        panel,_=iv.aggregate([event(typ=12,q='3')])
        self.assertEqual(panel[10,'2013-01']['receipts'],0)

    def test_zero_quantity_adjustment_still_counts(self):
        panel,_=iv.aggregate([event(typ=12,q='0')])
        self.assertEqual(panel[10,'2013-01']['adjustments'],1)

    def test_unknown_type_not_inferred_from_sign(self):
        with self.assertRaises(ValueError):iv.aggregate([event(typ=9)])

    def test_incoherent_type_sign_blocks(self):
        for typ,q in [(10,'5'),(10,'0'),(11,'-5'),(11,'0')]:
            with self.subTest(typ=typ,q=q),self.assertRaises(ValueError):iv.aggregate([event(typ=typ,q=q)])

    def test_missing_identity_never_zero(self):
        for identity in [None,0,-1]:
            with self.subTest(identity=identity),self.assertRaises(ValueError):iv.aggregate([event(item=identity)])

    def test_late_edit_blocks_availability_claim(self):
        e=event();e['edited_at']='2013-02-01 00:00:00'
        with self.assertRaises(ValueError):iv.aggregate([e])

    def test_zero_filling_requires_explicit_coverage(self):
        with self.assertRaises(ValueError):iv.candidate(example_panel(),{10:'2013-01'},'receipts')

    def test_unknown_statistic_not_authorized(self):
        with self.assertRaises(ValueError):iv.candidate({}, {10:'2013-01'},'current_stock',assume_coverage=True)

    def test_tie_negative_and_strict_excess_positive(self):
        p=example_panel();self.assertEqual(candidate(p)['monthly'][0]['positive'],0)
        p[10,'2013-04']['receipts']=3
        self.assertEqual(candidate(p)['monthly'][0]['positive'],1)

    def test_no_preappearance_history_or_targets(self):
        result=candidate({}, {10:'2014-12'})
        self.assertEqual(result['summary']['train']['observations'],0)
        self.assertEqual(result['monthly'][0]['month'],'2015-03')
        self.assertEqual(result['summary']['val']['observations'],10)

    def test_test_never_produced(self):
        result=candidate()
        self.assertEqual(result['monthly'][-1]['month'],'2015-12')
        self.assertEqual(result['targets_2016'],0)

    def test_target_change_does_not_change_historical_baseline(self):
        p=example_panel()
        # El baseline usa historia, no el objetivo: aislar el primer target.
        with patch.object(iv,'month_range',return_value=iter(['2013-04'])):
            a=candidate(p)['summary']['train']['historical']['confusion_matrix']
        p[10,'2013-04']['receipts']=100
        with patch.object(iv,'month_range',return_value=iter(['2013-04'])):
            b=candidate(p)['summary']['train']['historical']['confusion_matrix']
        self.assertEqual(a[0][1]+a[1][1],b[0][1]+b[1][1])
        self.assertEqual(a[0][1]+a[1][1],0)

    def test_all_zero_entity_retained_and_single_class_BA_undefined(self):
        r=candidate({})
        self.assertEqual(r['summary']['train']['observations'],21)
        self.assertEqual(r['summary']['train']['positive'],0)
        self.assertIsNone(r['summary']['train']['historical']['balanced_accuracy'])

    def test_majority_is_fit_on_train_only(self):
        p={(10,m):{'receipts':10**j} for j,m in enumerate(iv.month_range('2015-01','2015-12'),1)}
        r=candidate(p)
        self.assertEqual(r['summary']['train']['positive'],0)
        self.assertEqual(r['summary']['val']['positive'],12)
        self.assertEqual(r['summary']['val']['majority_class_from_train'],0)

    def test_count_label_independent_of_quantity_scale(self):
        a=event();b=copy.deepcopy(a);b['quantity']='900000'
        pa,fa=iv.aggregate([a]);pb,fb=iv.aggregate([b])
        self.assertEqual(iv.candidate(pa,fa,'receipts',assume_coverage=True),iv.candidate(pb,fb,'receipts',assume_coverage=True))

    def test_first_appearance_is_minimum_and_input_order_independent(self):
        a=event();b=event(2,when='2013-02-01 07:00:00')
        self.assertEqual(iv.aggregate([b,a]),iv.aggregate([a,b]))

    def test_catalog_is_causal_and_has_no_continuous_ids(self):
        fs=feature_catalog();self.assertEqual(len(fs),17)
        self.assertEqual(len({f['name'] for f in fs}),17)
        self.assertFalse(any('id'==f['name'] or 'stock_actual' in f['name'] for f in fs))
        self.assertTrue(all(f['formula'] and f['source'] and f['availability'] for f in fs))

    def test_existing_metrics_handle_single_class(self):
        self.assertIsNone(iv.binary_metrics([0,0],[0,0])['balanced_accuracy'])
        self.assertEqual(iv.binary_metrics([0,1],[0,0])['balanced_accuracy'],.5)


if __name__=='__main__':unittest.main()
