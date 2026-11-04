"""Fronteras temporales y conciliación de Pedidos, sin acceso directo a SQL."""
import datetime as dt
import json
import unittest

from src.db.conexion import validar_select
from src.db.especificar_pedidos import (BASE,OUT,TEST_START,STATES,Desarrollo,clasificar,estado_documento,reconciliar_auditoria,resumir,obtener)
from src.db.auditoria_dw import sha


class FakeReader:
    def __init__(self):self.calls=[]
    def select(self,sql,params):self.calls.append((sql,params));return [{}]


class EtiquetaPedidosTests(unittest.TestCase):
    def setUp(self):self.d=dt.date(2014,12,20);self.cut=dt.datetime(2015,1,1)
    def test_day_two_inclusive_and_calendar_not_48h(self):
        p=dt.datetime(2014,12,22,23,59,59)
        self.assertEqual(clasificar(self.d,p,p,self.cut),'within_two_days')
    def test_day_three_is_delayed(self):
        p=dt.datetime(2014,12,23)
        self.assertEqual(clasificar(self.d,p,p,self.cut),'after_two_days')
    def test_weekends_are_calendar_days(self):
        p=dt.datetime(2015,1,5,12)
        self.assertEqual(clasificar(dt.date(2015,1,2),p,p,dt.datetime(2016,1,1)),'after_two_days')
    def test_missing_pick_not_automatically_positive(self):
        self.assertEqual(clasificar(self.d,None,dt.datetime(2014,12,30),self.cut),'missing_pick_indeterminate')
    def test_invalid_date_is_not_labelled(self):
        p=dt.datetime(2014,12,19)
        self.assertEqual(clasificar(self.d,p,p,self.cut),'invalid')
    def test_fast_pick_inside_unfinished_horizon_excluded_uniformly(self):
        p=dt.datetime(2014,12,30,12)
        self.assertEqual(clasificar(dt.date(2014,12,30),p,p,self.cut),'insufficient_followup')
    def test_december_29_horizon_ends_exactly_at_cut(self):
        p=dt.datetime(2014,12,31,23,59,59)
        self.assertEqual(clasificar(dt.date(2014,12,29),p,p,self.cut),'within_two_days')
    def test_pick_at_cut_is_not_known_before_cut(self):
        self.assertEqual(clasificar(self.d,self.cut,self.cut,self.cut),'pick_not_observed_by_cut')
    def test_old_order_with_future_pick_does_not_leak(self):
        p=dt.datetime(2015,3,1)
        self.assertEqual(clasificar(self.d,p,p,self.cut),'pick_not_observed_by_cut')
    def test_later_version_excluded_even_when_pick_earlier(self):
        self.assertEqual(clasificar(self.d,dt.datetime(2014,12,21),dt.datetime(2015,2,1),self.cut),'row_not_available_by_cut')
    def test_document_logic_is_all_complete_or_known_counterexample(self):
        self.assertEqual(estado_documento(['within_two_days']*3),'within_two_days')
        self.assertEqual(estado_documento(['within_two_days','missing_pick_indeterminate']),'indeterminate_document')
        self.assertEqual(estado_documento(['after_two_days','missing_pick_indeterminate']),'after_two_days')
        self.assertEqual(estado_documento(['within_two_days','invalid']),'invalid')
    def test_document_insufficient_followup_has_priority(self):
        self.assertEqual(estado_documento(['insufficient_followup','within_two_days']),'insufficient_followup')


class SelloYConteosTests(unittest.TestCase):
    def test_development_reader_always_binds_sealed_date(self):
        r=FakeReader();d=Desarrollo(r)
        d.select('SELECT COUNT_BIG(*) FROM Sales.Orders o WHERE o.OrderDate < ?')
        self.assertEqual(r.calls[0][1],(dt.date(2016,1,1),))
    def test_missing_barrier_or_literal_fake_barrier_or_union_rejected(self):
        d=Desarrollo(FakeReader())
        for sql in ['SELECT * FROM Sales.Orders',"SELECT 'WHERE o.OrderDate < ?'",'SELECT * FROM Sales.Orders o WHERE o.OrderDate < ? UNION SELECT * FROM Sales.Orders']:
            with self.subTest(sql=sql),self.assertRaises(ValueError):d.select(sql)
    def test_all_actual_outcome_templates_are_select_with_development_gate(self):
        r=FakeReader();obtener(r)
        self.assertGreaterEqual(len(r.calls),10)
        for sql,params in r.calls:
            validar_select(sql);self.assertEqual(params,(TEST_START,));self.assertIn('WHERE o.OrderDate < ?',sql)
    def test_historical_reconciliation_is_exhaustive_without_new_test_query(self):
        r=reconciliar_auditoria()
        self.assertEqual(sum(r['states'].values()),231412)
        self.assertEqual(r['observed_mature'],227745)
        self.assertEqual(r['null_picking_total'],3147)
        self.assertEqual(r['null_picking_in_immature'],18)
        self.assertEqual(r['states']['sin_seguimiento_nominal_suficiente'],538)
    def test_rates_use_only_justified_labels(self):
        r=resumir([{'month':'2014-01','state':'after_two_days','lines':2},{'month':'2014-01','state':'within_two_days','lines':8},{'month':'2014-01','state':'missing_pick_indeterminate','lines':90}],'lines')
        self.assertEqual(r['train']['eligible'],10);self.assertEqual(r['train']['total'],100);self.assertEqual(r['train']['positive_proportion'],.2)
    def test_all_monthly_line_states_reconcile_to_raw_development(self):
        path=OUT/'agregados_desarrollo.json'
        if not path.exists():self.skipTest('Ejecutar agregados de desarrollo primero')
        a=json.loads(path.read_text(encoding='utf-8'))
        for month in a['line_structure_monthly']:
            self.assertLess(month['month'],'2016-01')
            self.assertEqual(month['stable_lines'],month['lines'])
            self.assertEqual(sum(r['lines'] for r in a['line_states_monthly'] if r['month']==month['month']),month['lines'])
        d=resumir(a['line_states_monthly'],'lines')
        self.assertEqual(d['train']['positive'],1734);self.assertEqual(d['validation']['positive'],698)
        self.assertEqual(d['train']['eligible'],126341);self.assertEqual(d['validation']['eligible'],71112)
    def test_query_ledger_has_no_test_outcome_reads(self):
        path=OUT/'consultas_desarrollo.json'
        if not path.exists():self.skipTest('Ejecutar script primero')
        for q in json.loads(path.read_text(encoding='utf-8')):
            validar_select(q['sql']);self.assertEqual(q['parameters'],['2016-01-01']);self.assertEqual(q['status'],'successful')
    def test_export_hashes_and_previous_files_preserved(self):
        path=OUT/'manifiesto.json'
        if not path.exists():self.skipTest('Ejecutar especificación primero')
        m=json.loads(path.read_text(encoding='utf-8'))
        for key in ['implementation_sha256','artifacts_sha256']:
            for name,digest in m[key].items():self.assertEqual(sha(BASE/name),digest,name)
        for name,digest in json.loads((OUT/'integridad_antes.json').read_text(encoding='utf-8')).items():self.assertEqual(sha(BASE/name),digest,name)


if __name__=='__main__':unittest.main()
