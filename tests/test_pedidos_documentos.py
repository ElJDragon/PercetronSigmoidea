"""Viabilidad documental: fixtures y agregados, sin conexiones ni escrituras SQL."""
import datetime as dt
import json
import unittest

from src.datamarts.pedidos.documentos import (BASE,OUT,COHORTS,tipo,base_sql,obtener,resumir_cohortes,dictamen,clasificar)
from src.db.auditoria_dw import sha
from src.db.conexion import validar_select
from src.db.especificar_pedidos import TEST_START


class FakeReader:
    def __init__(self):self.calls=[]
    def select(self,sql,params=()):self.calls.append((sql,params));return [{}]


class ReglaDocumentoTests(unittest.TestCase):
    def test_backorder_types_are_exclusive(self):
        self.assertEqual(tipo(0,False),'original_without_backorder')
        self.assertEqual(tipo(0,True),'original_with_backorder')
        self.assertEqual(tipo(1,False),'derived_terminal')
        self.assertEqual(tipo(1,True),'derived_with_backorder')
        self.assertEqual(tipo(2,False),'ambiguous_lineage')
    def test_own_document_date_is_used_not_parent_substitution(self):
        sql=base_sql()
        self.assertIn('o.OrderDate',sql)
        self.assertNotIn('p.OrderDate',sql)
        self.assertNotIn('OrderLines',sql)
        self.assertNotIn('line_edited',sql)
    def test_second_day_tie_is_zero(self):
        d=dt.date(2014,6,3);p=dt.datetime(2014,6,5,23,59,59)
        self.assertEqual(clasificar(d,p,p,dt.datetime(2015,1,1)),'within_two_days')
    def test_day_three_and_weekend_are_calendar(self):
        d=dt.date(2014,6,6);p=dt.datetime(2014,6,9,0,0)
        self.assertEqual(clasificar(d,p,p,dt.datetime(2015,1,1)),'after_two_days')
    def test_null_pick_and_invalid_dates_not_fabricated_labels(self):
        d=dt.date(2014,6,3);cut=dt.datetime(2015,1,1)
        self.assertEqual(clasificar(d,None,dt.datetime(2014,6,3),cut),'missing_pick_indeterminate')
        p=dt.datetime(2014,6,2);self.assertEqual(clasificar(d,p,p,cut),'invalid')
    def test_followup_and_availability_cuts(self):
        cut=dt.datetime(2015,1,1);p=dt.datetime(2014,12,30,12)
        self.assertEqual(clasificar(dt.date(2014,12,30),p,p,cut),'insufficient_followup')
        self.assertEqual(clasificar(dt.date(2014,12,20),cut,cut,cut),'pick_not_observed_by_cut')
        self.assertEqual(clasificar(dt.date(2014,12,20),dt.datetime(2014,12,21),cut,cut),'row_not_available_by_cut')
    def test_all_data_queries_are_select_and_seal_2016(self):
        r=FakeReader();obtener(r)
        self.assertEqual(len(r.calls),6)
        for sql,params in r.calls:
            validar_select(sql);self.assertEqual(params,(TEST_START,));self.assertIn('WHERE o.OrderDate < ?',sql)
            self.assertNotIn('Sales.OrderLines',sql)
    def test_derived_positives_do_not_rescue_original_cohort(self):
        rows=[{'month':'2014-01','cohort':'original_without_backorder','state':'within_two_days','documents':20},
              {'month':'2015-01','cohort':'original_without_backorder','state':'within_two_days','documents':10},
              {'month':'2014-01','cohort':'derived_terminal','state':'after_two_days','documents':10},
              {'month':'2015-01','cohort':'derived_terminal','state':'after_two_days','documents':10}]
        s=resumir_cohortes(rows)
        self.assertEqual(s['all_header_diagnostic']['validation']['positive'],10)
        self.assertFalse(dictamen(s)['sufficient_class_support_in_both_development_splits'])
        self.assertFalse(dictamen(s)['can_build_dataset_now'])


class EvidenciaDocumentosTests(unittest.TestCase):
    def setUp(self):
        if not (OUT/'etiquetas_desarrollo.json').exists():self.skipTest('Ejecutar verificación focalizada de desarrollo primero')
        self.summary=json.loads((OUT/'etiquetas_desarrollo.json').read_text(encoding='utf-8'))
        self.data=json.loads((OUT/'agregados_documentos.json').read_text(encoding='utf-8'))
    def test_unique_documents_and_date_consistency(self):
        d=self.data['identity_coverage']
        self.assertEqual(d['documents'],63978);self.assertEqual(d['distinct_documents'],63978)
        self.assertEqual(d['pick_before_order'],0)
        self.assertEqual(d['current_completed_flag']+d['current_pending_or_unobserved'],d['documents'])
        self.assertLess(d['max_date'],'2016-01-01')
    def test_states_and_cohorts_reconcile(self):
        self.assertEqual(sum(c['documents'] for c in self.data['cohort_counts']),63978)
        for split in ['train','validation']:
            whole=self.summary['all_header_diagnostic'][split]
            self.assertEqual(sum(v[split]['total'] for c,v in self.summary.items() if c!='all_header_diagnostic' and split in v),whole['total'])
            self.assertEqual(sum(whole['states'].values()),whole['total'])
            self.assertEqual(whole['positive']+whole['negative'],whole['eligible'])
        for r in self.data['monthly_states_by_cohort']:self.assertLess(r['month'],'2016-01')
    def test_isolated_original_cohort_has_no_positive_support(self):
        s=self.summary['original_without_backorder']
        self.assertEqual(s['train']['eligible'],32539);self.assertEqual(s['validation']['eligible'],18318)
        self.assertEqual(s['train']['positive'],0);self.assertEqual(s['validation']['positive'],0)
        self.assertEqual(self.summary['original_with_backorder']['train']['positive'],4)
        self.assertEqual(self.summary['original_with_backorder']['validation']['positive'],0)
    def test_backorder_dates_do_not_certify_creation(self):
        b=self.data['backorder_dates']
        self.assertEqual(b['edges'],6534);self.assertEqual(b['different_dates'],0)
        self.assertEqual(b['orphan_child'],0);self.assertEqual(b['non_increasing_ids'],0)
        v=json.loads((OUT/'dictamen.json').read_text(encoding='utf-8'))
        self.assertFalse(v['can_build_dataset_now']);self.assertFalse(v['matrices_generated'])
    def test_no_future_features_or_line_information_enabled(self):
        c=json.loads((OUT/'contrato_propuesto.json').read_text(encoding='utf-8'))
        self.assertEqual(len(c['calendar_features']),6)
        self.assertTrue(any('Picking' in n for n in c['forbidden']))
        self.assertTrue(any('líneas' in n for n in c['forbidden']))
        self.assertFalse(list(OUT.rglob('*.npy')))
    def test_hashes_and_prior_files_untouched(self):
        if not (OUT/'manifiesto.json').exists():self.skipTest('Generar manifiesto')
        m=json.loads((OUT/'manifiesto.json').read_text(encoding='utf-8'))
        for field in ['implementation_sha256','artifacts_sha256']:
            for name,digest in m[field].items():self.assertEqual(sha(BASE/name),digest,name)
        for name,digest in json.loads((OUT/'integridad_antes.json').read_text(encoding='utf-8')).items():self.assertEqual(sha(BASE/name),digest,name)


if __name__=='__main__':unittest.main()
