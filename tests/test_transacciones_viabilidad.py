"""Validación financiera focalizada y offline; eventos no equivalen a estados."""
import copy
import datetime as dt
import unittest
from unittest.mock import patch
from src.datamarts import transacciones_viabilidad as tv
from src.datamarts.reportes_transacciones import feature_catalog
from src.db.conexion import validar_select
from src.db.auditoria_dw import sha


def event(eid=1,entity=10,typ=3,date='2013-01-02'):
    return dict(event_id=eid,entity_id=entity,type_id=typ,event_date=date,edited_at=date+' 07:00:00',amount='-5.00',current_balance='0',current_finalized=True)


def panel_fixture():
    return {(10,'2013-01'):{'n':2,'known_at':'2013-01-31 07:00:00'},
            (10,'2013-02'):{'n':4,'known_at':'2013-02-28 07:00:00'},
            (10,'2013-03'):{'n':3,'known_at':'2013-03-31 07:00:00'},
            (10,'2013-04'):{'n':3,'known_at':'2013-04-30 07:00:00'}}


class ReglasFinancierasTests(unittest.TestCase):
    def test_domains_and_payment_types(self):
        self.assertEqual(tv.classify('customer',3),'payment_received')
        self.assertEqual(tv.classify('supplier',7),'payment_issued')
        self.assertEqual(tv.classify('customer',7),'unclassified')
        self.assertEqual(tv.classify('supplier',3),'unclassified')
        self.assertEqual(tv.classify('customer',2),'credit_note')
        self.assertEqual(tv.classify('customer',4),'refund')

    def test_negative_invoice_is_not_a_payment(self):
        g,first=tv.aggregate_payments([event(typ=1)],'customer')
        self.assertEqual(g,{});self.assertEqual(first,{})

    def test_distinct_events_same_day_are_not_collapsed_to_days(self):
        g,_=tv.aggregate_payments([event(1),event(2)],'customer')
        self.assertEqual(g[10,'2013-01']['n'],2)

    def test_duplicate_event_rejected(self):
        with self.assertRaises(ValueError):tv.aggregate_payments([event(),event()],'customer')

    def test_missing_identity_or_date_is_never_zero(self):
        for key,value in [('entity_id',None),('entity_id',0),('event_id',None),('event_date',None),('edited_at',None)]:
            r=event();r[key]=value
            with self.assertRaises(ValueError):tv.aggregate_payments([r],'customer')

    def test_counts_do_not_require_money_or_current_states(self):
        a=event();b=copy.deepcopy(a)
        b.update(amount=None,current_balance='9999999',current_finalized=False,current_finalization_date='2099-01-01')
        self.assertEqual(tv.aggregate_payments([a],'customer'),tv.aggregate_payments([b],'customer'))

    def test_composite_source_identity_separates_domains(self):
        a,_=tv.aggregate_payments([event(1,typ=3)],'customer')
        b,_=tv.aggregate_payments([event(1,typ=7)],'supplier')
        self.assertEqual(a[10,'2013-01']['n'],1);self.assertEqual(b[10,'2013-01']['n'],1)

    def test_future_events_are_rejected(self):
        with self.assertRaises(ValueError):tv.aggregate_payments([event(date='2016-01-01')],'customer')

    def test_tie_zero_and_strict_increase_positive(self):
        g=panel_fixture();first={10:'2013-01'}
        self.assertEqual(tv.candidate(g,first)['summary']['train']['positive'],0)
        g[10,'2013-04']['n']=4
        self.assertEqual(tv.candidate(g,first)['summary']['train']['positive'],1)

    def test_absence_zero_requires_explicit_coverage(self):
        g={(10,'2013-01'):{'n':1,'known_at':'2013-01-02 07:00:00'}}
        a=tv.candidate(g,{10:'2013-01'})
        b=tv.candidate(g,{10:'2013-01'},'business_ledger_academic')
        self.assertEqual(a['summary']['train']['eligible'],0)
        self.assertEqual(b['summary']['train']['eligible'],21)
        self.assertEqual(b['summary']['train']['target_zero'],21)
        self.assertEqual(b['summary']['train']['positive'],0)

    def test_history_is_three_consecutive_months_not_last_events(self):
        g=panel_fixture();del g[10,'2013-02']
        self.assertEqual(tv.candidate(g,{10:'2013-01'})['summary']['train']['eligible'],0)

    def test_version_at_prediction_is_not_prior_and_no_snapshot_required(self):
        g=panel_fixture();g[10,'2013-03']['known_at']='2013-04-01 00:00:00'
        self.assertEqual(tv.candidate(g,{10:'2013-01'})['summary']['train']['eligible'],0)
        # La comparación de eventos sin ese filtro es una sensibilidad explícita,
        # no una inferencia de estados anteriores a partir de saldos finales.
        self.assertEqual(tv.candidate(g,{10:'2013-01'},require_known_version=False)['summary']['train']['eligible'],1)

    def test_target_cut_exclusive_in_all_sensitivities(self):
        g={(10,f'2014-{m:02d}'):{'n':1,'known_at':f'2014-{m:02d}-15 07:00:00'} for m in [9,10,11,12]}
        g[10,'2014-12']['known_at']='2015-01-01 00:00:00'
        for gate in [True,False]:
            self.assertEqual(tv.candidate(g,{10:'2014-09'},require_known_version=gate)['summary']['train']['eligible'],0)

    def test_first_seen_and_future_target_do_not_create_past_entities(self):
        g=panel_fixture();g[20,'2015-12']={'n':1,'known_at':'2015-12-02 07:00:00'}
        v=tv.candidate(g,{10:'2013-01',20:'2015-12'})
        self.assertEqual(v['summary']['train']['entities'],1)
        self.assertEqual(v['summary']['validation']['eligible'],0)

    def test_cut_guard_rejects_missing_cut_and_writes(self):
        class Reader:
            def select(self,sql,params):return params
        self.assertEqual(tv.scoped(Reader(),'SELECT ct.TransactionDate FROM Sales.CustomerTransactions ct WHERE ct.TransactionDate < ?'),(dt.date(2016,1,1),))
        for sql in ['SELECT * FROM Sales.CustomerTransactions','SELECT * FROM t WHERE x < ?','DELETE FROM Sales.CustomerTransactions']:
            with self.assertRaises(ValueError):tv.scoped(Reader(),sql)

    def test_extract_templates_are_select_and_never_query_other_facts(self):
        calls=[]
        class Reader:
            consultas=[]
            def select(self,sql,params):validar_select(sql);calls.append((sql,params));return []
            def close(self):pass
        with patch.object(tv,'reader',return_value=Reader()),patch.object(tv,'guardar'):tv.extract()
        self.assertEqual(len(calls),13)
        for sql,params in calls:
            self.assertEqual(params,(tv.CUT,))
            self.assertFalse(any(x in sql for x in ['Fact.Sale','Fact.Purchase','Fact.Movement','Stock Holding']))


class EvidenciaFinancieraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (tv.OUT/'dictamen.json').exists():raise unittest.SkipTest('Generar auditoría financiera primero')
        cls.d=tv.load('comprobaciones_focalizadas.json');cls.c=tv.load('etiquetas_candidatas.json')

    def test_current_bijection_and_stable_identity(self):
        self.assertEqual(self.d['dw_count'],86475)
        self.assertEqual(self.d['source_counts'],{'customer':84338,'supplier':2137})
        self.assertTrue(self.d['side_scoped_identity_bijection'])
        self.assertEqual(self.d['mapping_mismatches'],{})
        self.assertEqual(self.d['dimension_known_identity_mismatches'],{})

    def test_operational_catalog_names_confirm_semantics(self):
        names={r['TransactionTypeID']:r['TransactionTypeName'] for r in tv.load('tipos_operacionales.json')}
        self.assertEqual(names[3],'Customer Payment Received');self.assertEqual(names[7],'Supplier Payment Issued')
        self.assertEqual({p['type_id'] for p in tv.load('perfiles_tipos.json')},{1,3,5,7})

    def test_observed_signs_dates_and_structural_nulls(self):
        for side,n in [('customer',23018),('supplier',320)]:
            p=self.d['payment_checks'][side]
            self.assertEqual(p['events'],n)
            for field in ['all_edited_on_event_day','all_negative_amount','all_null_document','all_finalized_same_day']:self.assertTrue(p[field])
        self.assertTrue(self.d['invoice_amount_identity'])
        self.assertTrue(all(p['edit_before_event']==0 and p['finalization_before_event']==0 for p in tv.load('perfiles_tipos.json')))

    def test_monthly_coverage_is_evidence_not_absolute_certificate(self):
        self.assertTrue(all(len(p['month_counts'])==36 for p in tv.load('perfiles_tipos.json')))
        coverage=tv.load('cobertura_y_supuestos.json')
        self.assertIn('academic',coverage['status']);self.assertTrue(coverage['unknowns'])
        self.assertGreaterEqual(len(coverage['assumptions']),4)
        self.assertTrue(all(not r['is_tracked_by_cdc'] for r in tv.load('metadatos_oltp.json')))

    def test_customer_account_is_recovered_from_oltp_not_unknown_dimension(self):
        self.assertEqual(self.d['dimension_unknown_rows']['customer'],43578)
        self.assertEqual(tv.load('granularidades.json')['customer']['identities'],257)
        self.assertEqual(self.d['dimensions']['customer']['max_versions'],1)

    def test_golden_customer_classes_and_within_entity_variation(self):
        v=self.c['customer_business_ledger_academic']['summary']
        self.assertEqual((v['train']['eligible'],v['train']['positive'],v['train']['negative']),(4612,1798,2814))
        self.assertEqual((v['validation']['eligible'],v['validation']['positive'],v['validation']['negative']),(2943,1198,1745))
        self.assertEqual((v['train']['entities_switching_class'],v['validation']['entities_switching_class']),(233,248))

    def test_supplier_support_is_separate_and_not_balanced_artificially(self):
        v=self.c['supplier_business_ledger_academic']['summary']
        self.assertEqual((v['train']['eligible'],v['train']['positive']),(126,16))
        self.assertEqual((v['validation']['eligible'],v['validation']['positive']),(72,8))
        self.assertEqual(v['train']['entities_single_class'],4)

    def test_coverage_sensitivity_and_edit_gate_agree_for_actual_payments(self):
        for side in ['customer','supplier']:
            self.assertEqual(self.c[side+'_business_ledger_academic']['summary'],self.c[side+'_event_semantics_without_edit_gate']['summary'])
        v=self.c['customer_unknown']['summary']
        self.assertEqual((v['train']['eligible'],v['train']['positive']),(3258,1253))
        self.assertEqual((v['validation']['eligible'],v['validation']['positive']),(2290,930))

    def test_monthly_yearly_and_entity_counts_reconcile(self):
        for c in self.c.values():
            for split,v in c['summary'].items():
                self.assertEqual(v['eligible'],v['positive']+v['negative'])
                self.assertEqual(v['eligible'],sum(m['eligible'] for m in v['monthly']))
                self.assertEqual(v['positive'],sum(m['positive'] for m in v['yearly']))
                self.assertEqual(v['eligible'],sum(e[split]['eligible'] for e in c['by_entity']))
                self.assertEqual(v['entities'],v['entities_switching_class']+v['entities_single_class'])
                self.assertTrue(all('2013-04'<=m['month']<'2016-01' for m in v['monthly']))

    def test_event_state_and_money_features_are_explicitly_separated(self):
        c=feature_catalog();by={r['nombre']:r for r in c}
        self.assertEqual(by['saldos_y_estados_actuales']['estado'],'prohibido')
        self.assertEqual(by['importe_cobrado_lag_k']['estado'],'condicionado_desactivado')
        self.assertEqual(by['cobros_lag_1']['estado'],'permitido_contrato_academico')
        self.assertEqual(by['mes_objetivo']['estado'],'prohibido')
        self.assertTrue(all(set(r)=={'nombre','formula','fuente','granularidad','disponibilidad','estado','justificacion'} for r in c))

    def test_approval_is_only_for_academic_count_contract_not_execution(self):
        d=tv.load('dictamen.json')
        self.assertTrue(d['dataset_construction_viable_under_documented_academic_contract'])
        for k in ['unconditional_or_operational_approval','construction_executed','matrices_generated','training_performed','pca_performed','test_2016_queried']:self.assertFalse(d[k])
        self.assertFalse(list(tv.OUT.rglob('*.npy')))

    def test_query_ledgers_are_read_only_and_development_sealed(self):
        for name in ['consultas_metadatos.json','consultas_oltp.json','consultas_dw.json']:
            for r in tv.load(name):
                validar_select(r['sql']);self.assertEqual(r['operation'],'SELECT');self.assertEqual(r['status'],'successful')
                if name!='consultas_metadatos.json':self.assertEqual(r['parameters'],['2016-01-01'])
        self.assertTrue(self.d['all_dates_before_2016'])
        for side in ['customer','supplier','dw']:self.assertEqual(tv.load(f'huella_{side}_antes.json'),tv.load(f'huella_{side}_despues.json'))

    def test_prior_evidence_and_manifest_integrity(self):
        for n,h in tv.load('integridad_antes.json').items():self.assertEqual(sha(tv.BASE/n),h,n)
        if (tv.OUT/'manifiesto.json').exists():
            for field in ['implementation_sha256','artifacts_sha256']:
                for n,h in tv.load('manifiesto.json')[field].items():self.assertEqual(sha(tv.BASE/n),h,n)


if __name__=='__main__':unittest.main()
