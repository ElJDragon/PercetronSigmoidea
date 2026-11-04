"""Pruebas de semántica temporal y evidencia exportada, sin conexión SQL."""
import copy
import datetime as dt
import unittest
from unittest.mock import patch

from src.datamarts import compras_viabilidad as cv
from src.datamarts.reportes_compras import catalogo
from src.db.conexion import validar_select
from src.db.auditoria_dw import sha


def group(q, known='2013-01-01 00:00:00'):
    return {'q':q,'known_at':known}


class ReglasComprasTests(unittest.TestCase):
    def sample(self):
        return {(1,'2013-01'):group(2),(1,'2013-02'):group(4),(1,'2013-03'):group(3),(1,'2013-04'):group(3)}

    def test_consecutive_month_arithmetic_and_year_boundary(self):
        self.assertEqual(cv.month_add('2014-12',1),'2015-01')
        self.assertEqual(cv.month_add('2015-01',-3),'2014-10')
        self.assertEqual(cv.month_start('2015-01'),'2015-01-01 00:00:00')

    def test_median_tie_is_zero(self):
        v=cv.panel(self.sample())['scenarios']['availability_gated']['train']
        self.assertEqual((v['eligible'],v['positive'],v['negative']),(1,0,1))

    def test_strict_increase_is_positive(self):
        g=self.sample();g[1,'2013-04']['q']=4
        self.assertEqual(cv.panel(g)['scenarios']['availability_gated']['train']['positive'],1)

    def test_not_last_three_active_months(self):
        g=self.sample();g[1,'2013-05']=g.pop((1,'2013-03'))
        self.assertEqual(cv.panel(g)['scenarios']['retrospective']['train']['eligible'],0)

    def test_absences_unknown_by_default(self):
        g={(1,'2013-01'):group(10)}
        self.assertEqual(cv.panel(g)['scenarios']['retrospective']['train']['eligible'],0)
        diagnostic=cv.panel(g,zero_policy='conditional_ledger')['scenarios']['retrospective']['train']
        self.assertEqual((diagnostic['eligible'],diagnostic['target_zero']),(21,21))

    def test_invalid_zero_policy_rejected(self):
        with self.assertRaises(ValueError):cv.panel(self.sample(),zero_policy='always_zero')
        with self.assertRaises(ValueError):cv.panel(self.sample(),mode='balanced_threshold')

    def test_whole_historical_aggregate_rejected_when_version_late(self):
        g=self.sample();g[1,'2013-03']['known_at']='2013-04-01 07:00:00'
        v=cv.panel(g)
        self.assertEqual(v['scenarios']['retrospective']['train']['eligible'],1)
        self.assertEqual(v['scenarios']['availability_gated']['train']['eligible'],0)
        self.assertEqual(v['excluded_reasons_nonexclusive']['historical_version_after_prediction'],1)

    def test_exact_prediction_timestamp_is_not_prior(self):
        g=self.sample();g[1,'2013-03']['known_at']='2013-04-01 00:00:00'
        self.assertEqual(cv.panel(g)['scenarios']['availability_gated']['train']['eligible'],0)

    def test_outcome_cut_applies_even_to_retrospective_diagnostic(self):
        g={(1,f'2014-{m:02}'):group(1) for m in [9,10,11,12]}
        g[1,'2014-12']['known_at']='2015-01-01 00:00:00'
        v=cv.panel(g)
        for scenario in v['scenarios'].values():self.assertEqual(scenario['train']['eligible'],0)
        self.assertEqual(v['excluded_reasons_nonexclusive']['target_version_after_split_cut'],1)

    def test_future_target_cannot_change_development(self):
        a=self.sample();b=copy.deepcopy(a);b[1,'2016-01']=group(999999,'2016-02-01 00:00:00')
        self.assertEqual(cv.panel(a),cv.panel(b))

    def test_activity_and_volume_are_different_targets(self):
        a=self.sample()
        self.assertEqual(cv.panel(a,mode='activity')['scenarios']['retrospective']['train']['positive'],1)
        self.assertEqual(cv.panel(a)['scenarios']['retrospective']['train']['positive'],0)

    def test_query_boundary_rejects_missing_cut_or_writes(self):
        class Reader:
            def select(self,sql,params):return params
        self.assertEqual(cv.scoped(Reader(),'SELECT po.OrderDate FROM Purchasing.PurchaseOrders po WHERE po.OrderDate < ?'),(dt.date(2016,1,1),))
        for sql in ['SELECT * FROM Purchasing.PurchaseOrders','DELETE FROM Purchasing.PurchaseOrders','SELECT x FROM t WHERE x < ?']:
            with self.assertRaises(ValueError):cv.scoped(Reader(),sql)

    def test_extract_templates_use_only_scoped_selects(self):
        calls=[]
        class Reader:
            consultas=[]
            def select(self,sql,params):
                validar_select(sql);calls.append((sql,params));return []
            def close(self):pass
        with patch.object(cv,'reader',return_value=Reader()),patch.object(cv,'guardar'):
            cv.extract()
        self.assertEqual(len(calls),11)
        for sql,params in calls:
            self.assertEqual(params,(cv.CUT,))
            self.assertNotIn('Fact.Movement',sql)
            self.assertNotIn('Fact.Transaction',sql)


class EvidenciaComprasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (cv.OUT/'dictamen.json').exists():raise unittest.SkipTest('Generar evidencia de Compras primero')
        cls.d=cv.load('comprobaciones_focalizadas.json');cls.c=cv.load('etiquetas_candidatas.json');cls.g=cv.load('granularidades.json')

    def test_current_source_and_dw_bijection_without_duplication(self):
        self.assertEqual((self.d['orders'],self.d['lines'],self.d['facts']),(1819,7307,7307))
        self.assertTrue(self.d['line_to_fact_bijection'])
        self.assertEqual(self.d['mapping_mismatches'],{})
        self.assertEqual(self.d['duplicate_order_item_pairs'],0)

    def test_development_scope_and_no_2016_target_months(self):
        self.assertEqual(self.d['date_max'],'2015-12-31');self.assertEqual(self.d['order_months'],36)
        for name in ['ordenes_desarrollo.json','hechos_desarrollo.json']:
            key='OrderDate' if name.startswith('ordenes') else 'order_date'
            self.assertTrue(all(r[key]<'2016-01-01' for r in cv.load(name)))
        for v in self.c.values():
            for ss in v['scenarios'].values():
                for summary in ss.values():
                    self.assertTrue(all('2013-04'<=r['month']<'2016-01' for r in summary['monthly']))

    def test_unit_history_does_not_certify_original_ordered_quantity(self):
        self.assertEqual(self.d['unit_history_issues'],{})
        self.assertEqual(self.d['product_unit_changes_before_2016'],0)
        self.assertEqual(self.d['current_partial_lines'],0)
        contract=cv.load('contrato_datos.json')
        self.assertFalse(contract['can_build_dataset'])
        self.assertIn('desconocida',contract['zero_policy'])

    def test_dimensional_issues_are_not_resolved_by_arbitrary_join(self):
        self.assertEqual(self.d['dimension_checks']['supplier']['fact_dimension_version_outside_order_midnight'],94)
        self.assertEqual(self.d['dimension_checks']['supplier']['overlapping_pairs'][0]['business_id'],3)
        self.assertEqual(self.d['dimension_checks']['item']['overlapping_pairs'],[])

    def test_granularity_and_intermittency(self):
        self.assertEqual(self.g['product']['identities'],219)
        self.assertEqual(self.g['supplier']['identities'],6)
        self.assertEqual(self.g['product']['active_entity_months'],521)
        self.assertEqual(self.g['supplier']['active_entity_months'],76)
        self.assertEqual(self.g['product']['activity_months_histogram']['1'],209)
        self.assertEqual(self.g['product']['unobserved_month_slots_conditional_not_certified_zero'],7363)

    def test_monthly_classes_reconcile_with_split_summary(self):
        for cc in self.c.values():
            for ss in cc['scenarios'].values():
                for v in ss.values():
                    self.assertEqual(v['eligible'],v['positive']+v['negative'])
                    self.assertEqual(v['eligible'],sum(m['eligible'] for m in v['monthly']))
                    self.assertEqual(v['positive'],sum(m['positive'] for m in v['monthly']))
                    self.assertEqual(v['entities'],v['entities_switching_class']+v['entities_single_class'])

    def test_cut_respecting_retrospective_counts(self):
        v=self.c['volume_product']['scenarios']['retrospective']
        self.assertEqual((v['train']['eligible'],v['train']['positive']),(4590,168))
        self.assertEqual((v['validation']['eligible'],v['validation']['positive']),(2619,89))
        f=self.c['order_frequency_supplier']['scenarios']['retrospective']
        self.assertEqual((f['train']['eligible'],f['train']['positive']),(124,18))
        self.assertEqual((f['validation']['eligible'],f['validation']['positive']),(70,7))

    def test_temporal_gate_removes_active_support(self):
        for key in ['volume_product_observed_only','volume_product_supplier_observed_only','order_frequency_supplier_observed_only']:
            self.assertEqual(self.c[key]['scenarios']['availability_gated']['train']['eligible'],0)
            self.assertEqual(self.c[key]['scenarios']['availability_gated']['validation']['eligible'],0)
        self.assertEqual(self.c['volume_product']['scenarios']['availability_gated']['validation']['positive'],0)

    def test_activity_does_not_change_within_entity(self):
        for split in self.c['order_activity_supplier']['scenarios']['retrospective'].values():
            self.assertEqual(split['entities_switching_class'],0)

    def test_features_explicitly_disable_unverified_history_and_outcomes(self):
        fs=catalogo()
        self.assertTrue(all(f['estado']=='condicionado_desactivado' for f in fs if f['nombre'].startswith('cantidad_')))
        self.assertEqual(sum(f['estado']=='permitido_candidato_sin_dataset' for f in fs),3)
        self.assertTrue(all(f['estado']=='prohibido' for f in fs if f['nombre'] in ['ultima_edicion_como_alta','recepcion_y_finalizacion','dimension_futura']))
        self.assertTrue(all(set(f)=={'nombre','formula','fuente','granularidad','disponibilidad','riesgo','estado'} for f in fs))

    def test_read_only_successful_query_ledgers_and_fingerprints(self):
        for name in ['consultas_metadatos.json','consultas_oltp.json','consultas_dw.json']:
            for row in cv.load(name):
                validar_select(row['sql']);self.assertEqual(row['operation'],'SELECT');self.assertEqual(row['status'],'successful')
                if name!='consultas_metadatos.json':self.assertEqual(row['parameters'],['2016-01-01'])
        for db in ['oltp','dw']:self.assertEqual(cv.load(f'huella_{db}_antes.json'),cv.load(f'huella_{db}_despues.json'))

    def test_no_dataset_models_or_new_mart_investigation(self):
        d=cv.load('dictamen.json')
        for field in ['dataset_authorized','training_viable_now','matrices_generated','training_performed','pca_performed']:self.assertFalse(d[field])
        self.assertEqual(d['authorized_observations'],{'train':0,'validation':0})
        self.assertEqual(d['next_separate_task'],'Transacciones financieras')
        self.assertFalse(list(cv.OUT.rglob('*.npy')))

    def test_prior_files_and_manifest_integrity(self):
        for n,h in cv.load('integridad_antes.json').items():self.assertEqual(sha(cv.BASE/n),h,n)
        if (cv.OUT/'manifiesto.json').exists():
            m=cv.load('manifiesto.json')
            for field in ['implementation_sha256','artifacts_sha256']:
                for n,h in m[field].items():self.assertEqual(sha(cv.BASE/n),h,n)


if __name__=='__main__':unittest.main()
