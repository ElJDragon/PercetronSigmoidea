"""Puerta temporal y perturbaciones con fixtures, sin completar datos de negocio."""
import datetime as dt
import json
import math
import unittest

from src.datamarts.pedidos.verificacion_temporal import (BASE,OUT,FEATURES,PROCESSED,calendario,verificar_existencia,autorizar_matrices,decision,datos_desarrollo,metadata)
from src.db.auditoria_dw import sha
from src.db.conexion import validar_select
from src.db.especificar_pedidos import TEST_START,clasificar


class FakeReader:
    def __init__(self):self.calls=[]
    def select(self,sql,params=()):self.calls.append((sql,params));return [{}]


class ExistenciaTests(unittest.TestCase):
    def test_last_edit_and_pick_cannot_be_used_as_birth(self):
        d=dt.date(2014,5,3)
        for evidence in [None,{'last_edited_when':dt.datetime(2014,5,3),'pick':dt.datetime(2014,5,3)}, {'kind':'last_edit','created_at':dt.datetime(2014,5,3),'provenance_verified':True}]:
            self.assertEqual(verificar_existencia(d,evidence),'creation_not_proven')
    def test_known_creation_without_provenance_is_not_a_certificate(self):
        self.assertEqual(verificar_existencia(dt.date(2014,5,3),{'created_at':dt.datetime(2014,5,3),'kind':'creation_audit'}),'creation_not_proven')
    def test_later_birth_cannot_be_retroactive(self):
        e={'created_at':dt.datetime(2014,5,4),'kind':'creation_audit','provenance_verified':True}
        self.assertEqual(verificar_existencia(dt.date(2014,5,3),e),'not_existing_on_reference_day')
    def test_other_creation_day_does_not_silently_change_target(self):
        e={'created_at':dt.datetime(2014,5,2),'kind':'creation_audit','provenance_verified':True}
        self.assertEqual(verificar_existencia(dt.date(2014,5,3),e),'creation_day_differs_from_reference')
    def test_independent_same_day_evidence_can_validate_fixture(self):
        e={'created_at':dt.datetime(2014,5,3,10),'kind':'creation_audit','provenance_verified':True}
        self.assertEqual(verificar_existencia(dt.date(2014,5,3),e),'verified_reference_day')
    def test_mixed_or_empty_population_does_not_pass_export_gate(self):
        for states in [[],['creation_not_proven'],['verified_reference_day','creation_not_proven']]:
            with self.subTest(states=states),self.assertRaises(ValueError):autorizar_matrices(states)
    def test_gate_has_no_partial_or_force_fallback(self):
        self.assertTrue(autorizar_matrices(['verified_reference_day']))
        with self.assertRaises(ValueError):autorizar_matrices(['not_existing_on_reference_day'])


class CalendarioYPerturbacionTests(unittest.TestCase):
    def test_six_names_and_formulas(self):
        x=calendario(dt.date(2024,1,1))
        self.assertEqual(tuple(x),FEATURES)
        self.assertAlmostEqual(x['mes_sin'],0);self.assertAlmostEqual(x['mes_cos'],1)
        self.assertAlmostEqual(x['semana_sin'],0);self.assertAlmostEqual(x['semana_cos'],1)
        self.assertEqual(x['dia_mes'],1);self.assertEqual(x['dias_mes'],31)
    def test_leap_year_and_cyclic_identity(self):
        x=calendario(dt.date(2024,2,29))
        self.assertEqual(x['dias_mes'],29)
        self.assertAlmostEqual(x['mes_sin']**2+x['mes_cos']**2,1)
        self.assertAlmostEqual(x['semana_sin']**2+x['semana_cos']**2,1)
        self.assertTrue(all(math.isfinite(v) for v in x.values()))
    def test_only_date_not_datetime_is_accepted(self):
        with self.assertRaises(ValueError):calendario(dt.datetime(2024,1,1))
    def test_future_perturbations_do_not_change_calendar(self):
        # Solo fixtures: nunca se infieren observaciones empresariales.
        original={'date':dt.date(2014,5,3),'picked':None,'price':10,'quantity':1,'customer':0,'last_edit':dt.datetime(2014,5,3)}
        perturbed={**original,'picked':dt.datetime(2016,5,31),'price':999999,'quantity':10000,'customer':999,'last_edit':dt.datetime(2016,6,1)}
        self.assertEqual(calendario(original['date']),calendario(perturbed['date']))
        self.assertEqual(verificar_existencia(original['date'],original),'creation_not_proven')
        self.assertEqual(verificar_existencia(perturbed['date'],perturbed),'creation_not_proven')
    def test_day_two_tie_and_null_reuse_frozen_specification(self):
        d=dt.date(2014,5,3);cut=dt.datetime(2015,1,1)
        p=dt.datetime(2014,5,5,23,59,59)
        self.assertEqual(clasificar(d,p,p,cut),'within_two_days')
        self.assertEqual(clasificar(d,None,dt.datetime(2014,5,3),cut),'missing_pick_indeterminate')
    def test_future_outcome_does_not_create_historical_training_label(self):
        d=dt.date(2014,5,3);p=dt.datetime(2016,1,1)
        self.assertEqual(clasificar(d,p,p,dt.datetime(2015,1,1)),'pick_not_observed_by_cut')


class IntegridadYAlcanceTests(unittest.TestCase):
    def test_all_data_templates_are_select_and_exclude_test(self):
        r=FakeReader();datos_desarrollo(r)
        for sql,params in r.calls:
            validar_select(sql);self.assertEqual(params,(TEST_START,));self.assertIn('WHERE o.OrderDate < ?',sql)
    def test_metadata_queries_do_not_read_business_outcomes(self):
        r=FakeReader();metadata(r)
        for sql,_ in r.calls:
            validar_select(sql);self.assertTrue('sys.' in sql or 'HAS_PERMS_BY_NAME' in sql)
            self.assertNotIn('FROM Sales.OrderLines',sql)
    def test_current_identity_is_not_treated_as_creation(self):
        meta={'columns':[],'history_configuration':[]};data={'identity':{'lines':10}}
        d=decision(meta,data)
        self.assertEqual(d['included_in_dataset'],0);self.assertEqual(d['blocked_from_materialization'],10)
        self.assertFalse(d['matrices_generated']);self.assertFalse(d['preprocessor_fitted'])
    def test_real_bridge_and_null_creation_evidence(self):
        if not (OUT/'trazabilidad_desarrollo.json').exists():self.skipTest('Ejecutar verificación de lectura primero')
        d=json.loads((OUT/'trazabilidad_desarrollo.json').read_text(encoding='utf-8'))
        self.assertEqual(d['identity']['lines'],201510);self.assertEqual(d['identity']['order_line_ids'],201510)
        self.assertEqual(d['dw_exact_bridge']['matched_lines'],201510)
        self.assertEqual(d['dw_exact_bridge']['distinct_lines'],201510)
        self.assertEqual(d['duplicate_current_order_stock']['excess'],0)
        self.assertEqual(d['backorder_lineage']['different_reference_dates'],0)
        for p in d['restricted_diagnostic_not_dataset']:self.assertEqual(p['positive_with_same_day_edit'],0)
    def test_no_matrix_or_preprocessor_created_when_gate_blocked(self):
        if not (OUT/'estado_construccion.json').exists():self.skipTest('Ejecutar verificación primero')
        d=json.loads((OUT/'estado_construccion.json').read_text(encoding='utf-8'))
        self.assertFalse(d['matrices_generated']);self.assertFalse(d['test_predictors_generated'])
        self.assertEqual(d['included_in_dataset'],0)
        self.assertFalse(list(OUT.rglob('*.npy')))
        self.assertFalse(PROCESSED.exists())
        features=json.loads((OUT/'caracteristicas_autorizadas.json').read_text(encoding='utf-8'))
        self.assertEqual(tuple(features['feature_order']),FEATURES);self.assertFalse(features['preprocessor']['fitted'])
    def test_manifest_and_previous_artifacts_unchanged(self):
        if not (OUT/'manifiesto.json').exists():self.skipTest('Ejecutar verificación primero')
        m=json.loads((OUT/'manifiesto.json').read_text(encoding='utf-8'))
        for field in ['implementation_sha256','artifacts_sha256']:
            for name,digest in m[field].items():self.assertEqual(sha(BASE/name),digest,name)
        for name,digest in json.loads((OUT/'integridad_antes.json').read_text(encoding='utf-8')).items():self.assertEqual(sha(BASE/name),digest,name)


if __name__=='__main__':unittest.main()
