"""Controles locales de SELECT y de evidencia; nunca modifican SQL Server."""
import datetime
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from src.db.conexion import Configuracion,Lector,identificador,validar_select
from src.db.auditoria_dw import OUT,BASE,cobertura,fechas,guardar,sha,metadatos


class Cursor:
    description=[('n',)]
    def __init__(self,connection):self.connection=connection;self.closed=False
    def execute(self,sql,*params):self.connection.calls.append((sql,params));return self
    def fetchall(self):return [(3,)]
    def close(self):self.closed=True


class Connection:
    def __init__(self):self.calls=[];self.cursors=[];self.closed=False
    def cursor(self):
        c=Cursor(self);self.cursors.append(c);return c
    def close(self):self.closed=True


class ConsultaSeguraTests(unittest.TestCase):
    def test_identifier_escapes_delimiter(self):
        self.assertEqual(identificador('Stock Holding'),'[Stock Holding]')
        self.assertEqual(identificador('a]b'),'[a]]b]')
        self.assertEqual(identificador('a]; DROP TABLE x--'),'[a]]; DROP TABLE x--]')
    def test_identifier_invalid(self):
        for v in ['',None,'a'*129,'a\0b']:
            with self.subTest(v=v),self.assertRaises(ValueError):identificador(v)
    def test_modifications_and_batches_rejected(self):
        for sql in ['DELETE FROM Fact.Orders','EXEC x','WITH q AS (SELECT 1 x) SELECT * FROM q','SELECT * INTO copy FROM t','SELECT 1; SELECT 2','SELECT 1 -- comentario','SELECT /* oculto */ 1','SELECT * FROM OPENROWSET(x)','UPDATE x SET a=2','SELECT NEXT VALUE FOR seq']:
            with self.subTest(sql=sql),self.assertRaises(ValueError):validar_select(sql)
    def test_literals_identifiers_and_parameterized_values(self):
        self.assertEqual(validar_select("SELECT 'DELETE; --' AS [DROP], [Order] FROM [Fact].[Order] WHERE [WWI Order ID]=?"),"SELECT 'DELETE; --' AS [DROP], [Order] FROM [Fact].[Order] WHERE [WWI Order ID]=?")
    def test_read_rejects_before_cursor_access(self):
        c=Connection();r=Lector(c)
        with self.assertRaises(ValueError):r.select('DROP TABLE x')
        self.assertEqual(c.calls,[]);self.assertEqual(c.cursors,[])
    def test_parameters_remain_separate(self):
        c=Connection();r=Lector(c);payload="x'; DROP TABLE y;--"
        self.assertEqual(r.select('SELECT ? AS n',(payload,)),[{'n':3}])
        self.assertEqual(c.calls,[('SELECT ? AS n',(payload,))])
        self.assertTrue(c.cursors[0].closed);self.assertEqual(r.consultas[0]['operation'],'SELECT')
        self.assertEqual(r.consultas[0]['status'],'successful')
    def test_connection_environment_requires_server(self):
        with patch.dict('os.environ',{},clear=True),self.assertRaisesRegex(ValueError,'SQLSERVER_SERVER'):Configuracion.entorno()
    def test_integrated_configuration_has_no_password_field(self):
        cfg=Configuracion('lpc:.\\AMBATO')
        self.assertNotIn('password',vars(cfg));self.assertEqual(cfg.database,'WideWorldImportersDW')
    def test_metadata_reads_catalogs_only(self):
        c=Connection();r=Lector(c);m=metadatos(r)
        self.assertEqual(set(m),{'columns','indexes','foreign_keys','check_constraints'})
        self.assertEqual(len(c.calls),4)
        for sql,params in c.calls:
            validar_select(sql);self.assertIn('sys.',sql);self.assertEqual(params,('Fact','Dimension'))


class EvidenciaTests(unittest.TestCase):
    def test_date_detection_uses_types_not_names(self):
        self.assertEqual(fechas([{'column_name':'Date Key','data_type':'date'},{'column_name':'Date Description','data_type':'nvarchar'},{'column_name':'Occurred','data_type':'datetime2'}]),['Date Key','Occurred'])
    def test_missing_months_and_changes_are_explicit(self):
        c=cobertura([{'month':'2020-01','n':10},{'month':'2020-03','n':40}])
        self.assertEqual(c['missing_months_between_extremes'],['2020-02'])
        self.assertEqual(c['months_observed'],2);self.assertEqual(c['volume_changes_outside_half_double'][0]['ratio'],4)
    def test_valid_to_sentinel_is_not_history(self):
        c=cobertura([{'month':'2013-01','n':1},{'month':'9999-12','n':1}],calendar=False)
        self.assertIsNone(c['missing_months_between_extremes'])
    def test_json_dates_and_hash_detect_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.json';guardar(p,{'event':datetime.date(2016,5,31),'nulls':2,'duplicates':0})
            self.assertEqual(json.loads(p.read_text(encoding='utf-8')),{'event':'2016-05-31','nulls':2,'duplicates':0})
            before=sha(p);p.write_text('{}',encoding='utf-8');self.assertNotEqual(before,sha(p))
    def test_real_counts_nulls_duplicates_and_temporal_totals(self):
        path=OUT/'perfiles_calidad.json'
        if not path.exists():self.skipTest('Ejecutar auditoría de lectura para disponer de la evidencia')
        p=json.loads(path.read_text(encoding='utf-8'))
        for name,expected in [('Order',231412),('Purchase',8367),('Movement',236667),('Stock Holding',227),('Transaction',99585)]:
            v=p['Fact.'+name];self.assertEqual(v['rows'],expected)
            self.assertEqual(v['primary_key_duplicates']['excess'],0)
            for column,t in v['temporal'].items():self.assertEqual(sum(r['n'] for r in t['monthly'])+v['columns'][column]['nulls'],expected)
        self.assertEqual(p['Fact.Order']['columns']['Picked Date Key']['nulls'],3147)
        self.assertFalse(p['Fact.Stock Holding']['temporal'])
    def test_export_manifest_integrity_and_only_select(self):
        path=OUT/'manifiesto.json'
        if not path.exists():self.skipTest('Ejecutar auditoría completa primero')
        m=json.loads(path.read_text(encoding='utf-8'))
        for name,digest in m['artifacts_sha256'].items():self.assertEqual(sha(BASE/name),digest,name)
        for file in ['consultas.json','consultas_oltp.json']:
            for q in json.loads((OUT/file).read_text(encoding='utf-8')):
                validar_select(q['sql']);self.assertEqual(q['operation'],'SELECT');self.assertEqual(q.get('status','successful'),'successful')
        self.assertTrue(m['database_fingerprints_match'])


if __name__=='__main__':unittest.main()
