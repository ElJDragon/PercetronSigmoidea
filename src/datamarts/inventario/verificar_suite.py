"""Registra la suite completa sin modificar pruebas previas.

Los ajustes de modelos y PCA que contiene la suite son fixtures unitarios,
no entrenamiento ni visualización del nuevo experimento de Inventario.
"""
import io
import json
import unittest

from src.db.auditoria_dw import guardar
from .construir import BASE,OUT,verify_prior,manifest

def main():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(BASE/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (OUT/'suite_existente.txt').write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    summary=dict(status='successful' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
        failures=[dict(test=str(test),traceback=trace) for test,trace in result.failures],
        errors=[dict(test=str(test),traceback=trace) for test,trace in result.errors],
        skipped=[dict(test=str(test),reason=reason) for test,reason in result.skipped],
        previous_tests_modified=False,experimental_training=False,experimental_PCA=False,
        synthetic_unit_fixtures_may_fit_models_or_PCA=True,preserved_files=len(verify_prior()))
    guardar(OUT/'suite_existente.json',summary)
    validation=OUT/'validaciones.json'
    if validation.exists():
        data=json.loads(validation.read_text(encoding='utf-8'))
        data['full_suite']=dict(status=summary['status'],tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),report='suite_existente.json')
        guardar(validation,data)
    manifest()
    print(json.dumps({k:summary[k] for k in ['status','tests_run','preserved_files']}))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
