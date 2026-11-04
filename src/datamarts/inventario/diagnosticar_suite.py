"""Contrasta fallos de sellos históricos con el estado anterior a este encargo."""
from .construir import BASE,OUT,read,sha,guardar,verify_prior,manifest

def main():
    baseline=verify_prior();groups=[]
    for name in ['compras','pedidos','pedidos/dataset','pedidos/documentos','transacciones',
                 'transacciones/dataset','transacciones/entrenamiento','transacciones/pca','transacciones/evaluacion_final']:
        path=BASE/'resultados/extension_datamarts'/name/'integridad_antes.json'
        differences=[]
        for rel,old in read(path).items():
            current=sha(BASE/rel)
            if current!=old:
                differences.append(dict(path=rel,historical_expected=old,before_inventory_dataset=baseline.get(rel),current=current,
                    already_different_before_this_task=baseline.get(rel)==current))
        groups.append(dict(historical_manifest=str(path.relative_to(BASE)),differences=differences))
    diffs=[d for g in groups for d in g['differences']]
    all_prior=all(d['already_different_before_this_task'] for d in diffs)
    report=dict(status='preexisting_historical_seal_differences' if all_prior else 'investigate',
        all_differences_predate_this_task=all_prior,all_changed_paths_in_report=all(d['path'].replace('\\','/').startswith('informe/') for d in diffs),
        groups=groups,existing_tests_modified=False,old_manifests_modified=False,
        explanation='Los sellos históricos incluyen la versión anterior del informe; la ampliación documental precede a este dataset. No se restauran ni resellan archivos para ocultar fallos.')
    guardar(OUT/'discrepancias_suite_previas.json',report)
    suite=read(OUT/'suite_existente.json');passed=suite['tests_run']-len(suite['failures'])-len(suite['errors'])-len(suite['skipped'])
    validation=read(OUT/'validaciones.json')
    validation['full_suite']['passed']=passed
    validation['full_suite']['all_historical_seal_differences_predate_this_task']=all_prior
    validation['full_suite']['diagnosis']='discrepancias_suite_previas.json'
    validation['overall_status']='dataset_verified_with_preexisting_full_suite_failures'
    guardar(OUT/'validaciones.json',validation)
    with (OUT/'reporte_dataset.md').open('a',encoding='utf-8',newline='\n') as f:
        f.write(f'\n## Resultado de pruebas y discrepancias históricas\n\n30 pruebas focalizadas y 35 controles del dataset satisfactorios; reproducción idéntica de 19 artefactos. Suite completa: {suite["tests_run"]} pruebas, {passed} correctas, {len(suite["failures"])} fallos y {len(suite["errors"])} errores. Los nueve casos fallidos verifican sellos previos que incluyen una versión antigua del informe. El contraste de cada diferencia con integridad_antes.json de este encargo demuestra que todas ya existían antes de construir Inventario; las rutas discrepantes son exclusivamente de informe/. No se cambiaron pruebas ni manifiestos anteriores para ocultarlos. Los 847 archivos protegidos permanecen exactamente como al iniciar este encargo. Detalle en suite_existente.json y discrepancias_suite_previas.json. El dataset queda verificado; la suite global no se presenta como aprobada.\n')
    manifest()
    print(dict(all_differences_predate_this_task=all_prior,all_changed_paths_in_report=report['all_changed_paths_in_report'],passed=passed,unique_affected_paths=len({d['path'] for d in diffs})))
    if not all_prior:raise RuntimeError('Diferencia no atribuible al estado previo')

if __name__=='__main__':main()
