"""Verificación documental: integridad, cifras, inclusión y reproducción sin modelos."""
from contextlib import ExitStack, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from generar_informe import BASE, OUT, main, read, sha

sys.path.insert(0,str(BASE))


def run():
    checks=[]; claims=[]
    def check(name,condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def claim(name,path,keys,expected):
        value=read(path)
        for key in keys:value=value[key]
        check(name,value==expected)
        claims.append({'claim':name,'source':path,'keys':keys,'value':value,'sha256':sha(BASE/path)})
    prior=read('informe/evidencias/integridad_antes.json')
    check('Archivos anteriores intactos antes de verificar',all(sha(BASE/p)==d for p,d in prior.items()))
    audit='resultados/auditoria/perfilado_datos.json'; ds=['datasets','datos_limpios.csv']
    for name,keys,value in [
        ('Líneas originales',ds+['rows'],228265),('Columnas de origen',ds+['column_count'],30),
        ('Facturas distintas',ds+['keys','InvoiceID','unique'],70510),('Nombres de producto',ds+['columns','NombreProducto','unique_non_null'],227),
        ('Duplicados completos',ds+['raw_duplicate_excess'],0),('Duplicados de línea',ds+['keys','InvoiceLineID','duplicate_excess'],0),
        ('Marca faltante',ds+['columns','Marca','missing'],209328),('Tamaño faltante',ds+['columns','Tamanio','missing'],67315),
        ('Grupo faltante',ds+['columns','GrupoCompra','missing'],84297),('Fecha inicial',ds+['columns','Fecha','min'],'2013-01-01 00:00:00+00:00'),
        ('Fecha final',ds+['columns','Fecha','max'],'2016-05-31 00:00:00+00:00'),
        ('Cantidad por precio',ds+['monetary_checks','cantidad_por_precio','violations'],0),('Base más impuesto',ds+['monetary_checks','base_mas_impuesto','violations'],0)]:claim(name,audit,keys,value)
    claim('Recortes de espacios en NombreCliente (no Vendedor)','resultados/auditoria/comparacion_csv.json',['columns','NombreCliente','whitespace_only_differences'],268)
    obs=pd.read_csv(BASE/'datos/Procesados/fase_03/observaciones.csv',usecols=['primer_mes_observado_conocido','particion'])
    calendar=pd.read_csv(BASE/'datos/Procesados/fase_03/producto_mes.csv',usecols=['producto_provisional'])
    check('Calendario producto-mes: 9019 filas',len(calendar)==9019)
    check('Observaciones conocidas al corte: 8792 filas',len(obs)==8792)
    check('Reconocidos antes del corte: 8792 filas',obs.primer_mes_observado_conocido.notna().sum()==8792)
    check('No se materializaron 288 pares anteriores a primera aparición',41*227-len(calendar)==288)
    claim('Train temporal','resultados/fase_03/manifiesto.json',['partitions','train'],['2013-04','2014-12'])
    claim('Validación temporal','resultados/fase_03/manifiesto.json',['partitions','val'],['2015-01','2015-12'])
    claim('Test temporal','resultados/fase_03/manifiesto.json',['partitions','test'],['2016-01','2016-05'])
    frozen=read('resultados/fase_06/congelacion_experimental.json')
    for folder,name,epoch,bias in [('fase_04','perceptron',70,0.010000000000000004),('fase_05','logistica',200,0.08539078966952132)]:
        path=f'resultados/{folder}/modelo_seleccionado.json';m=read(path)
        check('Estado exacto congelado '+name,m==frozen['models'][name]['state'])
        check('Hash congelado '+name,sha(BASE/path)==frozen['models'][name]['sha256'])
        claim('Checkpoint '+name,path,['selected_epoch'],epoch);claim('Sesgo '+name,path,['bias'],bias)
        tex=(OUT/f'generado/pesos_{folder}.tex').read_text(encoding='utf-8')
        for i,(feature,w) in enumerate(zip(m['feature_names'],m['weights'])):
            check(f'Peso {name} {feature} exacto',f'{{{feature}}}' in tex and f'& {repr(w)} '+r'\\' in tex)
            claims.append({'claim':f'Peso {name}: {feature}','source':path,'keys':['weights',i],'value':w,'sha256':sha(BASE/path)})
    stats=read('informe/evidencias/analisis_convergencia.json')
    check('Perceptrón: 0 épocas sin actualizaciones',stats['perceptron']['zero_updates_epochs']==0)
    check('Perceptrón: 49 transiciones crecientes',stats['perceptron']['increased_epochs']==49)
    check('Logística: ninguna subida estricta de pérdida',stats['logistica']['loss_increases_strict']==0)
    check('Logística: máximo de épocas, no tolerancia',not stats['logistica']['converged_by_tolerance'])
    loss=pd.read_csv(OUT/'evidencias/convergencia_logistica_por_epoca.csv',float_precision='round_trip')
    previous=np.r_[stats['logistica']['initial_train_loss'],loss.train_loss[:-1]]
    check('Delta L por época exacta',np.array_equal(previous-loss.train_loss,loss.delta_L))
    check('r por época exacta',np.array_equal(abs(loss.train_loss-previous)/np.maximum(1,abs(previous)),loss.relative_change))
    check('Pérdida final coincide con métricas guardadas',loss.train_loss.iloc[-1]==read('resultados/fase_05/metricas.json')['train']['cross_entropy'])
    check('Verificación numérica original de gradientes aprobada',all(c['passed'] for c in read('resultados/fase_05/verificacion_gradientes.json').values()))
    # Ejecutar la exportación con las operaciones prohibidas interceptadas.
    generated=list((OUT/'generado').glob('*.tex'))+list((OUT/'figuras').glob('*.png'))+list((OUT/'evidencias').glob('convergencia*.csv'))+[OUT/'evidencias/analisis_convergencia.json',OUT/'evidencias/trazabilidad.json']
    before={str(p.relative_to(BASE)):sha(p) for p in generated}
    guarded=[]
    with ExitStack() as stack:
        for mod,cls,names in [
            ('src.models.perceptron','PerceptronSimple',['fit','predict','decision_function']),
            ('src.models.logistic_regression','LogisticRegressionManual',['fit','predict','predict_proba','decision_function']),
            ('src.pipeline.preprocesamiento','PreprocesadorComun',['fit','transform']),
            ('src.visualization.pca','PCA2D',['fit','transform'])]:
            obj=getattr(importlib.import_module(mod),cls)
            for method in names:
                guarded.append(f'{cls}.{method}')
                stack.enter_context(patch.object(obj,method,side_effect=AssertionError('Operación prohibida en informe')))
        for mod in ['src.experiments.entrenar_perceptron','src.experiments.entrenar_logistica','src.experiments.evaluacion_final','src.pipeline.construir_dataset','src.experiments.visualizar_pca']:
            obj=importlib.import_module(mod);guarded.append(mod+'.run')
            stack.enter_context(patch.object(obj,'run',side_effect=AssertionError('No ejecutar experimentos')))
        with redirect_stdout(io.StringIO()):main()
    check('Reproducción binaria de tablas, figuras y evidencia calculada',before=={p:sha(BASE/p) for p in before})
    tex='\n'.join(p.read_text(encoding='utf-8') for p in (OUT/'secciones').glob('*.tex'))
    figures=re.findall(r'\\figura\{([^}]+)\}',tex)
    check('21 figuras únicas insertadas',len(figures)==len(set(figures))==21)
    check('Todas las imágenes existen',all((OUT/'figuras'/p).is_file() for p in figures))
    tables=re.findall(r'\\input\{generado/([^}]+)\}',tex)
    check('13 tablas generadas insertadas',len(tables)==len(set(tables))==13)
    check('Todas las tablas existen',all((OUT/'generado'/f'{p}.tex').is_file() for p in tables))
    check('Sin títulos académicos por fases',not re.search(r'\\(?:sub)*section\{[^}]*[Ff]ase\s+\d',tex))
    check('Alcance obligatorio explícito','obligatoria' in tex and 'demás Data Marts' in tex)
    log=(OUT/'main.log').read_text(encoding='utf-8',errors='replace')
    check('PDF compilado',bool(re.search(r'Output written on main.pdf',log)))
    check('Sin referencias o citas indefinidas','undefined' not in log)
    check('Sin desbordamientos horizontales o verticales','Overfull' not in log)
    check('Sin avisos de recompilación','Rerun to get' not in log)
    check('Sin destinos duplicados','duplicate ignored' not in log)
    aux=(OUT/'main.aux').read_text(encoding='utf-8')
    check('21 referencias de figuras resueltas',len(re.findall(r'\\newlabel\{fig:',aux))==21)
    check('13 referencias de tablas resueltas',len(re.findall(r'\\newlabel\{tab:',aux))==13)
    check('7 referencias bibliográficas resueltas',len(re.findall(r'\\bibcite\{',aux))==7)
    check('Archivos anteriores intactos después de verificar',all(sha(BASE/p)==d for p,d in prior.items()))
    (OUT/'evidencias/afirmaciones_verificadas.json').write_text(json.dumps(claims,ensure_ascii=False,indent=2),encoding='utf-8')
    result={'successful':True,'checks_passed':len(checks),'checks':checks,'protected_files':len(prior),'figures':len(figures),'tables':len(tables),'reproduced_artifacts':len(before),'forbidden_operations_guarded':guarded,'pdf_sha256':sha(OUT/'main.pdf'),'pdf_pages':int(re.search(r'Output written on main.pdf \((\d+) pages',log)[1]),'visual_review':'Pendiente de registrar la última inspección visual; el script no puede sustituirla.','prior_suite':'104 tests, 0 failures/errors/skipped; resultado histórico preservado en resultados/fase_08/validaciones.json','remaining_information':['Ciclo y fecha de entrega actuales','Validación operacional de cierres, claves y unidades','Ampliación obligatoria a los demás Data Marts, sin implementar en este informe']}
    (OUT/'evidencias/verificacion_informe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['successful','checks_passed','protected_files','figures','tables','reproduced_artifacts','pdf_pages']},indent=2))


if __name__=='__main__':run()
