"""Verificación documental integral, sin entrenar ni recalcular predicciones/PCA."""
from contextlib import ExitStack,redirect_stdout
import importlib
import io
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

from generar_transacciones import BASE,ROOT,OUT,EVIDENCE,SOURCE,FIGURES,main,read,sha,pct
sys.path.insert(0,str(BASE))


def run():
    checks=[];claims=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def claim(name,path,keys,expected):
        value=read(path)
        for key in keys:value=value[key]
        check(name,value==expected);claims.append(dict(claim=name,source=str(path.relative_to(BASE)),keys=keys,value=value,sha256=sha(path)))
    previous=read(EVIDENCE/'integridad_experimentos_antes.json');sales=read(EVIDENCE/'ventas_preservadas.json')
    check('Experimentos intactos al iniciar',all(sha(BASE/n)==h for n,h in previous.items()))
    check('Figuras, tablas y desarrollo detallado de Ventas intactos',all(sha(BASE/n)==h for n,h in sales.items()))
    guarded=[]
    generated=list(OUT.glob('*.tex'))+list((ROOT/'figuras/transacciones').glob('*.png'))+[ROOT/'secciones/titulos_transacciones.tex',EVIDENCE/'trazabilidad.json',EVIDENCE/'cifras_contrastadas.json']
    before={str(p.relative_to(BASE)):sha(p) for p in generated}
    with ExitStack() as stack:
        for module,cls,methods in [('src.models.perceptron','PerceptronSimple',['fit','predict','decision_function']),
                ('src.models.logistic_regression','LogisticRegressionManual',['fit','predict','predict_proba','decision_function']),
                ('src.visualization.pca','PCA2D',['fit','transform']),('src.datamarts.transacciones.preprocesamiento','PreprocesadorFinanciero',['fit','transform']),
                ('src.db.conexion','Configuracion',['conectar'])]:
            obj=getattr(importlib.import_module(module),cls)
            for method in methods:
                guarded.append(cls+'.'+method);stack.enter_context(patch.object(obj,method,side_effect=AssertionError('Operación experimental prohibida en informe')))
        with redirect_stdout(io.StringIO()):main()
    check('Tablas, figuras y evidencia documental reproducibles byte a byte',before=={n:sha(BASE/n) for n in before})
    global_test=SOURCE/'evaluacion_final/metricas_globales.json';counts=SOURCE/'evaluacion_final/conteos.json'
    for name,keys,value in [('Train filas',['train','eligible'],4612),('Validación filas',['val','eligible'],2943)]:claim(name,SOURCE/'dataset/conteos.json',keys,value)
    for name,keys,value in [('Test filas',['primary_n'],1284),('Test común',['common_n'],1276),('Exclusión solo histórico',['excluded_only_from_historical'],8)]:claim(name,global_test,keys,value)
    claim('Test positivos',counts,['positive'],525);claim('Test negativos',counts,['negative'],759)
    for name,cm in [('perceptron',[[346,413],[69,456]]),('logistic',[[579,180],[266,259]])]:claim('Matriz principal '+name,global_test,['primary',name,'confusion_matrix'],cm)
    test=read(global_test)
    for scope,filename in [('primary','test_principal'),('common','test_comun')]:
        tex=(OUT/f'{filename}.tex').read_text(encoding='utf-8')
        for model,values in test[scope].items():
            for key in ['accuracy','precision','recall','specificity','f1','balanced_accuracy']:check(f'Tabla {scope}: {model}.{key}',pct(values[key]) in tex)
    frozen=read(SOURCE/'entrenamiento/congelacion.json');macro=(OUT/'cifras.tex').read_text(encoding='utf-8')
    for key,epoch,biasmacro in [('perceptron',3,'TFBiasP'),('logistic',399,'TFBiasL')]:
        p=BASE/frozen['models'][key]['path'];m=read(p)
        check('Hash congelado '+key,sha(p)==frozen['models'][key]['sha256'])
        claim('Checkpoint '+key,p,['selected_epoch'],epoch)
        check('16 pesos '+key,len(m['weights'])==len(m['feature_names'])==16)
        table=(OUT/f'pesos_{key}.tex').read_text(encoding='utf-8')
        for i,(name,w) in enumerate(zip(m['feature_names'],m['weights'])):
            check(f'Peso exacto {key}/{name}',r'\archivo{'+name+'}' in table and '& '+repr(w)+r' \\' in table)
            claims.append(dict(claim=f'Peso {key}/{name}',source=str(p.relative_to(BASE)),keys=['weights',i],value=w,sha256=sha(p)))
        check('Sesgo exacto '+key,'{\\'+biasmacro+'}{'+repr(m['bias'])+'}' in macro)
    claim('Umbral logístico',BASE/frozen['models']['logistic']['path'],['params','threshold'],.5)
    evidence=read(EVIDENCE/'cifras_contrastadas.json')
    check('Catálogo de 16 nombres y fórmulas',len(re.findall(r'\\archivo\{', (OUT/'caracteristicas.tex').read_text(encoding='utf-8')))==16)
    for key,value in [('PC1_percent',27.89633478995603),('PC2_percent',17.1250590770698),('retained_percent',45.02139386702583)]:claim('PCA '+key,SOURCE/'pca/varianza_explicada.json',[key],value)
    conv=evidence['convergence_selected'];check('Perceptrón sigue actualizando',conv['perceptron']['last_online_errors']==1810 and conv['perceptron']['zero_error_epochs']==[])
    check('Logística termina por máximo, sin tolerancia',conv['logistic']['termination']=='max_epochs' and not conv['logistic']['gradient_tolerance_reached'])
    for key,run,n in [('perceptron','p_seed7',100),('logistic','l_eta010',400)]:
        csv=SOURCE/'entrenamiento/runs'/run/'historial.csv'
        lines=csv.read_text(encoding='utf-8').splitlines();check('Historial completo '+key,len(lines)==n+1)
    claim('Nuevas cuentas evaluables',SOURCE/'evaluacion_final/cohortes.json',['new_in_test','accounts'],3)
    claim('Etiquetas nuevas: 4 negativas',SOURCE/'evaluacion_final/cohortes.json',['new_in_test','metrics','primary','perceptron','support','0'],4)
    claim('Etiquetas nuevas: 1 positiva',SOURCE/'evaluacion_final/cohortes.json',['new_in_test','metrics','primary','perceptron','support','1'],1)
    tex='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'secciones').glob('*.tex'))
    figures=re.findall(r'\\figura\{([^}]+)\}',tex)
    check('39 figuras únicas insertadas',len(figures)==len(set(figures))==39)
    check('Todas las figuras existen',all((ROOT/'figuras'/name).is_file() for name in figures))
    for source,dest,label,title in FIGURES:
        check('Copia binaria '+dest,sha(SOURCE/source)==sha(ROOT/'figuras/transacciones'/dest))
        check('Figura con referencia cruzada '+label,r'\ref{fig:'+label+'}' in tex)
    table_files=list((ROOT/'generado').glob('*.tex'))+list(OUT.glob('*.tex'))
    table_count=sum(len(re.findall(r'\\begin\{(?:table|longtable)\}',p.read_text(encoding='utf-8'))) for p in table_files)
    check('26 tablas académicas',table_count==26)
    aux=(ROOT/'main.aux').read_text(encoding='utf-8')
    check('39 referencias de figuras resueltas',len(re.findall(r'\\newlabel\{fig:',aux))==39)
    check('26 referencias de tablas resueltas',len(re.findall(r'\\newlabel\{tab:',aux))==26)
    check('7 referencias bibliográficas conservadas',len(re.findall(r'\\bibcite\{',aux))==7)
    check('Títulos sin nombres de fases',not re.search(r'\\(?:sub)*section\{[^}]*[Ff]ase\s+\d',tex))
    check('Pendientes no presentados como completados','Inventario/Movimientos continúa pendiente' in tex and 'sin dataset ni clasificadores aprobados' in tex)
    log=(ROOT/'main.log').read_text(encoding='utf-8',errors='replace')
    for name,token in [('Sin referencias/citas indefinidas','undefined'),('Sin desbordamientos','Overfull'),('Sin recompilación pendiente','Rerun to get'),('Sin destinos duplicados','duplicate ignored')]:check(name,token not in log)
    check('PDF compilado','Output written on main.pdf' in log)
    check('Experimentos intactos al finalizar',all(sha(BASE/n)==h for n,h in previous.items()))
    check('Ventas aprobadas intactas al finalizar',all(sha(BASE/n)==h for n,h in sales.items()))
    pages=int(re.search(r'Output written on main.pdf \((\d+) pages',log)[1])
    result=dict(status='successful',documentary_checks=len(checks),checks=checks,pages=pages,figures=39,tables=26,new_figures=18,new_tables=13,
        experimental_files_preserved=len(previous),sales_artifacts_preserved=len(sales),reproduced_documentary_artifacts=len(before),forbidden_operations_guarded=guarded,
        classifier_training=False,predictions_recomputed=False,PCA_recomputed=False,SQL_queries=0,experimental_metrics_modified=False,
        pdf_sha256=sha(ROOT/'main.pdf'),latex_sha256=sha(ROOT/'main.tex'),visual_review='registrar tras inspección PNG final',
        historical_experimental_suite=read(SOURCE/'evaluacion_final/validaciones.json')['tests_run'],experimental_suite_rerun_for_document=False)
    visual_path=EVIDENCE/'revision_visual.json'
    if visual_path.is_file():
        visual=read(visual_path)
        if visual.get('pdf_sha256')==result['pdf_sha256']:
            result['visual_review']=visual
    (EVIDENCE/'afirmaciones_verificadas.json').write_text(json.dumps(claims,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (EVIDENCE/'verificacion_informe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','documentary_checks','pages','figures','tables','experimental_files_preserved','reproduced_documentary_artifacts']}))


if __name__=='__main__':run()
