"""Exportación exclusivamente documental desde resultados congelados de Transacciones."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import pandas as pd

BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'informe'
SOURCE=BASE/'resultados/extension_datamarts/transacciones'
OUT=ROOT/'generado/transacciones'
EVIDENCE=ROOT/'evidencias/transacciones'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def guardar(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def num(x,d=3):return f'{float(x):.{d}f}'.replace('.',',')
def pct(x):return num(100*x)
def literal(s):return r'\archivo{'+str(s)+'}'


FIGURES=[
 ('entrenamiento/graficos/perceptron/errores_entrenamiento.png','p_errores.png','tf:p-errores','Errores online del perceptrón financiero'),
 ('entrenamiento/graficos/perceptron/actualizaciones.png','p_actualizaciones.png','tf:p-actualizaciones','Actualizaciones del perceptrón financiero'),
 ('entrenamiento/graficos/perceptron/accuracy_por_epoca.png','p_accuracy.png','tf:p-accuracy','Accuracy por época del perceptrón financiero'),
 ('entrenamiento/graficos/perceptron/balanced_accuracy_validacion.png','p_ba.png','tf:p-ba','BA de validación del perceptrón financiero'),
 ('entrenamiento/graficos/logistica/entropia_cruzada.png','l_perdida.png','tf:l-perdida','Entropía cruzada logística de Transacciones'),
 ('entrenamiento/graficos/logistica/norma_gradiente.png','l_gradiente.png','tf:l-gradiente','Gradiente logístico de Transacciones'),
 ('entrenamiento/graficos/logistica/accuracy_por_epoca.png','l_accuracy.png','tf:l-accuracy','Accuracy logística por época en Transacciones'),
 ('entrenamiento/graficos/logistica/balanced_accuracy_validacion.png','l_ba.png','tf:l-ba','BA logística por época en Transacciones'),
 ('pca/graficos/dispersion_train.png','pca_train.png','tf:pca-train','PCA financiero de entrenamiento'),
 ('pca/graficos/dispersion_val.png','pca_val.png','tf:pca-val','PCA financiero de validación'),
 ('pca/graficos/varianza_explicada.png','pca_varianza.png','tf:pca-varianza','Varianza explicada del PCA financiero'),
 ('evaluacion_final/graficos/01_balanced_accuracy.png','test_ba.png','tf:test-ba','BA financiera en soporte común'),
 ('evaluacion_final/graficos/02_metricas.png','test_metricas.png','tf:test-metricas','Métricas financieras en soporte común'),
 ('evaluacion_final/graficos/03_confusion_perceptron.png','test_conf_p.png','tf:test-conf-p','Confusión del perceptrón financiero en test'),
 ('evaluacion_final/graficos/04_confusion_logistic.png','test_conf_l.png','tf:test-conf-l','Confusión logística financiera en test'),
 ('evaluacion_final/graficos/05_BA_mensual.png','test_mensual.png','tf:test-mensual','BA financiera mensual en soporte común'),
 ('evaluacion_final/graficos/06_train_validacion_test.png','test_particiones.png','tf:test-particiones','Generalización temporal financiera'),
 ('evaluacion_final/graficos/07_sensibilidad_especificidad.png','test_sensibilidad.png','tf:test-sensibilidad','Sensibilidad y especificidad financieras'),
]


def main():
    before=read(EVIDENCE/'integridad_experimentos_antes.json')
    assert all(sha(BASE/n)==h for n,h in before.items()),'Se modificó un experimento'
    for stage in ['dataset','entrenamiento','pca','evaluacion_final']:
        m=read(SOURCE/stage/'manifiesto.json')
        for key in ['implementation_sha256','artifacts_sha256']:
            for n,h in m[key].items():assert sha(BASE/n)==h,n
    OUT.mkdir(parents=True,exist_ok=True);(ROOT/'figuras/transacciones').mkdir(exist_ok=True)
    trace=[]
    def evidence(kind,label,paths,derivation):
        trace.append(dict(type=kind,label=label,sources=[dict(path=str(p.relative_to(BASE)),sha256=sha(p)) for p in paths],derivation=derivation))
    def table(name,caption,label,headers,rows,paths,columns=None):
        text='\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
        text+='\\begin{tabular}{'+(columns or 'l'+'r'*(len(headers)-1))+'}\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n'
        text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n'
        (OUT/f'{name}.tex').write_text(text,encoding='utf-8',newline='\n');evidence('table',label,paths,'Transcripción de resultados; porcentajes redondeados solo para lectura.')
    ds=read(SOURCE/'dataset/conteos.json');test=read(SOURCE/'evaluacion_final/metricas_globales.json')
    dev=read(SOURCE/'entrenamiento/metricas.json');frozen=read(SOURCE/'entrenamiento/congelacion.json');runs=read(SOURCE/'entrenamiento/experimentos.json')
    models={k:read(BASE/v['path']) for k,v in frozen['models'].items()}
    assert all(m['feature_names']==frozen['feature_names'] and m['n_features']==16 for m in models.values())
    assert test['primary_n']==1284 and test['common_n']==1276 and ds['train']['eligible']==4612 and ds['val']['eligible']==2943
    partitions=[]
    for split,title,period,common in [('train','Entrenamiento','2013-04--2014-12',4376),('val','Validación','2015-01--2015-12',2927),('test','Prueba','2016-01--2016-05',1276)]:
        m=test['primary']['perceptron'] if split=='test' else dev['full_support'][split]['perceptron']
        partitions.append([title,period,m['n'],m['support']['0'],m['support']['1'],common])
    table('particiones','Transacciones: particiones temporales y poblaciones principal y común.','tab:tf-particiones',
          ['Conjunto','Mes objetivo','N principal','Clase 0','Clase 1','N común'],partitions,[SOURCE/'dataset/conteos.json',SOURCE/'entrenamiento/metricas.json',SOURCE/'evaluacion_final/metricas_globales.json'],'llrrrr')
    formulas=[r'$n_1$',r'$n_2$',r'$n_3$',r'$(n_1+n_2+n_3)/3$',r'$\operatorname{mediana}(n_1,n_2,n_3)$',r'$\operatorname{std}(n_1,n_2,n_3)$, $ddof=0$',
              r'$n_1-n_2$',r'$(n_1-n_3)/2$',r'$\sum_{k=1}^3\mathbf1[n_k>0]$',r'$\mathbf1[n_1=0]$',r'$(n_1-n_2)/n_2$ si $n_2>0$; nulo si no',r'$\mathbf1[n_2=0]$',
              r'$\operatorname{ordinal}(t)-\operatorname{ordinal}(t_{\mathrm{primero}})$',r'$\sin(2\pi(\operatorname{mes}(t)-1)/12)$',r'$\cos(2\pi(\operatorname{mes}(t)-1)/12)$',r'Días calendario del mes objetivo']
    names=frozen['feature_names'];assert len(formulas)==len(names)==16
    tex=r'\begin{longtable}{p{5.7cm}p{9.3cm}}'+'\n'+r'\caption{Las 16 características financieras antes de estandarizar. $n_k=N_{c,t-k}$.}\label{tab:tf-caracteristicas}\\'+'\n'
    tex+=r'\toprule Característica & Fórmula \\ \midrule\endfirsthead'+'\n'+r'\toprule Característica & Fórmula \\ \midrule\endhead'+'\n'
    tex+='\n'.join(literal(n)+' & '+f+r' \\' for n,f in zip(names,formulas))+'\n'+r'\bottomrule\end{longtable}'+'\n'
    (OUT/'caracteristicas.tex').write_text(tex,encoding='utf-8',newline='\n');evidence('table','tab:tf-caracteristicas',[SOURCE/'dataset/caracteristicas.json',BASE/'datos/Procesados/transacciones/feature_names.json'],'Nombres/formulas del catálogo autorizado, sin variables condicionadas.')
    budget=[]
    for r in runs:budget.append(['Perceptrón' if r['algorithm']=='perceptron' else 'Logística',num(r['params']['learning_rate'],2),r['params']['seed'],r['params']['max_epochs'],r['selected_epoch'],pct(r['metrics']['val']['balanced_accuracy'])])
    table('configuraciones','Presupuesto financiero predefinido: cuatro ejecuciones, sin ampliación adaptativa.','tab:tf-config',['Modelo',r'$\eta$','Semilla','Máx. épocas','Checkpoint',r'BA val. (\%)'],budget,[SOURCE/'entrenamiento/configuracion_experimental.json',SOURCE/'entrenamiento/experimentos.json'],'lrrrrr')
    for key,title in [('perceptron','perceptrón'),('logistic','logística')]:
        m=models[key];rows=[[i+1,literal(n),repr(w)] for i,(n,w) in enumerate(zip(names,m['weights']))]
        table('pesos_'+key,'Pesos financieros congelados: '+title+'. Precisión decimal suficiente para recuperar float64.','tab:tf-pesos-'+key,['Índice','Característica estandarizada','Peso'],rows,[BASE/frozen['models'][key]['path']],'rlr')
    rows=[]
    for key,title in [('perceptron','Perceptrón'),('logistic','Logística')]:
        for split,label in [('train','Train'),('val','Validación')]:
            m=dev['full_support'][split][key];rows.append([title,label]+[pct(m[k]) for k in ['accuracy','precision','recall','specificity','f1','balanced_accuracy']])
    table('desarrollo',r'Métricas financieras de los checkpoints seleccionados en soporte principal (\%).','tab:tf-desarrollo',['Modelo','Conjunto','Acc.','Prec.','Recall','Espec.','F1','BA'],rows,[SOURCE/'entrenamiento/metricas.json'],'llrrrrrr')
    pca=read(SOURCE/'pca/interpretacion_componentes.json');rows=[]
    for pc,values in pca['top_absolute_coefficients'].items():
        for r in values[:5]:rows.append([pc,literal(r['feature']),num(r['coefficient'],9)])
    table('coeficientes_pca','Coeficientes de mayor magnitud del PCA financiero en entradas estandarizadas.','tab:tf-pca-coef',['Comp.','Característica','Coeficiente'],rows,[SOURCE/'pca/interpretacion_componentes.json'],'llr')
    names_models={'perceptron':'Perceptrón','logistic':'Logística','majority':'Mayoritario','historical':'Histórico'}
    for scope,filename,caption,label in [('primary','test_principal',r'Test financiero: métricas sobre las 1\,284 filas principales (\%).','tab:tf-test'),('common','test_comun',r'Comparación financiera secundaria: los cuatro métodos en 1\,276 filas (\%).','tab:tf-comun')]:
        rows=[[names_models[n]]+[pct(m[k]) for k in ['accuracy','precision','recall','specificity','f1','balanced_accuracy']] for n,m in test[scope].items()]
        table(filename,caption,label,['Método','Acc.','Prec.','Recall','Espec.','F1','BA'],rows,[SOURCE/'evaluacion_final/metricas_globales.json'])
    rows=[]
    for n,m in test['primary'].items():rows.append([names_models[n]]+[m[k] for k in ['TN','FP','FN','TP']]+[pct(m['false_alarm_rate']),pct(1-m['recall']),pct(m['positive_prediction_proportion'])])
    table('confusiones',r'Conteos financieros principales y tasas FPR/FNR y predicción positiva (\%).','tab:tf-confusiones',['Método','TN','FP','FN','TP','FPR','FNR','Pred. 1'],rows,[SOURCE/'evaluacion_final/metricas_globales.json'])
    monthly=read(SOURCE/'evaluacion_final/metricas_mensuales.json');rows=[]
    for month,v in monthly.items():
        p,l=v['primary']['perceptron'],v['primary']['logistic'];rows.append([month,v['primary_n'],v['common_n'],f"{p['support']['0']}/{p['support']['1']}",pct(p['balanced_accuracy']),pct(l['balanced_accuracy'])])
    table('mensual','Todos los meses de test financiero: soporte principal y BA principal, sin sustituirla por la común.','tab:tf-mensual',['Mes','N principal','N común','Clase 0/1',r'BA perc. (\%)',r'BA log. (\%)'],rows,[SOURCE/'evaluacion_final/metricas_mensuales.json'],'lrrlrr')
    cohorts=read(SOURCE/'evaluacion_final/cohortes.json');rows=[]
    for c,v in cohorts.items():
        p=v['metrics']['primary']['perceptron'];l=v['metrics']['primary']['logistic'];rows.append(['Vistas en desarrollo' if c=='known_development' else 'Nuevas en test',v['accounts'],v['metrics']['primary_n'],f"{p['support']['0']}/{p['support']['1']}",pct(p['balanced_accuracy']),pct(l['balanced_accuracy'])])
    table('cohortes','Cuentas financieras conocidas y nuevas; cinco filas nuevas no acreditan generalización estable.','tab:tf-cohortes',['Cohorte','Cuentas','Filas','Clase 0/1',r'BA perc. (\%)',r'BA log. (\%)'],rows,[SOURCE/'evaluacion_final/cohortes.json'],'lrrlrr')
    sales=read(BASE/'resultados/fase_07/comparacion_con_desarrollo.json');rows=[]
    for process,source,pairs in [('Ventas',sales,[('perceptron','Perceptrón'),('logistica','Logística')]),('Transacciones',test['primary'],[('perceptron','Perceptrón'),('logistic','Logística')])]:
        for key,title in pairs:
            m=source[key]['test'] if process=='Ventas' else source[key]
            rows.append([process,title,m['n']]+[pct(m[k]) for k in ['accuracy','balanced_accuracy','recall','specificity']])
    table('comparacion','Contraste descriptivo entre problemas distintos: porcentajes de test, sin comparación estadística emparejada.','tab:tf-comparacion',['Proceso','Modelo','N','Acc.','BA','Recall','Espec.'],rows,[BASE/'resultados/fase_07/comparacion_con_desarrollo.json',SOURCE/'evaluacion_final/metricas_globales.json'],'llrrrrr')
    short=[]
    for source,dest,label,title in FIGURES:
        src=SOURCE/source;target=ROOT/'figuras/transacciones'/dest;shutil.copyfile(src,target);assert sha(src)==sha(target)
        short.append(r'\expandafter\def\csname titulo@fig:'+label+r'\endcsname{'+title+'}')
        evidence('figure','fig:'+label,[src],'Copia binaria del PNG aprobado; sin regenerar curvas, PCA o predicciones.')
    (ROOT/'secciones/titulos_transacciones.tex').write_text('\n'.join(short)+'\n',encoding='utf-8',newline='\n')
    conv=read(SOURCE/'entrenamiento/convergencia.json');p=conv['p_seed7'];l=conv['l_eta010'];variance=read(SOURCE/'pca/varianza_explicada.json')
    values={'TFBiasP':repr(models['perceptron']['bias']),'TFBiasL':repr(models['logistic']['bias']),
            'TFBAPTest':pct(test['primary']['perceptron']['balanced_accuracy']),'TFBALTest':pct(test['primary']['logistic']['balanced_accuracy']),
            'TFPCAUno':num(variance['PC1_percent']),'TFPCADos':num(variance['PC2_percent']),'TFPCARetenida':num(variance['retained_percent']),'TFPCAPerdida':num(variance['discarded_percent']),
            'TFErroresInicialesP':str(p['first_online_errors']),'TFErroresFinalesP':str(p['last_online_errors']),'TFActualizacionesP':str(p['updates_total']),
            'TFMinErroresP':str(p['online_error_min']),'TFMaxErroresP':str(p['online_error_max']),'TFSubidasP':str(p['error_increase_steps']),
            'TFBAPUltima':pct(p['last_epoch_metrics']['val']['balanced_accuracy']),'TFLInicial':num(l['initial_train_loss'],9),
            'TFLFinal':num(l['last_train_loss'],9),'TFLValFinal':num(l['last_val_loss'],9),'TFGradFinal':num(l['last_gradient_norm'],9),
            'TFLogAPTest':num(test['logistic_AP_primary'],6),'TFDeltaBAP':num(100*(test['primary']['perceptron']['balanced_accuracy']-dev['full_support']['val']['perceptron']['balanced_accuracy'])),
            'TFDeltaBAL':num(100*(test['primary']['logistic']['balanced_accuracy']-dev['full_support']['val']['logistic']['balanced_accuracy']))}
    (OUT/'cifras.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in values.items())+'\n',encoding='utf-8',newline='\n')
    guardar(EVIDENCE/'cifras_contrastadas.json',dict(macros=values,source_models={k:dict(path=v['path'],sha256=sha(BASE/v['path'])) for k,v in frozen['models'].items()},
        dataset_shapes=read(SOURCE/'dataset/matrices.json')['shapes'],test_counts=read(SOURCE/'evaluacion_final/conteos.json'),
        variance=variance,convergence_selected={'perceptron':p,'logistic':l},read_only_documental_export=True))
    guardar(EVIDENCE/'trazabilidad.json',trace)
    assert all(sha(BASE/n)==h for n,h in before.items()),'Experimento alterado al exportar'
    print(json.dumps({'new_tables':13,'copied_figures':len(FIGURES),'experimental_files_intact':len(before),'predictions_or_PCA_recalculated':False}))


if __name__=='__main__':main()
