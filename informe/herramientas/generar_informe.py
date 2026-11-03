"""Exporta tablas y figuras documentales desde artefactos aprobados. No entrena."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'informe'
sys.path.insert(0, str(BASE))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads((BASE / p).read_text(encoding='utf-8'))
def esc(s):
    for a,b in [('\\',r'\textbackslash{}'),('_',r'\_'),('%',r'\\%'),('&',r'\&'),('#',r'\#')]: s=str(s).replace(a,b)
    return s
def num(x, digits=6): return f'{float(x):.{digits}f}'
def pct(x): return num(100*x,3)
def literal(s): return r'\texttt{\detokenize{'+str(s).replace('\\','/')+'}}'


def main():
    before=read('informe/evidencias/integridad_antes.json')
    for p,d in before.items(): assert sha(BASE/p)==d,p
    # La comprobación de integridad y métricas solo lee predicciones guardadas.
    from src.experiments.analisis_estadistico import verificar_entradas
    frozen, predictions, evaluation, agreements = verificar_entradas()
    for field in ['artifacts_sha256','implementation_sha256']:
        for p,d in read('resultados/fase_08/manifiesto.json')[field].items(): assert sha(BASE/p)==d,p
    for folder in ['generado','figuras','evidencias']: (OUT/folder).mkdir(exist_ok=True)
    trace=[]
    def record(kind,label,sources,detail):
        trace.append({'type':kind,'label':label,'sources':[{'path':s,'sha256':sha(BASE/s)} for s in sources], 'derivation':detail})
    def table(name,caption,label,headers,rows,sources,columns=None):
        columns=columns or ('l'+'r'*(len(headers)-1))
        text='\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{'+label+'}\n'
        text+='\\begin{tabular}{'+columns+'}\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n'
        text+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)
        text+='\n\\bottomrule\\end{tabular}\n\\end{table}\n'
        (OUT/'generado'/f'{name}.tex').write_text(text,encoding='utf-8')
        record('table',label,sources,'Valores leídos de las claves originales; porcentajes redondeados solo en presentación.')
    pmodel=read('resultados/fase_04/modelo_seleccionado.json'); lmodel=read('resultados/fase_05/modelo_seleccionado.json')
    features=pmodel['feature_names']; assert features==lmodel['feature_names']==frozen['dataset']['feature_names']
    for folder,model in [('fase_04',pmodel),('fase_05',lmodel)]:
        assert model==frozen['models']['perceptron' if folder=='fase_04' else 'logistica']['state']
        rows=[[str(i+1),literal(n),repr(w)] for i,(n,w) in enumerate(zip(features,model['weights']))]
        table('pesos_'+folder,'Pesos del modelo congelado '+('perceptrón' if folder=='fase_04' else 'logístico')+'. Valores float64 conservados con representación decimal de ida y vuelta.', 'tab:pesos-'+folder,['Índice','Característica estandarizada','Peso'],rows,[f'resultados/{folder}/modelo_seleccionado.json'],'rlr')
    comp=read('resultados/fase_07/comparacion_con_desarrollo.json')
    names={'perceptron':'Perceptrón','logistica':'Logística','mayoritario':'Mayoritario','historico':'Histórico'}
    rows=[]
    for n in names:
        for split,title in [('train','Train'),('val','Validación'),('test','Test')]:
            m=comp[n][split]
            rows.append([names[n],title]+[pct(m[k]) for k in ['accuracy','precision','recall','specificity','f1','balanced_accuracy']])
    table('metricas','Métricas finales por partición (\\%). Estados seleccionados en validación; no corresponden a la última época del perceptrón.','tab:metricas',['Método','Conjunto','Acc.','Prec.','Recall','Espec.','F1','BA'],rows,['resultados/fase_07/comparacion_con_desarrollo.json'])
    derived=read('resultados/fase_08/metricas_derivadas.json')
    table('confusiones','Conteos de test y tasas de error (\\%). Filas reales y columnas predichas en orden 0,1.','tab:confusiones',['Método','TN','FP','FN','TP','FPR','FNR','Pred. 1'],[[names[n]]+[str(derived[n][k]) for k in ['TN','FP','FN','TP']]+[pct(derived[n][k]) for k in ['false_positive_rate','false_negative_rate']]+[str(derived[n]['predicted_positive_count'])] for n in names],['resultados/fase_08/metricas_derivadas.json'])
    manifest=read('resultados/fase_03/manifiesto.json')
    rows=[]
    for s,t in [('train','Entrenamiento'),('val','Validación'),('test','Prueba')]:
        m=comp['perceptron'][s]
        rows.append([t,'--'.join(manifest['partitions'][s]),str(m['n']),str(m['support']['0']),str(m['support']['1']),pct(m['class_proportions']['1'])])
    table('particiones','Particiones por mes objetivo y soporte de clases; historia disponible desde enero de 2013.','tab:particiones',['Conjunto','Mes inicial--final','N','Clase 0','Clase 1','Clase 1 (\\%)'],rows,['resultados/fase_03/manifiesto.json','resultados/fase_07/comparacion_con_desarrollo.json'])
    monthly=read('resultados/fase_08/analisis_temporal.json')
    rows=[]
    for month,values in monthly['monthly'].items():
        p,l=values['perceptron'],values['logistica']
        rows.append([month,f"{l['support']['0']}/{l['support']['1']}",pct(p['balanced_accuracy']),pct(l['balanced_accuracy']),pct(l['precision']),pct(l['recall']),pct(l['specificity'])])
    table('mensual','Todos los meses de test. Tasas en \\%; soporte negativo/positivo.','tab:mensual',['Mes','0/1','BA perc.','BA log.','Prec. log.','Recall log.','Espec. log.'],rows,['resultados/fase_08/analisis_temporal.json'])
    rows=[]
    for cohort,cohort_data in evaluation['cohorts'].items():
        for n,m in cohort_data['methods'].items():
            c=m['confusion_matrix']; rows.append([cohort,names[n],str(m['n']),f"{m['support']['0']}/{m['support']['1']}",str(c[0][0]),str(c[0][1]),str(c[1][0]),str(c[1][1]),pct(m['balanced_accuracy'])])
    table('cohortes','Comparación de cohortes en test, sin excluir los productos nuevos.','tab:cohortes',['Cohorte','Método','N','0/1','TN','FP','FN','TP','BA (\\%)'],rows,['resultados/fase_07/metricas_cohortes.json'])
    pc=pd.read_csv(BASE/'resultados/fase_06/coeficientes_componentes.csv',float_precision='round_trip')
    rows=[]
    for component in ['PC1','PC2']:
        for _,r in pc.reindex(pc[f'{component}_coefficient'].abs().sort_values(ascending=False).index).head(5).iterrows():
            rows.append([component,literal(r.feature),num(r[f'{component}_coefficient'],9),pct(r[f'{component}_squared_contribution'])])
    table('pca_cargas','Cinco mayores coeficientes por magnitud en cada componente. El cuadrado es participación en la norma del vector, no importancia predictiva.','tab:pca-cargas',['Comp.','Característica','Coeficiente','Cuadrado (\\%)'],rows,['resultados/fase_06/coeficientes_componentes.csv'])
    gradients=read('resultados/fase_05/verificacion_gradientes.json')
    table('gradientes','Verificación original por diferencias finitas centrales. Todos los casos cumplen tolerancias.','tab:gradientes',['Caso','$h$','atol','rtol','Error abs. máx.','Error rel. máx.'],[[esc(n)]+[f'{m[k]:.5g}' for k in ['h','atol','rtol','max_absolute_error','max_relative_error']] for n,m in gradients.items()],['resultados/fase_05/verificacion_gradientes.json'])
    experiments=read('resultados/fase_05/experimentos.json')
    table('tasas','Presupuesto logístico previo: 200 épocas por tasa; checkpoint elegido por BA de validación a umbral 0,5.','tab:tasas',['Tasa','$k$ elegido','BA val. (\\%)','Pérdida final train','Gradiente final'],[[num(r['params']['learning_rate'],2),str(r['selected_epoch']),pct(r['metrics_at_05']['val']['balanced_accuracy']),num(pd.read_csv(BASE/f"resultados/fase_05/historial_{r['id']}.csv",float_precision='round_trip').train_loss.iloc[-1],9),num(pd.read_csv(BASE/f"resultados/fase_05/historial_{r['id']}.csv",float_precision='round_trip').gradient_norm.iloc[-1],9)] for r in experiments['runs']],['resultados/fase_05/experimentos.json']+[f"resultados/fase_05/historial_{r['id']}.csv" for r in experiments['runs']])
    hp_path='resultados/fase_04/historial_orden_barajado.csv'; hl_path='resultados/fase_05/historial_eta_010.csv'
    hp=pd.read_csv(BASE/hp_path,float_precision='round_trip'); hl=pd.read_csv(BASE/hl_path,float_precision='round_trip')
    assert np.array_equal(hp.updates,hp.online_errors)
    run=next(r for r in experiments['runs'] if r['id']==experiments['selected_id'])
    initial=run['initial_train_loss']; previous=np.r_[initial,hl.train_loss.to_numpy()[:-1]]
    hl['delta_L']=previous-hl.train_loss; hl['relative_change']=np.abs(hl.train_loss-previous)/np.maximum(1,np.abs(previous))
    hl.to_csv(OUT/'evidencias/convergencia_logistica_por_epoca.csv',index=False,float_format='%.17g')
    hp.to_csv(OUT/'evidencias/convergencia_perceptron_por_epoca.csv',index=False,float_format='%.17g')
    summary={'perceptron':{'selected_epoch':pmodel['selected_epoch'],'last_epoch':int(hp.epoch.iloc[-1]),'first_errors':int(hp.online_errors.iloc[0]),'last_errors':int(hp.online_errors.iloc[-1]),'min_errors':int(hp.online_errors.min()),'max_errors':int(hp.online_errors.max()),'increased_epochs':int((hp.online_errors.diff()>0).sum()),'zero_updates_epochs':int((hp.updates==0).sum()),'first10_mean_errors':float(hp.online_errors.head(10).mean()),'last10_mean_errors':float(hp.online_errors.tail(10).mean()),'weights_norm_selected':float(hp.loc[hp.epoch.eq(70),'weights_norm'].iloc[0]),'weights_norm_last':float(hp.weights_norm.iloc[-1])},'logistica':{'initial_train_loss':initial,'last_train_loss':float(hl.train_loss.iloc[-1]),'initial_val_loss':run['initial_val_loss'],'last_val_loss':float(hl.val_loss.iloc[-1]),'absolute_drop':initial-float(hl.train_loss.iloc[-1]),'relative_drop':(initial-float(hl.train_loss.iloc[-1]))/initial,'loss_increases_strict':int((hl.delta_L<0).sum()),'first10_mean_delta_L':float(hl.delta_L.head(10).mean()),'last10_mean_delta_L':float(hl.delta_L.tail(10).mean()),'last_delta_L':float(hl.delta_L.iloc[-1]),'last_relative_change':float(hl.relative_change.iloc[-1]),'initial_gradient_norm':float(hl.gradient_used_norm.iloc[0]),'last_gradient_norm':float(hl.gradient_norm.iloc[-1]),'last_weight_norm':float(hl.weights_norm.iloc[-1]),'stopped_by':'max_epochs; tolerance=None','converged_by_tolerance':run['converged_by_tolerance']}}
    (OUT/'evidencias/analisis_convergencia.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for k in [1,70,100]:
        r=hp.loc[hp.epoch.eq(k)].iloc[0]; rows.append([str(k),str(int(r.online_errors)),str(int(r.updates)),str(int(r.train_errors_end)),pct(r.train_accuracy),pct(r.val_accuracy),pct(r.val_balanced_accuracy),num(r.weights_norm,6)])
    table('convergencia_perceptron','Recorrido seleccionado del perceptrón: errores online y al cierre son magnitudes diferentes.','tab:conv-p',['Época','Online','Actualiz.','Error cierre','Acc. train','Acc. val','BA val','Norma $w$'],rows,[hp_path])
    rows=[]
    for k in [1,50,100,150,200]:
        r=hl.loc[hl.epoch.eq(k)].iloc[0]; rows.append([str(k),num(r.train_loss,9),num(r.val_loss,9),f'{r.delta_L:.4g}',f'{r.relative_change:.4g}',num(r.gradient_norm,7),pct(r.val_balanced_accuracy)])
    table('convergencia_logistica','Diagnóstico documental por época. $\\Delta L$ y $r$ se derivan del historial sin modificar el criterio de parada.','tab:conv-l',['Época','$L$ train','$L$ val','$\\Delta L$','$r$','Norma grad.','BA val (\\%)'],rows,[hl_path,'resultados/fase_05/experimentos.json'])
    # Figuras previas: copias exactas, con nombres cortos portables para LaTeX.
    copies={
      'perceptron_errores':'fase_04/errores_entrenamiento','perceptron_accuracy':'fase_04/accuracy_por_epoca','perceptron_ba':'fase_04/balanced_accuracy_validacion',
      'logistica_perdida':'fase_05/entropia_cruzada','logistica_gradiente':'fase_05/norma_gradiente','logistica_accuracy':'fase_05/accuracy_por_epoca','logistica_ba':'fase_05/balanced_accuracy_validacion',
      'pca_train':'fase_06/pca_train','pca_val':'fase_06/pca_val','pca_varianza':'fase_06/varianza_explicada',
      'confusion_perceptron':'fase_07/confusion_perceptron','confusion_logistica':'fase_07/confusion_logistica','ba_mensual':'fase_07/balanced_accuracy_mensual',
      'sensibilidad_especificidad':'fase_08/sensibilidad_especificidad','acuerdos':'fase_08/acuerdos_desacuerdos','cambios_medias':'fase_08/cambios_medias_caracteristicas'}
    for dest,source in copies.items():
        phase,name=source.split('/'); path=f'resultados/{phase}/graficos/{name}.png'
        shutil.copyfile(BASE/path,OUT/'figuras'/f'{dest}.png')
        record('figure',dest,[path],'Copia binaria idéntica de figura aprobada.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    def save(fig,name,sources,detail):
        fig.tight_layout(); fig.savefig(OUT/'figuras'/f'{name}.png',dpi=180,metadata={'Software':'PerceptronReal: informe documental'});plt.close(fig)
        record('figure',name,sources,detail)
    fig,ax=plt.subplots(figsize=(10,4.6))
    metrics=['accuracy','precision','recall','f1','balanced_accuracy']; positions=np.arange(len(metrics))
    for i,(name,color) in enumerate(zip(names,['#355a7a','#d27b2c','#56845b','#8b70a4'])):
        ax.bar(positions+(i-1.5)*.18,[comp[name]['test'][k] for k in metrics],width=.18,label=names[name],color=color)
    ax.set(xticks=positions,xticklabels=['Accuracy','Precision positiva','Recall positivo','F1 positivo','Balanced accuracy'],ylabel='Proporción',ylim=(0,1.16),title='Test WWI: comparación global de cinco métricas')
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_yticks(np.arange(0,1.01,.2));ax.legend(ncol=4,fontsize=9,loc='upper center');ax.grid(axis='y',alpha=.2)
    save(fig,'metricas_globales',['resultados/fase_07/comparacion_con_desarrollo.json'],'Cinco métricas finales guardadas; se añade BA a la vista previa de cuatro métricas, sin inferencia o evaluación nueva.')
    fig,axs=plt.subplots(1,2,figsize=(11,3.5))
    labels=['Train','Validación','Test'];xx=np.arange(3)
    for c,color in [('0','#355a7a'),('1','#e29a39')]:
        counts=[comp['perceptron'][s]['support'][c] for s in ['train','val','test']]
        bottom=np.zeros(3) if c=='0' else np.array([comp['perceptron'][s]['support']['0'] for s in ['train','val','test']])
        axs[0].bar(xx,counts,bottom=bottom,label=f'Clase {c}',color=color)
        for i,v in enumerate(counts):axs[0].text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white',fontsize=10)
    axs[0].set(xticks=xx,xticklabels=labels,ylabel='Observaciones producto-mes',title='Soporte real por partición');axs[0].legend(loc='upper right')
    dates=pd.period_range('2013-01','2016-05',freq='M')
    for y,(s,label,color) in enumerate([('train','Train','#355a7a'),('val','Validación','#e29a39'),('test','Test','#57957d')]):
        start,end=manifest['partitions'][s];left=dates.get_loc(start);right=dates.get_loc(end)
        axs[1].barh(y,right-left+1,left=left-.5,color=color,height=.5);axs[1].text((left+right)/2,y,f'{start} a {end}',ha='center',va='center',color='white',fontsize=9)
    axs[1].set(yticks=range(3),yticklabels=labels,xticks=[0,12,24,36,40],xticklabels=['2013-01','2014-01','2015-01','2016-01','2016-05'],title='Mes objetivo; enero-marzo 2013: historia');axs[1].tick_params(axis='x',rotation=30)
    save(fig,'datos_particiones',['resultados/fase_03/manifiesto.json','resultados/fase_07/comparacion_con_desarrollo.json'],'Soportes reales y calendario congelado, sin partición nueva.')
    fig,axs=plt.subplots(2,2,figsize=(10,6))
    axs[0,0].plot(np.r_[0,hl.epoch],np.r_[initial,hl.train_loss],label='Train');axs[0,0].plot(np.r_[0,hl.epoch],np.r_[run['initial_val_loss'],hl.val_loss],label='Validación');axs[0,0].set(ylabel='Entropía cruzada media',title='Trayectoria completa');axs[0,0].legend()
    last=hl.epoch>=160
    axs[0,1].plot(hl.epoch[last],hl.train_loss[last],color='#355a7a');axs[0,1].set(ylabel='Entropía cruzada train',title='Acercamiento: épocas 160–200')
    axs[1,0].semilogy(hl.epoch,hl.delta_L,label='Delta L');axs[1,0].semilogy(hl.epoch,hl.relative_change,'--',label='r (coincide: L previa < 1)');axs[1,0].set(ylabel='Cambio por época (log)',title='Descenso positivo, todavía no nulo');axs[1,0].legend(fontsize=8)
    axs[1,1].semilogy(hl.epoch,hl.gradient_norm);axs[1,1].set(ylabel='Norma L2 de gradiente',title='Incluye pesos y sesgo')
    for ax in axs.flat:ax.set_xlabel('Época');ax.grid(alpha=.2)
    save(fig,'convergencia_detallada',[hl_path,'resultados/fase_05/experimentos.json'],'Pérdidas reales, diferencias consecutivas y gradiente del historial seleccionado.')
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    axs[0].plot(hp.epoch,hp.online_errors,label='Errores online = actualizaciones');axs[0].plot(hp.epoch,hp.train_errors_end,label='Errores al cierre',alpha=.7);axs[0].axvline(70,color='black',ls='--',label='Checkpoint 70');axs[0].set(xlabel='Época',ylabel='Conteo de observaciones',title='Perceptrón: conteos distintos');axs[0].legend(fontsize=8)
    axs[1].plot(hl.epoch,hl.train_loss,label='Train');axs[1].plot(hl.epoch,hl.val_loss,label='Validación');axs[1].set(xlabel='Época',ylabel='Entropía cruzada media',title='Logística: pérdida diferenciable');axs[1].legend()
    save(fig,'errores_paneles',[hp_path,hl_path],'Paneles y escalas separados; no se equiparan conteo y entropía cruzada.')
    fig,ax=plt.subplots(figsize=(9,3.8))
    for i,n in enumerate(names):ax.bar(xx+(i-1.5)*.18,[1-comp[n][s]['accuracy'] for s in ['train','val','test']],width=.18,label=names[n])
    ax.set(xticks=xx,xticklabels=labels,ylabel='Error de clasificación = 1 − accuracy',title='Estados finales seleccionados; sin curva de test por época');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncol=4,fontsize=9,loc='upper center');ax.set_ylim(0,.66)
    save(fig,'error_particiones',['resultados/fase_07/comparacion_con_desarrollo.json'],'Complemento de accuracy final guardada para cuatro métodos y tres particiones.')
    # Catálogo completo de fórmulas autorizadas, sin recrear características.
    rules=read('datos/Procesados/fase_03/caracteristicas.json')['rules']
    text=r'\begin{longtable}{r>{\raggedright\arraybackslash}p{6.7cm}>{\raggedright\arraybackslash}p{6.2cm}}\caption{Entradas efectivas y fórmulas del contrato. $q_j=Q_{p,t-j}$.}\label{tab:caracteristicas}\\\toprule Índice & Nombre & Fórmula registrada\\\midrule\endfirsthead\toprule Índice & Nombre & Fórmula registrada\\\midrule\endhead'+'\n'
    for i,n in enumerate(features):text+=f'{i+1} & '+literal(n)+' & '+esc(rules[n]['formula'])+r' \\'+'\n'
    text+=r'\bottomrule\end{longtable}'
    (OUT/'generado/caracteristicas.tex').write_text(text,encoding='utf-8')
    record('table','tab:caracteristicas',['datos/Procesados/fase_03/caracteristicas.json'],'Fórmulas literales de las 20 entradas efectivas.')
    # Registro de afirmaciones cuantitativas compartidas con la narrativa.
    record('claims','convergencia',['resultados/fase_04/modelo_seleccionado.json','resultados/fase_05/modelo_seleccionado.json',hp_path,hl_path,'resultados/fase_05/experimentos.json'],'Ecuaciones Delta L y r; estadísticas conservadas en analisis_convergencia.json y CSV por época.')
    record('claims','interpretacion-estadistica',['resultados/fase_08/analisis_temporal.json','resultados/fase_08/metricas_derivadas.json','resultados/fase_08/acuerdos_desacuerdos.json','resultados/fase_08/distribuciones_descriptivas.json','resultados/fase_08/sensibilidad_mensual.json'],'Tablas y afirmaciones de resultados, dependencia y cambios descriptivos de población.')
    record('claims','identificacion-academica',[str(p.relative_to(BASE)).replace('\\','/') for p in (BASE/'referencias').glob('*.pdf')],'Autores/nivel según Dashboard y OpenRefine; docente según solicitud actual. Ciclo de entrega pendiente por divergencia documental de fechas.')
    record('claims','auditoria-y-metodologia',['resultados/auditoria/perfilado_datos.json','resultados/auditoria/comparacion_csv.json','resultados/fase_03/manifiesto.json','datos/Procesados/fase_03/caracteristicas.json','datos/Procesados/fase_03/preprocesador.json'],'Grano, cardinalidades, faltantes, calendario, descartes e imputación contrastados con claves estructuradas; reglas de disponibilidad según contrato.')
    record('claims','pca-varianzas',['resultados/fase_06/varianza_explicada.json','resultados/fase_06/parametros_pca.json'],'Proporciones PC1/PC2/2D/descartada, rango y varianza total: lectura de parámetros aprobados, sin SVD nueva.')
    (OUT/'evidencias/trazabilidad.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(sha(BASE/p)==d for p,d in before.items()),'Cambió evidencia previa'
    print(json.dumps({'protected_files':len(before),'figures':sum(t['type']=='figure' for t in trace),'tables':sum(t['type']=='table' for t in trace),'convergence':summary},ensure_ascii=True,indent=2))


if __name__=='__main__': main()
