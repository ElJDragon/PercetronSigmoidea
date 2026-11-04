"""Gráficos y documentación de desarrollo; no consume outcomes de test."""
import numpy as np
from src.db.auditoria_dw import guardar,sha
from src.experiments.entrenar_inventario import OUT,PREVIOUS,PLAN,read,FEATURES

def plots(selected,comparison,monthly,out):
    from src.evaluation.graficos import graficar_perceptron,plt,PercentFormatter
    from src.evaluation.graficos_logistica import graficar_logistica
    root=out/'graficos';root.mkdir(exist_ok=True)
    p=selected['perceptron'];l=selected['logistic_standard'];w=selected['logistic_weighted']
    graficar_perceptron(p['history'],p['metrics']['val'],p['selected_epoch'],17,root/'perceptron')
    graficar_logistica(l['history'],l['metrics']['val'],l['selected_epoch'],.5,17,root/'logistica_estandar')
    def save(fig,name):
        path=root/name;path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=170,metadata={'Software':'PerceptronReal Inventario'});plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.plot([h['epoch'] for h in p['history']],[h['updates'] for h in p['history']]);ax.axvline(p['selected_epoch'],ls='--',color='orange')
    ax.set(title='Inventario: actualizaciones del perceptrón',xlabel='Época',ylabel='Actualizaciones');ax.grid(alpha=.2);save(fig,'perceptron/actualizaciones.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ep=[h['epoch'] for h in w['history']]
    axes[0].plot(ep,[h['train_loss'] for h in w['history']],label='Train: objetivo ponderado')
    axes[0].set(title='BCE ponderada de entrenamiento',ylabel='Pérdida ponderada')
    axes[1].plot(ep,[h['unweighted_train_BCE'] for h in w['history']],label='Train sin ponderar')
    axes[1].plot(ep,[h['val_loss'] for h in w['history']],label='Validación sin ponderar')
    axes[1].set(title='BCE estándar de evaluación',ylabel='Pérdida sin ponderar')
    for ax in axes:ax.set_xlabel('Época');ax.axvline(w['selected_epoch'],ls='--',color='gray');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'logistica_ponderada/perdidas_definiciones_separadas.png')
    for key,label in [('gradient_norm','Norma del gradiente ponderado'),('val_balanced_accuracy','BA de validación sin ponderar')]:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained');ax.plot(ep,[h[key] for h in w['history']]);ax.axvline(w['selected_epoch'],ls='--',color='orange')
        ax.set(title='Logística ponderada: '+label,xlabel='Época',ylabel=label)
        if key=='val_balanced_accuracy':ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2);save(fig,'logistica_ponderada/'+key+'.png')
    cm=np.array(w['metrics']['val']['confusion_matrix']);fig,ax=plt.subplots(figsize=(6,5),layout='constrained');im=ax.imshow(cm,cmap='Blues')
    ax.set(title=f'Validación: logística ponderada, época {w["selected_epoch"]}',xlabel='Predicción',ylabel='Clase real',xticks=[0,1],yticks=[0,1])
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=15,color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.colorbar(im,ax=ax);save(fig,'logistica_ponderada/matriz_confusion_validacion.png')
    vals=comparison['primary']['val'];methods=list(vals)
    for filename,keys,title in [('metricas_validacion.png',['accuracy','balanced_accuracy','precision','recall','f1'],'Validación: métricas sin ponderar; mismas 2628 filas'),
                                ('comparacion_baselines.png',['balanced_accuracy','recall','specificity'],'Validación: clasificadores y baselines causales')]:
        fig,ax=plt.subplots(figsize=(12,5),layout='constrained');x=np.arange(len(methods));width=.8/len(keys)
        for j,key in enumerate(keys):ax.bar(x+(j-(len(keys)-1)/2)*width,[vals[m][key] for m in methods],width,label=key)
        ax.set(xticks=x,xticklabels=methods,title=title,ylabel='Proporción',ylim=(0,1.05));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncols=3);ax.grid(axis='y',alpha=.2);save(fig,filename)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    for m,v in vals.items():ax.scatter(v['recall'],v['precision'],s=65,label=m)
    ax.set(title='Validación: recall frente a precision positiva (umbral fijo)',xlabel='Recall positivo',ylabel='Precision positiva',xlim=(-.03,1.08),ylim=(-.03,1.08));ax.legend(loc='upper left');ax.grid(alpha=.2);save(fig,'recall_precision.png')
    months=sorted({r['month'] for r in monthly if r['partition']=='val'})
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for name in methods:
        rows=[next(r for r in monthly if r['partition']=='val' and r['month']==m and r['method']==name)['metrics'] for m in months]
        ax.plot(months,[r['positive_prediction_proportion'] for r in rows],marker='o',label=name)
    ax.set(title='Validación: proporción de alertas por mes',xlabel='Mes',ylabel='Proporción');ax.tick_params(axis='x',rotation=45);ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'alertas_por_mes.png')

def reports(runs,selected,comp,conv):
    selections='\n'.join(f"| {g} | {r['id']} | {r['variant']} | {r['selected_epoch']} |" for g,r in selected.items())
    table=[]
    for r in runs:
        for split,m in r['metrics'].items():
            tn,fp=m['confusion_matrix'][0];fn,tp=m['confusion_matrix'][1]
            table.append(f"| {r['id']} | {split} | {r['selected_epoch']} | {m['accuracy']:.6f} | {m['balanced_accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['specificity']:.6f} | {tn}/{fp}/{fn}/{tp} | {m['positive_prediction_proportion']:.6f} |")
    conclusions=[]
    for group in ['perceptron','logistic_standard','logistic_weighted']:
        r=selected[group];m=r['metrics']['val'];b=comp['primary']['val']['historical']['balanced_accuracy']
        conclusions.append(f"- {group}: BA {m['balanced_accuracy']:.6f}, diferencia frente al histórico {m['balanced_accuracy']-b:+.6f}; {'predice únicamente clase 0' if m['only_majority_predicted'] else 'predice ambas clases'}. AP={r.get('validation_average_precision','no aplicable al perceptrón')}.")
    convergence_lines=[]
    for r in runs:
        c=conv[r['id']]
        if r['algorithm']=='perceptron':
            convergence_lines.append(f"- {r['id']}: errores online inicial/final {c['first_online_errors']}/{c['last_online_errors']}, actualizaciones {c['updates_total']}, subidas de errores {c['error_increase_steps']}, épocas sin errores {c['zero_error_epochs']}; checkpoint {c['selected_epoch']}. No equivale a la última época ni a convergencia a cero si persisten actualizaciones.")
        else:
            convergence_lines.append(f"- {r['id']} ({r['loss_definition']}): pérdida train inicial/final {c['initial_train_loss']:.9f}/{c['last_train_loss']:.9f}; BCE val inicial/final {c['initial_val_loss']:.9f}/{c['last_val_loss']:.9f}; gradiente final {c['last_gradient_norm']:.9g}; terminación {c['termination']}, checkpoint {c['selected_epoch']}. Tolerancia 1e-6, alcanzada={c['gradient_tolerance_reached']}.")
        convergence_lines.append(f"  BA val del checkpoint/última época: {r['metrics']['val']['balanced_accuracy']:.9f}/{r['last_epoch_metrics']['val']['balanced_accuracy']:.9f}; BA train del checkpoint/última época: {r['metrics']['train']['balanced_accuracy']:.9f}/{r['last_epoch_metrics']['train']['balanced_accuracy']:.9f}.")
    text='''# Entrenamiento, validación y congelación de Inventario

Se preservan las matrices aprobadas y los 17 predictores, sin SQL, reconstrucción, acceso a 2016, PCA o cambio del informe. Se utilizan las clases matemáticas originales para perceptrón y logística estándar, con un observador de métricas por época que no cambia actualizaciones. La variante ponderada está aislada en src/models/logistic_inventario.py; reutiliza validadores, sigmoide, predicción, métricas y selección de la implementación previa.

## Presupuesto registrado antes de observar resultados

Seis ejecuciones: perceptrón eta 0,01, 100 épocas, semillas 42 y 7; logística estándar eta 0,05 y 0,10, 400 épocas; logística ponderada las mismas dos tasas y épocas. Logísticas con semilla 42 e inicialización cero, lote completo, tolerancia de gradiente 1e-6 y umbral fijo 0,5; perceptrón z>=0 y permutación por época. Todas parten de pesos/sesgo nuevos a cero. No se amplía el espacio tras observar métricas. Configuración y registro UTC con hash en configuracion_experimental.json y registro_previo.json.

La ponderación es una enmienda expresamente autorizada por este encargo al anterior valor por defecto sin pesos; no modifica dataset, población ni validación. Pesos calculados solo con train: n/(2*n_clase), 4599/(2*4513) para clase 0 y 4599/(2*86) para clase 1. Lw=sum(w_i*(logaddexp(0,z_i)-y_i*z_i))/sum(w_i); gradiente = X.T@(w_i*(sigmoid(z_i)-y_i)/sum(w_i)), con análogo para sesgo. Diferencias finitas moderadas y de logits extremos verifican la derivada. No hay SMOTE, replicación ni muestreo sintético de negocio.

Selección por BA sin ponderar de validación, después F1, época más temprana y orden de configuración. Se documentan los mejores estándar/ponderado y se elige la logística final entre ambos mediante la misma regla. No se optimizan umbrales, no se selecciona por AP y no se reentrena con train+val.

| Grupo | Ejecución | Variante | Época |
|---|---|---|---:|
'''+selections+'''

## Métricas de todos los checkpoints candidatos

Todas las métricas se calculan sin pesos; orden de matriz [[TN,FP],[FN,TP]], clase positiva 1. Se conservan las 4599/2628 observaciones, 86/35 positivas; solo 10/9 artículos tienen positivos.

| Ejecución | Partición | Época | Accuracy | BA | Precision | Recall | F1 | Specificity | TN/FP/FN/TP | Alertas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
'''+ '\n'.join(table)+'''

## Baselines y clase minoritaria

El mayoritario 0 se determina únicamente de train; accuracy cercana a 99% sin detectar positivos no es éxito predictivo. Histórico exacto de Inventario: 1[N(t-1)>mediana(N(t-1),N(t-2),N(t-3))]. Usa los mismos tres meses que cada fila elegible, no requiere cuarto mes. Población principal y común son idénticas: 4599 train y 2628 val, cero exclusiones por baseline. No se sustituye por la fórmula financiera. metricas.json conserva métricas completas y matrices de los cinco métodos, con comparación estándar/ponderada como variantes del mismo algoritmo.

'''+ '\n'.join(conclusions)+'''

AP describe el ordenamiento de puntuaciones; no es precision al umbral 0,5 ni demuestra calibración. Los scores ponderados no deben interpretarse automáticamente como probabilidades calibradas. Su entrenamiento optimiza una BCE ponderada, mientras que toda evaluación de validación permanece sin pesos. No se comparan numéricamente ambas pérdidas como un mismo objetivo. Los gráficos ponderados separan explícitamente objetivo de entrenamiento y BCE estándar de evaluación.

## Convergencia y checkpoints

'''+ '\n'.join(convergence_lines)+'''

Los historiales JSON/CSV incluyen errores/actualizaciones, accuracy/BA, precision/recall/F1, normas de pesos y gradiente cuando corresponde; traza_parametros.json conserva métricas completas, pesos y sesgo por época. La última época y el checkpoint se comparan sin confundirlos. Descenso de pérdida no prueba convergencia completa si termina por máximo de épocas.

## Concentración, incertidumbre y límites

concentracion_positivos.json conserva por artículo verdaderos positivos, falsas alertas y confusiones; metricas_mensuales.json y predicciones_desarrollo.csv conservan meses, soporte y alertas. Hay 35 positivos de validación en solo nueve artículos, frente a 210 artículos con todos sus meses negativos. En los tres clasificadores, todos los TP y FP se concentran en esos nueve artículos; los otros 210 no generan alertas. La BA global alta se beneficia de reconocer los numerosos artículos inactivos en recepción, y no equivale a discriminar con precisión los meses de aumento dentro de los artículos que reciben. Este desglose es descriptivo y no redefine la población ni selecciona modelos. Se mantienen meses monoclase con BA nula/no definida, sin borrarlos para mejorar promedios. Intervalos Wilson de recall al 95% son descriptivos bajo supuesto binomial: no se ha demostrado independencia temporal o por artículo y pueden ser optimistas. No sostienen significación estadística, superioridad universal ni utilidad económica. No se redefine población según errores. La cobertura y conservación histórica siguen siendo supuestos académicos.

## Integridad, reproducción y cierre

preflight.json contrasta hashes, catálogo, X/y/metadatos, preprocesador train-only y fechas. Los nueve fallos históricos incluyen hashes de una versión anterior del informe; discrepancias_previas.json documenta que preceden a este trabajo y no afectan matrices, algoritmos o reproducibilidad actuales. No se cambian pruebas, hashes o archivos previos para ocultarlos. Los modelos de modelos/ incluyen perceptrón, logística final y ganadores estándar/ponderado para trazabilidad; son dos algoritmos, no tres.

congelacion.json vincula modelos con dataset, preprocesador, configuración, umbrales, métricas y hashes. protocolo_test.md y protocolo_test.json definen una futura apertura, sin ejecutarla. pruebas_focalizadas.txt, suite_global.json, reproducibilidad.json y validaciones.json distinguen controles nuevos, nueve casos históricos y cualquier regresión. No presentar la suite global como aprobada mientras existan esos casos. Se detiene aquí, sin abrir test ni avanzar a PCA.

Reproducción: `python -m src.experiments.entrenar_inventario verificar` (repite únicamente el presupuesto ya fijado en directorio temporal y compara artefactos); suite: `python -m src.experiments.entrenar_inventario suite`. No volver a preparar un plan distinto ni sobrescribir modelos congelados.
'''
    (OUT/'reporte_entrenamiento.md').write_text(text,encoding='utf-8',newline='\n')
    guardar(OUT/'discrepancias_previas.json',dict(prior_suite=read(PREVIOUS/'dataset/suite_existente.json'),
        prior_diagnosis=read(PREVIOUS/'dataset/discrepancias_suite_previas.json'),preflight_verified=True,
        old_tests_modified=False,old_hashes_overwritten=False))

def protocol():
    frozen=read(OUT/'congelacion.json');spec=read(PREVIOUS/'dataset/especificacion_test.json')
    guardar(OUT/'protocolo_test.json',dict(status='reserved_not_executed',months=spec['target_months'],outcomes_queried=False,
        dataset_recipe=spec,models=frozen['models'],final_models=frozen['final_models'],frozen_manifest_sha256=sha(OUT/'congelacion.json'),
        majority_class=0,historical=PLAN['baselines']['historical'],thresholds=PLAN['thresholds'],primary_and_common_expected_equal=True,
        policy='If any method lacks history preserve principal results and explicitly report separate common support; never invent zeros for unknown coverage',
        required_opening_record_before_queries=True,fit_or_threshold_search_on_test=False))
    (OUT/'protocolo_test.md').write_text('''# Protocolo futuro de evaluación de Inventario (no ejecutado)

Ventana fijada por cobertura auditada: enero–mayo de 2016, eventos [2016-01-01,2016-06-01). No se consultaron outcomes ni construyeron matrices de test. Antes de abrir, verificar todos los hashes de congelacion.json, modelos finales, dataset, preprocesador, configuración y protocolo; persistir registro de apertura antes de la primera consulta. No rediseñar por resultados ni usar train+val para reajustar estos modelos.

Extraer por SELECT únicamente recepciones tipo 11, IDs operativos únicos y fecha TransactionOccurredWhen; reconciliar con DW sin multiplicar joins, separar versiones por identidad operativa y verificar unidades base y ediciones. Reutilizar el snapshot de desarrollo intacto para historia inicial octubre–diciembre de 2015. La primera aparición debe proceder de cualquier movimiento conocido antes de cada t, sin seleccionar por actividad futura; nuevas identidades sin tres meses cubiertos quedan excluidas con motivo. No usar Fact.Stock Holding, saldos, estados de compra actuales o identidades como magnitudes en X.

Modo event_ledger_academic de conservación/cierre: solo ausencia en mes cubierto es cero; meses desconocidos o incompletos no se imputan. Construir cada X al inicio de t con sus tres meses cerrados y las 17 fórmulas congeladas; meses anteriores de test ya concluidos pueden alimentar historia posterior. Después del cierre calcular N(i,t) y etiqueta de exceso estricto sobre mediana histórica; empate a 0. No usar N(i,t) como entrada ni criterio de elegibilidad.

Aplicar únicamente transform con las medias, desviaciones poblacionales y orden congelados; no fit, cambio de columnas, escalas, pesos o umbrales. Perceptrón z>=0; logística seleccionada sigmoid(z)>=0,5, con su variante y pesos de clase de train. Si es ponderada, scores no automáticamente calibrados. Mayoría 0 de train e histórico 1[n1>mediana(n1,n2,n3)], sobre las mismas claves; se espera población principal=común pues todos requieren tres meses. Si aparece insuficiencia de algún método, registrar comparación común separada sin reducir silenciosamente resultados principales.

Persistir predicciones, scores, claves, etiquetas, disponibilidad, exclusiones y hashes. Reportar BA, accuracy, precision, recall, F1, specificity, confusiones y proporción de alertas globales y por mes. BA mensual sin una clase es no definida; AP logística es ranking, distinta de precision al umbral. Desglosar artículos vistos en train, vistos solo en val y nuevos, positivos por artículo/mes, TP y falsas alertas, sin redefinir población. Examinar sensibilidad descriptiva a pocos positivos y dependencia temporal/por artículo; no afirmar superioridad universal o utilidad operacional. No cambiar decisiones después del test. Este protocolo se prepara pero no se ejecuta en este encargo.
''',encoding='utf-8',newline='\n')
