"""Tres figuras fieles y dictamen de auditoría/PCA de Inventario."""
import numpy as np
from src.experiments.pca_inventario import OUT,PREVIOUS,read,sha,FEATURES

def figures(pca,z,labels,out):
    from src.evaluation.graficos import plt,PercentFormatter
    root=out/'graficos';root.mkdir(exist_ok=True)
    joined=np.vstack([z['train'],z['val']]);lo=joined.min(axis=0);hi=joined.max(axis=0);margin=np.maximum((hi-lo)*.05,.1)
    for split,coords in z.items():
        fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
        # Todos los negativos permanecen; positivos grandes y al final para legibilidad.
        for cls,color,size,alpha,order in [(0,'#246aa0',16,.24,1),(1,'#d85e16',62,.9,3)]:
            mask=labels[split]==cls
            ax.scatter(coords[mask,0],coords[mask,1],c=color,s=size,alpha=alpha,zorder=order,
                edgecolors='white' if cls else 'none',linewidths=.5 if cls else 0,
                label=f'Clase {cls}: {int(mask.sum())} observaciones')
        ax.set(title=f'Inventario: {"entrenamiento" if split=="train" else "validación"} por clase\nPCA de 17 entradas, ajuste solo en train; visualización, no clasificación',
            xlabel=f'PC1 ({100*pca.explained_variance_ratio_[0]:.2f}% de varianza train)',
            ylabel=f'PC2 ({100*pca.explained_variance_ratio_[1]:.2f}% de varianza train)',
            xlim=(lo[0]-margin[0],hi[0]+margin[0]),ylim=(lo[1]-margin[1],hi[1]+margin[1]))
        ax.legend(fontsize=10);ax.grid(alpha=.18)
        fig.savefig(root/f'dispersion_{split}.png',dpi=200,metadata={'Software':'PerceptronReal Inventario PCA'});plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained');x=np.arange(1,18)
    ax.bar(x,pca.explained_variance_ratio_all_,color=['#246aa0']*2+['#aebdcb']*15,label='Individual (train)')
    ax.plot(x,np.cumsum(pca.explained_variance_ratio_all_),marker='o',color='#d85e16',label='Acumulada (train)')
    ax.axvline(2,color='#665275',ls='--',label=f'Dos componentes: {100*pca.explained_variance_ratio_.sum():.2f}%')
    ax.set(title='Inventario: varianza explicada por SVD de entrenamiento',xticks=x,xlabel='Componente',ylabel='Varianza explicada',ylim=(0,1.03))
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'varianza_explicada.png',dpi=200,metadata={'Software':'PerceptronReal Inventario PCA'});plt.close(fig)

def reports(audit,causal,conc,baseline,conv,pca):
    def write(name,text):(OUT/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')
    vals=audit['val'];table='\n'.join(f"| {n} | {m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} | {100*m['accuracy']:.6f} | {100*m['balanced_accuracy']:.6f} | {100*m['precision']:.6f} | {100*m['recall']:.6f} | {100*m['f1']:.6f} | {100*m['specificity']:.6f} | {100*m['false_positive_rate']:.6f} |" for n,m in vals.items())
    diag=conc['historical_activity_diagnostic'];d=diag['metrics']
    write('auditoria_metricas.md',f'''
# Auditoría de métricas de validación de Inventario

Se verificaron todos los SHA-256 del manifiesto de entrenamiento y los modelos de congelacion.json, checkpoint 30 del perceptrón (seed42, eta 0,01) y 156 de logística estándar (eta 0,05, umbral 0,5). Pesos y sesgos coinciden exactamente con sus trazas de la época seleccionada; 17 entradas en idéntico orden. Las predicciones, márgenes y probabilidades persistidos se reproducen exactamente al cargar, sin fit, nuevas épocas o cambios de parámetros. Evidencia en verificacion_modelos.json.

## Soporte y conciliación independiente

Ambos clasificadores se evalúan sobre las mismas 2628 filas elegibles: 35 positivas y 2593 negativas, 219 artículos, enero–diciembre de 2015. No es un subconjunto seleccionado por aciertos. Train conserva 4599 filas (86/4513). Se calculó de nuevo cada matriz desde etiquetas originales y predicciones persistidas y se cotejó con métricas congeladas.

| Modelo | TN | FP | FN | TP | Accuracy % | BA % | Precision % | Recall % | F1 % | Specificity % | FPR % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

La BA alta es aritméticamente coherente con recall 1 y especificidad cercana a 0,975. La precision positiva es solo 35%/33,333%: hay muchas falsas alertas en relación con los 35 positivos. No se usa accuracy alta por sí sola como prueba de éxito.

## Estructura de actividad que explica parte del resultado

Diagnóstico descriptivo único, sin búsqueda ni sustitución de baselines: predecir 1 si existió alguna recepción en los tres meses anteriores, `1[n1+n2+n3>0]`. En las mismas 2628 filas da matriz {d['confusion_matrix']}, BA {100*d['balanced_accuracy']:.6f}%, precision {100*d['precision']:.6f}% y recall {100*d['recall']:.6f}%. Concuerda con perceptrón en {100*diag['agreement_perceptron']:.6f}% y con logística en {100*diag['agreement_logistic']:.6f}% de filas.

Esta regla usa solo historia y muestra que identificar artículos con actividad recurrente de recepción explica gran parte del rendimiento global. No se seleccionó ni optimizó otra política y no se incorpora como baseline oficial ni modelo final. Los 35 TP y las falsas alertas de los clasificadores están en nueve artículos; los otros 210, con todos sus meses negativos, no generan alertas. Diferenciar esos grupos es más fácil que anticipar qué mes de un artículo receptor supera su referencia. Esto no prueba fuga ni superioridad universal, y no redefine la población oficial.

La revisión causal no detectó información del mes objetivo en X: revision_causal.json contrasta todas las fórmulas, agregados, referencias, etiquetas y disponibilidad anterior a t. Los supuestos académicos de conservación y cierre siguen siendo limitaciones; no existe un registro independiente de creación/borrados que certifique completitud operativa. No se modificaron métricas ni predicciones anteriores.
''')
    rows='\n'.join(f"| `{f['name']}` | {f['formula']} | {f['source']} | {f['availability']} | {f['max_absolute_formula_error']:.3g} |" for f in causal['features'])
    write('revision_causal.md',f'''
# Revisión causal de las 17 características

Estado: {causal['status']}. Se contrastaron las 7227 observaciones de desarrollo con los agregados artículo-mes y estos con los 7298 IDs de recepción del snapshot aprobado, sin reconstruir ni exportar un nuevo dataset. Cada entrada se evaluó únicamente como comprobación y se comparó con su valor crudo persistido. El preprocesador congelado reproduce las matrices procesadas y su ajuste es solo train.

| Característica | Fórmula | Fuente original | Disponibilidad contractual | Error máximo absoluto |
|---|---|---|---|---:|
{rows}

Las ventanas se forman por clave explícita artículo–mes t-1,t-2,t-3; no hay shift/rolling posicional que cruce artículos o use el mes t. Se verifican cronología, fechas de última edición de historia anteriores al inicio de t, primera aparición conocida antes de t por cualquier movimiento y agregados por identidad operativa. No se usan versiones posteriores, IDs como magnitudes, cantidades objetivo, existencias actuales, supervivencia futura ni últimas apariciones.

Mes cubierto sin recepción vale cero solo con event_ledger_academic. Los tres meses iniciales no fabrican historia anterior a la aparición; unknown/incomplete no son cero. La mediana de etiqueta usa exactamente los tres meses anteriores, y empate es 0. Calendario de t es conocido antes de t. LastEditedWhen coincide con fecha del evento en esta evidencia, pero no demuestra fecha real independiente de creación; el cierre/conservación de eventos es un supuesto académico respaldado por reconciliación, no una certificación productiva.

Las pruebas focalizadas perturban fixtures/copias: objetivo y futuro no alteran X anterior, cambiar etiquetas no cambia PCA y elegibilidad no depende de actividad posterior. No se cambia el snapshot, las matrices ni ninguna etiqueta aprobada. Ninguna evidencia temporal observada obliga a bloquear por fuga; las limitaciones de archivo retrospectivo se mantienen.
''')
    items=[i for i in conc['by_item'] if i['val_positive']]
    table='\n'.join(f"| {i['item_id']} | {i['train_positive']} | {i['val_positive']} | {i['val_negative']} | {i['perceptron']['TP']} | {i['perceptron']['FP']} | {i['logistic']['TP']} | {i['logistic']['FP']} |" for i in items)
    months='\n'.join(f'| {m} | {p} |' for m,p in conc['positive_by_month'].items())
    write('concentracion_positivos.md',f'''
# Concentración de positivos y alertas

Nueve identidades operativas con 35 positivos de validación: {conc['positive_item_ids']}. Son observaciones repetidas, no 35 réplicas independientes ni nueve poblaciones elegidas para el resultado principal. Artículos positivos en val sin antecedente positivo en train: {conc['positive_items_without_train_positives']}. El detalle de los 219 artículos, incluidos los 210 siempre negativos en validación, permanece en concentracion_positivos.json.

| Artículo | Pos. train | Pos. val | Neg. val | TP perceptrón | FP perceptrón | TP logística | FP logística |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

| Mes | Positivos val |
|---|---:|
{months}

TP compartidos: {conc['shared_true_positives']}. FP compartidos: {conc['shared_false_positives']}; FP exclusivos del perceptrón: {conc['perceptron_only_false_positives']}; de logística: {conc['logistic_only_false_positives']}. Desacuerdos totales: {conc['disagreements']}; ambos correctos: {conc['both_correct']}; ambos incorrectos: {conc['both_wrong']}. Las claves exactas de los errores están registradas en JSON. No se eliminan ni se utilizan para nuevas características o selección.

Los intervalos Wilson descriptivos del entrenamiento no establecen independencia temporal o por artículo. Con tan pocos positivos, meses sin positivos y alertas concentradas, un recall de 100% en esta validación no garantiza recall futuro ni utilidad económica. El diagnóstico de actividad histórica del informe de métricas explica la estructura dominante sin redefinir población ni modificar los baselines congelados.
''')
    write('conciliacion_baselines.md',f'''
# Conciliación de soportes y baselines

Mayoría: clase 0, determinada exclusivamente de train. Histórico exacto de Inventario: `{baseline['formula']}`. Requiere tres meses; no el cuarto antecedente del experimento financiero. Se volvió a calcular desde las entradas crudas y se comparó con las predicciones persistidas por las mismas claves.

| Partición | Principal | Común cuatro métodos | Mayoritario | Histórico | Sin histórico |
|---|---:|---:|---:|---:|---:|
| Train | 4599 | 4599 | 4599 | 4599 | 0 |
| Validación | 2628 | 2628 | 2628 | 2628 | 0 |

Elegibilidad: identidad operativa, tres meses consecutivos anteriores cubiertos desde primera aparición conocida antes de t y etiqueta válida de mes completo. Principal son todas las filas elegibles para ambos clasificadores; común es su intersección con ambos baselines. Aquí coinciden exactamente en claves y orden, sin exclusión por aciertos ni errores. No se fabrican antecedentes. BA val mayoritario={100*baseline['val']['methods']['majority']['balanced_accuracy']:.6f}%; histórico={100*baseline['val']['methods']['historical']['balanced_accuracy']:.6f}%. La política simple de actividad analizada es un diagnóstico posterior separado y no reemplaza estos baselines.
''')
    interp=read(OUT/'interpretacion_componentes.json');variance=read(OUT/'varianza_explicada.json')
    top='\n'.join(f"| {pc} | `{r['feature']}` | {r['coefficient']:.9f} | {r['squared_loading_share']:.9f} |" for pc,rows in interp['top_coefficients'].items() for r in rows)
    write('interpretacion_pca.md',f'''
# PCA bidimensional de Inventario

Se reutiliza PCA2D por SVD NumPy sin modificarla. Fit exclusivamente sobre X_train procesada 4599×17, centrada por su media de train. Validación 2628×17 recibe la misma media y matriz de componentes; no se ajusta un escalador nuevo, no se usan etiquetas en fit y PCA no entra en los clasificadores. Se conservan todos los puntos: negativos transparentes y positivos grandes dibujados después, sin jitter ni submuestreo. Varias observaciones pueden coincidir visualmente.

PC1 explica {variance['PC1_percent']:.9f}%, PC2 {variance['PC2_percent']:.9f}%; acumulada {variance['retained_percent']:.9f}% y fuera del plano {variance['lost_percent']:.9f}%. Rango numérico {variance['rank']} de 17. Las redundancias entre lags, media, mediana y cambios pueden reducir rango; no se eliminaron entradas de los clasificadores por ello. Ver comprobaciones_matematicas.json: ortogonalidad, autovalores de covarianza, varianza proyectada, suma de ratios y residuo de reconstrucción.

| Componente | Característica | Coeficiente real | Cuadrado del coeficiente |
|---|---|---:|---:|
{top}

Los nombres y signos de esta tabla definen la interpretación: PC1 combina principalmente {', '.join(r['feature'] for r in interp['top_coefficients']['PC1'][:3])}; PC2, {', '.join(r['feature'] for r in interp['top_coefficients']['PC2'][:3])}. En términos empresariales, PC1 resume intensidad habitual de recepciones, en contraste con ausencia de actividad. PC2 recoge principalmente tendencia, cambio reciente y variabilidad de los conteos; describe una dirección de variación, no una predicción de aumentos futuros. Los coeficientes son de autovectores unitarios sobre variables estandarizadas, no efectos causales ni importancias predictivas; su signo se fija por el coeficiente absoluto mayor positivo. Los cuadrados suman uno por componente, no son porcentajes del objetivo.

La agrupación repetida por artículo se cuantifica con centroides y fracción de suma de cuadrados entre artículos: train PC1={interp['grouping']['train']['between_item_SS_fraction_PC1']:.6f}, PC2={interp['grouping']['train']['between_item_SS_fraction_PC2']:.6f}; val PC1={interp['grouping']['val']['between_item_SS_fraction_PC1']:.6f}, PC2={interp['grouping']['val']['between_item_SS_fraction_PC2']:.6f}. Son descriptores geométricos de observaciones repetidas, no evidencia de grupos independientes o separabilidad causal. Los positivos pertenecen a pocos artículos receptores y se muestran junto a sus negativos; el agrupamiento por historia de recepción no garantiza separar aumentos mensuales. La información descartada y los solapamientos impiden deducir el rendimiento en 17 dimensiones de la apariencia 2D.

Parámetros, 17×2 coeficientes y proyecciones por clave se guardan sin tocar X/y. Las etiquetas en CSV se agregan exclusivamente como anotación visual después del ajuste. No hay proyección ni outcomes de test. La reproducibilidad y las perturbaciones de etiquetas se verifican separadamente.
''')
    cp=conv['perceptron'];cl=conv['logistic']
    write('verificacion_checkpoints.md',f'''
# Checkpoints y convergencia, sin nuevas épocas

Perceptrón: checkpoint 30, 100 épocas originales, {cp['updates_total']} actualizaciones, {cp['last_online_errors']} errores online en la última época, cero épocas sin errores. BA de validación seleccionada {cp['selected_checkpoint_metrics']['val']['balanced_accuracy']:.9f} frente a última {cp['last_epoch_metrics']['val']['balanced_accuracy']:.9f}. Continúa actualizándose: no hay convergencia a cero demostrada.

Logística estándar: checkpoint 156 de 400 épocas. Pérdida inicial/final train {cl['initial_train_loss']:.9f}/{cl['last_train_loss']:.9f}; gradiente final {cl['last_gradient_norm']:.9g}, superior a 1e-6. Terminación max_epochs, no tolerancia; no convergencia completa. BA val del checkpoint {cl['selected_checkpoint_metrics']['val']['balanced_accuracy']:.9f}, última {cl['last_epoch_metrics']['val']['balanced_accuracy']:.9f}.

Se cotejaron historiales JSON y trazas de parámetros originales con el diagnóstico guardado; no se alteraron curvas, pesos, sesgos, checkpoints ni hiperparámetros. No se llamó a entrenamiento.
''')
    write('protocolo_test_definitivo.md','''
# Protocolo definitivo previo a test de Inventario

Fuente normativa estructurada: protocolo_test_definitivo.json, con SQL, parámetros, hashes, modelos y reglas exactas. Estado cerrado antes de observar outcomes; evaluación no ejecutada. Ventana enero–mayo de 2016, intervalo [2016-01-01,2016-06-01), corte exclusivo de conocimiento de etiquetas 2016-06-01 00:00:00. No ampliar ventana o cambiar umbrales después de resultados.

Antes de cualquier consulta futura: verificar manifiestos y hashes de protocolo, código, dataset, preprocesador y modelos; registrar UTC, finalidad y hashes en archivo de apertura persistente de una sola creación. Si existe registro, no volver a tratar test como reserva independiente. El dictamen actual recomienda metodológicamente una primera evaluación posterior, pero no la ejecuta ni avanza automáticamente.

SELECT de recepciones tipo 11 con proyección mínima aprobada y límites exactos incluidos en JSON; autenticación integrada y lectura únicamente. Reconciliar ID único, identidad StockItemID/WWI Stock Item ID, tipo, fecha, cantidad y documentos con DW, sin multiplicación por joins. Primera aparición por cualquier movimiento conocido estrictamente antes del inicio de cada objetivo; el SELECT agregado de identidad va acotado por ese instante y no por supervivencia o actividad futura. Verificar disponibilidad de ediciones y versiones históricas, no usar nombres o claves DW como magnitudes.

Cobertura event_ledger_academic: conservación y cierre supuestos explícitos; meses desconocidos/incompletos se registran y excluyen, nunca cero automático. Tres meses consecutivos cubiertos desde primera aparición, sin requisito de evento en el objetivo; artículos nuevos esperan historia suficiente. Historia inicial octubre–diciembre de 2015 congelada; meses anteriores de test ya cerrados pueden alimentar objetivos posteriores. Al inicio de t generar solo las 17 fórmulas aprobadas; después del cierre generar N(t) y y=1[N(t)>mediana(n1,n2,n3)], empate 0. N(t) no participa en X ni en elegibilidad.

Solo transform del preprocesador congelado: 17 entradas, mismas medias, desviaciones ddof=0 y orden. Cargar perceptrón checkpoint30 seed42 eta0,01, activación z>=0 y logística estándar checkpoint156 eta0,05, umbral0,5. Sin fit, ponderación nueva, cambio de columnas ni umbrales. PCA conserva fit de train solo para visualización y nunca alimenta clasificación.

Mayoría 0 de train; histórico 1[n1>mediana(n1,n2,n3)]. Principal incluye todas las filas elegibles para clasificadores; común es intersección exacta de los cuatro métodos, esperada igual por requerir todos tres meses. Si faltase baseline, conservar principal y documentar soporte común y motivos por separado; no inventar cuarto mes ni eliminar casos según error. El diagnóstico exploratorio de actividad no es baseline nuevo ni se incorpora a la selección.

Guardar predicciones, scores, claves, eventos, disponibilidad, exclusiones, X/y y hashes separados; reproducir offline sin fit. Reportar globalmente, cada mes y por artículo: soportes, TP/TN/FP/FN, accuracy, BA, precision, recall, F1, especificidad, FPR y proporción de alertas; AP logística por scores agrupando empates, no equivalente a precision del umbral. Denominadores cero: precision/F1=0 sin alertas con anotación, recall/especificidad ausentes=None, BA=None si falta una clase real, AP=None si no hay positivos; conservar meses monoclase sin fingir 50%. Distinguir artículos vistos en train, vistos solo en val, nuevos elegibles y excluidos por historia. Conservar concentración de eventos positivos/TP/falsas alertas, dependencia temporal/por artículo y supuestos de cobertura. No seleccionar nuevos modelos, umbrales o población con test; no reentrenar train+val para esta comparación.
''')
    write('dictamen_pretest.md','''
# Dictamen previo a primera evaluación de test

Las predicciones, scores, pesos/sesgos, checkpoints 30/156 y métricas de validación se reproducen exactamente sobre las 2628 filas oficiales. La revisión focalizada no detecta fuga temporal en las 17 características; no certifica completitud operativa independiente del archivo retrospectivo. La BA superior al 98% es coherente con los conteos y con una estructura de recepción muy concentrada: los 35 positivos y las falsas alertas están en nueve artículos. Una política descriptiva de actividad previa explica parte importante del resultado. No se cambió ni optimizó el experimento a partir de ese hallazgo.

Principal y comparación común coinciden: 4599 train/2628 val; el histórico usa tres meses y no excluye filas. PCA se ajusta solo en train por SVD y es exclusivamente visual. No se reentrenaron clasificadores, reajustó preprocesador ni abrieron outcomes de 2016. Modelos, dataset, entrenamiento, informe y demás Data Marts permanecen protegidos por hashes.

Se recomienda metodológicamente autorizar la primera evaluación posterior de test conforme al protocolo definitivo, condicionada a verificación satisfactoria de hashes, pruebas focalizadas, ausencia de regresiones nuevas y registro de apertura antes de consultas. La reserva sigue sin abrir; este encargo se detiene en el dictamen. La suite global y los nueve casos históricos se declaran explícitamente en validaciones.json y suite_global.json; no se presentan como una suite totalmente aprobada ni se cambian sus valores esperados.
''')
