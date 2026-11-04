# Entrenamiento, validación y congelación de Inventario

Se preservan las matrices aprobadas y los 17 predictores, sin SQL, reconstrucción, acceso a 2016, PCA o cambio del informe. Se utilizan las clases matemáticas originales para perceptrón y logística estándar, con un observador de métricas por época que no cambia actualizaciones. La variante ponderada está aislada en src/models/logistic_inventario.py; reutiliza validadores, sigmoide, predicción, métricas y selección de la implementación previa.

## Presupuesto registrado antes de observar resultados

Seis ejecuciones: perceptrón eta 0,01, 100 épocas, semillas 42 y 7; logística estándar eta 0,05 y 0,10, 400 épocas; logística ponderada las mismas dos tasas y épocas. Logísticas con semilla 42 e inicialización cero, lote completo, tolerancia de gradiente 1e-6 y umbral fijo 0,5; perceptrón z>=0 y permutación por época. Todas parten de pesos/sesgo nuevos a cero. No se amplía el espacio tras observar métricas. Configuración y registro UTC con hash en configuracion_experimental.json y registro_previo.json.

La ponderación es una enmienda expresamente autorizada por este encargo al anterior valor por defecto sin pesos; no modifica dataset, población ni validación. Pesos calculados solo con train: n/(2*n_clase), 4599/(2*4513) para clase 0 y 4599/(2*86) para clase 1. Lw=sum(w_i*(logaddexp(0,z_i)-y_i*z_i))/sum(w_i); gradiente = X.T@(w_i*(sigmoid(z_i)-y_i)/sum(w_i)), con análogo para sesgo. Diferencias finitas moderadas y de logits extremos verifican la derivada. No hay SMOTE, replicación ni muestreo sintético de negocio.

Selección por BA sin ponderar de validación, después F1, época más temprana y orden de configuración. Se documentan los mejores estándar/ponderado y se elige la logística final entre ambos mediante la misma regla. No se optimizan umbrales, no se selecciona por AP y no se reentrena con train+val.

| Grupo | Ejecución | Variante | Época |
|---|---|---|---:|
| perceptron | p_seed42 | standard | 30 |
| logistic | l_eta005 | standard | 156 |
| logistic_standard | l_eta005 | standard | 156 |
| logistic_weighted | lw_eta005 | weighted | 1 |

## Métricas de todos los checkpoints candidatos

Todas las métricas se calculan sin pesos; orden de matriz [[TN,FP],[FN,TP]], clase positiva 1. Se conservan las 4599/2628 observaciones, 86/35 positivas; solo 10/9 artículos tienen positivos.

| Ejecución | Partición | Época | Accuracy | BA | Precision | Recall | F1 | Specificity | TN/FP/FN/TP | Alertas |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| p_seed42 | train | 30 | 0.984127 | 0.969100 | 0.543046 | 0.953488 | 0.691983 | 0.984711 | 4444/69/4/82 | 0.032833 |
| p_seed42 | val | 30 | 0.975266 | 0.987466 | 0.350000 | 1.000000 | 0.518519 | 0.974933 | 2528/65/0/35 | 0.038052 |
| p_seed7 | train | 98 | 0.979561 | 0.978179 | 0.477273 | 0.976744 | 0.641221 | 0.979614 | 4421/92/2/84 | 0.038269 |
| p_seed7 | val | 98 | 0.972222 | 0.985924 | 0.324074 | 1.000000 | 0.489510 | 0.971847 | 2520/73/0/35 | 0.041096 |
| l_eta005 | train | 156 | 0.979126 | 0.926629 | 0.468750 | 0.872093 | 0.609756 | 0.981166 | 4428/85/11/75 | 0.034790 |
| l_eta005 | val | 156 | 0.973364 | 0.986502 | 0.333333 | 1.000000 | 0.500000 | 0.973004 | 2523/70/0/35 | 0.039954 |
| l_eta010 | train | 73 | 0.978691 | 0.932111 | 0.463415 | 0.883721 | 0.608000 | 0.980501 | 4425/88/10/76 | 0.035660 |
| l_eta010 | val | 73 | 0.972983 | 0.986309 | 0.330189 | 1.000000 | 0.496454 | 0.972619 | 2522/71/0/35 | 0.040335 |
| lw_eta005 | train | 1 | 0.932811 | 0.954359 | 0.214834 | 0.976744 | 0.352201 | 0.931974 | 4206/307/2/84 | 0.085018 |
| lw_eta005 | val | 1 | 0.972222 | 0.985924 | 0.324074 | 1.000000 | 0.489510 | 0.971847 | 2520/73/0/35 | 0.041096 |
| lw_eta010 | train | 1 | 0.932811 | 0.954359 | 0.214834 | 0.976744 | 0.352201 | 0.931974 | 4206/307/2/84 | 0.085018 |
| lw_eta010 | val | 1 | 0.972222 | 0.985924 | 0.324074 | 1.000000 | 0.489510 | 0.971847 | 2520/73/0/35 | 0.041096 |

## Baselines y clase minoritaria

El mayoritario 0 se determina únicamente de train; accuracy cercana a 99% sin detectar positivos no es éxito predictivo. Histórico exacto de Inventario: 1[N(t-1)>mediana(N(t-1),N(t-2),N(t-3))]. Usa los mismos tres meses que cada fila elegible, no requiere cuarto mes. Población principal y común son idénticas: 4599 train y 2628 val, cero exclusiones por baseline. No se sustituye por la fórmula financiera. metricas.json conserva métricas completas y matrices de los cinco métodos, con comparación estándar/ponderada como variantes del mismo algoritmo.

- perceptron: BA 0.987466, diferencia frente al histórico +0.389973; predice ambas clases. AP=no aplicable al perceptrón.
- logistic_standard: BA 0.986502, diferencia frente al histórico +0.389009; predice ambas clases. AP=0.3668857240075474.
- logistic_weighted: BA 0.985924, diferencia frente al histórico +0.388430; predice ambas clases. AP=0.3548707735111173.

AP describe el ordenamiento de puntuaciones; no es precision al umbral 0,5 ni demuestra calibración. Los scores ponderados no deben interpretarse automáticamente como probabilidades calibradas. Su entrenamiento optimiza una BCE ponderada, mientras que toda evaluación de validación permanece sin pesos. No se comparan numéricamente ambas pérdidas como un mismo objetivo. Los gráficos ponderados separan explícitamente objetivo de entrenamiento y BCE estándar de evaluación.

## Convergencia y checkpoints

- p_seed42: errores online inicial/final 112/68, actualizaciones 6983, subidas de errores 49, épocas sin errores []; checkpoint 30. No equivale a la última época ni a convergencia a cero si persisten actualizaciones.
  BA val del checkpoint/última época: 0.987466255/0.945815658; BA train del checkpoint/última época: 0.969099604/0.978733272.
- p_seed7: errores online inicial/final 123/66, actualizaciones 7116, subidas de errores 49, épocas sin errores []; checkpoint 98. No equivale a la última época ni a convergencia a cero si persisten actualizaciones.
  BA val del checkpoint/última época: 0.985923641/0.892771748; BA train del checkpoint/última época: 0.978179317/0.975245930.
- l_eta005 (unweighted train and validation BCE): pérdida train inicial/final 0.693147181/0.077311004; BCE val inicial/final 0.693147181/0.079603937; gradiente final 0.051118254; terminación max_epochs, checkpoint 156. Tolerancia 1e-6, alcanzada=False.
  BA val del checkpoint/última época: 0.986502121/0.652707840; BA train del checkpoint/última época: 0.926629273/0.764672084.
- l_eta010 (unweighted train and validation BCE): pérdida train inicial/final 0.693147181/0.050948795; BCE val inicial/final 0.693147181/0.053607003; gradiente final 0.0256563065; terminación max_epochs, checkpoint 73. Tolerancia 1e-6, alcanzada=False.
  BA val del checkpoint/última época: 0.986309294/0.653479147; BA train del checkpoint/última época: 0.932110853/0.753154968.
- lw_eta005 (weighted train BCE / unweighted validation BCE): pérdida train inicial/final 0.693147181/0.131733488; BCE val inicial/final 0.693147181/0.205781600; gradiente final 0.0468718018; terminación max_epochs, checkpoint 1. Tolerancia 1e-6, alcanzada=False.
  BA val del checkpoint/última época: 0.985923641/0.985923641; BA train del checkpoint/última época: 0.954359241/0.977957735.
- lw_eta010 (weighted train BCE / unweighted validation BCE): pérdida train inicial/final 0.693147181/0.107962956; BCE val inicial/final 0.693147181/0.155189366; gradiente final 0.025504319; terminación max_epochs, checkpoint 1. Tolerancia 1e-6, alcanzada=False.
  BA val del checkpoint/última época: 0.985923641/0.985923641; BA train del checkpoint/última época: 0.954359241/0.977957735.

Los historiales JSON/CSV incluyen errores/actualizaciones, accuracy/BA, precision/recall/F1, normas de pesos y gradiente cuando corresponde; traza_parametros.json conserva métricas completas, pesos y sesgo por época. La última época y el checkpoint se comparan sin confundirlos. Descenso de pérdida no prueba convergencia completa si termina por máximo de épocas.

## Concentración, incertidumbre y límites

concentracion_positivos.json conserva por artículo verdaderos positivos, falsas alertas y confusiones; metricas_mensuales.json y predicciones_desarrollo.csv conservan meses, soporte y alertas. Hay 35 positivos de validación en solo nueve artículos, frente a 210 artículos con todos sus meses negativos. En los tres clasificadores, todos los TP y FP se concentran en esos nueve artículos; los otros 210 no generan alertas. La BA global alta se beneficia de reconocer los numerosos artículos inactivos en recepción, y no equivale a discriminar con precisión los meses de aumento dentro de los artículos que reciben. Este desglose es descriptivo y no redefine la población ni selecciona modelos. Se mantienen meses monoclase con BA nula/no definida, sin borrarlos para mejorar promedios. Intervalos Wilson de recall al 95% son descriptivos bajo supuesto binomial: no se ha demostrado independencia temporal o por artículo y pueden ser optimistas. No sostienen significación estadística, superioridad universal ni utilidad económica. No se redefine población según errores. La cobertura y conservación histórica siguen siendo supuestos académicos.

## Integridad, reproducción y cierre

preflight.json contrasta hashes, catálogo, X/y/metadatos, preprocesador train-only y fechas. Los nueve fallos históricos incluyen hashes de una versión anterior del informe; discrepancias_previas.json documenta que preceden a este trabajo y no afectan matrices, algoritmos o reproducibilidad actuales. No se cambian pruebas, hashes o archivos previos para ocultarlos. Los modelos de modelos/ incluyen perceptrón, logística final y ganadores estándar/ponderado para trazabilidad; son dos algoritmos, no tres.

congelacion.json vincula modelos con dataset, preprocesador, configuración, umbrales, métricas y hashes. protocolo_test.md y protocolo_test.json definen una futura apertura, sin ejecutarla. pruebas_focalizadas.txt, suite_global.json, reproducibilidad.json y validaciones.json distinguen controles nuevos, nueve casos históricos y cualquier regresión. No presentar la suite global como aprobada mientras existan esos casos. Se detiene aquí, sin abrir test ni avanzar a PCA.

Reproducción: `python -m src.experiments.entrenar_inventario verificar` (repite únicamente el presupuesto ya fijado en directorio temporal y compara artefactos); suite: `python -m src.experiments.entrenar_inventario suite`. No volver a preparar un plan distinto ni sobrescribir modelos congelados.

## Resultado final de verificaciones

19 pruebas nuevas y 35 controles de entrenamiento satisfactorios; 48 artefactos reproducidos byte a byte, incluidos 18 gráficos. Suite global: 386 pruebas, 377 correctas y exactamente los nueve casos históricos (cinco fallos y cuatro errores), sin regresiones nuevas. Los 890 archivos previos permanecen íntegros. Diferencias finitas registradas en gradientes_ponderados.json; revisión visual en revision_visual.json. Test 2016 continúa cerrado.
