# Auditoría de métricas de validación de Inventario

Se verificaron todos los SHA-256 del manifiesto de entrenamiento y los modelos de congelacion.json, checkpoint 30 del perceptrón (seed42, eta 0,01) y 156 de logística estándar (eta 0,05, umbral 0,5). Pesos y sesgos coinciden exactamente con sus trazas de la época seleccionada; 17 entradas en idéntico orden. Las predicciones, márgenes y probabilidades persistidos se reproducen exactamente al cargar, sin fit, nuevas épocas o cambios de parámetros. Evidencia en verificacion_modelos.json.

## Soporte y conciliación independiente

Ambos clasificadores se evalúan sobre las mismas 2628 filas elegibles: 35 positivas y 2593 negativas, 219 artículos, enero–diciembre de 2015. No es un subconjunto seleccionado por aciertos. Train conserva 4599 filas (86/4513). Se calculó de nuevo cada matriz desde etiquetas originales y predicciones persistidas y se cotejó con métricas congeladas.

| Modelo | TN | FP | FN | TP | Accuracy % | BA % | Precision % | Recall % | F1 % | Specificity % | FPR % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perceptron | 2528 | 65 | 0 | 35 | 97.526636 | 98.746626 | 35.000000 | 100.000000 | 51.851852 | 97.493251 | 2.506749 |
| logistic | 2523 | 70 | 0 | 35 | 97.336377 | 98.650212 | 33.333333 | 100.000000 | 50.000000 | 97.300424 | 2.699576 |

La BA alta es aritméticamente coherente con recall 1 y especificidad cercana a 0,975. La precision positiva es solo 35%/33,333%: hay muchas falsas alertas en relación con los 35 positivos. No se usa accuracy alta por sí sola como prueba de éxito.

## Estructura de actividad que explica parte del resultado

Diagnóstico descriptivo único, sin búsqueda ni sustitución de baselines: predecir 1 si existió alguna recepción en los tres meses anteriores, `1[n1+n2+n3>0]`. En las mismas 2628 filas da matriz [[2520, 73], [0, 35]], BA 98.592364%, precision 32.407407% y recall 100.000000%. Concuerda con perceptrón en 99.695586% y con logística en 99.885845% de filas.

Esta regla usa solo historia y muestra que identificar artículos con actividad recurrente de recepción explica gran parte del rendimiento global. No se seleccionó ni optimizó otra política y no se incorpora como baseline oficial ni modelo final. Los 35 TP y las falsas alertas de los clasificadores están en nueve artículos; los otros 210, con todos sus meses negativos, no generan alertas. Diferenciar esos grupos es más fácil que anticipar qué mes de un artículo receptor supera su referencia. Esto no prueba fuga ni superioridad universal, y no redefine la población oficial.

La revisión causal no detectó información del mes objetivo en X: revision_causal.json contrasta todas las fórmulas, agregados, referencias, etiquetas y disponibilidad anterior a t. Los supuestos académicos de conservación y cierre siguen siendo limitaciones; no existe un registro independiente de creación/borrados que certifique completitud operativa. No se modificaron métricas ni predicciones anteriores.
