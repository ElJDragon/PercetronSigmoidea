# Evaluación predictiva final WWI — Fase 7

## Integridad y apertura

Primera evaluación predictiva de test autorizada por esta fase. Registro previo e inmutable de primera apertura en registro_apertura_test.json (fecha/hora UTC−05:00, versiones, hashes, reglas y métricas); intentos/repeticiones en registro_ejecuciones.jsonl, sin sobrescribir eventos. Antes de abrir y_test se verificaron manifiestos, datos, modelos, transformaciones, contratos y parámetros congelados, y se reprodujeron exactamente las métricas train/val aprobadas. Congelación SHA-256 `e65994faf713b8feb4ab440befeb0711845bc10534b1f473421538576f0d28a4`; contrato 2.0. Ninguna inconsistencia se corrige reentrenando.

Se evaluaron **1.111 filas** con **20 características completas**, mismas claves/orden para cuatro métodos. X_test float64, y_test int8, finitos y 0/1. No se recalculó el dataset, preprocesamiento o PCA; no hubo fit, selección de nuevos parámetros ni exclusión por rendimiento. La cubierta temporal completa es el supuesto académico aprobado, no certificación del origen.

Reglas: perceptrón z>=0 (checkpoint 70), logística probabilidad>=0,5 (checkpoint 200), mayoritario=1 verificado con 2.382/2.217 etiquetas train, histórico=1[unidades_lag_1>R_referencia] usando solo agregados históricos aprobados (empate→0). La referencia se comprobó con los tres lags, no se construyó con respuestas del objetivo.

## Resultados globales

Principal: balanced accuracy; no se cambió tras ver los resultados. Positiva=ventas del mes superiores a mediana reciente. Los cuatro métodos usan exactamente el mismo soporte real.

| Método | N | Soporte 0 | Soporte 1 | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perceptron | 1111 | 558 | 553 | 53.915 % | 52.008 % | 96.022 % | 67.471 % | 54.104 % |
| logistica | 1111 | 558 | 553 | 62.286 % | 57.666 % | 91.139 % | 70.638 % | 62.415 % |
| mayoritario | 1111 | 558 | 553 | 49.775 % | 49.775 % | 100.000 % | 66.466 % | 50.000 % |
| historico | 1111 | 558 | 553 | 51.395 % | 51.884 % | 32.369 % | 39.866 % | 51.310 % |

| Método | TN | FP | FN | TP | Errores |
|---|---:|---:|---:|---:|---:|
| perceptron | 68 | 490 | 22 | 531 | 512 |
| logistica | 188 | 370 | 49 | 504 | 419 |
| mayoritario | 0 | 558 | 0 | 553 | 558 |
| historico | 392 | 166 | 374 | 179 | 540 |

En cada método TN+FP+FN+TP=N. Las matrices tienen filas reales/columnas predichas, orden 0/1. Métricas y convenciones de denominador cero se reutilizan del módulo aprobado; no se inventan estimaciones si falta una clase.

## Generalización frente a desarrollo

| Método | BA train | BA validación | BA test final | Test−validación |
|---|---:|---:|---:|---:|
| perceptron | 64.684 % | 61.632 % | 54.104 % | -7.528 pp |
| logistica | 65.742 % | 61.199 % | 62.415 % | +1.216 pp |
| mayoritario | 50.000 % | 50.000 % | 50.000 % | +0.000 pp |
| historico | 49.824 % | 49.222 % | 51.310 % | +2.088 pp |

Train/validación se reutilizaron de fase 5 y se verificó la reproducción de checkpoints. Validación participó en selección; sus métricas son optimistas respecto a una evaluación independiente. Test es el resultado final reservado de esta versión, no otra oportunidad para elegir modelo, época o umbral. Diferencias descriptivas no establecen significación ni superioridad universal; los métodos tienen presupuestos/objetivos distintos.

En esta reserva temporal agregada, la logística tiene BA 62.415 %, frente a 54.104 % del perceptrón, una diferencia descriptiva de 8.311 puntos. La BA del perceptrón cambia -7.528 puntos respecto a validación; la logística cambia +1.216. Son resultados finales observados; no se elige ni ajusta un nuevo modelo con estas diferencias.

## Comportamiento mensual

| Mes objetivo | N | Soporte 0/1 | BA perceptrón | BA logística | BA mayoritario | BA histórico |
|---|---:|---:|---:|---:|---:|---:|
| 2016-01 | 219 | 125/94 | 55.740 % | 63.753 % | 50.000 % | 56.081 % |
| 2016-02 | 219 | 155/64 | 54.667 % | 65.071 % | 50.000 % | 50.514 % |
| 2016-03 | 219 | 81/138 | 51.235 % | 56.065 % | 50.000 % | 49.799 % |
| 2016-04 | 227 | 107/120 | 51.238 % | 56.012 % | 50.000 % | 54.540 % |
| 2016-05 | 227 | 90/137 | 52.620 % | 51.127 % | 50.000 % | 46.630 % |

metricas_mensuales.json contiene también accuracy, precision, recall, F1, matriz y conteos por método/mes; promedios_mensuales.json conserva el promedio **no ponderado** y número de meses estimables por métrica. Las métricas globales se obtienen de todos los conteos conjuntos; no son necesariamente el promedio mensual. Un grupo sin ambas clases tiene BA no estimable y no se elimina. Los cinco meses, incluidos los desfavorables, se reportan completos.

La BA mensual media de logística es 58.406 %, distinta de su BA global 62.415 %. La BA global combina las tasas por clase con soportes temporales distintos, no es una media aritmética de las cinco BA. La composición real cambia entre meses (soportes 0/1 en tabla). El resultado mensual de logística va de 51.127 % a 65.071 %; su resultado agregado no representa igual comportamiento en cada período.

## Cohortes de productos

Conocidos: nombres exactos presentes en la lista de entrenamiento congelada; nuevos: nombres ausentes de esa lista. No se trata de altas comerciales certificadas. Se confirmaron 219 productos conocidos/1.095 filas y ocho nuevos/16 filas (abril-mayo). No se excluyó ningún producto nuevo por sus resultados.

| Cohorte | Productos | Método | N | Soporte 0/1 | Accuracy | Precision | Recall | F1 | BA |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| conocidos | 219 | perceptron | 1095 | 552/543 | 53.973 % | 51.914 % | 97.422 % | 67.734 % | 54.327 % |
| conocidos | 219 | logistica | 1095 | 552/543 | 62.192 % | 57.457 % | 91.529 % | 70.597 % | 62.431 % |
| conocidos | 219 | mayoritario | 1095 | 552/543 | 49.589 % | 49.589 % | 100.000 % | 66.300 % | 50.000 % |
| conocidos | 219 | historico | 1095 | 552/543 | 51.598 % | 51.917 % | 32.413 % | 39.909 % | 51.442 % |
| nuevos | 8 | perceptron | 16 | 6/10 | 50.000 % | 100.000 % | 20.000 % | 33.333 % | 60.000 % |
| nuevos | 8 | logistica | 16 | 6/10 | 68.750 % | 77.778 % | 70.000 % | 73.684 % | 68.333 % |
| nuevos | 8 | mayoritario | 16 | 6/10 | 62.500 % | 62.500 % | 100.000 % | 76.923 % | 50.000 % |
| nuevos | 8 | historico | 16 | 6/10 | 37.500 % | 50.000 % | 30.000 % | 37.500 % | 40.000 % |

metricas_cohorte_mes.json conserva el cruce completo, incluyendo grupos vacíos como no estimables. Dieciséis casos no demuestran estabilidad ni capacidad general para productos nuevos; cada error puede cambiar apreciablemente las cifras.

En conocidos, BA de perceptrón 54.327 % y logística 62.431 %. En nuevos, BA 60.000 % y 68.333 %, respectivamente. Una precisión alta en esa cohorte puede coexistir con recall bajo; se deben leer los conteos de ambas clases y no extrapolar con N=16.

## Errores y desacuerdos

Los clasificadores coinciden en 928 casos y discrepan en 183. Ambos aciertan 554, ambos fallan 374; solo acierta perceptrón en 45 y solo logística en 138. Estas últimas cuatro categorías (ambos aciertan, ambos fallan, solo perceptrón acierta, solo logística acierta) son excluyentes y suman N. Errores y desacuerdos por mes/cohorte están en errores_desacuerdos.json; predicciones_alineadas.csv permite rastrear FP/FN/TP/TN por observación y método, con probabilidades logísticas y scores del perceptrón claramente identificados.

El perceptrón predice clase 1 en 1021 casos y la logística en 874, frente a 553 positivos reales. El recall elevado coexiste con falsos positivos abundantes: 490 y 370, respectivamente. La especificidad es 12.186 % y 33.692 %. F1 y recall positivos aislados no reflejan bien esta dificultad para reconocer meses no superiores.

- perceptron: mes con mayor tasa de error 2016-02, 132/219 (60.274 %). Todos los meses se conservan.
- logistica: mes con mayor tasa de error 2016-04, 95/227 (41.850 %). Todos los meses se conservan.
- mayoritario: mes con mayor tasa de error 2016-02, 155/219 (70.776 %). Todos los meses se conservan.
- historico: mes con mayor tasa de error 2016-03, 128/219 (58.447 %). Todos los meses se conservan.

FP significa anticipar ventas superiores sin que se observe ese resultado; FN significa no anticipar una subida realmente observada. No se conoce su coste monetario y no se presume que uno sea más grave. Acuerdo entre modelos no implica acierto. Estos análisis no se utilizaron para modificar pesos, referencias ni umbrales.

No se detectó una contradicción demostrable de construcción, alineación o implementación en las comprobaciones. Los desaciertos son resultados predictivos bajo el contrato y no prueba automática de un bug. La verificación no certifica el linaje comercial ausente ni que el problema sea completamente predecible.

## Gráficos y reproducción

Se exportan seis PNG: BA global, accuracy/precision/recall/F1 globales, confusión perceptrón, confusión logística, BA mensual y comparación BA train/validación/test de los cuatro métodos. Ejes/proporciones identificados, sin intervalos o precisión inferencial no justificada. PCA no interviene.

```powershell
python -m src.experiments.evaluacion_final
python -m unittest discover -s tests -p 'test_evaluacion_final.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

El comando solo verifica/carga artefactos congelados, predice, calcula métricas y exporta. Las repeticiones son técnicas del mismo protocolo, no búsquedas. El registro de primera apertura se conserva y los eventos de cada intento se agregan; sus timestamps no se confunden con reproducibilidad de resultados científicos. Cualquier intento fallido y su motivo quedan registrados, incluso si ocurrió antes de abrir y_test.

## Límites y cierre

Solo cinco meses de prueba, observaciones correlacionadas por producto/mes e historia compartida; no se realizó bootstrap iid ni se atribuye significación a diferencias pequeñas. La distribución de clases puede variar entre meses/cohortes; por ello accuracy y F1 aisladas no reemplazan BA/matriz/soportes. Las probabilidades no se recalibraron ni se consideran perfectamente calibradas.

Persisten identidad por nombre sin StockItemID, cobertura académica no certificada, unidad comercial sin validación externa y ventas observadas distintas de demanda no atendida o inventario. La clase indica superar una mediana reciente, no rentabilidad ni utilidad económica garantizada.

Conclusión de esta versión: los resultados anteriores constituyen la comparación final de los cuatro métodos sobre la misma reserva temporal; se documenta su rendimiento observado, variación temporal y límites sin seleccionar nuevas configuraciones a partir de test. Modelos, umbrales, características, particiones, población y etiqueta quedan intactos después de observarlo. Cualquier investigación posterior requiere otra versión/protocolo y nueva reserva independiente, sin reutilizar este test como validación. No se avanzó a fase 8 ni se propone un reentrenamiento automático.
