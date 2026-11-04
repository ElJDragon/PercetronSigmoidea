# Conciliación de validación financiera

**Dictamen: evaluación consistente; no hay discrepancia de alineación.** La población principal de clasificadores y mayoría contiene 2.943 observaciones elegibles con tres meses históricos completos y etiqueta válida. La comparación de los cuatro métodos contiene 2.927: el baseline histórico requiere un cuarto mes, según el diseño fijado antes del entrenamiento. La diferencia es exclusivamente disponibilidad del baseline, no falta de etiqueta ni error de los clasificadores.

Las 16 observaciones tienen antigüedad observada de tres meses, estado eligible, etiqueta válida (7 positivas, 9 negativas), predicción de perceptrón/logística/mayoría y ningún cuarto mes anterior a su primera aparición en la población. No se rellena ese mes con cero ni se fabrica historia previa a la primera observación. No se filtró por acierto, error o clase. Reconciliación reconstruida desde metadatos, elegibilidad, etiquetas, mensualidad de cobros y predicciones mediante lecturas/joins uno a uno; sin tocar los archivos originales.

| observation_id | month | first_observed_month | fourth_month | y | perceptron | logistic | majority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| customer:1036:2015-02 | 2015-02 | 2014-11 | 2014-10 | 0 | 0 | 0 | 0 |
| customer:1037:2015-02 | 2015-02 | 2014-11 | 2014-10 | 0 | 0 | 0 | 0 |
| customer:1035:2015-03 | 2015-03 | 2014-12 | 2014-11 | 1 | 0 | 0 | 0 |
| customer:1038:2015-03 | 2015-03 | 2014-12 | 2014-11 | 0 | 0 | 0 | 0 |
| customer:1039:2015-04 | 2015-04 | 2015-01 | 2014-12 | 1 | 1 | 0 | 0 |
| customer:1040:2015-04 | 2015-04 | 2015-01 | 2014-12 | 0 | 0 | 0 | 0 |
| customer:1041:2015-04 | 2015-04 | 2015-01 | 2014-12 | 0 | 0 | 0 | 0 |
| customer:1042:2015-04 | 2015-04 | 2015-01 | 2014-12 | 1 | 0 | 0 | 0 |
| customer:1043:2015-05 | 2015-05 | 2015-02 | 2015-01 | 0 | 0 | 0 | 0 |
| customer:1044:2015-05 | 2015-05 | 2015-02 | 2015-01 | 0 | 1 | 0 | 0 |
| customer:1045:2015-05 | 2015-05 | 2015-02 | 2015-01 | 0 | 0 | 0 | 0 |
| customer:1046:2015-09 | 2015-09 | 2015-06 | 2015-05 | 1 | 0 | 0 | 0 |
| customer:1047:2015-11 | 2015-11 | 2015-08 | 2015-07 | 0 | 0 | 0 | 0 |
| customer:1049:2015-12 | 2015-12 | 2015-09 | 2015-08 | 1 | 0 | 0 | 0 |
| customer:1050:2015-12 | 2015-12 | 2015-09 | 2015-08 | 1 | 1 | 1 | 0 |
| customer:1052:2015-12 | 2015-12 | 2015-09 | 2015-08 | 1 | 0 | 0 | 0 |

La lista completa con estados, cortes, conteo objetivo, referencia y predicciones está en observaciones_excluidas.csv. Distribución por mes: {"2015-02": 2, "2015-03": 2, "2015-04": 4, "2015-05": 3, "2015-09": 1, "2015-11": 1, "2015-12": 3}. historical_defined=false en todas; la regla exacta es edad>=4 más cuatro meses cubiertos y conocidos antes de inicio(t). Aquí edad=3 y t-4 precede primera aparición. No hay otra limitación de disponibilidad para estas filas.

Evaluación principal original: perceptrón BA=66,782699%, logística BA=62,081956%, n=2.943. Comparación secundaria común: perceptrón BA=66,840058%, logística BA=62,114500%, mayoría=50%, histórico=47,788231%, n=2.927. Las pequeñas diferencias se explican por soporte, no por nuevos modelos o corrección de predicciones. En train se conservan igualmente n=4.612 principal y n=4.376 común. No se reemplazan las métricas ni los checkpoints seleccionados con validación completa.

Predicciones, scores y todas las métricas completas/comunes/mensuales se reprodujeron con los modelos cargados, sin fit. El umbral logístico es 0,5; checkpoints 3 y 399; ambos reciben exactamente las mismas 16 características estandarizadas y ordenadas. Los hashes coinciden con los aprobados. No se modifica el manifiesto anterior para adaptar este dictamen.
