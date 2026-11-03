# Protocolo cerrado de evaluación final — Fase 7

## Estado y alcance

Este documento es especificación: la fase 6 no implementa ni ejecuta predicciones/métricas de test. Su autoridad es congelacion_experimental.json, enlazado mediante SHA-256 en manifiesto.json de fase 6. Esa congelación fija los modelos originales, datos, transformaciones, reglas y cohortes. Ninguna etiqueta o predicción de test se usó para elegir PCA o el protocolo.

## Secuencia obligatoria antes de abrir test

1. Verificar el hash del archivo de congelación contra el manifiesto de fase 6; verificar los hashes de modelos, matrices X/y, metadatos, características, observaciones, preprocesador y contratos contra la congelación. Si cualquier archivo difiere, detener y documentar la discrepancia; no reentrenar ni regenerar datos para forzar coincidencias.
2. Cargar los JSON originales mediante PerceptronSimple.load y LogisticRegressionManual.load, sin llamar fit. Confirmar que estado, pesos, sesgos, 20 nombres/orden, checkpoints y reglas coinciden con la congelación: perceptrón η=0,01, época 70, semilla 42; logística η=0,1, época 200, semilla 42 y umbral 0,5. No elegir umbral nuevo ni reajustar modelos con train+val.
3. Cargar exactamente X_test.npy, y_test.npy y metadatos_test.csv de fase 3, una sola población común. Verificar float64/int8, finitud, etiquetas 0/1, forma 1.111×20 y 1.111, correspondencia de indice_matriz, claves producto×mes únicas y orden por mes/nombre exacto. El modelo no recibe nombres/IDs como predictores. Los 1.111 casos dependen del supuesto académico de cobertura, no de una certificación externa de cierre.
4. Verificar mismo orden de columnas con caracteristicas.json y estado congelado. No aplicar el escalador otra vez: X_test ya está preparado. No seleccionar/excluir columnas ni observaciones por sus resultados. Si se detecta cobertura realmente incompleta, detener la evaluación y registrar la inconsistencia; no escoger períodos por mejorar métricas.

## Reglas fijas de predicción

Perceptrón: z=X_test w+b, clase 1 si z>=0, clase 0 en otro caso. Scores no son probabilidades. Logística: p=sigmoid(X_test w+b), clase 1 si p>=0,5; clase 0 en otro caso. Usar los mismos índices, sin PCA. La etiqueta no cambia: y=1[Q_objetivo>mediana de los tres meses previos], empate→0. No reconstruirla con umbrales estimados sobre test.

Baseline mayoritario: predecir clase 1, fijada por 2.382 positivos y 2.217 negativos de entrenamiento (empate hubiera sido clase 0). No recalcular mayoría de test ni por mes/cohorte. Baseline histórico: 1[unidades_lag_1>R_referencia], empate→0, usando los predictores históricos crudos de observaciones.csv alineados por observacion_id a las mismas filas de metadatos_test. No comparar directamente columnas estandarizadas con medias/escalas diferentes. No usar Q_objetivo ni y_test para construir baseline.

La información histórica de meses previos de test puede estar ya incorporada causalmente en X_test: son pronósticos mensuales de un paso. Enero de 2016 usa octubre-diciembre de 2015; abril puede usar enero-marzo de 2016. Esto no autoriza fit, búsqueda o actualización de parámetros después de mirar los errores. No es un pronóstico de cinco meses emitido en diciembre.

## Métricas y tablas fijadas

Reutilizar metricas_binarias de fase 4. Principal: balanced accuracy; secundarias: accuracy, precision, recall y F1 de clase positiva, matriz de confusión y soporte/proporción por clase. Matriz con filas reales/columnas predichas, orden 0,1: [[TN,FP],[FN,TP]]. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2TP/(2TP+FP+FN); BA=(recall+TN/(TN+FP))/2.

Con denominador cero: precision=0 y bandera; recall no estimable sin positivos reales; especificidad no estimable sin negativos reales; BA no estimable si falta una clase real; F1=0 si su denominador es cero, con bandera. N=0 no genera métrica. No sustituir BA no estimable por 0,5 ni borrar grupos sin ambas clases.

Reportar para los cuatro métodos: global (todos los casos), cada uno de los cinco meses ene-may de 2016 y cohortes conocidas/nuevas. Publicar N, soportes, matriz y métricas de cada grupo, incluidos resultados desfavorables. N esperados por mes: 219/219/219/227/227 bajo cobertura académica. El promedio mensual se calcula solo sobre métricas estimables, declarando cuántos meses entran; no sustituye las métricas globales calculadas a partir de conteos sumados.

Productos nuevos: nombre exacto no incluido en la lista train_products congelada; esa lista se obtuvo exclusivamente de train. Esperados ocho productos y 16 filas elegibles (abril-mayo); conocidos 1.095 filas. No significa fecha de alta comercial: solo ausencia del nombre en entrenamiento. No incorporarlos retrospectivamente al vocabulario ni modificar la elegibilidad para evaluar más meses. Reportar por cohorte y, si resulta estimable, cohorte×mes con los mismos métodos; reconocer soportes pequeños.

Comparar descriptivamente los resultados finales con train/validación ya guardados en fases 4/5 y con baselines congelados. Indicar que validación se utilizó para selección y tiene optimismo, mientras test es la evaluación reservada de configuraciones previamente fijadas. No presentar diferencias pequeñas como significativas ni usar test para elegir y volver a evaluar otro checkpoint. No inventar pérdida probabilística del perceptrón/baselines; la BCE logística es un diagnóstico adicional opcional con logits congelados, no criterio para retocar el modelo.

## Cierre y límites

Exportar predicciones de los cuatro métodos con IDs y orden común, probabilidades solo de logística y scores identificados del perceptrón. Guardar métricas JSON global/mes/cohorte, matrices, configuración/hashes y las incidencias de cobertura. Mantener las respuestas separadas de predictores. No sobrescribir los modelos congelados.

Tras observar test queda prohibido cambiar pesos, sesgos, umbrales, características, escaladores/imputadores, particiones, población, definición de etiqueta o baselines dentro de esta evaluación final. Cualquier investigación nueva requerirá otra versión y una nueva reserva de evaluación; no reutilizar este test como validación encubierta.

Solo hay cinco meses y productos/ventanas correlacionados: no asumir 1.111 réplicas independientes ni bootstrap iid de líneas para afirmar significación. Cualquier incertidumbre debe reconocer dependencia temporal y pocos bloques; la cohorte nueva tiene solo 16 observaciones. No se define un coste monetario que no haya sido aprobado. Las probabilidades logísticas no se consideran calibradas por el solo uso de sigmoidea.

PCA está fuera de este proceso: Z_train/Z_val de fase 6 son solo gráficos, no entradas de los clasificadores ni sustitutos de X_test. No usar separación/solapamiento visual para reajustar modelos después de test. Este protocolo queda preparado para la fase 7, que requiere autorización independiente.
