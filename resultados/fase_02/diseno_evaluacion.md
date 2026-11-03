# Diseño de evaluación y visualización — Fase 2

## Protocolo temporal

Usar la etiqueta B y universo elegible de contrato_datos.md. Entrenamiento por mes objetivo abril de 2013–diciembre de 2014 (4.599 filas); validación enero–diciembre de 2015 (2.628); prueba enero–mayo de 2016 (1.111 potenciales, sujetas a cobertura). Se escoge el diseño con train/validación, nunca usando resultados finales de 2016. La auditoría previa global no constituye prueba de rendimiento ni habilita optimización con test.

En cada objetivo t se emite una predicción usando exclusivamente meses cerrados hasta t−1. Puede actualizarse la información observada en ventanas históricas de validación/test, sin reentrenar modelos ni reestimar escalas/vocabularios con esos conjuntos. Este protocolo de un paso mensual no equivale a un horizonte fijo de cinco meses desde un solo origen. Las respuestas solo se comparan al cerrar t.

Particiones por mes objetivo idénticas para ambos clasificadores y baselines. Si falta una respuesta por cobertura, se marca pendiente para todos; no se descartan observaciones según error de un algoritmo. Validación permite selección de hiperparámetros; prueba es una evaluación final del diseño congelado. No se recomienda refit automático train+validación en la primera comparación académica: conservar modelos ajustados con train y transformaciones train para no cambiar el contrato de validación. Cualquier refit posterior requiere protocolo explícito independiente.

## Baselines definidos, todavía no evaluados

### Mayoritario de entrenamiento

\[
c^*=\arg\max_{c\in\{0,1\}}\sum_{(p,t)\in E_{train}}\mathbf1[y_{p,t}=c],
\qquad\widehat y^{maj}_{p,t}=c^*.
\]

Empate en conteo: escoger 0 por regla fija. Con los conteos exploratorios de B, la clase mayoritaria es 1 (2.382 frente a 2.217). Esto fija el baseline, no reporta una métrica de evaluación definitiva. La decisión se congela en entrenamiento; no recalcular mayoría por mes ni por conjunto a evaluar.

### Histórico del último mes

Primero un pronóstico ingenuo de unidades `Q_estimado(p,t)=Q(p,t−1)`. Convertirlo usando exactamente la referencia de la etiqueta B:

\[
\widehat y^{hist}_{p,t}=\mathbf1[Q_{p,t-1}>R_{p,t}],
\qquad R_{p,t}=\operatorname{mediana}(q_1,q_2,q_3).
\]

Todos sus términos son conocidos al corte, con misma elegibilidad y empate→0. No reutilizar `y(p,t−1)` como si tuviera la misma referencia R: su umbral se calculó en otro corte. Tampoco usar como pronóstico R y compararlo consigo mismo, pues eso produciría siempre 0 de forma trivial. La regla histórica permite probar si el modelo aporta más que persistencia reciente. No se calcularon predicciones ni métricas de estos baselines en esta fase.

## Métricas y convenciones

Clase positiva: ventas observadas superiores a referencia reciente (y=1). Sean TP, TN, FP, FN los conteos con matriz:

| Real / Predicción | 0 | 1 |
|---|---:|---:|
| 0 | TN | FP |
| 1 | FN | TP |

\[
Accuracy=\frac{TP+TN}{N},\quad Precision=\frac{TP}{TP+FP},\quad
Recall=\frac{TP}{TP+FN},
\]
\[
F1=\frac{2TP}{2TP+FP+FN},\qquad
BalancedAccuracy=\frac12\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right).
\]

Reportar N y soporte por clase, conteos de matriz y proporciones, en cada conjunto/período/cohorte. Si no se predicen positivos, precision se presenta con convención computacional 0 y bandera de denominador cero. Si no hay positivos reales, recall es no estimable; si falta alguna clase real, balanced accuracy es no estimable, no inventar la tasa de esa clase. F1 con denominador cero se reporta 0 con bandera. Accuracy requiere N>0; conjunto vacío no genera métrica. Estas convenciones deben ser idénticas para todos los métodos.

**Métrica principal propuesta:** balanced accuracy, porque interesa reconocer tanto aumentos como meses ordinarios y el balance cambia por mes/cohorte. **Secundarias:** F1 positiva, precision y recall positiva, accuracy y matriz completa. El balance agregado cercano a 50 % no elimina variación temporal ni la necesidad de las otras métricas. No existe costo empresarial aprobado para FP/FN, por lo que no se inventa una función monetaria de utilidad.

Comparar baseline mayoritario, baseline histórico, perceptrón y logística en una tabla con mismo N/soporte, métricas globales y por mes; reportar también promedio de métricas mensuales estimables sin sustituir los conteos globales. Separar productos conocidos y ocho recientes (solo 16 filas potenciales en test, alta incertidumbre). Si una cohorte solo tiene una clase, mostrarlo y no inventar balanced accuracy. Los resultados de cada mes se reportan aunque sean desfavorables; no elegir períodos después de ver errores.

La correlación entre meses consecutivos y productos expuestos al mismo período invalida tratar todas las filas como réplicas independientes. No afirmar significación por diferencias pequeñas de accuracy. Si se necesitan intervalos en fase posterior, usar remuestreo por bloques temporales y reconocer los pocos meses de test; nunca bootstrap iid de líneas de factura.

## Salidas de los clasificadores y selección

Perceptrón: combinación lineal con dimensión dinámica y escalón; si usa bipolar, +1 representa la misma clase positiva, −1 la negativa. Su salida no es probabilidad calibrada. La regla del caso z=0 se definirá de manera fija (propuesta +1) antes de comparar.

Logística matemática: probabilidad P(y=1|X). Propuesta inicial de decisión: 0,5; si se selecciona otro umbral, usar únicamente train y validación. Procedimiento futuro predefinido: barrer `0,05, 0,10, …, 0,95` en validación con modelo entrenado en train; maximizar balanced accuracy, desempatar por F1 positiva y luego cercanía a 0,5; si persiste empate, escoger menor umbral. Congelar antes de test y usar `probabilidad >= umbral`. Esta regla está diseñada, no ejecutada. No confundir umbral probabilístico con la referencia de unidades R que define la etiqueta.

Hiperparámetros y paradas se seleccionarán en desarrollo con un presupuesto explícito en la fase de modelos; no se adoptan los 200 epochs del ejercicio previo automáticamente. Mismas X e índices, con ambos modelos ajustados con train. Guardar pesos, sesgo, nombres de columnas, semillas, hiperparámetros, transformación y codificación de clase para reproducir salidas. Prohibida la clase LogisticRegression de scikit-learn; biblioteca de gradientes pendiente. No se presupone autorización de PyTorch, TensorFlow o JAX.

## Procesos futuros independientes

**A. Entrenamiento y evaluación:** matriz completa de predictores válidos preparados según contrato; número de columnas dinámico tras selección, imputación/codificación y escalado ajustados con entrenamiento. Perceptrón y logística reciben exactamente X_train, X_val, X_test y las mismas respuestas canónicas/observaciones; solo cambia la representación bipolar cuando corresponda. No limitar las entradas a dos.

**B. Visualización:** proyección de esas características a dos componentes, inicialmente PCA mediante NumPy si se autoriza. Ajustar centrado/componentes con X_train; reutilizarlos sin reajustar para val/test. Reportar varianza explicada y nombres de los componentes. Las respuestas pueden colorear puntos retrospectivamente, sin intervenir en el ajuste de PCA.

X_2D no sustituye X del proceso A. La separación visual no demuestra separabilidad lineal de la matriz completa; una frontera 2D solo puede ser corte/aproximación identificada, nunca representación exacta general del clasificador multidimensional. No implementar PCA, perceptrón ni regresión logística durante esta fase. MNIST continúa como referencia conceptual y no se descarga.

## Verificación de esta fase y paso siguiente

El auxiliar exporta resúmenes y hashes de 16 archivos protegidos (dos CSV, ocho referencias y seis entregables fase 1), comprobados antes/después. No vuelve a ejecutar auditar_datos.py. Las pruebas focalizadas comprueban exclusión de objetivo/futuro de R, huecos no rellenados, frontera entre particiones, exclusión estadística de cantidades de test y elegibilidad de productos recientes. Las pruebas existentes de auditoría siguen aplicándose como controles de conservación. En la ejecución de esta fase pasaron las 16 pruebas: 11 existentes y cinco focalizadas nuevas, sin entrenar clasificadores.

En fase 3, implementar agregación causal, registro de cobertura/elegibilidad y pipeline train-only con el catálogo base permitido, conservando índices y nombres transformados. Verificar primero conteos y pruebas de ausencia de fuga, sin ajustar la etiqueta con prueba. Confirmar cierre de extracción y momento real de disponibilidad; las características condicionadas quedan apagadas hasta justificar su origen histórico. Esperar autorización para comenzar esa implementación. Esta fase termina con los cuatro documentos y el procedimiento exploratorio, sin modelos ni datasets definitivos.
