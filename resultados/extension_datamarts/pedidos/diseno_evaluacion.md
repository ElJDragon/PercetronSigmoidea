# Evaluación propuesta para retrasos raros

## Particiones por demanda original, sin aleatoriedad

| Partición | Fecha del pedido original | Corte exclusivo de conocimiento |
|---|---|---|
| Train | 2013-01-01 a 2014-12-31 | 2015-01-01 00:00 |
| Validación | 2015-01-01 a 2015-12-31 | 2016-01-01 00:00 |
| Test reservado | 2016-01-01 a 2016-05-31 | Propuesta: 2016-06-01 00:00, cobertura por certificar |

Los meses coinciden parcialmente con Ventas por la cobertura real, pero la unidad, etiqueta, elegibilidad y purga son propias de Pedidos. No se usan sus matrices ni su test como entrenamiento. El último día observado en la extracción no certifica un cierre administrativo: el corte test es una propuesta y no autoriza abrirlo ahora.

Exigir d+3 días <= corte. En train se excluyen uniformemente demandas de 30–31 de diciembre de 2014 y en validación las de 30–31 de diciembre de 2015, aunque alguna ya tenga picking rápido. Un pedido del 29 vence al comenzar el corte y puede ser maduro; su picking/versión debe seguir siendo estrictamente anterior al corte. Además se excluyen del aprendizaje picks o versiones conocidos posteriormente. El embargo de dos días por sí solo no resuelve un outcome registrado meses después.

En cada fold/ventana se recalcula esta disponibilidad; no reutilizar todas las etiquetas finales de train antes del corte de un fold interno. Modelo y transformaciones se congelan al terminar train; selección solo con validación. La historia secuencial en validación/test podrá avanzar únicamente con eventos que ya se hayan conocido, sin refit del preprocesador/modelo.

Mantener juntas las líneas de cada pedido y familia de backorders; si una familia atraviesa un corte, aplicar política de exclusión/purga de la familia, sin usar sus outcomes para elegirla. Los 6534 vínculos de desarrollo tienen la misma fecha, por lo que no se detecta ese cruce aquí. La prueba estructural se repetirá antes de materializar splits. Nada de partición aleatoria por línea ni inferencia iid.

## Métricas

Registrar soportes, matrices [TN FP; FN TP], precision/recall/F1 de clase retrasada, especificidad, BA, FPR, proporción de alertas y accuracy secundaria. FPR=FP/(TN+FP) mide negativos que reciben alerta; la fracción de falsas alertas entre alertas emitidas es FP/(TP+FP)=1-precision. No confundir denominadores.

BA es importante para comparar ambas clases, pero un valor agregado no demuestra utilidad. Reportar por mes y familia, además de cobertura/estados excluidos. No considerar exitoso un modelo por accuracy cercana a 99 % con recall de retrasos casi nulo. Train tiene 1734 positivos y validación 698; la dependencia y heterogeneidad reducen la información efectiva.

Cuando existan puntuaciones, calcular average precision (AP) mediante incrementos de recall ponderados por precision, agrupando empates. Probabilidad logística y margen del perceptrón son puntuaciones válidas para ranking; el margen no es probabilidad. AP no es automáticamente igual a integrar por trapecios toda la curva PR. No se calculó AP aquí porque no hay predicciones nuevas. Usar AP/métricas de clase positiva como evidencia complementaria al criterio de selección previamente acordado; no elegir una métrica después de mirar test. Tasas sin denominador real se marcan no estimables conforme a las convenciones del proyecto.

## Baselines y decisiones de entrenamiento

Mayoritaria: congelar **clase 0**, determinada únicamente desde las etiquetas justificadas de train (124607 negativas frente a 1734 positivas). Reportar que su recall de retraso será cero cuando haya positivos, sin entrenarla ahora.

Regla histórica candidata: predecir la última etiqueta de línea conocida del mismo producto; desempatar registros simultáneos determinísticamente por OrderLineID y utilizar mayoría train si no hay historia. Exigir known_at anterior al inicio del día de predicción. No está activada: comprobar antes asociación y disponibilidad; no usar una etiqueta posterior solo porque su OrderDate es anterior. Esta regla evita introducir umbrales numéricos arbitrarios para equilibrar clases.

No se fijan ponderaciones, sobremuestreo, datos sintéticos ni selección de umbral. Los algoritmos propios no incorporan sample_weight como opción intercambiable: cualquier cambio exige justificación, versión nueva y verificación de su matemática. El umbral empresarial de **dos días** define y y no debe confundirse con el umbral de clasificación de un modelo. El futuro presupuesto de tasas/épocas/umbrales se fijará con train/validación antes de una única apertura test; no se hereda automáticamente la selección de Ventas.

No estimar significación con líneas iid ni bootstrap de líneas; estudiar bloques temporales y familias con suficiente soporte si luego se autoriza inferencia. No se calculan métricas o prevalencias test para tomar estas decisiones.

## Viabilidad y pendientes

Hay ambas clases en todos los meses de desarrollo y un número suficiente para intentar una primera comparación, sin garantía de utilidad. El dictamen es **viable con ajustes explícitos** para la línea con demanda original fechada, picking observado y disponibilidad conservadora. Aceptar o rechazar esta población seleccionada y sus limitaciones de snapshot es una decisión previa al ETL. La emisión exacta de documentos hijos o el cumplimiento integral original necesitan historial adicional y no se sustituyen por el objetivo de línea.

Confirmar el significado empresarial y acción de una alerta a dos días; aceptar unidad de línea/ancla original; aprobar censura y seguimiento; certificar disponibilidad y linaje; decidir entradas iniciales e historial habilitados; fijar joins as-of y protocolo familiar. No se construyen matrices, entrenan modelos, cambian PCA, editan el informe ni realizan operaciones Git.

## Reproducción local

`python -m src.db.especificar_pedidos --server 'lpc:.\AMBATO'` consulta solo pedidos anteriores a 2016 y sustituye únicamente esta carpeta de especificación. Usa pyodbc/conexión integrada ya existentes. Los módulos operativos guardados son texto leído de sys.sql_modules, nunca ejecutado. consultas_desarrollo.json registra SQL y parámetros; hashes y huellas acotadas constan en manifiesto/validaciones. El módulo no ejecuta el auditor global ni reentrena nada.
