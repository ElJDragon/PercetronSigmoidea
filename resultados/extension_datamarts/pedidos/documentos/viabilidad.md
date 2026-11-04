# Pedidos por documento: dictamen de viabilidad

## Dictamen

**No corresponde autorizar la construcción del dataset con el objetivo solicitado en esta copia.** Una observación por OrderID es estructuralmente posible y OrderDate es una fecha de negocio razonable para un documento original; no es una marca auditada de inserción ni contiene hora. Pero la población en que el campo puede interpretarse como cumplimiento completo sin transferencias no tiene eventos positivos ni en train ni en validación.

Este dictamen es específico del objetivo «picking del pedido completo después de dos días». No afirma que Pedidos carezca de cualquier problema predictivo posible. No se vuelve a abrir indefinidamente la investigación ni se cambia la etiqueta para obtener equilibrio. Se recomienda continuar con **Fact.Movement**, bajo un contrato propio y otra tarea; no se consulta ni implementa ese proceso aquí.

## Evidencia local focalizada

Se consultó Sales.Orders y su relación consigo misma por BackorderOrderID, junto con sus metadatos. No se agregaron líneas para obtener etiquetas de documento. Hay 63978 documentos de desarrollo y 63978 OrderID distintos, desde 2013-01-01 hasta 2015-12-31. El estado actual muestra 61550 picking no nulos y 2428 nulos; hay 0 picking anteriores a OrderDate. Son estados de la copia actual, no estados reconstruidos al corte. La elegibilidad temporal se calcula aparte.

No se consultaron outcomes ni prevalencias de documentos fechados en 2016. Tampoco se repitió la auditoría del DW. Los datos globales previos no se usan como nuevos conteos de documento.

## Cohortes y resultados

| Cohorte | Partición | Documentos de origen | Etiquetables al corte | Positivos | Negativos | Prevalencia |
|---|---|---:|---:|---:|---:|---:|
| Todas las cabeceras (solo diagnóstico) | train | 40649 | 38911 | 1732 | 37179 | 4.4512 % |
| Todas las cabeceras (solo diagnóstico) | validation | 23329 | 21791 | 698 | 21093 | 3.2032 % |
| Original sin backorder | train | 32687 | 32539 | 0 | 32539 | 0.0000 % |
| Original sin backorder | validation | 18445 | 18318 | 0 | 18318 | 0.0000 % |
| Original con pendientes trasladados | train | 3916 | 3903 | 4 | 3899 | 0.1025 % |
| Original con pendientes trasladados | validation | 2396 | 2381 | 0 | 2381 | 0.0000 % |
| Derivado terminal | train | 3916 | 2347 | 1645 | 702 | 70.0895 % |
| Derivado terminal | validation | 2396 | 1009 | 654 | 355 | 64.8167 % |
| Derivado que vuelve a trasladar | train | 130 | 122 | 83 | 39 | 68.0328 % |
| Derivado que vuelve a trasladar | validation | 92 | 83 | 44 | 39 | 53.0120 % |
| Linaje ambiguo | train | 0 | 0 | 0 | 0 | No estimable |
| Linaje ambiguo | validation | 0 | 0 | 0 | 0 | No estimable |

El diagnóstico de todas las cabeceras produce 38911 etiquetas train (1732 positivas) y 21791 de validación (698 positivas). **No son tamaños autorizados de un dataset de pedido completo.** Mezclan eventos con distinta semántica y anclas no válidas para emisión del documento derivado.

En originales sin backorder hay 32539 etiquetas train y 18318 de validación, todas clase 0. Si se ampliara a todos los originales incluyendo los que trasladan pendientes, solo aparecerían cuatro positivos en train y ninguno en validación; además se debilitaría el significado de cumplimiento completo. No se justifica entrenar sobre un problema sin clase retrasada evaluable ni rescatar positivos de documentos con fecha heredada.

## Por qué los backorders no resuelven el bloqueo

Existen 6534 vínculos de desarrollo, 0 hijos ausentes, 0 diferencias de OrderDate padre/hijo y 0 vínculos con clave no creciente. Los documentos se clasifican por links entrantes y salientes: original sin salida, original con salida, derivado terminal y derivado que vuelve a trasladar. El agrupamiento previo de padres impide multiplicar filas. Los IDs crecientes apoyan ausencia de ciclos en los vínculos examinados, no otorgan una fecha de emisión.

La plantilla local de simulación copia OrderDate del padre al crear el hijo, traslada líneas no suministradas y marca al padre como picking completado para facturar lo restante. Esa lectura de código es evidencia de la semántica posible, no una ejecución realizada ni una prueba de la historia completa de cada fila. La igualdad de fechas observada es compatible con ella.

Por tanto, no se puede tratar al hijo como emitido realmente en su OrderDate heredado; no se sustituye ese campo por la fecha del padre ni por última edición. Tampoco se interpreta el cierre del padre tras separar pendientes como prueba de satisfacción de toda su solicitud inicial. Las líneas no acreditadas al registro permanecen fuera de los predictores. Estudiar familias completas requeriría otro universo y otra definición, no el documento propuesto silenciosamente renombrado.

La cohorte sin backorders se usa solo para evaluar si existe una población retrospectiva defendible. Su pertenencia depende de links finales que podrían conocerse después de emitir: es una selección retrospectiva y no un filtro de despliegue o predictor. El descarte de las otras cohortes obedece a ancla y significado del evento, definido antes de consultar su prevalencia; no es una selección de meses o clases convenientes.

## Evidencia, supuestos y límites

Demostrado: unicidad actual de OrderID, fechas y tipos de metadatos, estados agregados de desarrollo, cortes aplicados y relaciones de backorder. El módulo Website asigna OrderDate al registrar documentos y la plantilla de simulación genera fecha de negocio; no se deduce que conozcamos cuándo se insertó físicamente cada fila histórica.

Supuestos académicos explícitos para la alternativa: interpretar OrderDate del original como emisión de negocio a precisión diaria; considerar fiable el timestamp observado de picking; utilizar la versión persistida solo si su última edición es anterior al corte. No son certificaciones de ingestión, ausencia de backdating, composición original ni snapshot completo. La configuración de estos supuestos no compensa la falta de positivos de la cohorte admisible.

Los nulos no se etiquetan como retrasados: pueden ser pendientes o ausencia de evento verificable. No hay columna dedicada de cancelación en el esquema Orders consultado y no se deduce cancelación de comentarios, nulos o falta de factura. Se preserva el motivo indeterminado. Los extremos observados no certifican completitud del período.

Calendario del OrderDate propio puede conocerse al emitir. Atributos actuales de cabecera no se habilitan por existir: cliente/vendedor pueden servir como claves de asociación tras verificar temporalidad, pero no como números continuos; las demás entradas siguen desactivadas. Quedan prohibidos picking, picker, estado final, links finales, últimas ediciones como X y cualquier información futura o de líneas sin existencia probada. No se aplican joins dimensionales.

## Decisión final y continuidad

Mantener bloqueados tanto el contrato por línea anterior como esta alternativa para el objetivo evaluado. Preservar sus entregables y no construir X/y, preprocesador, modelos o PCA. No se modifica el informe ni se realizan operaciones Git.

Recomendación: cerrar esta investigación de Pedidos y pasar a Fact.Movement, ya auditado, porque registra eventos de movimiento y permite una formulación distinta sobre flujos sin exigir un supuesto de creación de líneas. No equivale a autorizar reconstrucción de stock ni predicción de stockout: faltan saldo inicial e historia reconciliada para eso. No se inicia una nueva auditoría de Movimientos ni se cambia automáticamente el trabajo a Compras.
