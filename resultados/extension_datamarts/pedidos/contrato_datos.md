# Contrato de datos: pedidos-1.0-especificacion

La versión estructurada está en contrato_datos.json. No hay X/y ni transformaciones ajustadas.

## Identidad y linaje

Unidad: OrderLineID estable de OLTP. Order Key es clave DW; WWI Order ID es asociación al documento actual; Backorder ID es un vínculo posterior, no predictor. Usar StockItemID y CustomerID operacionales solo como claves verificadas de asociación. No tratarlos como magnitudes continuas. Conservar IDs y estados en metadatos separados de X. Validar unicidad de cada puente DW–OLTP y no multiplicar líneas con joins.

La emisión propuesta es la demanda original por línea con fecha calendario OrderDate heredada en backorders. La hora de emisión no está observada. El modelo base solo admite información conocida por calendario o cerrada **antes del inicio de ese día**, evitando usar transacciones del mismo día ordenadas ficticiamente. El cambio a emisión real del documento hijo requiere recuperar su creación; no se sustituye silenciosamente el objetivo.

## Etiqueta, cortes y censura

Dos días calendario: DATE(p)>d+2 es 1, empate al día 2 es 0. Fin de seguimiento homogéneo: inicio de d+3. Deben cumplirse fin de horizonte <= corte, p >= inicio de d, p < corte y ambas últimas ediciones < corte. `known_at` es el máximo de esos timestamps. Las fechas inconsistentes quedan en estado invalid sin etiqueta. Nulos, picks futuros o versiones aún no observables quedan sin y; nunca se imputan.

No se acredita ausencia del evento a partir de NULL actual. Una etiqueta de retraso por ausencia al vencimiento queda deshabilitada hasta disponer de bitácora o snapshot fiable del horizonte. La disponibilidad de ingestión y la inmutabilidad de timestamps no están certificadas: este es un modo retrospectivo conservador explícito, no garantía operacional.

## Inventario de entradas

| Familia | Política | Razón |
|---|---|---|
| Calendario de d: mes/día de semana con codificación cíclica, día de mes y días del mes | Permitida en diseño base | Seis entradas definidas en contrato_datos.json; conocidas al emitir, sin usar test |
| Cantidad solicitada, precio y total de línea originales | Condicional, desactivada en base | Se asignan al crear, pero el valor actual no acredita por sí solo el inicial |
| Número de líneas, variedad y total del pedido original | Desactivada | Los traslados a backorder alteran la composición actual |
| Historial de actividad de producto/cliente | Condicional | Claves estables y hechos conocidos antes del día objetivo, sin reconstrucción futura |
| Historial de retrasos | Condicional | Solo etiquetas conocidas antes del corte de features; no basta OrderDate anterior |
| Atributos de dimensión Cliente/Ciudad/Vendedor/Producto | Desactivados hasta validar as-of | Solapamientos, desconocidos, versiones fuera de fecha y falta de hora |
| IDs | Solo asociación, grupos y metadatos | No variables continuas; codificación categórica requeriría contrato y vocabulario train-only |
| Picking objetivo, PickedQuantity objetivo, Picker, Backorder final, resultados posteriores | Prohibidos | Fuga directa del outcome |
| Últimas ediciones | Solo puertas de disponibilidad, no X | Pueden codificar el propio cumplimiento |

La ventana histórica de retrasos debe filtrar por `known_at < inicio(d)` y no por fecha del pedido. Para una línea de enero cuyo picking se registró en marzo, no hay etiqueta utilizable en historia de febrero. Se aplica también la disponibilidad de su versión persistida. No usar el vencimiento del día 3 como fecha de conocimiento del positivo en ausencia de registro fiable. Ventanas y soportes mínimos se fijarán antes de extraer, sin test.

El entrenamiento no debe heredar 20 variables, BOUNDS ni etiqueta de Ventas. La dimensión final será dinámica tras codificar y descartar constantes solo en train. Imputación, vocabularios, escalado y PCA se ajustarán únicamente en train cuando se autorice el pipeline; aquí no se calculan.

## Cliente desconocido

La auditoría histórica identificó 85469 líneas DW con cliente cero y 261 IDs OLTP sin correspondencia en Dimension.Customer. En desarrollo son 73772 líneas, 0 nulos y 255 IDs fuente; las 73772 líneas afectadas carecen de su ID en la dimensión. Son centinelas por ausencia de correspondencia, no 85469 identidades nuevas ni simples NULL.

No eliminar esas observaciones por cliente desconocido. En diseño base no se usan predictores de cliente; si se habilita una categoría posteriormente, mantener `desconocido` explícito y ajustar vocabulario en train. El ID CustomerID recuperable de Orders es una asociación real, no identidad inventada, pero su valor al registro y uso como predictor requieren prueba de disponibilidad. No usar la marca de desconocido de un ETL posterior como si fuera información conocida al emitir.

## Joins históricos

Reutilizar la evidencia de auditoría: City tiene 5948 pares superpuestos, Supplier uno y Employee uno. En hechos de Pedidos, 377 claves City, 2492 Picker y 2632 Salesperson no coinciden con el intervalo a medianoche. Son cifras históricas previas, no nuevos outcomes test. La vigencia a la hora exacta puede diferir de medianoche; no declararlos automáticamente errores ni usarlos como predictores seguros.

Un join permitido deberá usar ID operacional, instante as-of y [Valid From,Valid To), con exactamente una coincidencia. Cero: desconocido; más de una: ambigüedad, desactivar característica y registrar motivo. Prohibido resolver con TOP 1, clave máxima o última versión. Ni una FK válida ni un intervalo sin solapamientos acreditan que un atributo se conocía al emitir. Base sin atributos dimensionales hasta resolverlo.

## Gates para un ETL posterior

Reconciliar estados y unicidad; conservar las filas sin etiqueta con motivo; probar traslado y raíz de familias sin ciclos; confirmar la reconstrucción de fecha original; acordar censura/población; fijar entradas y ventanas efectivamente conocidas; congelar particiones y presupuesto de evaluación. Solo después crear extracción versionada de Pedidos en rutas nuevas. No modificar Ventas, auditoría anterior ni informe.
