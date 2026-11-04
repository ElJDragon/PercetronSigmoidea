# Pedidos: especificación predictiva

## Dictamen y alcance

**Viable con ajustes explícitos**, para estudiar retraso de picking de la **línea de demanda original**, con outcome observado y controles temporales conservadores. No se afirma utilidad predictiva, disponibilidad certificada al segundo exacto de emisión ni viabilidad del cumplimiento completo del documento original con su composición inicial. No hay matrices ni entrenamiento. La especificación se apoya en desarrollo 2013–2015 y conserva 2016 reservado.

La unidad recomendada es `Sales.OrderLines.OrderLineID`, no `Order Key` (clave técnica DW), `WWI Order ID` (documento actual) ni una factura. `Fact.Order` omite OrderLineID; la auditoría ya verificó correspondencia uno a uno de la pareja OrderID+StockItemID actual en esta copia. Esa unicidad se debe revalidar al extraer y no sustituye un identificador de línea estable.

## Instante y significado del evento

Se propone emitir la predicción cuando se registra la demanda original de la línea. `Order Date Key` deriva de `Orders.OrderDate`, de tipo date; `Picked Date Key` deriva de `OrderLines.PickingCompletedWhen`, reduciendo un timestamp a fecha. La hora exacta de creación no está disponible como columna dedicada; `LastEditedWhen` cambia y no es fecha de creación. El contrato base usa la fecha calendario de la demanda y solo historia cerrada antes de las 00:00 de ese día, sin ordenar artificialmente eventos del mismo día.

La plantilla local de simulación, almacenada dentro de `DataLoadSimulation.Configuration_ApplyDataLoadSimulationProcedures`, define el traslado de líneas pendientes a un nuevo OrderID y copia `OrderDate` del padre. **No se ejecutó esa plantilla**. Los 6534 vínculos de desarrollo tienen fechas padre/hijo iguales y IDs estrictamente crecientes, patrón compatible con el código. Esto apoya usar la fecha heredada como ancla original de la línea; no la convierte en fecha de emisión real del documento hijo ni acredita toda la historia de modificaciones.

La fuente usa date/datetime2 sin zona horaria incorporada. Se interpreta su calendario registrado, sin asignar UTC o la zona del equipo como zona del negocio. La certificación de cortes requiere aclarar esa convención.

`Website.InsertCustomerOrders` muestra que cantidad, precio, producto, cliente y vendedor se asignan al insertar. Eso demuestra su existencia conceptual al registro, pero no que el último valor almacenado sea siempre el inicial de las filas históricas. Cantidad/precio se dejan condicionados; conteos y totales del documento actual no representan automáticamente el pedido original después de traslados.

## Etiqueta exacta y disponibilidad

Sea d la fecha original y p el timestamp fiable de picking completo de la línea. Para outcomes observados: **y=1 si DATE(p)>d+2 días calendario; y=0 si DATE(p)<=d+2**. Picking el mismo día o el día 1 o 2 es clase 0; el día 3 es clase 1. El horizonte exclusivo termina a las 00:00 de d+3. No se trata de 48 horas exactas ni días laborables. Dos días es el candidato solicitado, sin evidencia de un SLA empresarial; no se cambió para equilibrar las clases.

Se rechaza p anterior al inicio de d como inconsistencia. Picking ausente queda **indeterminado**, sin y, aunque nominalmente haya transcurrido el horizonte. La copia actual no es un log completo que permita acreditar que no existió picking al vencer el plazo. Un futuro registro de ausencia fiable al vencimiento podría permitir y=1 desde ese instante, pero eso requiere otro contrato y evidencia; no se aplica ahora.

En esta versión la etiqueta solo se usa desde `known_at=max(p, última edición de línea, última edición de encabezado)`. Para cada partición se exige known_at anterior al corte y horizonte completo. Así, un picking de 2015 no etiqueta un ejemplo de train cerrado al iniciar 2015; tampoco se usa en train la mera inferencia retrospectiva de que habría estado retrasado antes. Esta política es conservadora y puede excluir estados que antes tuvieron información suficiente. Última edición anterior al corte respalda la versión persistida, pero no certifica latencia ETL, ausencia de backdating ni un snapshot histórico íntegro.

## Línea frente a documento

En desarrollo hay 201510 líneas de 63978 documentos actuales, con 1–5 líneas. No se observan documentos actuales con fechas de picking distintas entre sus líneas ni mezcla de nulos/observados; hay 2428 con todas sus líneas sin picking. No se interpreta esa homogeneidad como independencia: el traslado de pendientes cambia la composición. En 6534 documentos, el picking de encabezado difiere del último de sus líneas actuales; también hay 6534 documentos derivados. La igualdad numérica no se toma como una prueba causal individual.

Si el negocio exigiera cumplimiento integral del pedido **original**, se necesitaría la lista original de líneas y la familia completa. Con ese universo verificado, cumplimiento es max(p_i) solo si todas las líneas están completadas; basta una línea conocida posterior al día 2 para probar retraso, y todas conocidas dentro para probar no retraso. Si no hay un testigo tardío y faltan outcomes, el documento es indeterminado. No se usa promedio, mayoría de líneas ni timestamp final del encabezado como regla arbitraria. Las cifras de documento publicadas aquí son diagnóstico de la composición actual, no aprobación de esa etiqueta empresarial.

Las líneas de un documento o familia son dependientes. Agrupar por raíz de backorders en evaluación; nunca repartir líneas hermanas aleatoriamente. Los vínculos de desarrollo no cruzan fechas, pero esa condición se verificará de nuevo con los IDs al construir las particiones sin leer sus outcomes reservados.
