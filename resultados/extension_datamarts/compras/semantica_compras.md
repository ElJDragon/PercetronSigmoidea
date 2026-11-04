# Compras: semántica y temporalidad

## Alcance

Se leyó primero la auditoría general existente. La nueva consulta es focalizada: Purchasing.PurchaseOrders, PurchaseOrderLines, Suppliers, Warehouse.StockItems únicamente para unidad, Fact.Purchase y sus dimensiones Supplier/Stock Item. No se investigaron Pedidos, Transacciones ni Inventario. No se ejecutó ningún módulo de integración o simulación. Todas las consultas propias son SELECT; readonly/ApplicationIntent no sustituyen permisos de servidor. Se preserva el informe y todos los entregables anteriores.

## Grano y relaciones demostradas

Fact.Purchase es una **línea de orden**, no una orden ni un recibo individual. En desarrollo hay 7307 hechos, 7307 líneas y 1819 órdenes, hasta 34 líneas por orden. Purchase Key es la PK del hecho. El DW no conserva PurchaseOrderLineID; la pareja (WWI Purchase Order ID, StockItemID de negocio) es única en esta copia: hay 0 parejas repetidas y el cotejo constituye una biyección, sin discrepancias de fecha, proveedor, cantidades o estado. No se presupone que dicha pareja sea una restricción única general del sistema.

Se enlaza Fact.Purchase a cada dimensión por su PK sustituta para recuperar únicamente el ID de negocio, luego a PurchaseOrders por ID y PurchaseOrderLines por orden+producto. SupplierID enlaza Purchasing.Suppliers. Esta identidad actual no acredita que todos los atributos actuales fueran conocidos al emitir.

## Fechas, cantidades y estados

Integration.GetPurchaseUpdates asigna Date Key=CAST(po.OrderDate AS date), Ordered Outers=pol.OrderedOuters, Ordered Quantity=pol.OrderedOuters*si.QuantityPerOuter, Received Outers=pol.ReceivedOuters e Is Order Finalized=pol.IsOrderLineFinalized. El nombre DW de finalización se refiere a la **línea**, no a la cabecera. La extracción incremental usa el máximo de las últimas ediciones de cabecera y línea; no crea snapshots de cada valor histórico.

OrderDate es fecha de negocio de la orden, con precisión diaria. La plantilla local de simulación inserta cabecera y líneas con @StartingWhen y CAST(@StartingWhen AS date). Es evidencia del mecanismo posible, no prueba de la hora real de inserción de cada fila. No hay columna de creación auditada. Las 1819 cabeceras se editaron después de su OrderDate. **LastEditedWhen no es fecha de alta**: aquí solo sirve para una comprobación conservadora de disponibilidad de la versión persistida.

LastReceiptDate es última recepción registrada, no necesariamente primera recepción; ExpectedDeliveryDate es fecha esperada actual, no promesa original auditada. La plantilla de recepción actualiza ReceivedOuters, finalización y última recepción después de emitir. En la copia actual de desarrollo hay 7307 líneas finalizadas, 0 parcialmente recibidas y 0 sin recepción. Esto no prueba que nunca existieran recepciones parciales. Hay 0 recepciones anteriores a la fecha de orden. No se construye un target de recepción tardía: hay 0 positivos observados antes del corte frente a la fecha esperada actual; tampoco se rescata cambiando el umbral a uno o dos días.

PurchaseOrders y PurchaseOrderLines son no temporales, sin CDC, change tracking ni triggers observados; no permiten reconstruir modificaciones o altas/bajas pasadas. **OrderedOuters no se certifica como cantidad original.** El código de simulación observado no basta para probar inmutabilidad universal. ReceivedOuters y finalizaciones son estados actuales posteriores. Sus valores actuales no se habilitan como X.

La historia temporal de StockItems anterior a 2016 sí se leyó exclusivamente para QuantityPerOuter: 0 productos con cambios de conversión y ninguna discrepancia de unidad al OrderDate para las líneas examinadas. Cada producto conserva PackageTypeID en esta muestra. Eso permite interpretar unidades dentro del mismo producto, pero no sumar unidades de productos diferentes como una magnitud física homogénea. La unidad correcta de frecuencia es **órdenes distintas**, no líneas ni bultos.

## Identidad y dimensiones

Hay 219 productos de negocio, seis proveedores de compra y 219 pares producto-proveedor: cada producto se observa con un único proveedor en desarrollo. Agrupar por clave sustituta fragmentaría identidades; IDs no se convierten en números continuos. No hay miembros desconocidos en los hechos de compra examinados.

Las dimensiones se leen con Valid From anterior a 2016. Supplier presenta un par de intervalos superpuestos para el ID 3, que no pertenece a los seis proveedores comprados; no se inventa que afecte a todos. Hay 94 hechos cuyo Supplier Key apunta a una versión no vigente al OrderDate a medianoche. La precisión diaria y actualizaciones posteriores impiden usar directamente ese atributo como predictor. Stock Item tiene cero solapamientos y cero hechos fuera de vigencia al OrderDate en este análisis. Suppliers operacional es temporal, pero esa historia no reconstruye las órdenes ni sus cantidades. El contrato deshabilita atributos dimensionales: un futuro join tendría que devolver exactamente una versión válida al instante, sin escoger arbitrariamente la clave mayor.

## Granularidad y cobertura

| Grano | Identidades | Meses-entidad con actividad | Casillas desde primera observación | Ausencias en esas casillas |
|---|---:|---:|---:|---:|
| product | 219 | 521 | 7884 | 7363 |
| supplier | 6 | 76 | 216 | 140 |
| product_supplier | 219 | 521 | 7884 | 7363 |

Los 36 meses de desarrollo tienen órdenes y el cotejo DW/OLTP es completo **para las filas persistidas actuales**. Eso demuestra ausencia observada en el libro leído, no certifica cierres, ausencia de borrados, latencia o actividad real nula. Las casillas vacías se mantienen desconocidas por defecto. El escenario conditional_ledger supone explícitamente un libro mensual completo y evalúa ceros solo después de primera observación; no autoriza esos ceros. El escenario observed_only no fabrica ceros, pero condiciona a actividad futura del mes objetivo y sirve solo como diagnóstico de selección.

209/219 productos y cuatro/seis proveedores aparecen en un único mes, todos observados inicialmente en enero de 2013; siete productos aparecen los 36 meses. No se eliminan entidades usando su vida futura. Los pares no añaden información de cambio de proveedor. Producto-mes conserva unidad física consistente, pero es extremadamente intermitente; proveedor-mes es apropiado para frecuencia de órdenes, con muy pocas series efectivamente activas. La elección no se basa en generar más filas.

## Reproducción

`python -m src.datamarts.compras_viabilidad discover` lee metadatos; `extract` lee exclusivamente órdenes con OrderDate<2016-01-01 y hechos con Date Key<2016-01-01; `analyze` y `python -m src.datamarts.reportes_compras` son offline. Las fechas de fin de vigencia no son outcomes ni fijan el universo futuro. Algunas órdenes de diciembre de 2015 tienen última edición/recepción posterior al corte: se registra su inelegibilidad, sin construir prevalencias de órdenes de 2016. Las huellas COUNT/CHECKSUM antes/después no cambiaron; son controles no criptográficos, sin aislamiento snapshot, y no certifican historia ni permisos. Las consultas y evidencias se conservan en JSON.
