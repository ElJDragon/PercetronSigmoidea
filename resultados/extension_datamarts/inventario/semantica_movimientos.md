# Semántica de movimientos de Inventario

Investigación local focalizada de WideWorldImportersDW y WideWorldImporters en AMBATO, mediante pyodbc y autenticación integrada. Se reutiliza `../auditoria_dw/`: el total histórico conocido de 236667 movimientos y la cobertura enero de 2013–mayo de 2016 proceden de esa auditoría; no se volvieron a perfilar outcomes de 2016. `Fact.Stock Holding` tiene 227 filas de estado sin fecha histórica y no se consultó ni se usa como historial.

## Tipos efectivamente observados en desarrollo

206143 eventos únicos entre 2013-01-01 12:00:00 y 2015-12-31 12:00:00; 219 artículos. El catálogo contiene más tipos, pero solo estos tres aparecen en el intervalo investigado:

| Tipo | Eventos | Quantity < 0 | > 0 | = 0 | Con factura | Con orden compra |
|---|---:|---:|---:|---:|---:|---:|
| 10 Stock Issue | 198807 | 198807 | 0 | 0 | 198807 | 0 |
| 11 Stock Receipt | 7298 | 0 | 7298 | 0 | 0 | 7298 |
| 12 Stock Adjustment at Stocktake | 38 | 18 | 16 | 4 | 0 | 0 |

Tipo 10: todas las salidas tienen CustomerID y InvoiceID, ninguna SupplierID ni PurchaseOrderID. Tipo 11: todas las recepciones tienen SupplierID y PurchaseOrderID, ninguna factura ni cliente. Tipo 12: ajustes de recuento físico sin esos documentos, con ambos signos y cuatro eventos de cantidad cero. Estos cuatro siguen siendo eventos de ajuste para el candidato de frecuencia. No se encontraron transferencias tipo 9 ni tipos explícitos de devolución en los movimientos de desarrollo; los tipos financieros de abonos/reembolsos no se reinterpretan como movimientos físicos. Esto no certifica ausencia de devoluciones en la operación real ni en test.

`Quantity` es un cambio de unidades físicas, no importe ni saldo. Los ajustes negativos prueban que negativo no equivale automáticamente a venta; los positivos incluyen recepciones y ajustes, por lo que tampoco equivalen todos a compras. Las unidades se comparan dentro del mismo artículo y nunca se suman entre artículos para definir la etiqueta.

## Evidencia de origen y unidades

`modulos_movimientos.json` conserva las definiciones leídas, sin ejecutarlas. `Website.InvoiceCustomerOrders` registra `0 - InvoiceLines.Quantity` como Stock Issue. La definición de configuración de simulación contiene la creación de las rutinas de recepción y recuento: Stock Receipt se registra como `ReceivedOuters * QuantityPerOuter`, con fecha del evento, proveedor y orden; Stock Adjustment at Stocktake registra el delta físico. El código de configuración es evidencia de generación, no un procedimiento de recepción ejecutado en este encargo ni prueba de todas las operaciones posibles. `Integration.GetMovementUpdates` exporta el ID del evento, fecha, tipo, documentos y `CAST(Quantity AS int)`, con selección incremental por LastEditedWhen. No ejecutamos la integración.

La cantidad OLTP es decimal. Los 206143 valores de desarrollo son enteros exactos, por lo que no se observó truncamiento en DW. Las 219 versiones operativas históricas (una por artículo de desarrollo) mantienen UnitPackageID y QuantityPerOuter dentro de cada artículo; no faltan metadatos. Las recepciones ya están en unidades base: no volver a multiplicarlas por el empaque actual. Esta comparabilidad respalda B en desarrollo; si posteriormente cambia la unidad o aparecen fracciones, se debe detener y resolver, sin inferir valores desde estado actual. No se usan ReceivedOuters ni cantidades ordenadas actuales como predictores históricos.

## Identidad y reconciliación

Identidad estable: StockItemID operativo, equivalente a WWI Stock Item ID. Clave de evento: StockItemTransactionID. La clave sustituta DW se conserva solo como trazabilidad. En las versiones DW anteriores a 2016 hay 219 identidades reales con una versión por identidad y la fila desconocida 0; ningún movimiento analizado utiliza el desconocido. El diseño admite futuras versiones reuniéndolas por la identidad operativa, sin usar etiquetas o atributos posteriores.

Se enlazaron los 206143 eventos DW por ID con OLTP: cero ausentes en ambas direcciones, cero duplicados y cero diferencias de artículo, tipo, cantidad, fecha, factura u orden. Las huellas de lectura antes/después de ambas fuentes coinciden. No es una garantía criptográfica de toda la BD ni una auditoría de borrados, sino una comprobación focalizada de estabilidad durante las consultas.

## Relación con Ventas y flujo neto

En desarrollo, 198807 salidas coinciden con 198807 líneas de Fact.Sale en 198807 grupos fecha–factura–artículo. No hay grupos ausentes, diferencias de cantidad absoluta ni diferencias de número de líneas: cobertura 100% en ambos sentidos. Un objetivo de volumen mensual elevado de salidas sería la misma medida de unidades vendidas para esta población, y se descarta como nuevo experimento.

El flujo neto tiene significado de suma firmada de cambios, pero no de existencias, disponibilidad, rotura de stock ni demanda. En 6594 de 7227 ventanas objetivo+3 meses no hay recepciones ni ajustes; en todas esas ventanas la etiqueta de neto alto equivale exactamente a unidades vendidas bajas respecto de su mediana. D reintroduce principalmente Ventas con sentido invertido. Su mejor equilibrio de clases no lo vuelve una aplicación empresarial independiente.

## Cobertura, ausencia y disponibilidad

Los 36 meses de desarrollo están representados, todos los 219 artículos aparecen en enero de 2013 y tienen movimientos en los 36 meses: 7884 artículo-meses y cero meses sin cualquier movimiento. Solo 521 artículo-meses tienen recepciones y 7363 no las tienen. Después de enero de 2013 solo diez artículos vuelven a recibir; en 2015 hay 108 artículo-meses con recepción, concentrados en nueve artículos. Esta concentración es una limitación sustantiva y no se oculta filtrando por actividad futura.

Los 206143 registros tienen LastEditedWhen igual a TransactionOccurredWhen. Esto respalda disponibilidad por evento en el extracto conservado; LastEditedWhen no es por sí solo un registro de creación ni demuestra inmutabilidad. StockItemTransactions es NON_TEMPORAL_TABLE: no permite descartar retrospectivamente borrados o correcciones nunca conservadas. StockItems sí tiene historia temporal, consultada solo para identidad/unidades anteriores a 2016.

Se admite el supuesto académico explícito de archivo de eventos completo y conservado por artículo a partir de su primera aparición y de cierre del mes anterior. Bajo ese supuesto, un mes cubierto sin tipo 11 vale cero recepciones; un mes no cubierto/desconocido no se convierte en cero. Tener todos los meses y reconciliación exacta respalda el supuesto, pero no certifica cierres operativos. No se requieren snapshots de existencias porque se predicen eventos de recepción, no niveles de stock. Test no se abrió para elegir población ni etiqueta.

Fuentes detalladas: `tipos_y_signos.json`, `calidad_cobertura.json`, `cobertura_articulos.json`, `solapamiento_ventas.json`, archivos de consultas y extracciones acotadas. Cada consulta de eventos usa una frontera estricta anterior a 2016-01-01.
