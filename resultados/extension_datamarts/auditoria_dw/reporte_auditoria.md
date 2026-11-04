# Auditoría local de procesos WWI

La conexión integrada funcionó en la instancia AMBATO (SQL Server 16.0.1210.3), con ODBC Driver 17. BI solo contiene OLTP. La fuente principal fue WideWorldImportersDW; WideWorldImporters se consultó para resolver fechas, unidades y estados. No se ejecutaron procedimientos de integración: solo se leyó su definición. La ampliación académica a los otros procesos es obligatoria; esta tarea entrega auditoría, no implementación predictiva.

## Cobertura y grano

| Tabla | Filas | Fecha principal real |
|---|---:|---|
| Fact.Order | 231412 | 2013-01-01 a 2016-05-31; 41 meses |
| Fact.Purchase | 8367 | 2013-01-01 a 2016-05-31; 41 meses |
| Fact.Movement | 236667 | 2013-01-01 a 2016-05-31; 41 meses |
| Fact.Stock Holding | 227 | Sin fecha/historia demostrada |
| Fact.Transaction | 99585 | 2013-01-01 a 2016-05-31; 41 meses |

Los cuatro hechos de evento abarcan enero de 2013 a mayo de 2016, con 41 meses observados sin huecos internos mensuales; Stock Holding carece de fecha. Filas por año/mes y fechas secundarias completas están en cobertura_temporal.json. No se certifica completitud ni latencia a partir de presencia de filas. Dimension.Date tiene 1461 días; su calendario no prueba actividad del negocio. Valid To=9999 es un extremo de vigencia, no miles de meses de actividad.

Pedidos contiene 231412 líneas y 73595 órdenes (hasta 5 líneas por documento); Compras, 8367 líneas y 2074 órdenes (hasta 34). Movimientos tiene 236667 identificadores de origen distintos. Transacciones separa 97147 IDs de cliente y 2438 de proveedor; no comparar sus valores como un único espacio de IDs. La factura identificada se observa una vez en este hecho, mientras un pago puede agrupar obligaciones. El DW de pedidos no retiene OrderLineID; la pareja OrderID+StockItemID resulta única en OLTP y enlaza las 231412 filas sin discrepancias de cantidad/fechas. La conversión de unidades de compra enlaza las 8367 filas, sin discrepancias actuales, sin certificar inmutabilidad histórica.

## Calidad y convenciones

No se detectaron duplicados de PK ni repeticiones del contenido de los cinco hechos al excluir PK/linaje. Las FK auditadas tienen 0 huérfanos no nulos. Esto no equivale a identidad completa: comprobar claves cero en comprobaciones_adicionales.json, usadas como miembro desconocido. Customer Key/Supplier Key pueden estar pobladas con ese miembro en vez de NULL. Se exportan nulos, cardinalidad, extremos, ceros, negativos y cuantiles/IQR por variable; los atípicos son señales descriptivas, no errores ni eliminaciones.

Hay 85469 líneas de Pedidos con Customer Key=0, 92699 movimientos con Customer Key=0 y 52968 transacciones con Customer/Bill To Customer Key=0. Tener cero huérfanos FK no permite considerarlas identidades comerciales resueltas. El cotejo de los pedidos afectados contra OLTP se conserva en order_unknown_customer_source; no se deben eliminar esas filas ni usar nombre como sustituto silencioso. Recuperar el ID operacional y validar su disponibilidad o comenzar con historia por producto, dejando explícita la limitación. Hay 70510 transacciones con Payment Method Key=0; debe comprobarse su significado por tipo y no interpretarse automáticamente como dato erróneo de pago.

Hay 3147 fechas de picking nulas y 9 líneas de compra sin recepción/no finalizadas. Compras no tiene nulos DW en campos relevantes: cero recibido y estado false también representan información pendiente. En transacciones, IDs cliente/proveedor y campos de documento son mutuamente dependientes del tipo; su ausencia es frecuentemente estructural.

Stock Issue presenta 228265 cantidades negativas; Stock Receipt, 8358 positivas; Stock Adjustment at Stocktake tiene 44 movimientos (21 negativos, 19 positivos y 4 cero). Los signos son convenciones del proceso. No se reconstruye stock histórico: faltan saldo inicial reconciliado, continuidad y snapshots fechados. Stock Holding enlaza 227 productos y sus versiones dimensionales recientes, pero esa fecha de dimensión no fecha el saldo.

## Vigencia dimensional y fuga

| Dimensión | IDs de negocio (incluye miembro desconocido) | Máx. versiones | Pares de intervalos superpuestos |
|---|---:|---:|---:|
| Stock Item | 228 | 5 | 0 |
| Customer | 403 | 1 | 0 |
| Supplier | 14 | 3 | 1 |
| Transaction Type | 14 | 2 | 0 |
| Payment Method | 5 | 2 | 0 |
| City | 37941 | 6 | 5948 |
| Employee | 20 | 18 | 1 |

Se encontraron intervalos superpuestos en City, Supplier y Employee, sin intervalos invertidos. Los ejemplos acotados muestran versiones coincidentes que pueden multiplicar joins por ID+vigencia. No resolver el conflicto eligiendo arbitrariamente la clave mayor. En Pedidos hay 377 filas City, 2492 Picker y 2632 Salesperson fuera del intervalo de su fecha de evento a medianoche; Compras tiene 94 Supplier. No son necesariamente errores definitivos: las fechas DW pierden hora y algunas claves pueden resolverse al actualizar/cumplir. Requieren cotejo con timestamps y regla ETL; no reutilizar el atributo futuro como predictor del registro.

Picked Date Key procede de la finalización de la **línea**. En Compras, Date Key es OrderDate e Is Order Finalized se toma de **IsOrderLineFinalized**, no del encabezado. Ordered Quantity multiplica por QuantityPerOuter de la fuente actual; historial de unidad debe confirmarse. Outstanding Balance e Is Finalized se copian de estados actuales; FinalizationDate está disponible en OLTP, no en Fact.Transaction. Los hechos OLTP inspeccionados no están declarados como tablas temporales de sistema; sus columnas LastEditedWhen no reconstruyen todos los valores previos. El historial de una dimensión no es un historial de estados de cada hecho.

## Viabilidad y orden recomendado

1. **Pedidos primero para formalizar el contrato.** La línea con outcome de picking observado ofrece 227745 resultados maduros posibles: 3154 después de dos días y 224591 dentro; 3129 líneas maduras sin picking requieren política de censura. Son cotas antes de exigir historia y purgar fronteras, no un dataset construido. No convertir nulos automáticamente en atraso. Excluirlos puede sesgar hacia líneas completadas y debe justificarse. Mantener líneas del pedido/familia juntas al evaluar. Definir días calendario frente a plazo operativo y disponibilidad inicial de cantidad/precio. Backorder tiene una fecha de pedido igual a la del padre en esta copia: no demuestra cuándo se generó el vínculo ni permite fijar su horizonte.
2. **Compras segundo.** El objetivo de recepción posterior a fecha esperada tiene cero positivos entre 8235 líneas cerradas observadas y se descarta. El plazo de última recepción sí varía entre 1, 2 y 3 días: es una alternativa exploratoria, no un compromiso de usar >1 día ni de confundir última recepción con primer recibo. Hay 8358 recepciones observadas antes de filtros causales; 574 producto-meses observados. Confirmar recepciones parciales, unidad, estado y soporte por proveedor.
3. **Movimientos tercero.** Balance neto del próximo mes es candidato diferente de la etiqueta de Ventas: 352 grupos positivos y 8667 negativos entre 9019 producto-meses observados. Requiere justificar utilidad, cobertura y elegibilidad; no mide riesgo de stockout ni utiliza Stock Holding actual como historia.
4. **Transacciones requiere redefinición.** Las 70426 facturas cliente finalizadas lo hacen exactamente un día después; 84 siguen pendientes. En 70051 facturas maduras a 7 días y 68562 a 30 días no hay positivos para impago a esos horizontes. No entrenar una etiqueta monoclase ni clasificar pagos contra facturas usando su signo como supuesto problema empresarial. Solicitar otra evidencia/objetivo antes de diseñar el pipeline.
5. **Stock Holding está bloqueado para serie temporal** con una sola instantánea. Solicitar snapshots fechados y movimientos reconciliados si se quiere predecir existencias.

propuestas_predictivas.json conserva para cada candidato problema, unidad, horizonte, etiqueta posible, entradas históricas, elegibilidad, particiones, baseline, riesgos y conteos. matriz_viabilidad.json/csv compara los ocho criterios solicitados sin puntajes arbitrarios. Ninguno es aprobación de una etiqueta definitiva ni de entrenamiento.

## Decisiones antes del próximo dataset

Acordar objetivo y acción comercial; elegir unidad línea/documento y agrupación de familias; resolver censura y nulos; fijar instante de predicción, horizonte y embargo; certificar cierres/disponibilidad; definir joins dimensionales únicos as-of, unidades e historial necesario. Propuesta temporal inicial: 2013–2014 desarrollo, 2015 validación y enero–mayo de 2016 reserva de este **nuevo** proceso, con purga del horizonte y sin elegir por resultados. Los casos potenciales después de historia/purga aún no se cuantifican porque el contrato no está aprobado.

## Arquitectura reutilizable y verificaciones

conexion.py separa configuración y frontera SELECT; auditoria_dw.py separa inventario, perfiles y comprobaciones OLTP; reportes_dw.py exporta evidencia y decisiones. Para cada proceso futuro separar extracción/agregación, etiqueta, características y particiones. Reutilizar interfaces de métricas, preprocesador train-only, PerceptronSimple, LogisticRegressionManual y PCA2D sin refactorizarlos ahora. Las reglas de Ventas (etiqueta, constantes, BOUNDS, conteos) son específicas y no deben importarse como contrato universal. No se construyeron datasets, entrenaron modelos o recalculó PCA.

Consultas parametrizadas y nombres escapados: solo SELECT y lectura de catálogos; se bloquean lotes, comentarios y verbos de modificación. readonly/ApplicationIntent son indicaciones del driver, no permisos reducidos concedidos en SQL Server. No se crean usuarios/tablas ni se cambian permisos. No hay aislamiento de snapshot ni prueba absoluta contra cambios concurrentes: las huellas COUNT/CHECKSUM de las 13 tablas DW y ocho OLTP comparan sus intervalos documentados y son no criptográficas; no sustituyen una bitácora del servidor. Los agregados deben interpretarse con esa limitación de consistencia.

integridad_antes.json preserva el estado inicial de los archivos, incluidos cambios previos del usuario; se contrasta al finalizar. validaciones.json/manifiesto.json registran pruebas, hashes, consultas y control de lectura. No se tocó el informe, datos ni modelos de Ventas. No se ejecutan commits, push, merge ni cambios de rama.
