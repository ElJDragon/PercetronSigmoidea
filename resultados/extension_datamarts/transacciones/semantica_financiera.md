# Transacciones financieras: semántica y evidencia

## Alcance focalizado

Se leyeron las auditorías existentes antes de consultar. Las nuevas consultas solo inspeccionan eventos de Sales.CustomerTransactions y Purchasing.SupplierTransactions, Fact.Transaction, tipos y medios de pago, y dimensiones relacionadas para identidad/versiones. No se consulta Fact.Sale, no se reauditan otros procesos ni se modifica su contrato. Todas las consultas emitidas son SELECT; las definiciones SQL se leen como texto, nunca se ejecutan. readonly/ApplicationIntent son indicaciones del driver, no una concesión de permisos reducidos.

## Qué representa una fila

Cada fila de Fact.Transaction representa una transacción operacional de **uno de dos dominios**, cliente o proveedor; no una línea de venta, un saldo mensual ni una asignación pago-factura. La identidad correcta es (dominio, ID transacción operacional), no solo el número. En desarrollo hay 86475 filas DW: 84338 de cliente y 2137 de proveedor. El cotejo por ID de origen es uno a uno y exhaustivo para las filas actuales, sin discrepancias de fechas, tipos, importes, documento o estado actual y sin multiplicación. No hay IDs numéricos compartidos entre dominios en esta muestra; aun así se mantienen espacios separados.

| Dominio | Tipo observado | Eventos desarrollo | Importes positivos | Negativos | Cero | Entidades |
|---|---|---:|---:|---:|---:|---:|
| customer | 1: Customer Invoice | 61320 | 61320 | 0 | 0 | 257 |
| customer | 3: Customer Payment Received | 23018 | 0 | 23018 | 0 | 257 |
| supplier | 5: Supplier Invoice | 1817 | 1817 | 0 | 0 | 6 |
| supplier | 7: Supplier Payment Issued | 320 | 0 | 320 | 0 | 6 |

El catálogo contiene Customer Invoice (1), Customer Credit Note (2), Customer Payment Received (3), Customer Refund (4), Supplier Invoice (5), Supplier Credit Note (6), Supplier Payment Issued (7), Supplier Refund (8) y Customer Contra (13). Los tipos 9..12 corresponden a movimientos de stock, no eventos de cobro. En las fuentes financieras de desarrollo solo aparecen 1,3,5,7: **no hay notas de crédito, reembolsos, contra ni importes cero observados**. Su ausencia en esta copia no demuestra que sean imposibles. Las notas de crédito ajustan la obligación y los refunds son devoluciones; no se agrupan con pagos ni se les atribuye un signo observado inexistente. Los medios operacionales son Cash, Check, Credit-Card y EFT; los pagos examinados son EFT. Los nombres se verifican en catálogos locales, no se infieren del signo.

## Signos, documentos e importes

Una factura cliente positiva registra una obligación a cobrar; no acredita cobro efectivo. Un cobro cliente tipo 3 negativo reduce la cuenta a cobrar y corresponde a entrada registrada de dinero. Una factura proveedor positiva registra una obligación a pagar; no es ingreso. Un pago proveedor tipo 7 negativo reduce esa obligación y corresponde a salida registrada. Por tanto, **el mismo signo negativo describe flujos de caja opuestos según dominio y tipo**. El análisis mide registros del libro, no una conciliación con extractos bancarios.

Las 61.320 facturas cliente tienen InvoiceID y los 1.817 documentos proveedor tienen PurchaseOrderID. Los 23.018 cobros y 320 pagos tienen documento nulo: es ausencia estructural, no pérdida de identidad. La plantilla de simulación agrupa obligaciones pendientes por cuenta y crea un registro de pago con importe negativo de su suma; no conserva una tabla de aplicación pago-factura. No se atribuye cada cobro a una sola factura ni se inventa una conciliación exacta. En facturas, TransactionAmount=AmountExcludingTax+TaxAmount; en pagos los dos componentes fiscales son cero y el importe de pago sigue siendo negativo: esa igualdad fiscal no se impone a tipos de pago.

TransactionAmount es el importe del evento, no OutstandingBalance. Su signo y reconocimiento se cotejan con OLTP y el código local. Las fuentes consultadas no aportan código monetario ni certifican moneda única/conversiones: no se etiqueta volumen financiero ni se habilitan importes como predictores del contrato recomendado. Nunca se afirma AUD/USD u otra divisa sin evidencia local. Los conteos de eventos no dependen de la moneda.

## Fechas: evento frente a estado

Integration.GetTransactionUpdates asigna Date Key=CAST(TransactionDate AS date) en ambos dominios. TransactionDate de una factura es su día empresarial de registro; para tipo 3/7 es el día del cobro/pago registrado. No se confunde con la fecha de factura subyacente: el pago no lleva InvoiceID ni PurchaseOrderID. FinalizationDate y saldos son estados de la obligación, modificados al liquidar; LastEditedWhen es última modificación, **no nacimiento auditado**. No conocemos la hora exacta de creación de cada evento.

Los 23.338 pagos/cobros tienen última edición el mismo día de TransactionDate y finalización ese mismo día en la copia. Los módulos ProcessCustomerPayments/PaySuppliers insertan un pago con TransactionDate=CAST(@StartingWhen AS date), separadamente de actualizar OutstandingBalance, FinalizationDate y LastEditedWhen de obligaciones. Esa evidencia respalda reconstruir conteos de pagos, sin exigir conocer el saldo anterior de la factura. No equivale a una bitácora completa ni prueba que ningún cliente externo pudo modificar o borrar una transacción.

Ambas tablas de transacción son no temporales y no tienen CDC/change tracking/triggers observados. El código financiero local inspeccionado actualiza estados y no muestra borrados de los eventos; no se auditó el log transaccional ni todas las escrituras externas. Se distingue evidencia textual del mecanismo y prueba de la historia individual. Los snapshots serían necesarios para consultar saldos históricos; **su ausencia no invalida automáticamente el conteo de eventos de pago** con identidad, fecha y cobertura justificadas. Campos de estado quedan prohibidos como X y no deciden la inclusión del evento de pago.

## Identidad y dimensiones

La cuenta de cobro es CustomerTransactions.CustomerID: corresponde a Bill To, no necesariamente al cliente destinatario de una factura. Integration usa COALESCE(Invoice.CustomerID, ct.CustomerID) para Customer, pero ct.CustomerID para Bill To. Se usa el ID operacional de cuenta mediante el evento, sin reagrupar por nombre o clave sustituta. Hay 43.578 transacciones de cliente con Bill To desconocido en DW, incluidos 21.142 cobros; no se eliminan ni se fusionan en una cuenta cero. Todas recuperan su cuenta desde el ID único de transacción OLTP, sin discrepancias en los matches dimensionales conocidos.

Customer tiene 403 IDs incluyendo desconocido y una versión por ID en la ventana dimensional; Supplier tiene hasta tres versiones y un par de intervalos superpuestos para ID 3. Tipo y medio se consultan por claves sustitutas para cotejo y por sus IDs operacionales estables para semántica. No se usan atributos dimensionales como X ni joins de rango arbitrarios. Si se agregaran atributos, exigir exactamente una versión [Valid From,Valid To) válida y conocida al instante de predicción; no elegir clave mayor ni versión futura. La vigencia hasta 9999 no es un período observado.

## Cobertura y diferencia respecto a Ventas

Los cuatro tipos observados están presentes en los 36 meses enero 2013..diciembre 2015, y cada evento de la copia OLTP concilia con DW. No se usó 2016. Las huellas COUNT/CHECKSUM antes/después coinciden; son controles no criptográficos sin aislamiento snapshot, no certificados de completitud histórica.

El objetivo nuevo cuenta cobros asentados por cuenta pagadora, frente al objetivo de Ventas sobre unidades por producto. No se vuelve a etiquetar Fact.Sale: las 61.320 facturas y los 23.018 cobros son eventos distintos con distintos IDs, fechas y grano. Un pago puede cubrir varias obligaciones y no revela unidades vendidas. En WWI la simulación sí vincula los pagos a obligaciones generadas por facturación: no se presume independencia económica ni se presenta como prueba de comportamiento bancario real.

La cobertura es razonable **bajo supuestos académicos explícitos de libro de eventos conservado**, no una certificación absoluta. La conciliación y los 36 meses sustentan ausencia observada en el libro, sin descartar borrados/backdating no registrados. No se impone bloqueo automático por carecer de snapshots; tampoco se convierte una ausencia fuera del alcance o con extracción incompleta en cero. El detalle y riesgos se conservan en cobertura_y_supuestos.json.
