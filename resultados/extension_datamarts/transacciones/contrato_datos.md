# Contrato financiero: conteo de cobros por cuenta

**Estado: viable para un experimento retrospectivo académico bajo los supuestos explícitos de este contrato.** No es certificación operativa y no se construye ningún dataset en este encargo. Las variantes proveedor/importe quedan fuera del alcance autorizado. Fuente estructurada: contrato_datos.json y dictamen.json.

## Identidad, objetivo y tiempo

Clave de observación futura: (dominio customer, CustomerTransactions.CustomerID, mes objetivo). ID de evento: (dominio, CustomerTransactionID). La cuenta es la pagadora Bill To operacional, recuperada mediante el ID de transacción cuando DW contiene miembro desconocido. No se agrupa por nombres o claves sustitutas ni se incluyen IDs como magnitudes continuas.

Población: cuentas con un cobro previamente observado y tres meses calendario completos desde su primer mes de cobro, incluyendo meses posteriores sin actividad dentro del alcance cubierto. Primera observación no se presenta como alta real. No se usan últimas apariciones, actividad del target o identidad del catálogo actual para seleccionar filas. Las cuentas nuevas y nunca vistas en cobros quedan fuera de este primer universo, explícitamente.

Instante: inicio del primer día de t en tiempo empresarial local. Horizonte: mes calendario completo [inicio(t),inicio(t+1)). No se inventa hora de inserción: TransactionDate es fecha empresarial diaria. N(c,t)=número de IDs únicos **tipo 3 Customer Payment Received**, sin filtrar por saldo o estado actual. R=mediana de los conteos de t-1,t-2,t-3 consecutivos; y=1[N(c,t)>R]; empate=0. Solo N(t) participa como outcome, nunca como X. La etiqueta significa frecuencia registrada superior a la referencia, no probabilidad de impago, importe cobrado ni unidades vendidas.

Train: objetivos 2013-04..2014-12, outcome conocido antes de 2015-01-01 00:00. Validación: objetivos 2015-01..2015-12, outcome antes de 2016-01-01 00:00. Cortes exclusivos. Test de 2016 reservado, sin consulta ni prevalencia en este diseño. Historia de validación puede usar meses anteriores de 2015 si eran conocidos antes de t; el preprocesador y modelos seguirán congelados tras train.

## Cobertura y disponibilidad

El modo de cobertura deberá registrarse expresamente como business_ledger_academic; no activar ceros por defecto. La ausencia se convierte en cero solo para un mes cubierto del libro y después de primera observación. Cobertura desconocida, extracto fallido, identidad/tipo/fecha faltante o duplicado => error o indeterminado; nunca imputación a clase 0. Los tres lags esenciales no se imputan para fabricar historia. En un mes parcialmente extraído no se cuentan solo las filas disponibles como si fueran el total.

Se exige identidad de evento estable y fecha empresarial confiable dentro de los siguientes supuestos, apoyados por el cotejo y las fechas actuales:

1. TransactionDate representa el día de registro empresarial del pago/cobro; no se afirma una hora auditada de inserción.
2. El libro de eventos de desarrollo conserva los pagos registrados sin borrados o reclasificaciones retrospectivas relevantes no observados. Los módulos locales y las versiones actuales son evidencia de apoyo, no bitácora exhaustiva.
3. Las extracciones conciliadas OLTP–DW y presencia de pagos en los 36 meses justifican cobertura académica mensual del libro desde enero de 2013; no certifican conciliación bancaria ni completitud absoluta.
4. Bajo esa cobertura explícita, ausencia de un pago en un mes posterior a primera observación significa cero eventos registrados, no incapacidad de pagar ni ausencia de ventas.
5. La disponibilidad retrospectiva de eventos se aproxima por su día empresarial y el control observable de versión. Para todos los pagos seleccionados, última edición coincide con ese día; no se necesita inferir saldos anteriores.

La versión persistida de los pagos tiene última edición el mismo día empresarial; su máximo por agregado es anterior a inicio(t) para lags y anterior al corte para y. Se verifica como control adicional observable, sin llamarlo fecha de alta ni imponerlo a la historia de estados de facturas. No se exige snapshot mensual para contar estos eventos. Si un evento apareciera con versión posterior o se detectara una corrección, conservar el agregado como desconocido o reconstruirlo con evidencia, sin eliminar solo esa fila ni asumir que nunca cambió.

No reconstruir OutstandingBalance o IsFinalized de meses anteriores a partir del último valor. FinalizationDate no es la fecha empresarial del pago para otras clases; no se usa para seleccionar ni para generar X. No se cruza un pago con una factura suponiendo que NULL InvoiceID equivale a dato perdido.

## Catálogo causal

Permitido_contrato_academico requiere cobertura y supuestos anteriores. Permitido refiere a calendario conocido. Condicionado permanece desactivado; prohibido no entra en el futuro dataset.

| Nombre | Fórmula | Fuente | Grano | Disponibilidad | Estado | Justificación |
|---|---|---|---|---|---|---|
| cobros_lag_1 | N(c,t-1) | CustomerTransactions, tipo 3, DISTINCT CustomerTransactionID | Cuenta cliente-mes | Antes de inicio(t), mes anterior cerrado | permitido_contrato_academico | Eventos de pago con identidad y fecha verificadas; sin saldos. Cobertura académica explícita. |
| cobros_lag_2 | N(c,t-2) | CustomerTransactions, tipo 3, DISTINCT CustomerTransactionID | Cuenta cliente-mes | Antes de inicio(t), mes anterior cerrado | permitido_contrato_academico | Eventos de pago con identidad y fecha verificadas; sin saldos. Cobertura académica explícita. |
| cobros_lag_3 | N(c,t-3) | CustomerTransactions, tipo 3, DISTINCT CustomerTransactionID | Cuenta cliente-mes | Antes de inicio(t), mes anterior cerrado | permitido_contrato_academico | Eventos de pago con identidad y fecha verificadas; sin saldos. Cobertura académica explícita. |
| cobros_media_3 | (n1+n2+n3)/3 | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto. |
| cobros_mediana_3 | mediana(n1,n2,n3) | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto. |
| cobros_std_3 | std(n1,n2,n3), ddof=0 | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto. |
| cobros_cambio_1 | n1-n2 | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto. |
| cobros_tendencia_3 | (n1-n3)/2 | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto. |
| meses_con_cobros_3 | SUM 1[nk>0], k=1..3 | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto. |
| mes_anterior_sin_cobros | 1[n1=0] | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto. |
| variacion_relativa_cobros | (n1-n2)/n2 si n2>0; nulo en otro caso | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | No dividir entre cero; opcional y acompañada por indicador de base cero. |
| base_variacion_cero | 1[n2=0] | Agregados de eventos tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Depende de cobertura que permita el cero. |
| antiguedad_observada_meses | ordinal(t)-ordinal(primer mes de cobro conocido antes de t) | CustomerTransactions tipo 3 | Cuenta cliente-mes | Antes de inicio(t) | permitido_contrato_academico | Primera observación, no fecha real de alta; no usar último evento futuro ni supervivencia. |
| mes_sin | sin(2*pi*(mes(t)-1)/12) | Calendario civil | Cuenta cliente-mes | Conocida antes de inicio(t) | permitido | Calendario conocido; no acredita actividad ni cobertura. |
| mes_cos | cos(2*pi*(mes(t)-1)/12) | Calendario civil | Cuenta cliente-mes | Conocida antes de inicio(t) | permitido | Calendario conocido; no acredita actividad ni cobertura. |
| dias_mes_objetivo | Días calendario de t | Calendario civil | Cuenta cliente-mes | Conocida antes de inicio(t) | permitido | Calendario conocido; no acredita actividad ni cobertura. |
| importe_cobrado_lag_k | -SUM TransactionAmount(c,t-k), solo tipo 3 | CustomerTransactions.TransactionAmount | Cuenta cliente-mes | Antes de inicio(t), eventos íntegros y moneda verificada | condicionado_desactivado | Signo y reconocimiento comprobados, pero el esquema extraído no certifica código de moneda o ausencia de mezcla/conversión; fuera del dataset recomendado. |
| pagos_proveedor_lag_k | COUNT DISTINCT SupplierTransactionID(s,t-k), tipo 7 | SupplierTransactions | Proveedor-mes separado | Antes de inicio(t) | condicionado_otro_objetivo | No mezclar con cuentas cliente; seis proveedores y solo dos series persistentes: no es la población elegida. |
| atributos_dimensionales | Atributo en intervalo único [Valid From,Valid To) válido antes de t | Dimensiones relacionadas | Entidad-mes | As-of anterior a t | condicionado_desactivado | No son necesarios para conteos; miembros desconocidos y versiones pueden romper identidad o multiplicar filas. |
| saldos_y_estados_actuales | OutstandingBalance, IsFinalized, FinalizationDate actuales | OLTP/DW | Evento/entidad | No autorizable en esa forma | prohibido | No son snapshots de meses anteriores. |
| mes_objetivo | N(c,t), importe(c,t), evento de t como X | OLTP/DW | Evento/entidad | No autorizable en esa forma | prohibido | Información posterior al instante; N(t) solo para y. |
| identificadores_continuos | Customer Key, Supplier Key, IDs como cantidades | OLTP/DW | Evento/entidad | No autorizable en esa forma | prohibido | Identificadores se usan para unión/agrupación, no magnitudes. |
| fecha_edicion_como_alta | LastEditedWhen como creación del evento | OLTP/DW | Evento/entidad | No autorizable en esa forma | prohibido | Una edición no certifica la primera inserción. |
| cualquier_tipo_negativo_como_cobro | SUM/COUNT de importes negativos sin tipo y dominio | OLTP/DW | Evento/entidad | No autorizable en esa forma | prohibido | El mismo signo describe cobro de cliente o pago a proveedor; crédito/refund tiene otra semántica. |

No se necesitan atributos dimensionales para el objetivo elegido. Si se añadieran después, un join as-of debe devolver exactamente una versión válida y disponible; miembro desconocido, cero o múltiples matches no se resuelven eligiendo una PK mayor o una versión futura. El evento OLTP es la fuente de identidad financiera en este contrato.

## Alcance de la autorización y cierre

Es viable construir posteriormente el dataset académico **solo del candidato A**, con 4.612 observaciones train y 2.943 de validación bajo las reglas documentadas. Una implementación futura debe registrar los supuestos, comprobar nuevamente cobertura/identidad/cortes y conservar sus huellas; estos conteos no son una matriz definitiva ni garantía si la fuente cambia. El modo de cobertura desconocida mantiene los meses sin eventos como indeterminados y no puede presentarse como población completa.

La autorización no cubre importes, saldos, evaluación de test, uso operativo o entrenamiento en este encargo. No se pide certificación absoluta inexistente para un conteo académico justificable, pero el alcance y el riesgo deben acompañar sus resultados. Se preservan Ventas, Pedidos, Compras y el informe. No se avanza automáticamente a Inventario/Movimientos.
