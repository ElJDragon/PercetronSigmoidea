# Transacciones: etiquetas de desarrollo y sensibilidad

## Propuestas y criterio de elección

A: frecuencia de cobros registrados por cuenta cliente-mes. N(c,t)=COUNT DISTINCT CustomerTransactionID, **solo tipo 3**, por CustomerID operacional y TransactionDate del mes. R(c,t)=mediana(N(c,t-1),N(c,t-2),N(c,t-3)); y=1[N(c,t)>R(c,t)]. Empate=0. Se predice al inicio de t para el mes completo. La acción empresarial potencial es anticipar carga de conciliación/gestión de cobros, no impago, ingreso de ventas o liquidez monetaria elevada. No se selecciona por balance: se elige por semántica de eventos, continuidad y variación dentro de cuentas.

B: misma regla con SupplierTransactionID tipo 7 y SupplierID, población proveedor separada. Interpretable como mayor frecuencia de pagos emitidos, pero solo seis proveedores; cuatro tienen actividad únicamente en enero 2013 y solo dos son persistentes. La existencia de dos clases no ofrece por sí sola generalización suficiente. Se mantiene condicionado y no elegido para el primer dataset.

C: mayor importe mensual de cobros o pagos, usando -SUM TransactionAmount dentro de dominio y tipo correctos. El signo y reconocimiento del evento están comprobados, pero falta acreditar moneda/unidad única. No se calcula su prevalencia ni se habilitan importes en el contrato recomendado. No se suman signos de facturas y pagos ni se llama ingreso a toda cantidad positiva. Estado: condicionado a moneda y objetivo empresarial; no viable con evidencia confirmada todavía.

Cliente-tipo-mes con tipo 3 fijo es exactamente A; incorporar tipos de factura o mezclar clientes/proveedores cambia el problema y no crea nuevas observaciones válidas de cobros. No se proponen clasificadores que aprendan el tipo a partir de su signo. No se cambian mediana, horizonte o umbral para equilibrar clases.

## Grano, historia y ceros

Hay 257 cuentas con cobros y 7.709 cuenta-meses con eventos. Desde primera observación hasta diciembre 2015 hay 8.320 casillas posibles y 611 ausencias; nunca se fabrican meses anteriores a primera aparición. Solo 13 cuentas tienen cobros en los 36 meses: se incluye actividad intermitente y primeras apariciones, sin exigir supervivencia futura. En proveedores hay seis identidades, 76 meses-entidad activos y 140 ausencias entre 216 casillas posteriores a primera observación.

Elegibilidad: una cuenta ya observada en cobros, al menos tres meses calendario de historia posterior a su primera observación y meses cubiertos. El primer mes objetivo es tres meses después del primer mes de pago. Se mantienen cuentas cuando dejan de aparecer; no se selecciona por el último evento futuro. Las nuevas cuentas con menos historia quedan indeterminadas. Una cuenta solo facturada pero nunca vista en cobros no pertenece a esta población inicial de pagadores conocidos; no se afirma cubrir todo el universo de clientes.

business_ledger_academic interpreta ausencia dentro del libro de eventos cubierto como cero **eventos registrados**, bajo supuestos documentados. No significa cero ventas ni impago. unknown no rellena ceros: solo etiqueta casillas con actividad observada en t y tres meses anteriores. Esa segunda distribución es sensibilidad censurada por actividad del mes objetivo, no población alternativa de predicción aprobada. Sin cobertura, una ausencia se mantiene indeterminada.

## Conteos y clases

Train usa meses objetivo abril 2013..diciembre 2014 y etiquetas disponibles antes de 2015-01-01. Validación usa 2015 y etiquetas antes de 2016-01-01. En todas las variantes el objetivo debe estar disponible antes del corte, incluso en la sensibilidad sin filtro de edición de X. Los pagos tienen edición en su propio día: el control de disponibilidad observado no elimina ningún caso de los paneles académicos. Se excluyen estados posteriores y no se abren outcomes de 2016.

| Candidato / escenario | Partición | Elegibles | Positivos | Negativos | Positivos % | Entidades con cambio de clase | Entidades de una clase |
|---|---|---:|---:|---:|---:|---:|---:|
| customer_business_ledger_academic | train | 4612 | 1798 | 2814 | 38.9853% | 233 | 3 |
| customer_business_ledger_academic | validation | 2943 | 1198 | 1745 | 40.7068% | 248 | 4 |
| customer_unknown | train | 3258 | 1253 | 2005 | 38.4592% | 231 | 5 |
| customer_unknown | validation | 2290 | 930 | 1360 | 40.6114% | 240 | 11 |
| customer_event_semantics_without_edit_gate | train | 4612 | 1798 | 2814 | 38.9853% | 233 | 3 |
| customer_event_semantics_without_edit_gate | validation | 2943 | 1198 | 1745 | 40.7068% | 248 | 4 |
| supplier_business_ledger_academic | train | 126 | 16 | 110 | 12.6984% | 2 | 4 |
| supplier_business_ledger_academic | validation | 72 | 8 | 64 | 11.1111% | 2 | 4 |
| supplier_unknown | train | 42 | 16 | 26 | 38.0952% | 2 | 0 |
| supplier_unknown | validation | 24 | 8 | 16 | 33.3333% | 2 | 0 |
| supplier_event_semantics_without_edit_gate | train | 126 | 16 | 110 | 12.6984% | 2 | 4 |
| supplier_event_semantics_without_edit_gate | validation | 72 | 8 | 64 | 11.1111% | 2 | 4 |

## Interpretación y sensibilidad

A tiene 4.612 casos train (1.798 positivos, 2.814 negativos; 38,9853%) y 2.943 de validación (1.198 positivos, 1.745 negativos; 40,7068%). Hay 236/252 cuentas elegibles; 233/248 cambian de clase y 3/4 permanecen en una clase dentro de cada partición. No depende exclusivamente de separar identidades activas e inactivas. El escenario unknown reduce a 3.258/2.290 casos y 1.253/930 positivos, con prevalencias 38,4592%/40,6114%; el soporte cambia, pero el balance no nace exclusivamente de rellenar ceros. No se elige ese escenario por su prevalencia ni se interpreta como prueba absoluta de completitud.

En A académico hay 380/182 targets cero y 86/31 referencias cero. Cuando R=0, cualquier cobro del mes es clase 1 por definición; no se usa un percentil nuevo para evitarlo. En B académico hay 126/72 casos, 16/8 positivos y 110/64 negativos; los ceros de cuatro proveedores explican buena parte del soporte. Sin ausencias solo hay 42/24 casos, con 16/8 positivos, todos de dos proveedores. Esta diferencia es una advertencia sobre población y generalización, no motivo para borrar retrospectivamente proveedores inactivos.

etiquetas_candidatas.json incluye distribución mensual, anual y por entidad, cambios de clase, ceros y exclusiones. distribuciones_mensuales.csv permite revisar meses con ambas clases o soporte reducido. No se promedian silenciosamente prevalencias de meses con tamaños distintos. Los números describen un libro retrospectivo bajo los supuestos declarados y no constituyen resultados predictivos. No se construyeron matrices, estimaron baselines, entrenaron modelos o ajustaron umbrales.

## Dictamen por objetivo

- A: **viable bajo supuestos académicos explícitos**, con evidencia local de apoyo; autorizable para una construcción posterior del dataset de conteos y calendario bajo este contrato, sin autorización de uso operativo.
- B: condicionado por soporte y alcance empresarial; no elegido para entrenamiento de este proyecto ahora.
- C: condicionado a unidad monetaria verificable; sin etiquetas calculadas ni importes habilitados.

No hay certificación absoluta de eventos no borrados, backdating o completitud bancaria. Es una limitación del alcance académico, no razón automática para exigir snapshots de saldos que el objetivo no usa. Si se detectan lagunas o correcciones relevantes, esos meses dejan de estar cubiertos y deben permanecer indeterminados o reconstruirse con evidencia. Se detiene la investigación aquí.
