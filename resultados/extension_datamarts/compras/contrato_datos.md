# Compras: contrato propuesto, bloqueado

**Estado: no autorizado para construir X/y ni preprocesadores.** La especificación define condiciones y no transforma las sensibilidades en un dataset definitivo. Fuente formal: contrato_datos.json; decisión: dictamen.json.

## Objetivo preferido si se resuelve la evidencia

Una observación por (SupplierID de negocio, mes objetivo t), para anticipar al inicio de t si el número de órdenes del mes superará la mediana de los tres meses anteriores. N(s,t)=COUNT DISTINCT PurchaseOrderID con SupplierID=s y OrderDate dentro de [inicio(t),inicio(t+1)). R=mediana(N(s,t-1),N(s,t-2),N(s,t-3)); y=1[N(s,t)>R]. Empate=0. No se suman cantidades heterogéneas del proveedor. La alternativa física mantiene producto-mes y unidades base consistentes, pero no está aprobada.

La identidad debe ser observable antes de t y disponer de tres meses completos anteriores. Primera observación no es fecha de alta. No se usa el catálogo actual para fabricar identidades históricas ni se seleccionan solo proveedores que seguirán comprando. No se usan nombres o claves sustitutas como números continuos. La fecha empresarial es OrderDate propio de la orden, sin atribuir hora ni inserción auditada.

Train: objetivos 2013-04..2014-12 con outcome conocido antes de 2015-01-01 00:00; validación: 2015-01..2015-12 conocido antes de 2016-01-01 00:00. Ambos cortes son exclusivos. 2016 permanece reservado. Histórico antes de inicio(t), objetivo solo al cierre y después de verificar disponibilidad; nunca filtrar partes desconocidas de un agregado para obtener una suma aparentemente completa.

Mes sin filas = desconocido por defecto. La conciliación DW/OLTP demuestra ausencia en la copia persistida, pero no cierre ni ausencia de altas retroactivas/borrados. Cero solo con cobertura explícita y evidencia de cierre aplicable a la entidad y mes. No imputar lags esenciales ni convertir NULL en cero. La sensibilidad conditional_ledger está desactivada en el contrato de producción. Observed_only condiciona a actividad de t y no autoriza una población predictiva.

El control conservador de versión persistida usa última edición anterior a la predicción para historia, y anterior al corte para etiqueta. No equivale a fecha de creación, disponibilidad ETL o historia completa. Se exige también acreditar registro, composición e integridad del conjunto. Si se obtiene bitácora o snapshots confiables, puede reemplazarse la regla de rechazo mediante una reconstrucción demostrada; no mediante asumir que la cantidad nunca cambió.

## Inventario causal

Permitido significa candidato conocido en calendario, no autorización para construir matrices. Condicionado permanece desactivado hasta acreditar fuente, unidad, cobertura y disponibilidad.

| Nombre | Fórmula | Fuente | Grano | Disponibilidad | Riesgo | Estado |
|---|---|---|---|---|---|---|
| cantidad_lag_1 | Q(e,t-1) | PurchaseOrderLines.OrderedOuters × StockItems.QuantityPerOuter histórico | producto-mes | Antes del inicio de t; todas las versiones de t-1 acreditadas | Cantidad mutable; altas tardías; unidad y cobertura | condicionado_desactivado |
| ordenes_lag_1 | COUNT DISTINCT PurchaseOrderID(e,t-1) | PurchaseOrders.OrderDate, SupplierID; líneas solo para producto | proveedor-mes; producto-mes condicionado | Antes del inicio de t; identidad y mes de negocio acreditados | No confundir orden con línea; reasignación o alta retroactiva | condicionado_desactivado |
| cantidad_lag_2 | Q(e,t-2) | PurchaseOrderLines.OrderedOuters × StockItems.QuantityPerOuter histórico | producto-mes | Antes del inicio de t; todas las versiones de t-2 acreditadas | Cantidad mutable; altas tardías; unidad y cobertura | condicionado_desactivado |
| ordenes_lag_2 | COUNT DISTINCT PurchaseOrderID(e,t-2) | PurchaseOrders.OrderDate, SupplierID; líneas solo para producto | proveedor-mes; producto-mes condicionado | Antes del inicio de t; identidad y mes de negocio acreditados | No confundir orden con línea; reasignación o alta retroactiva | condicionado_desactivado |
| cantidad_lag_3 | Q(e,t-3) | PurchaseOrderLines.OrderedOuters × StockItems.QuantityPerOuter histórico | producto-mes | Antes del inicio de t; todas las versiones de t-3 acreditadas | Cantidad mutable; altas tardías; unidad y cobertura | condicionado_desactivado |
| ordenes_lag_3 | COUNT DISTINCT PurchaseOrderID(e,t-3) | PurchaseOrders.OrderDate, SupplierID; líneas solo para producto | proveedor-mes; producto-mes condicionado | Antes del inicio de t; identidad y mes de negocio acreditados | No confundir orden con línea; reasignación o alta retroactiva | condicionado_desactivado |
| cantidad_media_3 | (q1+q2+q3)/3 | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| cantidad_mediana_3 | mediana(q1,q2,q3) | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| cantidad_std_3 | std(q1,q2,q3), ddof=0 | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| cantidad_tendencia_3 | (q1-q3)/2 | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| cantidad_cambio_1 | q1-q2 | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| cantidad_variacion_relativa | (q1-q2)/q2 si q2>0; indeterminada en otro caso | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| frecuencia_media_3 | (n1+n2+n3)/3 | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| meses_activos_3 | SUM 1[nk>0], k=1..3 | Agregados previos acreditados | Entidad-mes | Antes del inicio de t | Hereda los bloqueos de lags y ceros | condicionado_desactivado |
| antiguedad_observada_meses | ordinal(t)-ordinal(primer mes observable antes de t) | Registro histórico de identidad | Entidad-mes | Antes de t, sin usar última compra futura | Primera observación no equivale a alta real; supervivencia | condicionado_desactivado |
| mes_sin | sin(2*pi*(mes(t)-1)/12) | Calendario civil | Entidad-mes | Conocida antes de t | Bajo; no certifica cobertura | permitido_candidato_sin_dataset |
| mes_cos | cos(2*pi*(mes(t)-1)/12) | Calendario civil | Entidad-mes | Conocida antes de t | Bajo; no certifica cobertura | permitido_candidato_sin_dataset |
| dias_mes | días calendario de t | Calendario civil | Entidad-mes | Conocida antes de t | Bajo; no certifica cobertura | permitido_candidato_sin_dataset |
| recepcion_y_finalizacion | ReceivedOuters, LastReceiptDate, IsOrderLineFinalized, IsOrderFinalized | OLTP/DW | Línea/documento/entidad | No acreditada en el momento requerido | Fuga, semántica o identidad | prohibido |
| cantidades_mes_objetivo | Q(e,t), N(e,t), referencia calculada con t | OLTP/DW | Línea/documento/entidad | No acreditada en el momento requerido | Fuga, semántica o identidad | prohibido |
| claves_numericas | Supplier Key, Stock Item Key, IDs como magnitudes | OLTP/DW | Línea/documento/entidad | No acreditada en el momento requerido | Fuga, semántica o identidad | prohibido |
| dimension_futura | Atributos fuera de un intervalo único válido al instante | OLTP/DW | Línea/documento/entidad | No acreditada en el momento requerido | Fuga, semántica o identidad | prohibido |
| ultima_edicion_como_alta | LastEditedWhen como fecha de creación | OLTP/DW | Línea/documento/entidad | No acreditada en el momento requerido | Fuga, semántica o identidad | prohibido |

Los atributos de proveedor/producto siguen desactivados salvo identidad para agrupación. Para habilitarlos, un join as-of debe devolver una sola versión válida antes de la predicción. Cero o múltiples matches implican indeterminación; nunca escoger la mayor PK, una versión posterior o un nombre actual. El hecho puede referenciar una dimensión resuelta al editar/recibir y no al emitir.

## Condiciones pendientes y autorización

Se requiere evidencia histórica del registro y de los valores de órdenes al inicio de mes, cierres verificables para los ceros, y soporte positivo/negativo útil sin depender de cuatro proveedores inactivos o de actividad futura. Actualmente: **0 observaciones autorizadas en train y 0 en validación**. Los conteos de análisis son escenarios diagnósticos, no una entrega de dataset.

No se propone adaptar silenciosamente el instante a después de recibir, desplazar ventanas para rescatar clases ni cambiar umbrales por balance. Se preservan Ventas y todos los contratos previos. Se recomienda pasar a Transacciones financieras en un encargo nuevo y dejar Inventario/Movimientos al final.
