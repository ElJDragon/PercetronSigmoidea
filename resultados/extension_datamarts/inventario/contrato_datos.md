# Contrato propuesto: frecuencia relativa de recepciones de inventario

Estado: diseño recomendado, viable bajo supuestos académicos explícitos. No hay dataset de aprendizaje construido ni matrices X/y. Fuentes: eventos Warehouse.StockItemTransactions tipo 11, reconciliados con Fact.Movement, Dimension.Transaction Type y Dimension.Stock Item; la historia temporal de Warehouse.StockItems respalda identidad/unidad. Fact.Stock Holding queda excluido. No se modifica ningún contrato o resultado de Ventas/Transacciones.

## Unidad, etiqueta e instante

Clave única `(StockItemID, mes_objetivo)`. Fecha de predicción: comienzo de t, después de cerrar t-1. Horizonte: mes natural t. N(i,m) = COUNT DISTINCT StockItemTransactionID tipo 11 de i con TransactionOccurredWhen dentro de m. R(i,t)=mediana(N(i,t-1),N(i,t-2),N(i,t-3)); y(i,t)=1[N(i,t)>R(i,t)]. Empate clase 0. N(i,t) e y(i,t) son solo objetivo, nunca entrada; R es calculable antes de t. No usar PurchaseOrderID como conteo alternativo sin una nueva definición.

## Elegibilidad y ausencia

Identidad operativa positiva y fecha/tipo/evento válidos. Sin duplicados de ID. Tres meses consecutivos anteriores a t, posteriores o iguales al primer mes con cualquier movimiento del artículo conocido antes de t y cubiertos por el archivo conservado. No exigir recepción en t, última actividad futura ni supervivencia. Nunca usar primera aparición calculada con eventos posteriores a t; si un artículo es nuevo, esperar tres meses completos de historial observable, sin rellenar historia anterior a su aparición con cero. Primera aparición de eventos es antigüedad observada, no fecha comercial de creación del artículo.

El supuesto académico es conservación completa de eventos y cierre del mes anterior a partir de primera aparición. La reconciliación exacta y los meses globales presentes lo respaldan sin certificarlo. Mes cubierto sin recepción tipo 11 se cuenta cero; mes de cobertura desconocida invalida la fila y queda en un registro de exclusiones. No imputar una etiqueta, ausencia de identidad ni historia esencial. En desarrollo todos los 219 artículos ya aparecen en enero de 2013; no se infieren fechas de alta desde versiones DW ni se filtra por sus recepciones futuras. No hay meses sin movimientos de cualquier tipo, pero sí meses sin recepciones.

Si una futura extracción muestra un tipo nuevo, signo incoherente, unidad cambiada, cantidad fraccionaria truncada en DW o edición posterior al corte, detenerse y documentar: no convertir el valor en cero ni excluir silenciosamente según etiqueta. LastEditedWhen solo se usa para control de disponibilidad; no como predictor. Para el extracto retrospectivo actual coincide con la fecha del evento; no constituye registro independiente de creación.

## Catálogo causal autorizable: 17 candidatas

Sea nk=N(i,t-k), qk=SUM Quantity de tipo 11 en i,t-k. Son candidatas definidas antes de construir matrices; la dimensión efectiva se determinará solo con train al descartar columnas constantes/nulas según reglas congeladas. No se fuerza dimensión 20 ni se introducen IDs continuos.

| Característica | Fórmula | Origen | Disponibilidad |
|---|---|---|---|
| `recepciones_lag_1` | n1 = N(i,t-1) | COUNT DISTINCT StockItemTransactionID, tipo 11 | Cierre de t-1 |
| `recepciones_lag_2` | n2 = N(i,t-2) | COUNT DISTINCT StockItemTransactionID, tipo 11 | Cierre de t-2 |
| `recepciones_lag_3` | n3 = N(i,t-3) | COUNT DISTINCT StockItemTransactionID, tipo 11 | Cierre de t-3 |
| `unidades_recibidas_lag_1` | q1 = Q(i,t-1) | SUM Quantity de eventos tipo 11; unidades base del mismo artículo | Cierre de t-1 |
| `unidades_recibidas_lag_2` | q2 = Q(i,t-2) | SUM Quantity de eventos tipo 11; unidades base del mismo artículo | Cierre de t-2 |
| `unidades_recibidas_lag_3` | q3 = Q(i,t-3) | SUM Quantity de eventos tipo 11; unidades base del mismo artículo | Cierre de t-3 |
| `recepciones_media_3` | (n1+n2+n3)/3 | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `recepciones_mediana_3` | mediana(n1,n2,n3) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `recepciones_std_3` | sqrt(sum((nk-media)^2)/3) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `recepciones_cambio_1` | n1-n2 | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `recepciones_tendencia_3` | (n1-n3)/2 | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `meses_con_recepciones_3` | sum(1[nk>0], k=1..3) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `mes_anterior_sin_recepciones` | 1[n1=0] | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `antiguedad_observada_meses` | ordinal(t)-ordinal(primer_mes_evento_conocido_antes_de_t) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `mes_sin` | sin(2*pi*(mes(t)-1)/12) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `mes_cos` | cos(2*pi*(mes(t)-1)/12) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |
| `dias_mes_objetivo` | dias_calendario(t) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t |

Las unidades previas son del mismo artículo, ya convertidas a base por el evento; no usar el empaque actual para reinterpretar el pasado. Nombres, IDs de artículo/proveedor/cliente/factura/orden, claves sustitutas, fecha de última actividad futura, estados de compra o finalización actuales, saldo actual, QuantityOnHand, ReorderLevel, TargetStockLevel, mes objetivo y cantidades futuras no son predictores. La cantidad histórica de entrada sí está autorizada; no se autoriza stock retrospectivo inventado.

## Particiones, tipos y preprocesado posterior

Train objetivos 2013–2014, con primer objetivo elegible abril de 2013; val objetivos 2015; test 2016 reservado. Los meses de historia inicial de val pueden proceder de train, y los de test de años previos; eso es historia causal, no mezcla de ajuste. Esperado en desarrollo: 4599 filas train, 2628 val; no se anticipa número de filas/clases de test.

Futuro preprocesado: misma lista y orden para ambos modelos, float64; ajuste único con train. Eliminar columnas constantes o completamente nulas en train y aplicar esa misma máscara a val/test. Historia esencial faltante implica exclusión, no imputación; si se autoriza alguna variable opcional más adelante, mediana train con trazabilidad. Escalado `(x-media_train)/std_train`, ddof=0; ni refit ni selección con val/test. No remuestrear ni ponderar clases en esta versión de contrato. No ejecutar estos pasos ahora.

Futuro paquete verificable: claves/etiquetas separadas de X, cobertura y motivos de exclusión, catálogo ordenado retenido/descartado, parámetros train-only, firmas de archivos y pruebas de causalidad por corte. Construir solo después del dictamen, en un encargo posterior. Congelar toda decisión experimental antes de consultar outcomes de test.
