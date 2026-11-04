# Revisión causal de las 17 características

Estado: no_temporal_leak_detected. Se contrastaron las 7227 observaciones de desarrollo con los agregados artículo-mes y estos con los 7298 IDs de recepción del snapshot aprobado, sin reconstruir ni exportar un nuevo dataset. Cada entrada se evaluó únicamente como comprobación y se comparó con su valor crudo persistido. El preprocesador congelado reproduce las matrices procesadas y su ajuste es solo train.

| Característica | Fórmula | Fuente original | Disponibilidad contractual | Error máximo absoluto |
|---|---|---|---|---:|
| `recepciones_lag_1` | n1 = N(i,t-1) | COUNT DISTINCT StockItemTransactionID, tipo 11 | Cierre de t-1 | 0 |
| `recepciones_lag_2` | n2 = N(i,t-2) | COUNT DISTINCT StockItemTransactionID, tipo 11 | Cierre de t-2 | 0 |
| `recepciones_lag_3` | n3 = N(i,t-3) | COUNT DISTINCT StockItemTransactionID, tipo 11 | Cierre de t-3 | 0 |
| `unidades_recibidas_lag_1` | q1 = Q(i,t-1) | SUM Quantity de eventos tipo 11; unidades base del mismo artículo | Cierre de t-1 | 0 |
| `unidades_recibidas_lag_2` | q2 = Q(i,t-2) | SUM Quantity de eventos tipo 11; unidades base del mismo artículo | Cierre de t-2 | 0 |
| `unidades_recibidas_lag_3` | q3 = Q(i,t-3) | SUM Quantity de eventos tipo 11; unidades base del mismo artículo | Cierre de t-3 | 0 |
| `recepciones_media_3` | (n1+n2+n3)/3 | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `recepciones_mediana_3` | mediana(n1,n2,n3) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `recepciones_std_3` | sqrt(sum((nk-media)^2)/3) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `recepciones_cambio_1` | n1-n2 | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `recepciones_tendencia_3` | (n1-n3)/2 | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `meses_con_recepciones_3` | sum(1[nk>0], k=1..3) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `mes_anterior_sin_recepciones` | 1[n1=0] | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `antiguedad_observada_meses` | ordinal(t)-ordinal(primer_mes_evento_conocido_antes_de_t) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `mes_sin` | sin(2*pi*(mes(t)-1)/12) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `mes_cos` | cos(2*pi*(mes(t)-1)/12) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |
| `dias_mes_objetivo` | dias_calendario(t) | Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido | Antes del comienzo de t | 0 |

Las ventanas se forman por clave explícita artículo–mes t-1,t-2,t-3; no hay shift/rolling posicional que cruce artículos o use el mes t. Se verifican cronología, fechas de última edición de historia anteriores al inicio de t, primera aparición conocida antes de t por cualquier movimiento y agregados por identidad operativa. No se usan versiones posteriores, IDs como magnitudes, cantidades objetivo, existencias actuales, supervivencia futura ni últimas apariciones.

Mes cubierto sin recepción vale cero solo con event_ledger_academic. Los tres meses iniciales no fabrican historia anterior a la aparición; unknown/incomplete no son cero. La mediana de etiqueta usa exactamente los tres meses anteriores, y empate es 0. Calendario de t es conocido antes de t. LastEditedWhen coincide con fecha del evento en esta evidencia, pero no demuestra fecha real independiente de creación; el cierre/conservación de eventos es un supuesto académico respaldado por reconciliación, no una certificación productiva.

Las pruebas focalizadas perturban fixtures/copias: objetivo y futuro no alteran X anterior, cambiar etiquetas no cambia PCA y elegibilidad no depende de actividad posterior. No se cambia el snapshot, las matrices ni ninguna etiqueta aprobada. Ninguna evidencia temporal observada obliga a bloquear por fuga; las limitaciones de archivo retrospectivo se mantienen.
