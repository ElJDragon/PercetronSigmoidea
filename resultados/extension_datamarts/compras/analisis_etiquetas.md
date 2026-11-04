# Compras: análisis de etiquetas candidatas

## Definiciones y momento de predicción

Se predice al inicio del mes t para ese mes completo. Train usa meses objetivo de abril de 2013 a diciembre de 2014 y etiquetas disponibles antes de 2015-01-01; validación usa 2015 con disponibilidad antes de 2016-01-01. Los tres meses previos son consecutivos, no las tres últimas compras. Empate con mediana = clase 0. No se leen órdenes ni prevalencias de 2016; los parámetros de diseño no usan test.

1. Volumen producto-mes: Q=sum(OrderedOuters*QuantityPerOuter de ese producto); y=1[Q(t)>mediana(Q(t-1),Q(t-2),Q(t-3))]. Significa volumen solicitado por encima de su referencia reciente, sin identificar demanda, recepción o rentabilidad. Producto-proveedor-mes usa la misma fórmula: en esta muestra no añade identidad porque cada producto tiene un solo proveedor. Volumen proveedor-mes se descarta por mezclar productos y unidades heterogéneas; tampoco se suman bultos de paquetes distintos.
2. Frecuencia proveedor-mes: N=COUNT DISTINCT PurchaseOrderID por SupplierID y OrderDate; y=1[N(t)>mediana(N(t-1),N(t-2),N(t-3))]. Significa necesidad de gestionar más órdenes que la referencia reciente, no más cantidad. Evita conversiones físicas, aunque requiere historia de identidad y emisión. Es la alternativa empresarial preferida **para un contrato futuro** si se acredita disponibilidad; no es dataset aprobado.
3. Actividad proveedor-mes: y=1[N(t)>0]. Predice si habrá al menos una orden en el próximo mes completo desde su inicio. No modifica umbrales para mejorar balance; se compara porque evita cantidad original. En esta copia separa dos proveedores siempre activos de cuatro inactivos tras enero: ninguna entidad cambia de clase en ninguno de los splits retrospectivos. No demuestra capacidad de anticipar cambios.

No se usa recepción tardía/cumplimiento: estado final no reconstruible y sin positivos reales de retraso frente a la fecha esperada actual. No se optimizan umbrales, percentiles ni meses según el balance.

## Escenarios y distribución de clases

Los nombres terminados en observed_only exigen actividad en t y tres meses previos: son diagnóstico condicionado por el resultado futuro, nunca población de despliegue. Los demás usan conditional_ledger, una sensibilidad que interpreta ausencias como cero después de primera observación bajo un supuesto de libro completo **no certificado**. La cobertura por defecto sigue siendo desconocida.

Retrospective calcula los predictores sobre valores persistidos actuales: no prueba su disponibilidad en el pasado. **Incluso ese escenario excluye outcomes cuya versión sea posterior al corte de su partición.** Availability_gated añade max(última edición cabecera/líneas relevantes)<inicio(t) en cada agregado histórico. Si una fila no pasa, se invalida el agregado completo; no se elimina esa fila para reducir Q o N. LastEditedWhen no se interpreta como nacimiento. Esta comprobación rechaza versiones sin prueba de disponibilidad; no demuestra que sus valores fueran necesariamente distintos o inexistentes antes. La falta de historia impide resolverlo.

| Candidato / cobertura | Escenario | Partición | Casos diagnósticos | Positivos | Negativos | Positivos % | Entidades que cambian clase | Entidades de una clase |
|---|---|---|---:|---:|---:|---:|---:|---:|
| volume_product | retrospective | train | 4590 | 168 | 4422 | 3.6601% | 4 | 215 |
| volume_product | retrospective | validation | 2619 | 89 | 2530 | 3.3982% | 8 | 211 |
| volume_product | availability_gated | train | 4423 | 3 | 4420 | 0.0678% | 2 | 209 |
| volume_product | availability_gated | validation | 2520 | 0 | 2520 | 0.0000% | 0 | 210 |
| volume_product_observed_only | retrospective | train | 162 | 160 | 2 | 98.7654% | 2 | 7 |
| volume_product_observed_only | retrospective | validation | 99 | 89 | 10 | 89.8990% | 8 | 1 |
| volume_product_observed_only | availability_gated | train | 0 | 0 | 0 | No estimable | 0 | 0 |
| volume_product_observed_only | availability_gated | validation | 0 | 0 | 0 | No estimable | 0 | 0 |
| volume_product_supplier | retrospective | train | 4590 | 168 | 4422 | 3.6601% | 4 | 215 |
| volume_product_supplier | retrospective | validation | 2619 | 89 | 2530 | 3.3982% | 8 | 211 |
| volume_product_supplier | availability_gated | train | 4423 | 3 | 4420 | 0.0678% | 2 | 209 |
| volume_product_supplier | availability_gated | validation | 2520 | 0 | 2520 | 0.0000% | 0 | 210 |
| volume_product_supplier_observed_only | retrospective | train | 162 | 160 | 2 | 98.7654% | 2 | 7 |
| volume_product_supplier_observed_only | retrospective | validation | 99 | 89 | 10 | 89.8990% | 8 | 1 |
| volume_product_supplier_observed_only | availability_gated | train | 0 | 0 | 0 | No estimable | 0 | 0 |
| volume_product_supplier_observed_only | availability_gated | validation | 0 | 0 | 0 | No estimable | 0 | 0 |
| order_frequency_supplier | retrospective | train | 124 | 18 | 106 | 14.5161% | 2 | 4 |
| order_frequency_supplier | retrospective | validation | 70 | 7 | 63 | 10.0000% | 2 | 4 |
| order_frequency_supplier | availability_gated | train | 84 | 0 | 84 | 0.0000% | 0 | 4 |
| order_frequency_supplier | availability_gated | validation | 48 | 0 | 48 | 0.0000% | 0 | 4 |
| order_activity_supplier | retrospective | train | 124 | 40 | 84 | 32.2581% | 0 | 6 |
| order_activity_supplier | retrospective | validation | 70 | 22 | 48 | 31.4286% | 0 | 6 |
| order_activity_supplier | availability_gated | train | 84 | 0 | 84 | 0.0000% | 0 | 4 |
| order_activity_supplier | availability_gated | validation | 48 | 0 | 48 | 0.0000% | 0 | 4 |
| order_frequency_supplier_observed_only | retrospective | train | 40 | 18 | 22 | 45.0000% | 2 | 0 |
| order_frequency_supplier_observed_only | retrospective | validation | 22 | 7 | 15 | 31.8182% | 2 | 0 |
| order_frequency_supplier_observed_only | availability_gated | train | 0 | 0 | 0 | No estimable | 0 | 0 |
| order_frequency_supplier_observed_only | availability_gated | validation | 0 | 0 | 0 | No estimable | 0 | 0 |

## Interpretación, censura y soporte

El volumen producto-mes aparenta 4.590/2.619 casos train/validación con 168/89 positivos si se aceptan todos los ceros. Sin ceros solo quedan 162/99 casos observados, con 160/89 positivos. El cambio de prevalencia de ~4% a ~99%/~90% muestra cuánto manda el supuesto de actividad: ambos resultados son descriptivos, no evidencia de un problema operativo resuelto. En el panel con ceros hay 4.420/2.520 targets cero y 4.422/2.520 referencias cero; la referencia cero vuelve la regla casi un detector de actividad. Un aumento desde cero es positivo por la fórmula, no una señal demostrada de valor empresarial.

Tras comprobar disponibilidad, el escenario de volumen con ceros conserva 4.423/2.520 casillas, con 3/0 positivos; la validación queda monoclase. Sin ceros, el volumen conserva 0/0. Para frecuencia, la sensibilidad retrospectiva ofrece 124/70 casos con 18/7 positivos; sin ceros, 40/22 con 18/7 positivos. Tras comprobar disponibilidad, solo quedan 84/48 ceros de cuatro proveedores inactivos, y sin ceros quedan 0/0. Tener dos clases en los datos retrospectivos no compensa la ausencia de historia acreditada y la concentración en dos proveedores.

No se rellenan meses previos al primer registro. Los tres meses completos previos deben existir y estar acreditados antes de t. No se purgan fronteras eliminando solo positivos ni se filtra una entidad por su actividad posterior. La disponibilidad del objetivo exige además cierre del mes y versión anterior al corte del split. Se guardan motivos no excluyentes de inelegibilidad en etiquetas_candidatas.json. La selección por versiones actuales es también un riesgo de censura y supervivencia: el resultado no estima desempeño de la población original.

Las distribuciones mensuales completas están en distribuciones_mensuales.csv y etiquetas_candidatas.json, con soporte, positivos, negativos y proporciones. Los JSON incluyen entidades de una clase, entidades que cambian, ceros y razones de exclusión. Meses sin casos elegibles no tienen prevalencia estimable. No se calculan métricas de modelos, baselines ajustados o prevalencias de 2016.

## Dictamen

**No autorizar dataset ni entrenamiento ahora.** Hay cero observaciones autorizadas, porque los conteos que incorporan ceros son condicionales a cobertura no acreditada, y las series activas sin ese supuesto no tienen historia disponible demostrable al instante propuesto. Esto es insuficiencia de evidencia para el protocolo, no prueba de que ninguna cantidad original existiera ni de que Compras sea universalmente impredecible. Se detiene aquí y se recomienda Transacciones financieras como próximo encargo separado, antes de Inventario/Movimientos.
