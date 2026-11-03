# Especificación del problema — Fase 2

## Estado y pregunta empresarial

**Decisión técnica propuesta para fase 3:** clasificación binaria de ventas unitarias observadas de un producto en un mes respecto a su nivel reciente. La especificación es reproducible; su valor económico sigue siendo una hipótesis, no un rendimiento demostrado. La fase 1 está aprobada. Esta fase sustituye sus propuestas preliminares de objetivo y partición; no altera sus archivos ni resultados.

**Pregunta:** al cierre del mes anterior, ¿las unidades facturadas observadas del producto p en el mes objetivo t superarán la mediana de sus unidades observadas en los tres meses inmediatamente anteriores?

Utilidad propuesta: identificar productos con posible aumento relativo para priorizar revisión comercial y planificación de ventas. No calcula órdenes de compra, unidades óptimas ni stock; no mide demanda no atendida ni causalidad. Un aumento de una unidad puede cambiar la clase: sin costos de decisión ni mínimo de incremento aprobado, su relevancia empresarial puede ser limitada. Esa restricción debe acompañar cualquier interpretación académica.

## Evidencia y decisiones

**Observado en fase 1:** 228.265 líneas, 30 columnas, 70.510 facturas, 227 nombres de producto, enero de 2013 a mayo de 2016; no hay StockItemID ni inventario histórico. Hay 219 nombres desde enero de 2013 y ocho primeras ventas en enero de 2016. No hay huecos producto-mes después de la primera observación. Las ventas representan facturación observada, no demanda total.

**Fuente principal:** `datos/originales/datos_limpios.csv`, SHA-256 `d4f3b3b7b018ee49c515915fe851ff7327d82fb198cbc8af117e8f0e73fff37e`. Se acepta para trazabilidad la normalización de espacios de cliente documentada en fase 1, sin modificar el CSV. El esquema, las claves y medidas se conservan respecto al otro CSV según la comparación existente.

**Decisiones propuestas:** horizonte mensual; mínimo tres meses previos completos; etiqueta relativa de mediana móvil 3; empate en clase 0; particiones por mes objetivo: entrenamiento hasta diciembre de 2014, validación 2015 y prueba enero-mayo de 2016. Son decisiones de diseño sustentadas en desarrollo, no requisitos textuales adicionales del docente.

**Supuestos de operación no demostrados por CSV:** las facturas de cada mes cerrado están completas y disponibles en el instante de corte; CantidadVendida mantiene una unidad comercial consistente para cada nombre de producto; un mismo nombre conserva la identidad del producto; la extracción no actualiza atributos retrospectivamente sin historia. No hay timestamps de ingestión para demostrar disponibilidad real. El cierre de mayo de 2016 requiere confirmación de extracción; terminar el día 31 no prueba completitud.

## Definición matemática

Sea I el conjunto de líneas facturadas. Para una línea i se conocen `NombreProducto_i`, `Fecha_i` y `CantidadVendida_i`. Definir:

\[
p=\mathrm{NombreProducto}\quad\text{(cadena exacta del CSV)},\qquad
t=\text{mes calendario objetivo, representado como YYYY-MM}.
\]

No usar ordinales generados como magnitud del producto; no fusionar nombres por similitud, mayúsculas o trim adicional. Un cambio de nombre crea provisionalmente otra entidad. Dos SKU con el mismo nombre quedarían mezclados: solicitar StockItemID para resolverlo. La clave de fila futura es la pareja única `(producto_provisional, mes_objetivo)`.

Sean s(t) el inicio del mes t y e(t)=s(t+1) su límite superior exclusivo:

\[
Q_{p,t}=\sum_{i:\,p_i=p,\;s(t)\le Fecha_i<e(t)} CantidadVendida_i.
\]

Q es una suma de unidades facturadas observadas; no es una suma de precios, beneficio ni número de líneas. La suma vacía solo se interpreta como cero observado en períodos completos de extracción y después de reconocer al producto; no reconstruye ventas anteriores a su primera aparición ni demuestra demanda cero.

La información predictora pertenece a:

\[
\mathcal F_{t-1}=\{\text{hechos efectivamente disponibles antes de }s(t)\}.
\]

El horizonte es el mes completo t y la predicción se emite en el límite `s(t)`, inmediatamente después del cierre de t−1, **antes de cualquier venta de t**. La fecha máxima de evento consultable es el último día de t−1, con condición computacional `Fecha < s(t)`; no se admiten ventas de t aunque lleguen antes de ejecutar un script retrospectivo. El calendario del mes t sí se conoce entonces. Los CSV solo ofrecen fechas, no el tiempo real de disponibilidad.

Si el cierre tiene latencia, el diseño actual no se presenta como operativo: se debe acordar otra fecha de emisión/horizonte y recalcular la elegibilidad, en lugar de incorporar facturas tardías como si se conocieran al corte. No se desplaza silenciosamente la predicción dentro del mes objetivo.

Para las observaciones elegibles:

\[
R_{p,t}=\operatorname{mediana}(Q_{p,t-1},Q_{p,t-2},Q_{p,t-3}),
\qquad y_{p,t}=\mathbf1[Q_{p,t}>R_{p,t}].
\]

Clase 1: venta mensual superior al nivel reciente del propio producto. Clase 0: venta igual o inferior. No llamarlas “rentable/no rentable”, “stock suficiente/insuficiente” ni “demanda absoluta alta/baja”. Con tres valores completos no hace falta interpolación de cuantiles; se usa el valor central ordenado. R puede ser predictor válido porque se conoce antes de t; no determina y sin conocer Q de t.

La referencia se actualiza causalmente en cada corte usando meses ya cerrados, incluso de validación o prueba previos. La regla (ventana 3, mediana y desigualdad estricta) queda congelada antes de evaluar prueba; actualizar historia no es optimizar la regla sobre prueba.

## Alcance y aceptación de la futura implementación

El contrato detallado está en `contrato_datos.md`; la comparación de etiquetas en `analisis_etiquetas.md`; evaluación en `diseno_evaluacion.md`. La fase 3 deberá construir únicamente los agregados y transformaciones autorizados por ese contrato, con pruebas temporales y trazabilidad. Ambos modelos consumirán todas las características válidas de una misma preparación, de dimensión dinámica. PCA tendrá una salida visual independiente, ajustada con entrenamiento.

No se implementaron clasificadores, PCA, preprocesamiento definitivo ni datasets finales. El auxiliar de esta fase construye únicamente agregados exploratorios en memoria y exporta resúmenes de desarrollo, no filas predictivas con X/y. No se repitió la auditoría integral.

## Pendientes y límites

Confirmar disponibilidad al cierre, integridad de la extracción y unidad/identidad del producto. El diseño se puede implementar académicamente con esos supuestos explícitos, pero sin su validación no se afirma utilidad productiva. Deben confirmarse autorización docente de biblioteca de gradientes y PCA antes de implementarlos; no bloquean la presente especificación matemática. Cualquier nuevo umbral comercial o cambio de ventana requerirá versión nueva y revisión sin usar resultados de prueba para escogerlo.
