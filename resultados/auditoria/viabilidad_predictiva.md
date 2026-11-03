# Viabilidad predictiva y preprocesamiento preliminar

Análisis conceptual y exploratorio de fase 1. No se fijan etiquetas definitivas, no se crean matrices de entrenamiento ni se ajustan transformaciones/modelos. Evidencia completa en perfilado_datos.json y reporte_auditoria.md.

## Comparación de problemas

| Alternativa | Evidencia disponible | Viabilidad preliminar | Riesgo principal |
|---|---|---|---|
| A. Beneficio negativo | 228.265 líneas, 4.626 negativas (2,026592 %) | Condicionada al instante de predicción y disponibilidad de costos/condiciones | 88,932123 % de pérdidas en cuatro productos; memorizar identidad y fuga monetaria |
| B. Demanda alta siguiente mes | 227 productos, 41 meses, 9.019 producto-mes observados | Principal candidata para formalizar | Umbral pendiente, escala heterogénea, productos nuevos y demanda censurada por stock |
| C. Compra futura de cliente | 663 nombres de cliente, 24.029 cliente-mes observados | Alternativa viable con población elegible bien definida | Pocos meses inactivos, supervivencia y censura del último período |
| D. Inventario | No hay series de stock/abastecimiento | No viable como problema real con estos CSV solos | Confundir cantidad vendida con existencias |

## A. Anticipar beneficio negativo

**Pregunta:** ¿qué línea comercial propuesta terminará con Beneficio < 0? Unidad: línea de factura futura/propuesta; positiva pérdida, negativa beneficio >= 0. Cero se integra en no negativo (aunque se observaron cero casos de beneficio cero).

**Momento propuesto:** antes de aprobar precio/condiciones y emitir la línea. Debe confirmarse si es un pedido, una cotización o una factura ya emitida; el CSV registra resultados facturados y no ofrece timestamps de cotización/aprobación. Una evaluación retrospectiva sobre factura no prueba anticipación operacional.

**Predictores posibles:** calendario planificado, atributos de producto y cliente vigentes entonces, historial de pérdidas y precios calculado solo con operaciones previas, cantidad solicitada/precio ofertado solo si existen antes de la decisión. Cliente, territorio y vendedor solo si ya fueron asignados y se conserva su estado histórico.

**Fuga:** Beneficio, costo reconstruido a partir de Beneficio, margen realizado, medidas fiscales/ingresos finales, descuentos efectivos posteriores; agregados del día o factura que incluyan la propia operación. InvoiceLineID e InvoiceID no son predictores ordinales. PrecioUnitario/CantidadVendida realizados no son automáticamente válidos por estar en la misma fila. Si costos y precios previos permiten calcular exactamente el margen, una regla empresarial puede ser más apropiada que ML.

**Desequilibrio y concentración:** positivas 223.639, negativas 4.626, cero 0. Para el problema binario la clase mayoritaria es no negativo con exactitud teórica 97,973408 %. De 227 nombres de producto, 18 tienen alguna pérdida; cuatro variantes de Halloween zombie mask (XL/S/M/L) tienen 1.071/1.044/1.017/982 pérdidas y pierden en todas sus líneas. Concentran 4.114 de las 4.626 pérdidas. No hay evidencia de causas como promociones o liquidaciones.

**Datos adicionales:** costo histórico, cotizaciones/pedidos, promociones, autorizaciones y tiempos de disponibilidad. **Evaluación futura:** partición temporal, métricas por clase/producto y análisis separado de productos nuevos; no interpretar alta exactitud aleatoria como aprendizaje general de rentabilidad. No elegir esta alternativa definitivamente por facilidad de etiqueta.

## B. Demanda alta el siguiente mes

**Pregunta:** al cerrar el mes t, ¿las unidades vendidas del producto durante t+1 superarán un umbral comercial de alta demanda? Unidad: producto-mes de origen t, identificado idealmente por StockItemID, con horizonte de un mes. Positiva alta demanda en t+1; negativa baja (incluido cero solo si el producto estaba elegible y el mes completamente observado).

**Evidencia:** 219 productos presentes desde enero de 2013; ocho aparecen por primera vez en enero de 2016 y tienen cinco meses. 227 × 41 = 9.307 posiciones en cuadrícula global, 9.019 observadas y 288 sin transacciones. Todas las ausencias globales son anteriores a las primeras ventas de los ocho productos nuevos; hay cero huecos mensuales después de primera venta por producto. La primera venta no es fecha probada de alta en catálogo. No inventar ceros en esos 288 casos.

Las unidades producto-mes observadas tienen mediana 192, percentiles 25/75 de 137/1.584, mínimo 41 y máximo 8.064. Son estadísticas descriptivas globales, **no** umbrales listos para entrenar: mezclan unidades comerciales y escalas de productos; no se usarán para fijar la etiqueta con información de validación/prueba.

**Umbral pendiente:** priorizar un umbral de negocio previamente establecido por producto/familia si existe. Como alternativa experimental, proponer cuantiles por producto estimados exclusivamente en el tramo de entrenamiento, con regla de respaldo entrenada para productos de poco historial y congelados después. Una mediana puede favorecer balance y un cuantil alto una clase rara; empates y ceros alteran el balance. Decidir si se desea volumen absoluto o demanda alta respecto al comportamiento habitual. No estimar un umbral sobre todo el histórico ni el propio mes futuro. Cuantiles globales de auditoría no eligen el objetivo.

**Predictores disponibles al cierre de t:** sumas de unidades/ingresos y conteos de facturas de t y meses anteriores; retardos 1/2/3/12 meses definidos respecto al objetivo t+1; medias y variabilidad de ventanas pasadas; tendencia y recencia; calendario conocido de t+1 (mes, trimestre, días); atributos históricos del producto. Participaciones de clientes/territorios y precios medios solo calculados sobre ventas hasta t. Debe fijarse latencia del cierre para evitar datos aún no registrados.

**Fuga:** unidades, ingresos, precio realizado o cartera real de t+1; medias móviles centradas; categorías o catálogo construidos mirando altas futuras; agregados de toda la serie; imputar con períodos posteriores; seleccionar variables o umbrales usando prueba; completar ausencias anteriores a alta con conocimiento retrospectivo no disponible.

**Productos nuevos y meses sin venta:** definir catálogo vigente en cada corte con IDs y fechas de alta/baja; no incluir entidades futuras en el universo de t. Con poco historial: indicadores de disponibilidad/antigüedad, política de mínimos o cohorte de arranque separada y fallback aprendido en entrenamiento. Después de alta, mes sin transacción puede equivaler a ventas cero si extracción completa y producto vigente; no equivale necesariamente a demanda cero (stockout). No usar prueba para decidir exclusiones.

**Estacionalidad:** tres años completos y cinco meses de 2016 permiten explorar calendario y retardos anuales con cautela. Solo hay alrededor de tres repeticiones por mes del año; miles de filas producto-mes no son miles de períodos independientes. Es necesario comparar baselines temporales sencillos (último mes/temporada) en fase posterior. No se ha probado separabilidad ni utilidad predictiva de un modelo lineal.

**Desequilibrio:** aún no es medible como etiqueta porque no se aprobó umbral/elegibilidad. Cuantificar en train/validation/test después de formularlo. **Datos adicionales deseables:** StockItemID, catálogo/vigencia, unidades de empaque, stockout, promociones, precios planificados, devoluciones y confirmación de extracción completa. Con estos CSV solos, llamar al objetivo ventas futuras o demanda atendida.

**Viabilidad:** candidata principal por historial repetido y alineación con análisis de productos; requiere formalizar umbral, población y cortes. No se selecciona definitivamente.

## C. Actividad futura de clientes

**Pregunta:** al cierre de t, ¿un cliente elegible comprará en t+1? Unidad: cliente-mes; positiva al menos una factura en el mes siguiente, negativa ninguna durante un mes de observación completo.

**Momento:** cierre de t. **Predictores:** recencia, frecuencia con facturas distintas, unidades/importe históricos, antigüedad conocida, categoría/grupo y territorio vigentes, calendario de t+1, tendencias de ventanas anteriores. Retardos y ceros se definen dentro de la vida observada elegible.

**Fuga:** cualquier factura/importe/fecha de última compra de t+1; segmentaciones RFM calculadas sobre toda la serie; lista de clientes que solo aparecen en el futuro; información dimensional actual sin historia; usar ausencia de registro más allá del corte como abandono definitivo. El mes final no puede generar etiqueta del mes siguiente sin observación de este.

**Evidencia de balance:** 24.029 cliente-mes con compras de 27.183 en cuadrícula global. Desde primera observación: 25.986 posiciones, 1.957 sin compra (7,530978 %). Son estados mensuales descriptivos, no distribución final de etiquetas t+1. Elegibilidad por apertura/actividad histórica y baja cambia el denominador; excluir último origen censurado y nuevos sin historia cambia aún más el balance. No rellenar meses previos al alta con ceros arbitrarios.

**Datos adicionales:** CustomerID, fechas de alta/baja y estado histórico, campañas/contactos, integridad del período futuro. FechaApertura existe, pero su significado y cobertura deben confirmarse. **Viabilidad:** posible; alta actividad mensual puede producir un clasificador mayoritario muy fuerte y poca utilidad incremental. Evaluar recall de inactividad y utilidad comercial además de exactitud. Es compra en un mes, no una afirmación permanente de churn.

## D. Inventario

**Pregunta posible con nuevas fuentes:** al cierre de t, ¿el producto tendrá quiebre de stock durante t+1? Unidad: producto-ubicación-período (o producto-período si no existen múltiples almacenes). Positiva stockout según definición acordada (p. ej. existencias <= 0 con demanda pendiente), negativa sin quiebre en un período observado completamente.

**Predictores potenciales:** stock disponible al corte, pedidos de compra pendientes, entradas planificadas conocidas, demanda pasada, tiempos históricos de reposición, stock de seguridad y restricciones de proveedor. **Fuga:** stock mínimo futuro, entradas efectivas futuras no conocidas y salidas de t+1. **Desequilibrio:** desconocido; no hay etiqueta observada.

Los CSV contienen ventas, no snapshots de stock ni movimientos/órdenes de abastecimiento. La Carta solicita QuantityOnHand pero este campo está ausente. Incluso un stock actual aislado no reconstruye inventario histórico. CantidadVendida no es stock; EsRefrigerado no informa disponibilidad. Se requieren snapshots o saldo inicial más movimientos fechados, backorders, proveedores/plazos, calendario y ubicación del inventario. No viable con las fuentes actuales como problema de inventario predictivo real.

## Pipeline futuro, solo diseño

1. **Contrato del problema:** aprobar unidad, instante t, horizonte, población elegible, etiqueta y política de empates/censura. Recuperar IDs y linaje; aceptar explícitamente el trim de cliente documentado. Conservar hashes y filas de origen para trazabilidad.
2. **Construcción temporal:** agregar por entidad-período usando claves estables. Calendario de períodos completos; distinguir ausencia, no vigencia y cero. Todas las características proceden de información disponible hasta el corte. La etiqueta usa el horizonte posterior exclusivamente. Guardar fecha de origen y fecha de fin de etiqueta; no exportar matrices definitivas hasta autorización.
3. **Partición cronológica común:** como propuesta a discutir, objetivos ene-2014 a dic-2014 para train, ene-jun-2015 para validación y jul-2015 a may-2016 para test; 2013 sirve como historial inicial de doce meses. Cada fila origen se asigna por fecha del mes objetivo, verificando que toda etiqueta de entrenamiento haya concluido antes del primer instante de predicción de validación. Un corte con latencia o horizontes solapados exige embargo. Estos cortes son preliminares, no fueron aplicados y pueden cambiar por balance/tamaño.
4. **Separación de desarrollo y prueba:** diseñar/seleccionar con entrenamiento y validación; mantener test final sin consultas para ajustar umbrales, hiperparámetros o transformaciones. Si se usan etiquetas por cuantiles de entrenamiento, estimarlas solo allí y congelarlas. La inspección global de fase 1 es auditoría descriptiva, no evaluación de rendimiento.
5. **Nulos:** preservar motivos; categorías con token explícito sin información cuando corresponda, marcador de ausencia y política de desconocidos. GrupoCompra sin grupo no equivale a error; Marca ausente no demuestra falta de marca comercial. Medianas u otra imputación numérica se ajustan solo con entrenamiento. No usar backfill futuro; indicadores de historial insuficiente para productos nuevos. Mantener FechaApertura como atributo solo si se valida su disponibilidad histórica.
6. **Codificación:** one-hot con vocabulario aprendido solo en train y categoría desconocida en validación/test; evaluar cardinalidad de nombres y no codificar IDs como números continuos. No usar target encoding de toda la muestra. No ordenar Tamanio de forma numérica: combina unidades y tallas heterogéneas. Calendario cíclico opcional sujeto a justificación.
7. **Selección y escala:** remover constantes detectadas en train (DescuentoEstandar es candidata), redundancias e información posterior. Ajustar media/desviación u otra escala solo con train; política explícita de varianza cero. Transformaciones idénticas y orden de columnas fijo para las otras particiones; conservar nombres transformados, categorías, parámetros y versión de esquema. Los atípicos no se eliminan por IQR de toda la población.
8. **Contrato de matrices:** X_train/X_val/X_test con número dinámico de columnas, finitas y mismas columnas/orden; IDs de filas, cortes y etiquetas inmutables. Ambos clasificadores reciben exactamente estas matrices y particiones. Si perceptrón usa {-1,+1} y logística {0,1}, derivar ambas codificaciones del mismo vector de clase con mapa explícito; definir la misma clase positiva.
9. **PCA independiente:** partir de las mismas características preparadas; centrar y ajustar componentes exclusivamente en train, proyectar val/test con parámetros guardados. Valorar NumPy mediante SVD/autodescomposición y documentar varianza explicada y restricciones. La proyección de dos componentes alimenta solo visualización; no reemplaza las matrices completas. Una frontera dibujada en el plano requiere aclarar si es un corte/aproximación; no es frontera exacta multidimensional.
10. **Evaluación posterior:** baselines y métricas por clase/cohorte/período, curvas de entrenamiento y pesos finales según docente; no inferir convergencia del perceptrón por número de épocas. Toda métrica comparativa debe usar las mismas observaciones. No implementar ninguna de estas etapas en fase 1.

## Recomendación para fase 2

Formalizar primero B: ventas unitarias altas en el siguiente mes por producto. Aprobar umbral por negocio o regla de cuantiles entrenados, unidad de medida, elegibilidad de catálogo, tratamiento de los ocho productos recientes, cierre de mayo de 2016 y cortes temporales. Confirmar biblioteca de gradientes y PCA. Si no se obtiene definición útil de alta demanda, comparar C con criterios de valor empresarial antes de elegir. A queda condicionada por concentración y disponibilidad previa; D exige nuevas fuentes. Esperar autorización para cualquier implementación posterior.
