# Contrato de datos — Producto × Mes objetivo

Versión propuesta 2.0, para implementar en fase 3 tras autorización. No es un dataset materializado. Prevalece sobre propuestas temporales alternativas de fase 1; conserva las restricciones arquitectónicas de esa fase.

## 1. Entrada, identidad y salida futura

Fuente: `datos/originales/datos_limpios.csv`, separador `;`, UTF-8 compatible según auditoría; hash en especificacion_problema.md y exploracion_etiquetas.json. No modificar originales ni referencias. Reusar esquema/calidad de fase 1, sin repetir todo el perfilado. Los 268 trims de nombre de cliente documentados no cambian el identificador de producto.

Identificador provisional: cadena exacta `NombreProducto`. No deducir StockItemID, no fusionar productos por similitud ni usar números de fila como identificación comercial. Solicitar claves estables; homónimos, renombres y unidades por empaque no quedan resueltos por este contrato.

Metadatos de cada futura fila (no todos son X): `producto_provisional`, `mes_objetivo`, `corte_exclusivo=s(t)`, `mes_primera_observacion_conocida`, `estado_elegibilidad`, `particion`, versión de fuente/contrato y trazabilidad de agregados. Clave única: producto × mes objetivo. Q objetivo e y son respuestas almacenadas aparte, no predictores.

Q, R e y se definen en especificacion_problema.md. Clase positiva canónica 1; negativa 0. Para perceptrón bipolar, derivar `y_perceptron=2y−1`, sin cambiar observaciones ni significado. No fijar todavía la dimensionalidad después de codificar categorías.

## 2. Meses completos y estados de observación

Un mes es completo solo si el origen cubre todo su intervalo y su carga está cerrada. Que existan ventas el último día no lo certifica. La primera venta observada es reconocimiento provisional del producto, no alta comercial probada.

| Situación | Estado y política |
|---|---|
| Mes anterior a primera venta conocida al corte | Producto todavía no reconocido en el universo; no generar fila ni inventar cero. No usar catálogo futuro. |
| Producto reconocido, mes cerrado, sin transacciones y extracción confirmada completa | Q observado = 0 para esta pregunta de facturación; registrar ausencia de transacciones. No inferir demanda cero, vigencia ni stock. |
| Mes sin transacciones y completitud no comprobada | Q desconocido, nunca cero automático. Bloquear etiqueta/historial dependiente hasta resolver cobertura. |
| Producto posiblemente fuera de comercialización | No se puede confirmar con el CSV. Se conserva en universo de productos previamente observados bajo esa definición explícita; no se declara activo. Si se obtiene catálogo histórico, revisar elegibilidad con versión nueva. |
| Producto con menos de tres meses previos completos | No elegible, estado `historial_insuficiente`; no compensar con meses posteriores. |
| Hueco desconocido dentro de los tres meses previos | No elegible hasta tener tres meses consecutivos completos; no contar tres meses dispersos. |
| Mes objetivo sin respuesta confirmada | Puede emitirse predicción si el historial era elegible; y queda pendiente. No excluirlo selectivamente de evaluación por no registrar una venta. Resolver cobertura primero. |

En estos CSV no hay meses sin transacciones posteriores a primera venta según fase 1. Por tanto, las políticas sobre huecos son preventivas y no cambian los conteos actuales. El auxiliar exploratorio no crea ceros: deja un hueco como desconocido y exige tres meses anteriores observados. Una futura extracción con huecos exige aplicar la tabla, no asumir que sus conteos seguirán siendo iguales.

## 3. Elegibilidad en el instante de predicción

Sea f_p el primer mes de venta del producto que ya se conocía al corte. Para t, E(p,t)=1 si:

1. El producto ya fue observado antes de s(t), usando únicamente la historia disponible entonces.
2. Los meses t−3, t−2 y t−1 son consecutivos, completos, con Q conocido; equivalen a `t−f_p >= 3` solo en esta copia sin huecos.
3. No hay fallo de identidad/unidad que invalide agregar historia. Si se descubre uno, bloquear/revisar, sin fusionar entidades automáticamente.

La inclusión para **emitir una predicción** no depende de tener ventas en t. Para entrenamiento/evaluación, además se exige que t sea completamente observado y tenga Q/y resueltos. Diferenciar `elegible_para_prediccion` de `respuesta_disponible` evita sesgo al filtrar solo meses positivos.

Primeros meses de 2013: enero-marzo sirven como contexto y no generan filas elegibles con k=3; abril de 2013 es primer objetivo. Los ocho productos primero observados en enero de 2016 no son elegibles en enero-marzo; abril es su primer objetivo, mayo el segundo. No usar esos productos para completar vocabulario de entrenamiento.

Último mes: mayo de 2016 es un objetivo válido con historia febrero-abril **si** se confirma cobertura de mayo. Junio puede predecirse después, pero no evaluarse con esta copia: no existe su respuesta. No crear etiquetas de junio ni convertirlas en cero. Si mayo es incompleto, dejarlo pendiente y reducir el test evaluable con una razón de cobertura externa, nunca por sus resultados.

## 4. Sensibilidad de historia mínima

Conteos derivados de la cobertura ya auditada, sin examinar cantidades ni etiquetas de 2016:

| Meses mínimos previos | Primer objetivo cohorte 2013 | Train | Validación | Test ene–may 2016 | Total |
|---:|---|---:|---:|---:|---:|
| 1 | 2013-02 | 5.037 | 2.628 | 1.127 | 8.792 |
| **3 (propuesto)** | **2013-04** | **4.599** | **2.628** | **1.111** | **8.338** |
| 6 | 2013-07 | 3.942 | 2.628 | 1.095 | 7.665 |
| 12 | 2014-01 | 2.628 | 2.628 | 1.095 | 6.351 |

Elegir 3 permite una mediana reciente, menos sensible que la media a un mes extremo, sin exigir un año ni dejar fuera a todos los productos recientes de la prueba. Una mediana de tres no garantiza robustez a cambios persistentes, estacionalidad o errores de identidad. La tabla es sensibilidad de **población**, no evidencia de que ventanas 1/6/12 produzcan mejores o peores clasificadores; no se entrenaron.

Con k=3, los 8.338 pares son candidatos bajo completitud. Respecto a 9.019 pares observados de fase 1, 681 carecen de historial previo suficiente: 219×3 + 8×3. Los 288 huecos anteriores a primera aparición nunca se incluyen como respuesta cero.

## 5. Partición por mes objetivo

| Conjunto | Banda contractual | Meses efectivamente elegibles k=3 | Productos por objetivo | n |
|---|---|---|---|---:|
| Entrenamiento | 2013-01–2014-12 | 2013-04–2014-12 (21) | 219 | 4.599 |
| Validación | 2015-01–2015-12 | 12 meses | 219 | 2.628 |
| Prueba | 2016-01–2016-05 | 5 meses | 219 ene–mar; 227 abr–may | 1.111 |

Toda fila del mismo mes pertenece al mismo conjunto. No partición aleatoria por líneas ni por producto-mes. Entrenamiento y validación cubren 219 productos; prueba cubre esos 219 (1.095 filas) más ocho productos nuevos (16 filas). Ningún producto nuevo aparece en entrenamiento/validación. Ese subgrupo no ha sido validado empíricamente en desarrollo: reportar su rendimiento por separado en la fase de evaluación futura.

**Historia entre particiones:** enero de 2015 utiliza octubre-diciembre de 2014. Enero de 2016 utiliza octubre-diciembre de 2015. Abril de 2016 puede usar enero-marzo de 2016 porque ya terminaron al emitir la predicción de abril. Esto es evaluación mensual secuencial de un paso, no cinco predicciones emitidas simultáneamente en diciembre de 2015. Los modelos y parámetros de transformación permanecen congelados durante test; se actualizan solo agregados históricos y R con reglas ya fijadas. No se eligen hiperparámetros ni ventana con los resultados de esos meses.

Entrenar antes de emitir validación requiere que las etiquetas de diciembre de 2014 ya estén cerradas. La frontera temporal es el mismo instante de cierre/apertura bajo latencia cero; si hay retraso real, introducir corte/embargo compatible y recalcular, sin cambiar la partición por conveniencia del rendimiento. No hay objetivos mensuales solapados; las ventanas históricas compartidas crean dependencia estadística y deben reconocerse.

## 6. Catálogo de características

Notación: q_j=Q(p,t−j); N(p,m)=COUNT DISTINCT InvoiceID del producto en m; V(p,m)=SUM VentaSinImpuesto; C(p,m)=COUNT DISTINCT NombreCliente; L(p,m)=número de líneas; P(p,m)=V/Q si Q>0. Todos los agregados comerciales usan exclusivamente meses anteriores completos. Fórmulas por producto × mes objetivo salvo que se indique calendario. Ventanas incluyen límites explícitos.

**Permitida** significa incorporable al conjunto base de fase 3 si pasa validaciones de datos y disponibilidad al corte. No equivale a prometer que sobreviva a selección de constantes/redundancia en train. **Condicionada** permanece desactivada hasta resolver la condición indicada. **Prohibida** nunca entra en X bajo este problema.

| Nombre | Fórmula/regla y origen | Grano | Disponible | Faltantes | Riesgo de fuga | Decisión |
|---|---|---|---|---|---|---|
| unidades_lag_1/2/3 | q1, q2, q3; CantidadVendida y Fecha | p×t | cierre t−1 | Historia esencial desconocida: no elegible | Desplazar antes de ventana; nunca Q de t | Permitida, tres columnas conceptuales |
| unidades_media_3 | (q1+q2+q3)/3; CantidadVendida | p×t | cierre t−1 | No elegible si falta historia | Media centrada incluiría futuro | Permitida |
| unidades_mediana_3 | mediana(q1,q2,q3)=R | p×t | cierre t−1 | No elegible si falta historia | No incluir Q objetivo | Permitida; referencia conocida, no etiqueta |
| unidades_std_3 | raíz de SUM(qj−media3)^2/3, ddof=0 | p×t | cierre t−1 | No elegible si falta historia | No estimar sobre serie completa | Permitida |
| cambio_unidades_1 | q1−q2 | p×t | cierre t−1 | No elegible si falta historia | No usar Q(t)−q1 | Permitida |
| variacion_relativa_1 | (q1−q2)/q2 si q2>0 | p×t | cierre t−1 | Si q2=0: nulo + indicador; imputación train | Dividir por cero, usar target | Permitida |
| base_variacion_cero | 1[q2=0] | p×t | cierre t−1 | No elegible si q2 desconocido | Indicador histórico, no de ventas futuras | Permitida |
| tendencia_unidades_3 | Pendiente OLS en posiciones 0,1,2: (q1−q3)/2 | p×t | cierre t−1 | No elegible si falta historia | No extender regresión a t | Permitida |
| facturas_lag_1 | N(p,t−1); InvoiceID | p×t | cierre t−1 | Cero solo si mes completo sin ventas | Contar facturas distintas, no líneas | Permitida |
| facturas_media_3 | SUM N(p,t−j)/3, j=1..3 | p×t | cierre t−1 | Desconocido: bloquear/revisar cobertura | No usar facturas de t | Permitida |
| ingresos_lag_1 | V(p,t−1); VentaSinImpuesto | p×t | cierre t−1 | Mes completo sin ventas: 0; otro: desconocido | Ingresos realizados de t prohibidos | Permitida |
| ingresos_media_3 | SUM V(p,t−j)/3 | p×t | cierre t−1 | Igual política de cobertura | No sumar horizonte objetivo | Permitida |
| precio_unitario_ponderado_lag_1 | V(p,t−1)/q1; VentaSinImpuesto/CantidadVendida | p×t | cierre t−1 | q1=0: nulo + indicador; imputación train | No usar precio efectivo futuro | Permitida |
| mes_anterior_sin_ventas | 1[q1=0], solo con mes completo | p×t | cierre t−1 | q1 desconocido: no elegible | No significa falta de demanda/stock | Permitida |
| clientes_distintos_lag_1 | C(p,t−1); NombreCliente | p×t | cierre t−1 | 0 si mes completo sin ventas; desconocido si falta cliente | No usar cartera de t; nombres sin CustomerID | Permitida con limitación de identidad declarada |
| unidades_por_factura_lag_1 | q1/N(p,t−1) si N>0 | p×t | cierre t−1 | N=0: nulo + indicador; imputación train | No confundir ticket monetario | Permitida |
| antiguedad_observada_meses | ordinal(t)−ordinal(f_p), f conocido al corte | p×t | cierre t−1 | Producto no reconocido: no elegible | No calcular f mirando productos futuros | Permitida; no es antigüedad comercial |
| mes_objetivo_sin/cos | sin/cos(2π(mes(t)−1)/12); calendario | t, repetido por p | antes de t | No faltantes; fecha inválida: error | Calendario futuro conocido es válido | Permitidas |
| dias_mes_objetivo | días calendario de t, incluyendo bisiestos | t | antes de t | No faltantes | No usar número real de días con ventas de t | Permitida |
| mes_numero/trimestre | mes(t), 1+floor((mes−1)/3); calendario | t | antes de t | No faltantes | Sin fuga, pero información redundante | Condicionadas; por defecto usar sin/cos, no agregar duplicados sin justificar |
| unidades_lag_12/media_12 | Q(p,t−12), media t−12..t−1 | p×t | cierre t−1 | Historia corta: nulo+indicador, nunca futuro | No excluir cohortes distintas según resultado | Condicionadas; evaluar manejo de historia corta solo en desarrollo, no elevar mínimo base sin versión nueva |
| Marca/Tamanio/EsRefrigerado históricos | último atributo no nulo observado con Fecha<s(t), desempate por Fecha e InvoiceLineID | p×t | cierre t−1 bajo linaje histórico | token desconocido/indicador; bandera nula no es 0 | Dimensiones actuales pudieron sobrescribir historia | Condicionadas a verificar linaje y disponibilidad; no tomar valor de venta de t ni de ficha global |
| identidad_producto_onehot | vocabulario NombreProducto visto solo en train, categoría desconocida en nuevos | p×t | producto conocido al corte | categoría desconocida explícita | Memorizar identidad; vocabulario global revela catálogo futuro | Condicionada; clave solo metadato por defecto, sin ordinal ni target encoding |
| mezcla_categoria_cliente_lag_1 | proporción de facturas distintas por CategoriaCliente en t−1 | p×t | cierre t−1 | Sin facturas: nulo+indicador; categorías desconocidas | Un InvoiceID por categoría; historia dimensional no demostrada | Condicionada a linaje; categorías aprendidas solo en train |
| mezcla_territorio_lag_1 | proporción de facturas por TerritorioVentas en t−1 | p×t | cierre t−1 | Igual política anterior | Ubicación actual vs histórica | Condicionada a linaje y justificación de cardinalidad |
| beneficio_historico | SUM Beneficio en t−1..t−3 | p×t | cierre t−1 si consolidado | Desconocido: imputación train+indicador | Beneficio actual prohibido; cierre puede tener latencia | Condicionada a confirmación del cierre; no necesaria para base |
| Q_objetivo, ingresos/impuestos/beneficio/precio/clientes realizados en t | medidas de cualquier línea de t | p×t | después del corte | No aplica | Información del horizonte | Prohibidas como X |
| InvoiceID/InvoiceLineID directos | identificadores de línea/factura como números | línea, no p×t | algunas líneas futuras no existen al corte | No aplica | Orden y grano erróneos | Prohibidos como X; permitidos solo para contar/trazar hechos pasados |
| stock/demanda_no_atendida | inferir a partir de CantidadVendida | no observado | no disponible | No inventar | No existe fuente de inventario | Prohibidas |
| medias globales futuras, etiqueta, referencia con Q(t), embeddings/target encoding de toda la serie | agregados no causales | varios | después del corte | No aplica | Fuga directa/indirecta | Prohibidas |

Mes, NombreMes, Dia, NombreDia, Trimestre y Anio originales son atributos por línea. Se reconstruye calendario del objetivo; no se trasladan filas de t al predictor. DescuentoEstandar es constante en la copia y no es un atributo de producto: excluir del conjunto base. Las demás columnas originales tampoco se incluyen automáticamente.

### Preparación futura común a ambos modelos

Imputación numérica y categorías desconocidas con parámetros ajustados **solo en train**; indicadores definidos antes de consultar test. No imputar historia esencial para fabricar elegibilidad. Si una columna opcional resulta completamente nula en train, excluirla y registrar motivo. Codificar categóricas autorizadas con vocabulario train y estado desconocido; no ordenar Tamanio como magnitud física.

Selección de constantes/redundancias y escalado solo con train; conservar columna→origen→transformación, orden y parámetros. Para z-score, usar media y desviación poblacional de train, con regla explícita para desviación cero (excluir constante). No ajustar otra escala por conjunto. No podar características usando resultados de prueba. Las dependencias algebraicas del catálogo (por ejemplo media3 y lags) pueden justificar retirar duplicados mediante criterio determinista documentado; no restringen el problema a dos variables.

Todas las características permitidas que sobrevivan a esas reglas alimentan X completa común. Las condicionadas solo se habilitan tras registrar la evidencia requerida. Mantener mismos índices, orden, particiones y máscaras para perceptrón y logística, sin eliminar filas de un solo modelo. Guardar nombres finales y codificación de clase; dimensión determinada por datos de entrenamiento y decisiones registradas, nunca fijada aquí.

## 7. Invariantes para verificar en fase 3

- Cada predictor de una fila depende de eventos con Fecha<corte y de calendario conocido; disponibilidad de carga confirmada o supuesto académico explícito.
- Cambiar unidades del mes t o posteriores no cambia X, R ni elegibilidad de esa fila; puede cambiar únicamente su respuesta Q/y. Cambiar un mes anterior conocido sí puede afectar cortes posteriores.
- Claves únicas, particiones disjuntas por mes objetivo; ningún futuro desconocido se codifica como cero. Meses completos sin filas no se pierden por filtrar únicamente ventas existentes.
- Antes de prueba, congelar contrato, lista de características, transformaciones, regla R, parámetros de modelos y decisión de umbral logístico. La historia puede avanzar mes a mes sin `fit` sobre test.
- Counts esperados k=3: 4.599/2.628/1.111 bajo cobertura asumida; toda diferencia exige explicación de calidad/cobertura, no un recorte para mejorar métricas.
- Hashes de originales, referencias y entregables de fase 1 sin cambios. Ningún agregado exploratorio de fase 2 se reutiliza como dataset definitivo sin construir y verificar el pipeline autorizado.
