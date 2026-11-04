"""Redacción reproducible del contrato, exclusivamente desde agregados existentes."""
import json
from .especificar_pedidos import OUT,STATES
from .auditoria_dw import guardar


def leer(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def escribir(name,text): (OUT/name).write_text(text,encoding='utf-8',newline='\n')


def exportar():
    a=leer('agregados_desarrollo.json');d=leer('distribucion_etiquetas_desarrollo.json');r=leer('reconciliacion_historica.json')
    tr=d['line']['train'];va=d['line']['validation'];doc=a['document_composition'];unknown=a['unknown_customer_development']
    names={'invalid':'Fechas inválidas/inconsistentes','insufficient_followup':'Horizonte incompleto al corte de partición','missing_pick_indeterminate':'Picking nulo, horizonte nominal completo: indeterminado','pick_not_observed_by_cut':'Picking registrado en/después del corte: no disponible','row_not_available_by_cut':'Última edición en/después del corte: versión no disponible','within_two_days':'Picking observado hasta día 2 inclusive','after_two_days':'Picking observado desde día 3'}
    state_table='\n'.join(f"| {names[s]} | {tr['states'].get(s,0)} | {va['states'].get(s,0)} |" for s in STATES)
    def month_table(unit):
        rows=[]
        for month,item in sorted(d[unit].items()):
            if len(month)!=7:continue
            rows.append(f"| {month} | {item['total']} | {item['eligible']} | {item['positive']} | {item['negative']} | {100*item['positive_proportion']:.4f} % |")
        return '\n'.join(rows)
    contract={'version':'pedidos-1.0-especificacion','verdict':'Viable con ajustes explícitos para líneas con picking observado; no validación de utilidad predictiva ni habilitación del ETL',
        'unit':{'primary_key':'Sales.OrderLines.OrderLineID','dw_bridge':['WWI Order ID','WWI Stock Item ID'],'document_id':'Sales.Orders.OrderID actual; cambia por traslado a backorder','group_id':'Raíz de la familia de backorders, solo asociación, nunca entrada numérica'},
        'prediction_instant':'Emisión de la demanda original de la línea; solo se conoce su fecha calendario heredada OrderDate, no hora exacta. Usar historia estrictamente anterior al inicio de ese día.',
        'target':'1[DATE(picking_completed)>OrderDate+2 días calendario], solo cuando picking observado y justificadamente disponible; igualdad es clase 0',
        'horizon_end_exclusive':'inicio de OrderDate+3 días; no 48 horas, días laborables ni SLA contractual',
        'known_at_conservative':'max(PickingCompletedWhen,OrderLines.LastEditedWhen,Orders.LastEditedWhen); última edición es cota conservadora del estado persistido, no prueba de latencia ni historial completo',
        'asof_mode':'Reconstrucción retrospectiva conservadora de la copia, no snapshot real de cada corte; disponibilidad de ingestión sin certificar',
        'missing_pick_policy':'Indeterminado incluso con seguimiento nominal suficiente: no evidencia histórica negativa fiable de ausencia del evento; conservar estado sin y',
        'development_eligibility':'OrderDate<2016-01-01, horizon_end<=cut, picking>=inicio OrderDate y picking<cut, ambas últimas ediciones<cut; no nulos; filtros de features posteriores aún no aplicados',
        'partitions':{'train':{'orders':['2013-01-01','2014-12-31'],'outcomes_before':'2015-01-01T00:00:00'},'validation':{'orders':['2015-01-01','2015-12-31'],'outcomes_before':'2016-01-01T00:00:00'},'test':{'orders':['2016-01-01','2016-05-31'],'proposed_followup_before':'2016-06-01T00:00:00','outcomes':'No consultados en esta tarea; certificar cobertura antes de apertura'}},
        'boundary_policy':'Excluir últimos dos días calendario de cada partición para horizonte homogéneo; además retener etiqueta solo si su observación y versión existían antes del corte. No etiquetar positivos en el vencimiento sin observabilidad de ausencia.',
        'features_enabled_design':{'mes_sin':'sin(2*pi*(month-1)/12)','mes_cos':'cos(2*pi*(month-1)/12)','semana_sin':'sin(2*pi*weekday/7), lunes=0','semana_cos':'cos(2*pi*weekday/7), lunes=0','dia_mes':'día de mes 1..31','dias_mes':'días calendario del mes, incluyendo bisiestos'},
        'features_conditional_disabled':['Cantidad solicitada y precio/importe originales: no tomar último estado como valor inicial sin prueba','Conteo y variedad original del pedido: la composición actual fue alterada por backorders','Historia de actividad por producto/cliente: requiere origen estable, corte estricto y versiones disponibles','Historia de retrasos: disponible solo desde known_at del evento, no desde OrderDate ni desde una supuesta ausencia al día 3','Atributos dimensionales/cliente: relaciones as-of únicas y comprobadas antes de habilitar'],
        'forbidden_features':['Picking objetivo','PickedQuantity objetivo','Picker final','BackorderOrderID final','Estado de cumplimiento','Fechas de última edición como predictores','Información posterior al registro','OrderKey/OrderID/OrderLineID/StockItemID/CustomerID como magnitudes continuas'],
        'unknown_customer':'Conservar observación; categoría desconocido si se habilita categórica y su resolución es conocida al corte. En base se desactivan entradas de cliente. IDs OLTP son evidencia, no identidad inventada; su uso requiere contrato.',
        'dimensional_join_policy':'No unir a última versión. Exigir intervalo semiabierto vigente en corte, ID operacional y exactamente una coincidencia; cero -> desconocido, más de una -> ambiguo/deshabilitado. Nunca arbitrar con TOP 1 o clave máxima.',
        'metrics':['balanced_accuracy','precision_delay','recall_delay','f1_delay','average_precision_from_scores_if_available','specificity','FPR','false_discovery_rate','alert_proportion','confusion_matrix','accuracy_secondary'],
        'baselines':{'majority':'Clase 0 congelada desde train justificado; regla/estado persisten antes de abrir test','causal_history_candidate':'Última etiqueta de línea conocida por producto, con known_at anterior al inicio del día objetivo; si falta, mayoría train. Desactivada hasta validar identidad y disponibilidad de historia.'},
        'training_strategy':'No ponderación, sobremuestreo, síntesis o búsqueda de umbral fijados aquí; modelos propios no aceptan automáticamente sample_weight. Estrategia posterior solo con train/validación y pruebas propias.',
        'test_seal':'Todas las nuevas consultas de outcomes llevan o.OrderDate<2016-01-01 parametrizado; cifras globales preexistentes solo se reutilizan para conciliación.'}
    guardar(OUT/'contrato_datos.json',contract)
    escribir('especificacion_problema.md',f'''# Pedidos: especificación predictiva

## Dictamen y alcance

**Viable con ajustes explícitos**, para estudiar retraso de picking de la **línea de demanda original**, con outcome observado y controles temporales conservadores. No se afirma utilidad predictiva, disponibilidad certificada al segundo exacto de emisión ni viabilidad del cumplimiento completo del documento original con su composición inicial. No hay matrices ni entrenamiento. La especificación se apoya en desarrollo 2013–2015 y conserva 2016 reservado.

La unidad recomendada es `Sales.OrderLines.OrderLineID`, no `Order Key` (clave técnica DW), `WWI Order ID` (documento actual) ni una factura. `Fact.Order` omite OrderLineID; la auditoría ya verificó correspondencia uno a uno de la pareja OrderID+StockItemID actual en esta copia. Esa unicidad se debe revalidar al extraer y no sustituye un identificador de línea estable.

## Instante y significado del evento

Se propone emitir la predicción cuando se registra la demanda original de la línea. `Order Date Key` deriva de `Orders.OrderDate`, de tipo date; `Picked Date Key` deriva de `OrderLines.PickingCompletedWhen`, reduciendo un timestamp a fecha. La hora exacta de creación no está disponible como columna dedicada; `LastEditedWhen` cambia y no es fecha de creación. El contrato base usa la fecha calendario de la demanda y solo historia cerrada antes de las 00:00 de ese día, sin ordenar artificialmente eventos del mismo día.

La plantilla local de simulación, almacenada dentro de `DataLoadSimulation.Configuration_ApplyDataLoadSimulationProcedures`, define el traslado de líneas pendientes a un nuevo OrderID y copia `OrderDate` del padre. **No se ejecutó esa plantilla**. Los {a['backorder_links']['links']} vínculos de desarrollo tienen fechas padre/hijo iguales y IDs estrictamente crecientes, patrón compatible con el código. Esto apoya usar la fecha heredada como ancla original de la línea; no la convierte en fecha de emisión real del documento hijo ni acredita toda la historia de modificaciones.

La fuente usa date/datetime2 sin zona horaria incorporada. Se interpreta su calendario registrado, sin asignar UTC o la zona del equipo como zona del negocio. La certificación de cortes requiere aclarar esa convención.

`Website.InsertCustomerOrders` muestra que cantidad, precio, producto, cliente y vendedor se asignan al insertar. Eso demuestra su existencia conceptual al registro, pero no que el último valor almacenado sea siempre el inicial de las filas históricas. Cantidad/precio se dejan condicionados; conteos y totales del documento actual no representan automáticamente el pedido original después de traslados.

## Etiqueta exacta y disponibilidad

Sea d la fecha original y p el timestamp fiable de picking completo de la línea. Para outcomes observados: **y=1 si DATE(p)>d+2 días calendario; y=0 si DATE(p)<=d+2**. Picking el mismo día o el día 1 o 2 es clase 0; el día 3 es clase 1. El horizonte exclusivo termina a las 00:00 de d+3. No se trata de 48 horas exactas ni días laborables. Dos días es el candidato solicitado, sin evidencia de un SLA empresarial; no se cambió para equilibrar las clases.

Se rechaza p anterior al inicio de d como inconsistencia. Picking ausente queda **indeterminado**, sin y, aunque nominalmente haya transcurrido el horizonte. La copia actual no es un log completo que permita acreditar que no existió picking al vencer el plazo. Un futuro registro de ausencia fiable al vencimiento podría permitir y=1 desde ese instante, pero eso requiere otro contrato y evidencia; no se aplica ahora.

En esta versión la etiqueta solo se usa desde `known_at=max(p, última edición de línea, última edición de encabezado)`. Para cada partición se exige known_at anterior al corte y horizonte completo. Así, un picking de 2015 no etiqueta un ejemplo de train cerrado al iniciar 2015; tampoco se usa en train la mera inferencia retrospectiva de que habría estado retrasado antes. Esta política es conservadora y puede excluir estados que antes tuvieron información suficiente. Última edición anterior al corte respalda la versión persistida, pero no certifica latencia ETL, ausencia de backdating ni un snapshot histórico íntegro.

## Línea frente a documento

En desarrollo hay {doc['lines']} líneas de {doc['documents']} documentos actuales, con 1–5 líneas. No se observan documentos actuales con fechas de picking distintas entre sus líneas ni mezcla de nulos/observados; hay {doc['all_missing']} con todas sus líneas sin picking. No se interpreta esa homogeneidad como independencia: el traslado de pendientes cambia la composición. En {doc['header_vs_last_line_timestamp_different']} documentos, el picking de encabezado difiere del último de sus líneas actuales; también hay {a['development_document_child_status']['derived_backorder_documents']} documentos derivados. La igualdad numérica no se toma como una prueba causal individual.

Si el negocio exigiera cumplimiento integral del pedido **original**, se necesitaría la lista original de líneas y la familia completa. Con ese universo verificado, cumplimiento es max(p_i) solo si todas las líneas están completadas; basta una línea conocida posterior al día 2 para probar retraso, y todas conocidas dentro para probar no retraso. Si no hay un testigo tardío y faltan outcomes, el documento es indeterminado. No se usa promedio, mayoría de líneas ni timestamp final del encabezado como regla arbitraria. Las cifras de documento publicadas aquí son diagnóstico de la composición actual, no aprobación de esa etiqueta empresarial.

Las líneas de un documento o familia son dependientes. Agrupar por raíz de backorders en evaluación; nunca repartir líneas hermanas aleatoriamente. Los vínculos de desarrollo no cruzan fechas, pero esa condición se verificará de nuevo con los IDs al construir las particiones sin leer sus outcomes reservados.
''')
    historical='\n'.join(f'| {k} | {n} |' for k,n in r['states'].items())
    annual=[]
    for unit in ['line','document_diagnostic']:
        for year in ['2013','2014','2015']:
            q=d[unit][year];annual.append(f"| {unit} | {year} | {q['eligible']} | {q['positive']} | {q['negative']} | {100*q['positive_proportion']:.4f} % |")
    escribir('analisis_etiquetas.md',f'''# Análisis de etiquetas y conciliación

## Reconciliación de la auditoría anterior

Esta tabla **reutiliza únicamente agregados históricos existentes**, de toda la copia, para explicar las cifras recibidas. No se volvió a consultar picking de pedidos fechados en 2016 ni se calcularon nuevas tasas o métricas test. Estos conteos globales no seleccionan diseño, umbral o granularidad.

| Estado histórico mutuamente excluyente | Líneas |
|---|---:|
{historical}
| **Total** | **{r['total']}** |

224591+3154=227745 maduras con picking observado. 227745+3129=230874 con seguimiento nominal de la auditoría. Faltan 538 para 231412: su horizonte nominal no estaba completo. Los 3147 nulos totales se descomponen en 3129 maduros y 18 de esos 538; los otros 520 inmaduros tienen picking observado, pero no se subdividen por clase. No hay una fila adicional de nulos que sumar al total: son atributos de los estados anteriores. La auditoría no detectó picking anterior al pedido; otros motivos de no elegibilidad no se inventan y se evalúan en el contrato temporal de desarrollo.

Tener seguimiento nominal no equivale a prueba de ausencia de picking. Por ello, las 3129 líneas siguen indeterminadas y no se asignan automáticamente a retraso ni se borran del inventario.

## Conteos nuevos al corte de cada partición, solo desarrollo

| Estado exhaustivo por línea | Train | Validación |
|---|---:|---:|
{state_table}
| **Total** | **{tr['total']}** | **{va['total']}** |

Train conserva {tr['eligible']} etiquetas justificadas: {tr['positive']} positivas y {tr['negative']} negativas ({100*tr['positive_proportion']:.4f} % positivas). Validación conserva {va['eligible']}: {va['positive']} positivas y {va['negative']} negativas ({100*va['positive_proportion']:.4f} %). Se exige picking y versión observables en cada corte, no solo fecha de pedido dentro de la partición. Los conteos son candidatos antes de filtros adicionales de características históricas; no certifican el tamaño de matrices futuras.

La selección de outcomes observados excluye demanda no completada y puede sesgar contra los casos más difíciles. Debe informarse siempre su cobertura junto a métricas futuras y no extrapolar rendimiento a indeterminados. No se interpreta la clase ausente en un mes como cero retrasos de toda la población.

## Año: línea y documento actual separados

| Unidad | Año | Elegibles | Positivos | Negativos | Positivos / elegibles |
|---|---:|---:|---:|---:|---:|
{chr(10).join(annual)}

Las tasas por documento dan el mismo peso a cada documento; por línea ponderan documentos con más líneas. No son intercambiables y no se eligió la unidad para mejorar el equilibrio. La recomendación de línea se basa en identidad estable, significado del evento y mutación de composición. Los períodos 2013/2014 usan el corte train de 2015; no se exige un corte separado al final de 2013 porque ambos años integran el mismo train.

En los estados elegibles de desarrollo, 2428 de las 2432 líneas tardías pertenecen hoy a documentos derivados de backorder y cuatro a documentos no derivados. Esto es asociación descriptiva, no prueba de causa; exige revisar linaje y reportar desempeño por familia. El indicador final de backorder sería fuga si se usara como entrada de emisión y permanece prohibido. Los traslados no se interpretan como nuevas demandas independientes.

## Mes a mes: líneas

Pedidos se refiere a documentos actuales del universo del mes; filas etiquetadas pueden representar solo parte de un documento. `line_structure_monthly` y `line_states_monthly` en agregados_desarrollo.json incluyen documentos distintos y productos por mes/estado. Las sumas de documentos entre estados no son un total válido si un documento aparece en varios estados.

| Mes | Líneas de origen | Líneas elegibles | Positivos | Negativos | Prevalencia |
|---|---:|---:|---:|---:|---:|
{month_table('line')}

## Mes a mes: documentos actuales (diagnóstico)

| Mes | Documentos de origen | Documentos etiquetables | Positivos | Negativos | Prevalencia |
|---|---:|---:|---:|---:|---:|
{month_table('document_diagnostic')}

Hay eventos positivos en cada mes de desarrollo; se dispone de {tr['positive']} en train y {va['positive']} en validación para una primera evaluación de clases raras. Eso no garantiza precisión estable por producto ni independencia estadística. Deben examinarse soportes por mes y familia. No se generaron datos sintéticos, duplicados de positivos, pesos ni balanceo.

## Censura y decisiones

La tabla de estados tiene prioridad definida: inconsistencia, horizonte incompleto, picking nulo, picking no observado al corte, versión posterior al corte y finalmente clases 0/1. Evita doble conteo. Los nulos actuales no se convierten en evidencia negativa histórica; tampoco se usan picks posteriores al corte de una partición para enriquecer sus clases. El test desde 2016 no se abre. Antes de ETL confirmar aceptación de esta población seleccionada por observación, ancla original heredada y limitación de disponibilidad de la copia.
''')
    escribir('contrato_datos.md',f'''# Contrato de datos: pedidos-1.0-especificacion

La versión estructurada está en contrato_datos.json. No hay X/y ni transformaciones ajustadas.

## Identidad y linaje

Unidad: OrderLineID estable de OLTP. Order Key es clave DW; WWI Order ID es asociación al documento actual; Backorder ID es un vínculo posterior, no predictor. Usar StockItemID y CustomerID operacionales solo como claves verificadas de asociación. No tratarlos como magnitudes continuas. Conservar IDs y estados en metadatos separados de X. Validar unicidad de cada puente DW–OLTP y no multiplicar líneas con joins.

La emisión propuesta es la demanda original por línea con fecha calendario OrderDate heredada en backorders. La hora de emisión no está observada. El modelo base solo admite información conocida por calendario o cerrada **antes del inicio de ese día**, evitando usar transacciones del mismo día ordenadas ficticiamente. El cambio a emisión real del documento hijo requiere recuperar su creación; no se sustituye silenciosamente el objetivo.

## Etiqueta, cortes y censura

Dos días calendario: DATE(p)>d+2 es 1, empate al día 2 es 0. Fin de seguimiento homogéneo: inicio de d+3. Deben cumplirse fin de horizonte <= corte, p >= inicio de d, p < corte y ambas últimas ediciones < corte. `known_at` es el máximo de esos timestamps. Las fechas inconsistentes quedan en estado invalid sin etiqueta. Nulos, picks futuros o versiones aún no observables quedan sin y; nunca se imputan.

No se acredita ausencia del evento a partir de NULL actual. Una etiqueta de retraso por ausencia al vencimiento queda deshabilitada hasta disponer de bitácora o snapshot fiable del horizonte. La disponibilidad de ingestión y la inmutabilidad de timestamps no están certificadas: este es un modo retrospectivo conservador explícito, no garantía operacional.

## Inventario de entradas

| Familia | Política | Razón |
|---|---|---|
| Calendario de d: mes/día de semana con codificación cíclica, día de mes y días del mes | Permitida en diseño base | Seis entradas definidas en contrato_datos.json; conocidas al emitir, sin usar test |
| Cantidad solicitada, precio y total de línea originales | Condicional, desactivada en base | Se asignan al crear, pero el valor actual no acredita por sí solo el inicial |
| Número de líneas, variedad y total del pedido original | Desactivada | Los traslados a backorder alteran la composición actual |
| Historial de actividad de producto/cliente | Condicional | Claves estables y hechos conocidos antes del día objetivo, sin reconstrucción futura |
| Historial de retrasos | Condicional | Solo etiquetas conocidas antes del corte de features; no basta OrderDate anterior |
| Atributos de dimensión Cliente/Ciudad/Vendedor/Producto | Desactivados hasta validar as-of | Solapamientos, desconocidos, versiones fuera de fecha y falta de hora |
| IDs | Solo asociación, grupos y metadatos | No variables continuas; codificación categórica requeriría contrato y vocabulario train-only |
| Picking objetivo, PickedQuantity objetivo, Picker, Backorder final, resultados posteriores | Prohibidos | Fuga directa del outcome |
| Últimas ediciones | Solo puertas de disponibilidad, no X | Pueden codificar el propio cumplimiento |

La ventana histórica de retrasos debe filtrar por `known_at < inicio(d)` y no por fecha del pedido. Para una línea de enero cuyo picking se registró en marzo, no hay etiqueta utilizable en historia de febrero. Se aplica también la disponibilidad de su versión persistida. No usar el vencimiento del día 3 como fecha de conocimiento del positivo en ausencia de registro fiable. Ventanas y soportes mínimos se fijarán antes de extraer, sin test.

El entrenamiento no debe heredar 20 variables, BOUNDS ni etiqueta de Ventas. La dimensión final será dinámica tras codificar y descartar constantes solo en train. Imputación, vocabularios, escalado y PCA se ajustarán únicamente en train cuando se autorice el pipeline; aquí no se calculan.

## Cliente desconocido

La auditoría histórica identificó 85469 líneas DW con cliente cero y 261 IDs OLTP sin correspondencia en Dimension.Customer. En desarrollo son {unknown['zero_customer']} líneas, {unknown['null_customer']} nulos y {unknown['operational_ids_for_unknown']} IDs fuente; las {unknown['missing_business_id_in_dimension']} líneas afectadas carecen de su ID en la dimensión. Son centinelas por ausencia de correspondencia, no 85469 identidades nuevas ni simples NULL.

No eliminar esas observaciones por cliente desconocido. En diseño base no se usan predictores de cliente; si se habilita una categoría posteriormente, mantener `desconocido` explícito y ajustar vocabulario en train. El ID CustomerID recuperable de Orders es una asociación real, no identidad inventada, pero su valor al registro y uso como predictor requieren prueba de disponibilidad. No usar la marca de desconocido de un ETL posterior como si fuera información conocida al emitir.

## Joins históricos

Reutilizar la evidencia de auditoría: City tiene 5948 pares superpuestos, Supplier uno y Employee uno. En hechos de Pedidos, 377 claves City, 2492 Picker y 2632 Salesperson no coinciden con el intervalo a medianoche. Son cifras históricas previas, no nuevos outcomes test. La vigencia a la hora exacta puede diferir de medianoche; no declararlos automáticamente errores ni usarlos como predictores seguros.

Un join permitido deberá usar ID operacional, instante as-of y [Valid From,Valid To), con exactamente una coincidencia. Cero: desconocido; más de una: ambigüedad, desactivar característica y registrar motivo. Prohibido resolver con TOP 1, clave máxima o última versión. Ni una FK válida ni un intervalo sin solapamientos acreditan que un atributo se conocía al emitir. Base sin atributos dimensionales hasta resolverlo.

## Gates para un ETL posterior

Reconciliar estados y unicidad; conservar las filas sin etiqueta con motivo; probar traslado y raíz de familias sin ciclos; confirmar la reconstrucción de fecha original; acordar censura/población; fijar entradas y ventanas efectivamente conocidas; congelar particiones y presupuesto de evaluación. Solo después crear extracción versionada de Pedidos en rutas nuevas. No modificar Ventas, auditoría anterior ni informe.
''')
    escribir('diseno_evaluacion.md',f'''# Evaluación propuesta para retrasos raros

## Particiones por demanda original, sin aleatoriedad

| Partición | Fecha del pedido original | Corte exclusivo de conocimiento |
|---|---|---|
| Train | 2013-01-01 a 2014-12-31 | 2015-01-01 00:00 |
| Validación | 2015-01-01 a 2015-12-31 | 2016-01-01 00:00 |
| Test reservado | 2016-01-01 a 2016-05-31 | Propuesta: 2016-06-01 00:00, cobertura por certificar |

Los meses coinciden parcialmente con Ventas por la cobertura real, pero la unidad, etiqueta, elegibilidad y purga son propias de Pedidos. No se usan sus matrices ni su test como entrenamiento. El último día observado en la extracción no certifica un cierre administrativo: el corte test es una propuesta y no autoriza abrirlo ahora.

Exigir d+3 días <= corte. En train se excluyen uniformemente demandas de 30–31 de diciembre de 2014 y en validación las de 30–31 de diciembre de 2015, aunque alguna ya tenga picking rápido. Un pedido del 29 vence al comenzar el corte y puede ser maduro; su picking/versión debe seguir siendo estrictamente anterior al corte. Además se excluyen del aprendizaje picks o versiones conocidos posteriormente. El embargo de dos días por sí solo no resuelve un outcome registrado meses después.

En cada fold/ventana se recalcula esta disponibilidad; no reutilizar todas las etiquetas finales de train antes del corte de un fold interno. Modelo y transformaciones se congelan al terminar train; selección solo con validación. La historia secuencial en validación/test podrá avanzar únicamente con eventos que ya se hayan conocido, sin refit del preprocesador/modelo.

Mantener juntas las líneas de cada pedido y familia de backorders; si una familia atraviesa un corte, aplicar política de exclusión/purga de la familia, sin usar sus outcomes para elegirla. Los {a['backorder_links']['links']} vínculos de desarrollo tienen la misma fecha, por lo que no se detecta ese cruce aquí. La prueba estructural se repetirá antes de materializar splits. Nada de partición aleatoria por línea ni inferencia iid.

## Métricas

Registrar soportes, matrices [TN FP; FN TP], precision/recall/F1 de clase retrasada, especificidad, BA, FPR, proporción de alertas y accuracy secundaria. FPR=FP/(TN+FP) mide negativos que reciben alerta; la fracción de falsas alertas entre alertas emitidas es FP/(TP+FP)=1-precision. No confundir denominadores.

BA es importante para comparar ambas clases, pero un valor agregado no demuestra utilidad. Reportar por mes y familia, además de cobertura/estados excluidos. No considerar exitoso un modelo por accuracy cercana a 99 % con recall de retrasos casi nulo. Train tiene {tr['positive']} positivos y validación {va['positive']}; la dependencia y heterogeneidad reducen la información efectiva.

Cuando existan puntuaciones, calcular average precision (AP) mediante incrementos de recall ponderados por precision, agrupando empates. Probabilidad logística y margen del perceptrón son puntuaciones válidas para ranking; el margen no es probabilidad. AP no es automáticamente igual a integrar por trapecios toda la curva PR. No se calculó AP aquí porque no hay predicciones nuevas. Usar AP/métricas de clase positiva como evidencia complementaria al criterio de selección previamente acordado; no elegir una métrica después de mirar test. Tasas sin denominador real se marcan no estimables conforme a las convenciones del proyecto.

## Baselines y decisiones de entrenamiento

Mayoritaria: congelar **clase 0**, determinada únicamente desde las etiquetas justificadas de train ({tr['negative']} negativas frente a {tr['positive']} positivas). Reportar que su recall de retraso será cero cuando haya positivos, sin entrenarla ahora.

Regla histórica candidata: predecir la última etiqueta de línea conocida del mismo producto; desempatar registros simultáneos determinísticamente por OrderLineID y utilizar mayoría train si no hay historia. Exigir known_at anterior al inicio del día de predicción. No está activada: comprobar antes asociación y disponibilidad; no usar una etiqueta posterior solo porque su OrderDate es anterior. Esta regla evita introducir umbrales numéricos arbitrarios para equilibrar clases.

No se fijan ponderaciones, sobremuestreo, datos sintéticos ni selección de umbral. Los algoritmos propios no incorporan sample_weight como opción intercambiable: cualquier cambio exige justificación, versión nueva y verificación de su matemática. El umbral empresarial de **dos días** define y y no debe confundirse con el umbral de clasificación de un modelo. El futuro presupuesto de tasas/épocas/umbrales se fijará con train/validación antes de una única apertura test; no se hereda automáticamente la selección de Ventas.

No estimar significación con líneas iid ni bootstrap de líneas; estudiar bloques temporales y familias con suficiente soporte si luego se autoriza inferencia. No se calculan métricas o prevalencias test para tomar estas decisiones.

## Viabilidad y pendientes

Hay ambas clases en todos los meses de desarrollo y un número suficiente para intentar una primera comparación, sin garantía de utilidad. El dictamen es **viable con ajustes explícitos** para la línea con demanda original fechada, picking observado y disponibilidad conservadora. Aceptar o rechazar esta población seleccionada y sus limitaciones de snapshot es una decisión previa al ETL. La emisión exacta de documentos hijos o el cumplimiento integral original necesitan historial adicional y no se sustituyen por el objetivo de línea.

Confirmar el significado empresarial y acción de una alerta a dos días; aceptar unidad de línea/ancla original; aprobar censura y seguimiento; certificar disponibilidad y linaje; decidir entradas iniciales e historial habilitados; fijar joins as-of y protocolo familiar. No se construyen matrices, entrenan modelos, cambian PCA, editan el informe ni realizan operaciones Git.

## Reproducción local

`python -m src.db.especificar_pedidos --server 'lpc:.\\AMBATO'` consulta solo pedidos anteriores a 2016 y sustituye únicamente esta carpeta de especificación. Usa pyodbc/conexión integrada ya existentes. Los módulos operativos guardados son texto leído de sys.sql_modules, nunca ejecutado. consultas_desarrollo.json registra SQL y parámetros; hashes y huellas acotadas constan en manifiesto/validaciones. El módulo no ejecuta el auditor global ni reentrena nada.
''')
