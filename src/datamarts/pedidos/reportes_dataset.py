"""Documenta la detención explícita; no escribe predictores o respuestas."""
import json
from src.db.auditoria_dw import BASE,guardar,sha
from .verificacion_temporal import OUT,SPEC,FEATURES,PROCESSED


def exportar(meta,data,verdict):
    contract=json.loads((SPEC/'contrato_datos.json').read_text(encoding='utf-8'))
    stats=json.loads((SPEC/'distribucion_etiquetas_desarrollo.json').read_text(encoding='utf-8'))
    features=[]
    for name in FEATURES:
        features.append({'name':name,'formula':contract['features_enabled_design'][name],'origin':'Fecha de registro original verificada de la línea, todavía sin resolver por fila','availability':'Calendario conocido al registro, condicionado a validar el ancla','authorized_by_existing_contract':True,'business_values_computed':False,'status':'authorized_definition_materialization_blocked','justification':'Solo calendario, sin información del propio día ni posterior; no se sustituye por fecha de última edición'})
    guardar(OUT/'caracteristicas_autorizadas.json',{
        'calendar_features':features,'feature_order':list(FEATURES),'all_six_retained_in_design':True,'preprocessor':{'fitted':False,'reason':'No hay train temporalmente validado; no ajustar imputación, z-score ni constantes sobre candidatos ambiguos'},
        'disabled':[{'feature':n,'enabled':False,'reason':why} for n,why in [('Quantity','El valor actual no certifica la cantidad original; no hay versiones de línea identificadas'),('UnitPrice','Asignado al insertar según módulo, pero el último estado puede ser posterior; no hay valores originales trazados'),('Atributos dimensionales','Relación as-of no garantizada, se mantienen desactivados'),('Cliente reconstruido/categorías','No se inventan identidades; no se usan en la base'),('Historia y agregados del propio día','Desactivados; estadísticas futuras solo podrían usar información anterior al día de registro')]],
        'test_policy':'No leer picking de 2016 ni seleccionar líneas test por su resultado. Sin ancla verificada, no se genera X_test tampoco.'})
    inclusion={'decision':verdict['status'],'included':0,'development_candidates_blocked':data['identity']['lines'],'reason':'No se identificó registro de creación independiente por OrderLineID. No se afirma que todas las líneas hayan sido creadas tarde.',
        'per_partition':[],'test':'No leído ni filtrado por outcome; predictores no materializados',
        'limited_metadata_sample':'trazabilidad_desarrollo.json: metadata_sample_no_picking; IDs/fecha nominal/última edición, sin picking, y ni cantidades/precios',
        'full_row_export':'No se vuelcan todas las filas ni se fabrica un inventario de matrices ante un bloqueo general; se conservan conteos exhaustivos de candidatos y motivos globales.'}
    by_partition={r['partition']:r for r in data['restricted_diagnostic_not_dataset']}
    for split in ['train','validation']:
        q=stats['line'][split];r=by_partition[split]
        inclusion['per_partition'].append({'partition':split,'source_lines':q['total'],'previously_label_eligible':q['eligible'],'previous_positive':q['positive'],
            'included_in_dataset':0,'source_lines_blocked_by_anchor':q['total'],'same_day_edit_diagnostic_only':r['same_day_edit'],'same_day_positive_diagnostic_only':r['positive_with_same_day_edit'],
            'reason':'Puerta de existencia al registro sin evidencia. Última edición del mismo día es una cota de presencia durante ese día, no el alta ni la hora de emisión.'})
    guardar(OUT/'observaciones_incluidas_excluidas.json',inclusion)
    modules_path=SPEC/'modulos_operativos.json'
    modules=json.loads(modules_path.read_text(encoding='utf-8'))
    snippets=[]
    for m in modules:
        text=m['definition'].replace('\r','')
        terms=['-- move the items that haven','-- create the backorder order','-- reserve the required stock'] if m['schema_name']=='DataLoadSimulation' else (['INSERT Sales.OrderLines','INSERT Sales.Orders'] if m['module_name']=='InsertCustomerOrders' else [])
        for term in terms:
            pos=text.find(term)
            if pos>=0:snippets.append({'schema':m['schema_name'],'module':m['module_name'],'search_term':term,'excerpt':text[max(0,pos-80):pos+1400]})
    guardar(OUT/'evidencia_codigo.json',{'source':str(modules_path.relative_to(BASE)),'source_sha256':sha(modules_path),'executed':False,'interpretation':'Definiciones leídas como texto. Explican posibles operaciones; no son una bitácora que certifique cada fila histórica.','excerpts':snippets})
    rows='\n'.join(f"| {x['partition']} | {x['source_lines']} | {x['previously_label_eligible']} | {x['previous_positive']} | {x['same_day_edit_diagnostic_only']} | {x['same_day_positive_diagnostic_only']} |" for x in inclusion['per_partition'])
    history='\n'.join(f"| {x['table_name']} | {x['temporal_type_desc']} | {x['is_tracked_by_cdc']} | {x['change_tracking']} | {x['history_table']} |" for x in meta['history_configuration'])
    report=f'''# Verificación temporal del dataset de Pedidos

## Decisión: construcción detenida

No se generaron X_train, y_train, X_val, y_val, X_test ni parámetros de preprocesamiento. La condición nueva de demostrar existencia al registro no queda resuelta por la copia actual. El usuario aceptó provisionalmente la población con picking observado; esa aceptación no demuestra que cada línea existiera en su fecha heredada ni autoriza a inventar un alta. Se aplica la instrucción de detenerse si la ambigüedad afecta la validez general.

La especificación anterior permanece intacta: sus conteos son candidatos retrospectivos, no matrices ya validadas. No se cambió el problema a predecir en la fecha del último cambio, picking, facturación o creación del documento hijo.

## Fuentes consultadas y trazabilidad actual

Se reutilizó AMBATO/pyodbc/autenticación integrada, emitiendo solo SELECT. Únicamente se consultaron metadatos de Orders/OrderLines, catálogos de posibles historiales y agregados de Pedidos de desarrollo con `OrderDate < 2016-01-01`. No se repitió la auditoría de Compras, Inventario o Transacciones. No se leyeron outcomes de test.

En desarrollo hay {data['identity']['lines']} filas, {data['identity']['order_line_ids']} OrderLineID distintos y {data['identity']['current_order_ids']} OrderID actuales. El puente DW por OrderID actual y StockItemID operacional enlaza {data['dw_exact_bridge']['matched_lines']} líneas distintas con {data['dw_exact_bridge']['dw_rows']} filas DW, sin ausentes ({data['dw_exact_bridge']['no_source']}) ni discrepancias de fecha ({data['dw_exact_bridge']['date_discrepancies']}). La pareja actual OrderID+StockItemID tiene {data['duplicate_current_order_stock']['excess']} repeticiones adicionales. Es trazabilidad de identidad **actual**, no prueba de existencia en la fecha de emisión.

Hay {data['backorder_lineage']['edges']} vínculos de backorders de desarrollo, todos con fecha heredada igual y claves crecientes. La plantilla de simulación guardada define mover las líneas pendientes manteniendo OrderLineID, cambiando OrderID y LastEditedWhen, y copiar OrderDate del padre al hijo. La fecha heredada no es la fecha real de creación del hijo y la pertenencia actual tampoco acredita la composición original. Conservar la clave de línea ayuda a vincular filas, pero una secuencia numérica no es un timestamp ni un log de inserción. Las definiciones se leyeron como texto y nunca se ejecutaron; no prueban qué operaciones se ejecutaron históricamente sobre cada fila.

## Ausencia de evidencia de creación por fila

OrderLines solo tiene PickingCompletedWhen y LastEditedWhen como timestamps; no tiene CreatedAt/CreatedWhen o equivalente en su esquema. LastEditedWhen tiene default sysdatetime(), pero puede sobrescribirse al recoger o trasladar una línea. Un default de inserción no conserva su valor inicial tras una actualización. Las descripciones de estas columnas están en metadatos_creacion.json y no las convierten en fecha de alta.

| Tabla | Temporal | CDC actual | Change Tracking actual | History table |
|---|---|---|---|---|
{history}

No hay triggers registrados en estas dos tablas. Los catálogos sí muestran archivos históricos para otras entidades, pero no una historia vinculada de Orders/OrderLines que documente sus inserciones y traslados. La visibilidad de metadatos se registra para no confundir falta de permisos con inexistencia. Esto no descarta copias externas, logs antiguos o archivos no proporcionados; tampoco demuestra que CDC/temporal nunca hayan existido. Significa que no se identificó evidencia recuperable suficiente en las fuentes de esta tarea.

No se puede contar cuántas líneas se crearon realmente después de OrderDate: falta precisamente esa fecha. Última edición posterior no implica creación posterior, y última edición el mismo día tampoco prueba la hora ni el valor inicial al emitir. No se etiquetan registros como «creados tarde» sin evidencia.

## Por qué no basta excluir los registros con última edición posterior

| Partición | Líneas de origen | Antes etiquetables | Positivos previos | Última edición el mismo día, diagnóstico | Positivos en ese diagnóstico |
|---|---:|---:|---:|---:|---:|
{rows}

Este es un contraste de trazabilidad, **no un dataset alternativo**. Si se tomara únicamente última edición el mismo día como sustituto de alta, el subconjunto de outcomes observados perdería todos los positivos. Es esperable: una fecha de picking posterior al día 2 viene acompañada por una actualización posterior. Seleccionar así usando el último estado introduce selección por el resultado; no permite concluir que las líneas descartadas no existieran al registrar la demanda.

No se fuerza la recuperación de los 126341/71112 registros explorados ni se presenta un problema monoclase como clasificación binaria válida. La puerta exige evidencia independiente por línea y bloquea los {data['identity']['lines']} candidatos actuales. Son candidatos no materializados, no una afirmación de inexistencia ni una eliminación empresarial. Los motivos agregados y una muestra de doce IDs sin picking se conservan en observaciones_incluidas_excluidas.json y trazabilidad_desarrollo.json.

## Características

Se conservan las seis definiciones autorizadas: mes_sin=sin(2*pi*(mes-1)/12), mes_cos=cos(...), semana_sin=sin(2*pi*weekday/7), semana_cos=cos(...), con lunes=0; dia_mes y dias_mes con bisiestos. Son calendario conocido para una fecha válida de registro. No se calculan sobre registros empresariales con ancla incierta y no se sustituye d por una fecha posterior.

Quantity y UnitPrice se asignan al insertar según el módulo Website, pero no hay una versión inicial trazada por OrderLineID ni garantía de inmutabilidad en los metadatos. Se mantienen desactivados, incluso si sus valores actuales fueran finitos o plausibles. Los agregados de calidad actuales no certifican disponibilidad original. También siguen desactivados atributos dimensionales ambiguos e historia; no se rellenan ni inventan clientes. Solo se considera historia anterior al día de emisión si una futura ancla se valida; ningún agregado del propio día se habilita.

No se ajustan medianas, escaladores o constantes porque no existe un train temporalmente validado. Cuando se resuelva la puerta, los seis predictores se conservarán inicialmente y cualquier descarte posterior se decidirá solo con train, con dimensión dinámica. No se entrenan clasificadores ni se ajusta PCA.

## Etiqueta y test

La regla aprobada no cambia: picking hasta d+2 inclusive es 0; posterior es 1; ausencia o inconsistencia queda indeterminada. Se conservan los cortes históricos y `known_at=max(picking,edición línea,edición encabezado)` sin admitir outcomes o versiones posteriores al corte. La nueva condición de existencia se aplica antes de esa elegibilidad: tener una etiqueta retrospectiva no demuestra que el predictor/observación existiera en t0.

No se construye X_test: tampoco se dispone de ancla certificada por línea de 2016, y seleccionarlas por disponibilidad posterior de picking sería indebido. No se consulta su picking, prevalencia o métrica. Su futuro protocolo de evaluación debe seleccionar predictores sin outcomes y resolver por separado la cobertura final de respuestas, conservando el sesgo por picking observado.

## Evaluación conservada y siguiente evidencia necesaria

Siguen documentadas BA, precision/recall/F1 del retraso, AP con puntuaciones cuando existan, especificidad, confusión y proporción de alertas. Mayoritario=0 congelado desde train del contrato, sin ejecutar evaluación ni elegir umbrales, pesos o remuestreo. No se extrapolará la población retrospectiva a todos los pedidos.

Para reanudar se necesita un registro independiente de inserción por OrderLineID (CreatedAt verificable o log/snapshot al registro), fecha de referencia y origen de demanda, historial de reasignaciones de backorder y, si se desean habilitar, cantidades/precios originales. Un log de cargas que solo demuestra que la línea existía al cargar el DW no certifica que existiera en OrderDate. Debe comprobarse además que el propio log no es una fecha rellenada a posteriori. Alternativa: acordar explícitamente otro instante/objetivo sustentado por los eventos disponibles, sin reemplazo silencioso del contrato actual.

La instrucción de «construir cuando exista evidencia suficiente» se resuelve mediante esta detención documentada. No se crean matrices vacías, fechas inferidas, registros empresariales sintéticos ni un preprocesador aparente. La carpeta datos/Procesados/pedidos no se crea para simular un entregable inexistente.

## Reproducción y controles

`python -m src.datamarts.pedidos.verificacion_temporal --server 'lpc:.\\AMBATO'` repite solo estas comprobaciones y produce este bloqueo. No existe un flag para ignorarlo o forzar matrices. La aceptación de otro tipo de evidencia exige validar su procedencia, no editar un booleano para eludir la condición.

Consultas y parámetros están en consultas_metadatos.json y consultas_desarrollo.json. Se compara una huella COUNT/CHECKSUM de desarrollo antes/después: control acotado no criptográfico, no certificación de ausencia de cambios concurrentes de toda la base. El registro contiene solo SELECT. Se conservan por hashes todos los archivos previos, incluidas la auditoría, especificación, Ventas y el PDF previamente modificado por el usuario. No se realizan operaciones Git.
'''
    (OUT/'reporte_verificacion_temporal.md').write_text(report,encoding='utf-8',newline='\n')
