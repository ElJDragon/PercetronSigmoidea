"""Contrato y dictamen de Compras a partir de evidencia local de desarrollo."""
import csv
import json
from .compras_viabilidad import OUT, BASE, load
from src.db.auditoria_dw import guardar, sha


def catalogo():
    rows=[]
    def add(name,formula,source,grain,available,risk,status):
        rows.append(dict(nombre=name,formula=formula,fuente=source,granularidad=grain,disponibilidad=available,riesgo=risk,estado=status))
    for k in [1,2,3]:
        add(f'cantidad_lag_{k}',f'Q(e,t-{k})','PurchaseOrderLines.OrderedOuters × StockItems.QuantityPerOuter histórico','producto-mes',f'Antes del inicio de t; todas las versiones de t-{k} acreditadas','Cantidad mutable; altas tardías; unidad y cobertura','condicionado_desactivado')
        add(f'ordenes_lag_{k}',f'COUNT DISTINCT PurchaseOrderID(e,t-{k})','PurchaseOrders.OrderDate, SupplierID; líneas solo para producto','proveedor-mes; producto-mes condicionado',f'Antes del inicio de t; identidad y mes de negocio acreditados','No confundir orden con línea; reasignación o alta retroactiva','condicionado_desactivado')
    for name,formula in [('cantidad_media_3','(q1+q2+q3)/3'),('cantidad_mediana_3','mediana(q1,q2,q3)'),('cantidad_std_3','std(q1,q2,q3), ddof=0'),('cantidad_tendencia_3','(q1-q3)/2'),('cantidad_cambio_1','q1-q2'),('cantidad_variacion_relativa','(q1-q2)/q2 si q2>0; indeterminada en otro caso'),('frecuencia_media_3','(n1+n2+n3)/3'),('meses_activos_3','SUM 1[nk>0], k=1..3')]:
        add(name,formula,'Agregados previos acreditados','Entidad-mes', 'Antes del inicio de t','Hereda los bloqueos de lags y ceros','condicionado_desactivado')
    add('antiguedad_observada_meses','ordinal(t)-ordinal(primer mes observable antes de t)','Registro histórico de identidad','Entidad-mes','Antes de t, sin usar última compra futura','Primera observación no equivale a alta real; supervivencia','condicionado_desactivado')
    for name,formula in [('mes_sin','sin(2*pi*(mes(t)-1)/12)'),('mes_cos','cos(2*pi*(mes(t)-1)/12)'),('dias_mes','días calendario de t')]:
        add(name,formula,'Calendario civil','Entidad-mes','Conocida antes de t','Bajo; no certifica cobertura','permitido_candidato_sin_dataset')
    for name,formula in [('recepcion_y_finalizacion','ReceivedOuters, LastReceiptDate, IsOrderLineFinalized, IsOrderFinalized'),('cantidades_mes_objetivo','Q(e,t), N(e,t), referencia calculada con t'),('claves_numericas','Supplier Key, Stock Item Key, IDs como magnitudes'),('dimension_futura','Atributos fuera de un intervalo único válido al instante'),('ultima_edicion_como_alta','LastEditedWhen como fecha de creación')]:
        add(name,formula,'OLTP/DW','Línea/documento/entidad','No acreditada en el momento requerido','Fuga, semántica o identidad','prohibido')
    return rows


def write(name,text):
    (OUT/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')


def exportar():
    d=load('comprobaciones_focalizadas.json');g=load('granularidades.json');c=load('etiquetas_candidatas.json')
    features=catalogo();guardar(OUT/'catalogo_caracteristicas.json',features)
    decision={'dataset_authorized':False,'training_viable_now':False,'recommended_objective_for_future_contract':'frecuencia mensual de órdenes por proveedor superior a mediana de tres meses previos',
              'recommended_granularity':'proveedor-mes, sin sumar unidades heterogéneas',
              'recommended_target':'1[N(s,t)>median(N(s,t-1),N(s,t-2),N(s,t-3))]',
              'status':'insufficient_historical_availability_and_coverage_for_training',
              'authorized_observations':{'train':0,'validation':0},
              'reason':'No hay reconstrucción de cantidades originales o estado de órdenes al inicio de mes. El control conservador elimina toda la historia activa de los escenarios sin ceros. Los paneles con ausencias como cero dependen de cobertura no certificada y no dan soporte positivo en validación tras el control.',
              'next_separate_task':'Transacciones financieras','work_order':['Compras','Transacciones financieras','Inventario/Movimientos'],
              'test_2016_used_for_design':False,'matrices_generated':False,'training_performed':False,'pca_performed':False}
    guardar(OUT/'dictamen.json',decision)
    contract={'status':'propuesto_bloqueado_no_construir','observation':'business SupplierID + target month; candidate only',
              'prediction_time':'primer instante del primer día calendario de t, hora empresarial local',
              'business_anchor':'OrderDate propio de PurchaseOrders; fecha empresarial diaria, no inserción auditada',
              'target':decision['recommended_target'],'quantity_candidate':'SUM OrderedOuters * QuantityPerOuter histórico, solo dentro del mismo producto',
              'tie_class':0,'missing_or_unknown':'indeterminado; nunca imputar cero por defecto',
              'zero_policy':'desconocida por defecto; conditional_ledger es sensibilidad explícita, no cobertura autorizada',
              'universe':'identidades observables antes de t con al menos tres meses completos previos; no seleccionar por actividad futura',
              'label_available':'mes t cerrado y todos los registros que lo componen conocidos antes del corte de la partición',
              'historical_available':'cada agregado completo de t-1,t-2,t-3 acreditado antes de inicio(t); un registro conocido tarde invalida el agregado, no se elimina solo esa fila',
              'version_gate':'max LastEditedWhen relevante < inicio(t) para historia; < corte para target. Condición conservadora necesaria de esta verificación, no prueba de creación ni de completitud.',
              'splits':{'train':['2013-04','2014-12'],'validation':['2015-01','2015-12'],'reserved':'2016, sin lectura de órdenes/outcomes de ese año para diseño'},
              'cuts':{'train':'2015-01-01 00:00:00 exclusivo','validation':'2016-01-01 00:00:00 exclusivo'},
              'identity_policy':'IDs de negocio para agrupación; PK sustituta para cotejar identidad, no atributos. Atributos solo si join as-of único; nunca escoger versión futura ni la clave mayor.',
              'features':features,'can_build_dataset':False,'requires':['Evidencia de registro y composición/versiones históricas de órdenes al inicio de cada mes','Cierres y cobertura verificables antes de convertir ausencia en cero','Soporte temporal y variación dentro de entidades, sin filtrar por futuro'],
              'decision':decision}
    guardar(OUT/'contrato_datos.json',contract)
    with (OUT/'distribuciones_mensuales.csv').open('w',encoding='utf-8',newline='') as fh:
        fields=['candidate','zero_policy','scenario','split','month','eligible','positive','negative','positive_proportion']
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
        for name,cc in c.items():
            for scenario,ss in cc['scenarios'].items():
                for split,summary in ss.items():
                    for row in summary['monthly']:writer.writerow(dict(candidate=name,zero_policy=cc['zero_policy'],scenario=scenario,split=split,**row))
    # Metadatos y código son evidencia semántica, nunca se ejecutan módulos SQL.
    modules=load('modulos_compras.json');evidence=[]
    for r in modules:
        text=r['definition'] or '';snippets=[]
        needles=['INSERT Purchasing.PurchaseOrders','INSERT Purchasing.PurchaseOrderLines','SET ReceivedOuters = OrderedOuters','pol.OrderedOuters * si.QuantityPerOuter']
        for needle in needles:
            pos=text.find(needle)
            if pos>=0:snippets.append({'needle':needle,'excerpt':text[max(0,pos-100):pos+1550]})
        evidence.append({'schema':r['schema_name'],'module':r['module_name'],'executed':False,'snippets':snippets})
    guardar(OUT/'evidencia_semantica.json',{'source_sha256':sha(OUT/'modulos_compras.json'),'modules':evidence})
    gran_table='\n'.join(f"| {name} | {v['identities']} | {v['active_entity_months']} | {v['ledger_month_slots_after_first_observation']} | {v['unobserved_month_slots_conditional_not_certified_zero']} |" for name,v in g.items())
    write('semantica_compras.md',f'''# Compras: semántica y temporalidad

## Alcance

Se leyó primero la auditoría general existente. La nueva consulta es focalizada: Purchasing.PurchaseOrders, PurchaseOrderLines, Suppliers, Warehouse.StockItems únicamente para unidad, Fact.Purchase y sus dimensiones Supplier/Stock Item. No se investigaron Pedidos, Transacciones ni Inventario. No se ejecutó ningún módulo de integración o simulación. Todas las consultas propias son SELECT; readonly/ApplicationIntent no sustituyen permisos de servidor. Se preserva el informe y todos los entregables anteriores.

## Grano y relaciones demostradas

Fact.Purchase es una **línea de orden**, no una orden ni un recibo individual. En desarrollo hay {d['facts']} hechos, {d['lines']} líneas y {d['orders']} órdenes, hasta {d['max_lines_per_order']} líneas por orden. Purchase Key es la PK del hecho. El DW no conserva PurchaseOrderLineID; la pareja (WWI Purchase Order ID, StockItemID de negocio) es única en esta copia: hay {d['duplicate_order_item_pairs']} parejas repetidas y el cotejo constituye una biyección, sin discrepancias de fecha, proveedor, cantidades o estado. No se presupone que dicha pareja sea una restricción única general del sistema.

Se enlaza Fact.Purchase a cada dimensión por su PK sustituta para recuperar únicamente el ID de negocio, luego a PurchaseOrders por ID y PurchaseOrderLines por orden+producto. SupplierID enlaza Purchasing.Suppliers. Esta identidad actual no acredita que todos los atributos actuales fueran conocidos al emitir.

## Fechas, cantidades y estados

Integration.GetPurchaseUpdates asigna Date Key=CAST(po.OrderDate AS date), Ordered Outers=pol.OrderedOuters, Ordered Quantity=pol.OrderedOuters*si.QuantityPerOuter, Received Outers=pol.ReceivedOuters e Is Order Finalized=pol.IsOrderLineFinalized. El nombre DW de finalización se refiere a la **línea**, no a la cabecera. La extracción incremental usa el máximo de las últimas ediciones de cabecera y línea; no crea snapshots de cada valor histórico.

OrderDate es fecha de negocio de la orden, con precisión diaria. La plantilla local de simulación inserta cabecera y líneas con @StartingWhen y CAST(@StartingWhen AS date). Es evidencia del mecanismo posible, no prueba de la hora real de inserción de cada fila. No hay columna de creación auditada. Las {d['orders_edited_after_order_date']} cabeceras se editaron después de su OrderDate. **LastEditedWhen no es fecha de alta**: aquí solo sirve para una comprobación conservadora de disponibilidad de la versión persistida.

LastReceiptDate es última recepción registrada, no necesariamente primera recepción; ExpectedDeliveryDate es fecha esperada actual, no promesa original auditada. La plantilla de recepción actualiza ReceivedOuters, finalización y última recepción después de emitir. En la copia actual de desarrollo hay {d['current_finalized_lines']} líneas finalizadas, {d['current_partial_lines']} parcialmente recibidas y {d['current_no_receipt_lines']} sin recepción. Esto no prueba que nunca existieran recepciones parciales. Hay {d['receipt_before_order']} recepciones anteriores a la fecha de orden. No se construye un target de recepción tardía: hay {d['late_expected_receipt_development_observed_before_cut']} positivos observados antes del corte frente a la fecha esperada actual; tampoco se rescata cambiando el umbral a uno o dos días.

PurchaseOrders y PurchaseOrderLines son no temporales, sin CDC, change tracking ni triggers observados; no permiten reconstruir modificaciones o altas/bajas pasadas. **OrderedOuters no se certifica como cantidad original.** El código de simulación observado no basta para probar inmutabilidad universal. ReceivedOuters y finalizaciones son estados actuales posteriores. Sus valores actuales no se habilitan como X.

La historia temporal de StockItems anterior a 2016 sí se leyó exclusivamente para QuantityPerOuter: {d['product_unit_changes_before_2016']} productos con cambios de conversión y ninguna discrepancia de unidad al OrderDate para las líneas examinadas. Cada producto conserva PackageTypeID en esta muestra. Eso permite interpretar unidades dentro del mismo producto, pero no sumar unidades de productos diferentes como una magnitud física homogénea. La unidad correcta de frecuencia es **órdenes distintas**, no líneas ni bultos.

## Identidad y dimensiones

Hay 219 productos de negocio, seis proveedores de compra y 219 pares producto-proveedor: cada producto se observa con un único proveedor en desarrollo. Agrupar por clave sustituta fragmentaría identidades; IDs no se convierten en números continuos. No hay miembros desconocidos en los hechos de compra examinados.

Las dimensiones se leen con Valid From anterior a 2016. Supplier presenta un par de intervalos superpuestos para el ID 3, que no pertenece a los seis proveedores comprados; no se inventa que afecte a todos. Hay 94 hechos cuyo Supplier Key apunta a una versión no vigente al OrderDate a medianoche. La precisión diaria y actualizaciones posteriores impiden usar directamente ese atributo como predictor. Stock Item tiene cero solapamientos y cero hechos fuera de vigencia al OrderDate en este análisis. Suppliers operacional es temporal, pero esa historia no reconstruye las órdenes ni sus cantidades. El contrato deshabilita atributos dimensionales: un futuro join tendría que devolver exactamente una versión válida al instante, sin escoger arbitrariamente la clave mayor.

## Granularidad y cobertura

| Grano | Identidades | Meses-entidad con actividad | Casillas desde primera observación | Ausencias en esas casillas |
|---|---:|---:|---:|---:|
{gran_table}

Los 36 meses de desarrollo tienen órdenes y el cotejo DW/OLTP es completo **para las filas persistidas actuales**. Eso demuestra ausencia observada en el libro leído, no certifica cierres, ausencia de borrados, latencia o actividad real nula. Las casillas vacías se mantienen desconocidas por defecto. El escenario conditional_ledger supone explícitamente un libro mensual completo y evalúa ceros solo después de primera observación; no autoriza esos ceros. El escenario observed_only no fabrica ceros, pero condiciona a actividad futura del mes objetivo y sirve solo como diagnóstico de selección.

209/219 productos y cuatro/seis proveedores aparecen en un único mes, todos observados inicialmente en enero de 2013; siete productos aparecen los 36 meses. No se eliminan entidades usando su vida futura. Los pares no añaden información de cambio de proveedor. Producto-mes conserva unidad física consistente, pero es extremadamente intermitente; proveedor-mes es apropiado para frecuencia de órdenes, con muy pocas series efectivamente activas. La elección no se basa en generar más filas.

## Reproducción

`python -m src.datamarts.compras_viabilidad discover` lee metadatos; `extract` lee exclusivamente órdenes con OrderDate<2016-01-01 y hechos con Date Key<2016-01-01; `analyze` y `python -m src.datamarts.reportes_compras` son offline. Las fechas de fin de vigencia no son outcomes ni fijan el universo futuro. Algunas órdenes de diciembre de 2015 tienen última edición/recepción posterior al corte: se registra su inelegibilidad, sin construir prevalencias de órdenes de 2016. Las huellas COUNT/CHECKSUM antes/después no cambiaron; son controles no criptográficos, sin aislamiento snapshot, y no certifican historia ni permisos. Las consultas y evidencias se conservan en JSON.
''')
    rows=[]
    for name,cc in c.items():
        for scenario,ss in cc['scenarios'].items():
            for split,v in ss.items():
                pct='No estimable' if v['positive_proportion'] is None else f"{v['positive_proportion']*100:.4f}%"
                rows.append(f"| {name} | {scenario} | {split} | {v['eligible']} | {v['positive']} | {v['negative']} | {pct} | {v['entities_switching_class']} | {v['entities_single_class']} |")
    write('analisis_etiquetas.md','''# Compras: análisis de etiquetas candidatas

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
'''+'\n'.join(rows)+'''

## Interpretación, censura y soporte

El volumen producto-mes aparenta 4.590/2.619 casos train/validación con 168/89 positivos si se aceptan todos los ceros. Sin ceros solo quedan 162/99 casos observados, con 160/89 positivos. El cambio de prevalencia de ~4% a ~99%/~90% muestra cuánto manda el supuesto de actividad: ambos resultados son descriptivos, no evidencia de un problema operativo resuelto. En el panel con ceros hay 4.420/2.520 targets cero y 4.422/2.520 referencias cero; la referencia cero vuelve la regla casi un detector de actividad. Un aumento desde cero es positivo por la fórmula, no una señal demostrada de valor empresarial.

Tras comprobar disponibilidad, el escenario de volumen con ceros conserva 4.423/2.520 casillas, con 3/0 positivos; la validación queda monoclase. Sin ceros, el volumen conserva 0/0. Para frecuencia, la sensibilidad retrospectiva ofrece 124/70 casos con 18/7 positivos; sin ceros, 40/22 con 18/7 positivos. Tras comprobar disponibilidad, solo quedan 84/48 ceros de cuatro proveedores inactivos, y sin ceros quedan 0/0. Tener dos clases en los datos retrospectivos no compensa la ausencia de historia acreditada y la concentración en dos proveedores.

No se rellenan meses previos al primer registro. Los tres meses completos previos deben existir y estar acreditados antes de t. No se purgan fronteras eliminando solo positivos ni se filtra una entidad por su actividad posterior. La disponibilidad del objetivo exige además cierre del mes y versión anterior al corte del split. Se guardan motivos no excluyentes de inelegibilidad en etiquetas_candidatas.json. La selección por versiones actuales es también un riesgo de censura y supervivencia: el resultado no estima desempeño de la población original.

Las distribuciones mensuales completas están en distribuciones_mensuales.csv y etiquetas_candidatas.json, con soporte, positivos, negativos y proporciones. Los JSON incluyen entidades de una clase, entidades que cambian, ceros y razones de exclusión. Meses sin casos elegibles no tienen prevalencia estimable. No se calculan métricas de modelos, baselines ajustados o prevalencias de 2016.

## Dictamen

**No autorizar dataset ni entrenamiento ahora.** Hay cero observaciones autorizadas, porque los conteos que incorporan ceros son condicionales a cobertura no acreditada, y las series activas sin ese supuesto no tienen historia disponible demostrable al instante propuesto. Esto es insuficiencia de evidencia para el protocolo, no prueba de que ninguna cantidad original existiera ni de que Compras sea universalmente impredecible. Se detiene aquí y se recomienda Transacciones financieras como próximo encargo separado, antes de Inventario/Movimientos.
''')
    feature_table='\n'.join('| '+' | '.join(str(r[k]) for k in ['nombre','formula','fuente','granularidad','disponibilidad','riesgo','estado'])+' |' for r in features)
    write('contrato_datos.md','''# Compras: contrato propuesto, bloqueado

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
'''+feature_table+'''

Los atributos de proveedor/producto siguen desactivados salvo identidad para agrupación. Para habilitarlos, un join as-of debe devolver una sola versión válida antes de la predicción. Cero o múltiples matches implican indeterminación; nunca escoger la mayor PK, una versión posterior o un nombre actual. El hecho puede referenciar una dimensión resuelta al editar/recibir y no al emitir.

## Condiciones pendientes y autorización

Se requiere evidencia histórica del registro y de los valores de órdenes al inicio de mes, cierres verificables para los ceros, y soporte positivo/negativo útil sin depender de cuatro proveedores inactivos o de actividad futura. Actualmente: **0 observaciones autorizadas en train y 0 en validación**. Los conteos de análisis son escenarios diagnósticos, no una entrega de dataset.

No se propone adaptar silenciosamente el instante a después de recibir, desplazar ventanas para rescatar clases ni cambiar umbrales por balance. Se preservan Ventas y todos los contratos previos. Se recomienda pasar a Transacciones financieras en un encargo nuevo y dejar Inventario/Movimientos al final.
''')
    write('diseno_evaluacion.md','''# Compras: diseño de evaluación condicionado

**Protocolo propuesto; no se ejecuta porque dataset y entrenamiento no están autorizados.** No hay ajuste de modelos, generación de matrices, cálculo nuevo de PCA ni apertura del test de 2016. Compras precede a Transacciones financieras y luego Inventario/Movimientos.

## Partición y disponibilidad

Instante de predicción: inicio de t. Horizonte: mes calendario t completo. Train=objetivos abril 2013..diciembre 2014, validación=2015, reserva=2016 sin uso para diseñar o seleccionar. La fecha empresarial solo tiene precisión diaria. Es necesario demostrar que la identidad, los tres meses previos y todos sus valores eran conocidos antes del instante. Los outcomes de train deben conocerse antes de enero 2015 y los de validación antes de enero 2016; los de frontera no maduros quedan indeterminados. No se usa estado final de diciembre conocido en enero como etiqueta disponible antes del corte. La validación puede usar historia de meses anteriores de 2015 si ya era conocida, sin reajustar modelos o preprocesador.

El conjunto de entidades elegibles se define usando información anterior a t, sin exigir compra futura o persistencia hasta fin del período. No se realiza división aleatoria. Se comprueba unicidad de entidad-mes, continuidad de identidad y ausencia de solapamiento de claves entre particiones. No se seleccionan meses o entidades porque equilibren etiquetas. Las filas repetidas de una orden no cuentan como órdenes distintas y no se multiplican mediante joins dimensionales.

## Baselines, métricas y selección

Baseline mayoritario: clase más frecuente calculada exclusivamente en train autorizado, con empate fijado en 0, congelada para validación y test. Baseline causal de persistencia: para frecuencia, predicción=1[N(s,t-1)>mediana(N(s,t-2),N(s,t-3),N(s,t-4))]. Requiere un cuarto mes de historia disponible y exactamente la misma población de comparación; se informa la pérdida adicional o se compara a todos los métodos en esa intersección predefinida. No se usa N(t) para predecir su propia clase. Para actividad: 1[N(s,t-1)>0]. No se ajusta ningún baseline en este encargo.

Principal: balanced accuracy, solo estimable cuando el soporte contiene ambas clases. Secundarias: accuracy, precision, recall, F1, matriz de confusión en orden [0,1], soporte y prevalencias por mes y proveedor. Precision/F1 con denominador nulo conservan las convenciones existentes; recall o balanced accuracy no estimables deben indicarse, no maquillarse. En meses monoclase se informa soporte y limitación. Average Precision es opcional si hay clase positiva suficiente y scores con orientación positiva: sigmoid de logística o margen firmado del perceptrón, sin interpretar este último como probabilidad calibrada. No se implementa aquí una métrica ausente ni se usa test para fijarla.

Selección de hiperparámetros y umbral solo en desarrollo, según protocolo congelado antes de test. Toda escala/imputación se ajusta solo en train. Métricas mensuales junto con métricas globales; distinguir cambios dentro de entidad de memorizar proveedor activo/inactivo. Generalización temporal 2013–2014→2015 no equivale a proveedores nuevos. Reportar entidades nuevas/observadas, grupos de frecuencia y cobertura, sin usar esas categorías futuras para seleccionar. Con solo dos series proveedor activas no se presume generalización amplia ni independencia de las observaciones. Si se estimara incertidumbre, usar bloques temporales y respetar entidad; no bootstrap i.i.d. de filas como evidencia de independencia.

## Reutilización real del código

| Componente | Reutilización | Adaptación futura necesaria |
|---|---|---|
| src/models/perceptron.py, PerceptronSimple | Algoritmo NumPy sin cambios | Recibir matrices autorizadas y catálogo propio |
| src/models/logistic_regression.py, LogisticRegressionManual | Sigmoid, pérdida y gradientes sin cambios | Orquestación/datos específicos |
| src/evaluation/metricas.py | metricas_binarias y convenciones 0/1 sin cambios | Scores/AP solo si se decide e implementa aparte |
| src/visualization/pca.py, PCA2D | SVD y transform sin cambios | Fit solo train, exclusivamente visualización futura |
| src/pipeline/preprocesamiento.py | Reglas estadísticas reutilizables; clase actual **no enchufable sin cambios** | PreprocesadorComun importa FEATURES/OPTIONAL de Ventas y exige ese catálogo exacto; necesita adaptador configurable separado que preserve Ventas |
| src/evaluation/graficos.py, gráficos logísticos y dispersión | Funciones de dibujo aprovechables tras revisar interfaz | Etiquetas, títulos y rutas de Compras en orquestador propio; no ejecutar ahora |
| src/experiments/evaluacion_final.py | Principios de congelación y evaluación | Contiene rutas, hashes, metadatos, baselines y contrato específicos de Ventas; no ejecutar/importar su run como evaluación universal |
| src/pipeline/caracteristicas.py y particiones.py | Principios de ventanas y cobertura | FEATURES, medidas, etiquetas y BOUNDS son de Ventas: no importar como contrato de Compras |

Los únicos módulos nuevos requeridos son extracción SQL de Compras, conciliación de identidad/unidad, agregación mensual con cobertura, etiqueta y características históricas, elegibilidad temporal y su orquestación. La auditoría implementa solo extracción focalizada y análisis diagnóstico, no un pipeline definitivo. No se duplican algoritmos matemáticos ni se refactoriza el experimento existente.

## Dictamen y cierre

No es viable entrenar ahora bajo el instante y la evidencia disponibles. Se recomienda frecuencia proveedor-mes como formulación futura más interpretable que sumar cantidades incompatibles, pero sigue bloqueada por historia y cobertura. Volumen producto-mes tampoco pasa la comprobación temporal; producto-proveedor-mes no mejora identidad, y actividad solo separa entidades persistentes. No hay autorización para dataset. Se recomienda continuar con Transacciones financieras en otra tarea; no se inicia automáticamente ninguna investigación adicional.
''')


if __name__=='__main__':
    exportar()
