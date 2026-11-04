"""Dictamen documental desde cabeceras agregadas, sin construir dataset."""
import csv
import io
import json
from src.db.auditoria_dw import BASE,guardar,sha
from .documentos import OUT,COHORTS


LABELS={'all_header_diagnostic':'Todas las cabeceras (solo diagnóstico)','original_without_backorder':'Original sin backorder','original_with_backorder':'Original con pendientes trasladados','derived_terminal':'Derivado terminal','derived_with_backorder':'Derivado que vuelve a trasladar','ambiguous_lineage':'Linaje ambiguo'}


def exportar(data,summary,verdict):
    def row(cohort,split):
        v=summary[cohort].get(split,{'total':0,'eligible':0,'positive':0,'negative':0,'positive_proportion':None,'states':{}})
        return [LABELS[cohort],split,v['total'],v['eligible'],v['positive'],v['negative'],f"{100*v['positive_proportion']:.4f} %" if v['positive_proportion'] is not None else 'No estimable']
    table='\n'.join('| '+' | '.join(map(str,row(c,s)))+' |' for c in ['all_header_diagnostic',*COHORTS] for s in ['train','validation'])
    yearly='\n'.join('| '+' | '.join(map(str,row(c,y)))+' |' for c in ['all_header_diagnostic',*COHORTS[:-1]] for y in ['2013','2014','2015'])
    monthly=[]
    for cohort,periods in summary.items():
        for period,v in sorted(periods.items()):
            if len(period)!=7:continue
            monthly.append({'cohort':cohort,'month':period,'documents':v['total'],'eligible':v['eligible'],'positive':v['positive'],'negative':v['negative'],'positive_proportion':v['positive_proportion'],'indeterminate_or_not_eligible':v['total']-v['eligible']})
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(monthly[0]));w.writeheader();w.writerows(monthly)
    (OUT/'prevalencias_mensuales.csv').write_text(buf.getvalue(),encoding='utf-8',newline='\n')
    month_table='\n'.join(f"| {r['month']} | {r['documents']} | {r['eligible']} | {r['positive']} | {r['negative']} | {100*r['positive_proportion']:.4f} % |" for r in monthly if r['cohort']=='all_header_diagnostic')
    original_table='\n'.join(f"| {r['month']} | {r['documents']} | {r['eligible']} | {r['positive']} | {r['negative']} | {100*r['positive_proportion']:.4f} % |" for r in monthly if r['cohort']=='original_without_backorder')
    source=OUT.parent/'modulos_operativos.json';modules=json.loads(source.read_text(encoding='utf-8'));excerpts=[]
    for m in modules:
        text=m['definition'].replace('\r','')
        terms=['-- create the backorder order','-- flag the original order as backordered and picking completed'] if m['schema_name']=='DataLoadSimulation' else (['INSERT Sales.Orders'] if m['module_name']=='InsertCustomerOrders' else [])
        for term in terms:
            i=text.find(term)
            if i>=0:excerpts.append({'schema':m['schema_name'],'module':m['module_name'],'excerpt':text[max(0,i-80):i+1800]})
    guardar(OUT/'evidencia_semantica.json',{'source':str(source.relative_to(BASE)),'source_sha256':sha(source),'executed':False,'excerpts':excerpts,'interpretation':'Código existente, no bitácora histórica; se contrasta con fechas y relaciones actuales sin atribuir precisión de inserción a OrderDate.'})
    raw=data['identity_coverage'];base=summary['original_without_backorder'];allrows=summary['all_header_diagnostic'];edge=data['backorder_dates']
    contract={'version':'pedidos-documentos-1.0-viabilidad','unit':'OrderID único de Sales.Orders; sin ponderación por líneas',
        'prediction_anchor':'OrderDate propio del documento, fecha de negocio sin hora y no inserción auditada. No reemplazar fecha de backorder por raíz ni por LastEditedWhen.',
        'candidate_label':'0 si DATE(PickingCompletedWhen)<=OrderDate+2; 1 si posterior; null/inconsistencia/seguimiento o disponibilidad insuficiente -> indeterminado o no elegible',
        'horizon':'Dos días calendario, igualdad incluida en clase 0; fin exclusivo al inicio de d+3, no SLA demostrado',
        'availability':'Horizonte completo al corte, PickingCompletedWhen<cut y LastEditedWhen<cut. Última edición solo limita versión, no fija nacimiento o instante de predicción.',
        'development':{'train':['2013-01-01','2014-12-31'],'train_labels_before':'2015-01-01T00:00:00','validation':['2015-01-01','2015-12-31'],'validation_labels_before':'2016-01-01T00:00:00','test':'OrderDate>=2016-01-01 sellado, sin outcomes nuevos'},
        'business_cohort_assessment':'El completo original solo puede interpretarse razonablemente sin traslado a backorder; originales con salida y derivados se separan para diagnóstico, no se incorporan para equilibrar clases.',
        'backorder_selection_risk':'La ausencia/presencia final de links es selección retrospectiva, no una condición operativa conocida al emitir. No usarla como predictor ni extrapolar la cohorte a todos los pedidos.',
        'calendar_features':['mes_sin','mes_cos','semana_sin','semana_cos','dia_mes','dias_mes'],
        'header_predictors':'CustomerID, SalespersonPersonID y datos comerciales conceptualmente asignados al insertar, pero valores actuales sin historial no certificados: desactivados salvo calendario.',
        'forbidden':['Picking del objetivo','PickedByPersonID','BackorderOrderID final','IsUndersupplyBackordered actual como proxy de outcome sin disponibilidad comprobada','LastEditedWhen como predictor o alta','Información de líneas o composición original no demostrada','Atributos dimensionales con joins ambiguos','Estadísticas de retraso no conocidas en el corte de features'],
        'cancellation':'No se encontró estado dedicado de cancelación en el esquema consultado. Sin picking no implica cancelado ni retrasado.',
        'verdict':verdict,'next_process':'Fact.Movement: nueva formulación explícita y nueva autorización; no se desarrolla aquí.'}
    guardar(OUT/'contrato_propuesto.json',contract)
    viability=f'''# Pedidos por documento: dictamen de viabilidad

## Dictamen

**No corresponde autorizar la construcción del dataset con el objetivo solicitado en esta copia.** Una observación por OrderID es estructuralmente posible y OrderDate es una fecha de negocio razonable para un documento original; no es una marca auditada de inserción ni contiene hora. Pero la población en que el campo puede interpretarse como cumplimiento completo sin transferencias no tiene eventos positivos ni en train ni en validación.

Este dictamen es específico del objetivo «picking del pedido completo después de dos días». No afirma que Pedidos carezca de cualquier problema predictivo posible. No se vuelve a abrir indefinidamente la investigación ni se cambia la etiqueta para obtener equilibrio. Se recomienda continuar con **Fact.Movement**, bajo un contrato propio y otra tarea; no se consulta ni implementa ese proceso aquí.

## Evidencia local focalizada

Se consultó Sales.Orders y su relación consigo misma por BackorderOrderID, junto con sus metadatos. No se agregaron líneas para obtener etiquetas de documento. Hay {raw['documents']} documentos de desarrollo y {raw['distinct_documents']} OrderID distintos, desde {raw['min_date']} hasta {raw['max_date']}. El estado actual muestra {raw['current_completed_flag']} picking no nulos y {raw['current_pending_or_unobserved']} nulos; hay {raw['pick_before_order']} picking anteriores a OrderDate. Son estados de la copia actual, no estados reconstruidos al corte. La elegibilidad temporal se calcula aparte.

No se consultaron outcomes ni prevalencias de documentos fechados en 2016. Tampoco se repitió la auditoría del DW. Los datos globales previos no se usan como nuevos conteos de documento.

## Cohortes y resultados

| Cohorte | Partición | Documentos de origen | Etiquetables al corte | Positivos | Negativos | Prevalencia |
|---|---|---:|---:|---:|---:|---:|
{table}

El diagnóstico de todas las cabeceras produce {allrows['train']['eligible']} etiquetas train ({allrows['train']['positive']} positivas) y {allrows['validation']['eligible']} de validación ({allrows['validation']['positive']} positivas). **No son tamaños autorizados de un dataset de pedido completo.** Mezclan eventos con distinta semántica y anclas no válidas para emisión del documento derivado.

En originales sin backorder hay {base['train']['eligible']} etiquetas train y {base['validation']['eligible']} de validación, todas clase 0. Si se ampliara a todos los originales incluyendo los que trasladan pendientes, solo aparecerían cuatro positivos en train y ninguno en validación; además se debilitaría el significado de cumplimiento completo. No se justifica entrenar sobre un problema sin clase retrasada evaluable ni rescatar positivos de documentos con fecha heredada.

## Por qué los backorders no resuelven el bloqueo

Existen {edge['edges']} vínculos de desarrollo, {edge['orphan_child']} hijos ausentes, {edge['different_dates']} diferencias de OrderDate padre/hijo y {edge['non_increasing_ids']} vínculos con clave no creciente. Los documentos se clasifican por links entrantes y salientes: original sin salida, original con salida, derivado terminal y derivado que vuelve a trasladar. El agrupamiento previo de padres impide multiplicar filas. Los IDs crecientes apoyan ausencia de ciclos en los vínculos examinados, no otorgan una fecha de emisión.

La plantilla local de simulación copia OrderDate del padre al crear el hijo, traslada líneas no suministradas y marca al padre como picking completado para facturar lo restante. Esa lectura de código es evidencia de la semántica posible, no una ejecución realizada ni una prueba de la historia completa de cada fila. La igualdad de fechas observada es compatible con ella.

Por tanto, no se puede tratar al hijo como emitido realmente en su OrderDate heredado; no se sustituye ese campo por la fecha del padre ni por última edición. Tampoco se interpreta el cierre del padre tras separar pendientes como prueba de satisfacción de toda su solicitud inicial. Las líneas no acreditadas al registro permanecen fuera de los predictores. Estudiar familias completas requeriría otro universo y otra definición, no el documento propuesto silenciosamente renombrado.

La cohorte sin backorders se usa solo para evaluar si existe una población retrospectiva defendible. Su pertenencia depende de links finales que podrían conocerse después de emitir: es una selección retrospectiva y no un filtro de despliegue o predictor. El descarte de las otras cohortes obedece a ancla y significado del evento, definido antes de consultar su prevalencia; no es una selección de meses o clases convenientes.

## Evidencia, supuestos y límites

Demostrado: unicidad actual de OrderID, fechas y tipos de metadatos, estados agregados de desarrollo, cortes aplicados y relaciones de backorder. El módulo Website asigna OrderDate al registrar documentos y la plantilla de simulación genera fecha de negocio; no se deduce que conozcamos cuándo se insertó físicamente cada fila histórica.

Supuestos académicos explícitos para la alternativa: interpretar OrderDate del original como emisión de negocio a precisión diaria; considerar fiable el timestamp observado de picking; utilizar la versión persistida solo si su última edición es anterior al corte. No son certificaciones de ingestión, ausencia de backdating, composición original ni snapshot completo. La configuración de estos supuestos no compensa la falta de positivos de la cohorte admisible.

Los nulos no se etiquetan como retrasados: pueden ser pendientes o ausencia de evento verificable. No hay columna dedicada de cancelación en el esquema Orders consultado y no se deduce cancelación de comentarios, nulos o falta de factura. Se preserva el motivo indeterminado. Los extremos observados no certifican completitud del período.

Calendario del OrderDate propio puede conocerse al emitir. Atributos actuales de cabecera no se habilitan por existir: cliente/vendedor pueden servir como claves de asociación tras verificar temporalidad, pero no como números continuos; las demás entradas siguen desactivadas. Quedan prohibidos picking, picker, estado final, links finales, últimas ediciones como X y cualquier información futura o de líneas sin existencia probada. No se aplican joins dimensionales.

## Decisión final y continuidad

Mantener bloqueados tanto el contrato por línea anterior como esta alternativa para el objetivo evaluado. Preservar sus entregables y no construir X/y, preprocesador, modelos o PCA. No se modifica el informe ni se realizan operaciones Git.

Recomendación: cerrar esta investigación de Pedidos y pasar a Fact.Movement, ya auditado, porque registra eventos de movimiento y permite una formulación distinta sobre flujos sin exigir un supuesto de creación de líneas. No equivale a autorizar reconstrucción de stock ni predicción de stockout: faltan saldo inicial e historia reconciliada para eso. No se inicia una nueva auditoría de Movimientos ni se cambia automáticamente el trabajo a Compras.
'''
    (OUT/'viabilidad.md').write_text(viability,encoding='utf-8',newline='\n')
    states=['invalid','insufficient_followup','missing_pick_indeterminate','pick_not_observed_by_cut','row_not_available_by_cut','within_two_days','after_two_days']
    state_table='\n'.join(f"| {state} | {allrows['train']['states'].get(state,0)} | {allrows['validation']['states'].get(state,0)} |" for state in states)
    analysis=f'''# Etiquetas de cabeceras: solo desarrollo

Cada observación es un OrderID contado una vez. Todas las consultas de datos filtran OrderDate < 2016-01-01 antes de clasificar; no hay outcomes test. Los conteos se recalcularon desde Sales.Orders, no se copiaron de las líneas. Se conserva el diagnóstico de todas las cabeceras separado de la cohorte empresarial defendible.

## Estados exhaustivos al corte

| Estado | Train | Validación |
|---|---:|---:|
{state_table}
| **Total de documentos** | **{allrows['train']['total']}** | **{allrows['validation']['total']}** |

Prioridad: fecha incoherente, seguimiento incompleto, picking nulo, picking posterior/al corte, versión no disponible y clases 0/1. Los motivos no se suman dos veces. El picking nulo es indeterminado: no se cuenta automáticamente como atraso, cancelación ni resultado negativo. Las cabeceras completas en la copia pero observadas después del corte no generan etiquetas disponibles en ese corte.

## Por cohorte y año

| Cohorte | Año | Documentos de origen | Etiquetables | Positivos | Negativos | Prevalencia |
|---|---|---:|---:|---:|---:|---:|
{yearly}

Train incluye 2013–2014 con corte común 2015-01-01, no cortes ficticios al final de cada año. Validación es 2015 con corte 2016-01-01. Las prevalencias son proporciones entre outcomes justificados al corte, con sesgo de selección por observación. Cabeceras pendientes/indeterminadas no desaparecen del inventario de estados.

## Mensual: todas las cabeceras, diagnóstico no aprobado como dataset

| Mes | Documentos | Etiquetables | Positivos | Negativos | Prevalencia |
|---|---:|---:|---:|---:|---:|
{month_table}

## Mensual: originales sin backorder

| Mes | Documentos | Etiquetables | Positivos | Negativos | Prevalencia |
|---|---:|---:|---:|---:|---:|
{original_table}

prevalencias_mensuales.csv incluye todos los tipos de backorder por mes; etiquetas_desarrollo.json contiene además estados, años y particiones. Los períodos sin positivos se conservan. La ausencia de clase 1 en train y validación de la cohorte admisible impide una evaluación binaria útil: no se modifica el horizonte para fabricar eventos ni se balancea con datos sintéticos.

No se interpretan las tasas altas de derivados como oportunidad para entrenar sin resolver su fecha heredada. Los cuatro positivos entre originales que generan backorders tampoco justifican el objetivo de cumplimiento completo y no aportan positivos de validación.
'''
    (OUT/'analisis_etiquetas.md').write_text(analysis,encoding='utf-8',newline='\n')
    proposed='''# Contrato propuesto por documento — no habilitado para dataset

Unidad: OrderID único de Sales.Orders. Instante candidato: fecha de emisión de negocio OrderDate del propio documento, sin precisión de hora ni afirmación de inserción auditada. No usar LastEditedWhen como alta ni reemplazar la fecha de un backorder por la de su raíz. Historia, si se habilitara, solo anterior al inicio del día de emisión; no agregados del mismo día.

Outcome candidato: PickingCompletedWhen del pedido completo hasta OrderDate+2 días calendario inclusive -> 0; posterior -> 1. Empate al día 2 pertenece a 0. Fin exclusivo del horizonte: inicio de d+3. No son 48 horas exactas, días laborables ni un SLA contractual demostrado. Fechas anteriores al pedido son inconsistentes y no etiquetables. Picking nulo queda indeterminado; no existe estado dedicado de cancelación que permita separar cancelados con evidencia suficiente.

Train: documentos fechados 2013–2014, con picking y última versión anteriores a 2015-01-01 00:00. Validación: documentos 2015, con conocimiento anterior a 2016-01-01 00:00. Exigir d+3 <= corte para seguimiento homogéneo; excluir últimos dos días de cada partición aunque alguno esté recogido rápido. Un evento de picking o edición posterior al corte no se utiliza para etiquetar aprendizaje anterior. LastEditedWhen es puerta conservadora de versión, no fecha de emisión. Los timestamps carecen de zona incorporada: conservar calendario fuente, no inventar un huso.

Backorders: separar originales sin salida, originales con salida, derivados terminales y derivados que vuelven a trasladar, comprobando padres únicos. Todas las cohortes usan su propio OrderDate en los diagnósticos. La fecha heredada invalida interpretar el hijo como emitido realmente ese día; cierre tras transferir pendientes no es cumplimiento íntegro de la demanda original. No unir familias para redefinir silenciosamente la unidad. La cohorte sin traslados es un subconjunto retrospectivo seleccionado con links finales, no un filtro disponible al emitir.

Predictores conceptualmente permitidos: seis campos de calendario del contrato previo, solo para una fecha de negocio admisible. Cliente/vendedor y otros atributos de cabecera no se habilitan sin disponibilidad histórica demostrada. No usar cantidades, totales, número de líneas, variedad ni otras entradas del universo de líneas no acreditado. No incorporar joins dimensionales ambiguos o versiones futuras. Identificadores son asociaciones/grupos, nunca medidas continuas.

Prohibidos: picking del objetivo, picker, cumplimiento final, links posteriores de backorder, edición como X, resultados futuros e historias de retraso cuyo conocimiento todavía no existía en el corte. No reconstruir valores iniciales a partir de los finales.

Evaluación futura, si hubiera una cohorte válida con ambas clases: soportes, matriz de confusión, balanced accuracy, precision/recall/F1 del retraso, especificidad, falsas alertas y proporción de alertas; AP solo con puntuaciones. No éxito por accuracy mayoritaria. Baseline mayoritario=0. No fijar remuestreo, pesos o umbrales por test ni ajustar el plazo para equilibrar. Agrupar familias y respetar fechas; no tratar documentos relacionados como réplicas independientes.

Decisión actual: no autorizar dataset. En originales sin backorder no hay positivos ni en train ni en validación; el agregado con positivos mezcla eventos de distinta interpretación y documentos con ancla no defendible. La aceptación de una fecha empresarial para originales no resuelve esa falta de eventos. Se recomienda continuar con Fact.Movement en otra solicitud explícita, sin iniciar aquí nuevos experimentos.

Reproducir esta verificación focalizada: `python -m src.datamarts.pedidos.documentos --server 'lpc:.\\AMBATO'`. Solo SELECT, sin consultas de outcomes desde 2016, matrices, modelos, PCA o cambios del informe. Se guardan consultas parametrizadas, huellas acotadas, contratos y hashes. No se modifica el contrato anterior por línea.
'''
    (OUT/'contrato_propuesto.md').write_text(proposed,encoding='utf-8',newline='\n')
