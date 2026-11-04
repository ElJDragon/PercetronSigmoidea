"""Contrato y dictamen de Inventario desde agregados exclusivamente de desarrollo."""
from src.datamarts.inventario_viabilidad import OUT,load
from src.db.auditoria_dw import guardar

def feature_catalog():
    features=[]
    for k in (1,2,3):
        features.append(dict(name=f'recepciones_lag_{k}',formula=f'n{k} = N(i,t-{k})',source='COUNT DISTINCT StockItemTransactionID, tipo 11',availability=f'Cierre de t-{k}'))
    for k in (1,2,3):
        features.append(dict(name=f'unidades_recibidas_lag_{k}',formula=f'q{k} = Q(i,t-{k})',source='SUM Quantity de eventos tipo 11; unidades base del mismo artículo',availability=f'Cierre de t-{k}'))
    for name,formula in [('recepciones_media_3','(n1+n2+n3)/3'),('recepciones_mediana_3','mediana(n1,n2,n3)'),
        ('recepciones_std_3','sqrt(sum((nk-media)^2)/3)'),('recepciones_cambio_1','n1-n2'),
        ('recepciones_tendencia_3','(n1-n3)/2'),('meses_con_recepciones_3','sum(1[nk>0], k=1..3)'),
        ('mes_anterior_sin_recepciones','1[n1=0]'),('antiguedad_observada_meses','ordinal(t)-ordinal(primer_mes_evento_conocido_antes_de_t)'),
        ('mes_sin','sin(2*pi*(mes(t)-1)/12)'),('mes_cos','cos(2*pi*(mes(t)-1)/12)'),('dias_mes_objetivo','dias_calendario(t)')]:
        features.append(dict(name=name,formula=formula,source='Historia tipo 11 y primera aparición en cualquier movimiento; calendario conocido',availability='Antes del comienzo de t'))
    return features

def pct(x):return 'no definida' if x is None else f'{100*x:.3f}%'

def write(name,text): (OUT/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')

def generate():
    types=load('tipos_y_signos.json');q=load('calidad_cobertura.json');overlap=load('solapamiento_ventas.json');candidates=load('exploracion_etiquetas.json')
    features=feature_catalog();guardar(OUT/'catalogo_caracteristicas.json',features)
    table='\n'.join(f"| {r['type_id']} {r['name']} | {r['events']} | {r['negative']} | {r['positive']} | {r['zero']} | {r['invoice_links']} | {r['purchase_links']} |" for r in types)
    write('semantica_movimientos.md',f"""
# Semántica de movimientos de Inventario

Investigación local focalizada de WideWorldImportersDW y WideWorldImporters en AMBATO, mediante pyodbc y autenticación integrada. Se reutiliza `../auditoria_dw/`: el total histórico conocido de 236667 movimientos y la cobertura enero de 2013–mayo de 2016 proceden de esa auditoría; no se volvieron a perfilar outcomes de 2016. `Fact.Stock Holding` tiene 227 filas de estado sin fecha histórica y no se consultó ni se usa como historial.

## Tipos efectivamente observados en desarrollo

206143 eventos únicos entre {q['first_event']} y {q['last_event']}; 219 artículos. El catálogo contiene más tipos, pero solo estos tres aparecen en el intervalo investigado:

| Tipo | Eventos | Quantity < 0 | > 0 | = 0 | Con factura | Con orden compra |
|---|---:|---:|---:|---:|---:|---:|
{table}

Tipo 10: todas las salidas tienen CustomerID y InvoiceID, ninguna SupplierID ni PurchaseOrderID. Tipo 11: todas las recepciones tienen SupplierID y PurchaseOrderID, ninguna factura ni cliente. Tipo 12: ajustes de recuento físico sin esos documentos, con ambos signos y cuatro eventos de cantidad cero. Estos cuatro siguen siendo eventos de ajuste para el candidato de frecuencia. No se encontraron transferencias tipo 9 ni tipos explícitos de devolución en los movimientos de desarrollo; los tipos financieros de abonos/reembolsos no se reinterpretan como movimientos físicos. Esto no certifica ausencia de devoluciones en la operación real ni en test.

`Quantity` es un cambio de unidades físicas, no importe ni saldo. Los ajustes negativos prueban que negativo no equivale automáticamente a venta; los positivos incluyen recepciones y ajustes, por lo que tampoco equivalen todos a compras. Las unidades se comparan dentro del mismo artículo y nunca se suman entre artículos para definir la etiqueta.

## Evidencia de origen y unidades

`modulos_movimientos.json` conserva las definiciones leídas, sin ejecutarlas. `Website.InvoiceCustomerOrders` registra `0 - InvoiceLines.Quantity` como Stock Issue. La definición de configuración de simulación contiene la creación de las rutinas de recepción y recuento: Stock Receipt se registra como `ReceivedOuters * QuantityPerOuter`, con fecha del evento, proveedor y orden; Stock Adjustment at Stocktake registra el delta físico. El código de configuración es evidencia de generación, no un procedimiento de recepción ejecutado en este encargo ni prueba de todas las operaciones posibles. `Integration.GetMovementUpdates` exporta el ID del evento, fecha, tipo, documentos y `CAST(Quantity AS int)`, con selección incremental por LastEditedWhen. No ejecutamos la integración.

La cantidad OLTP es decimal. Los 206143 valores de desarrollo son enteros exactos, por lo que no se observó truncamiento en DW. Las 219 versiones operativas históricas (una por artículo de desarrollo) mantienen UnitPackageID y QuantityPerOuter dentro de cada artículo; no faltan metadatos. Las recepciones ya están en unidades base: no volver a multiplicarlas por el empaque actual. Esta comparabilidad respalda B en desarrollo; si posteriormente cambia la unidad o aparecen fracciones, se debe detener y resolver, sin inferir valores desde estado actual. No se usan ReceivedOuters ni cantidades ordenadas actuales como predictores históricos.

## Identidad y reconciliación

Identidad estable: StockItemID operativo, equivalente a WWI Stock Item ID. Clave de evento: StockItemTransactionID. La clave sustituta DW se conserva solo como trazabilidad. En las versiones DW anteriores a 2016 hay 219 identidades reales con una versión por identidad y la fila desconocida 0; ningún movimiento analizado utiliza el desconocido. El diseño admite futuras versiones reuniéndolas por la identidad operativa, sin usar etiquetas o atributos posteriores.

Se enlazaron los 206143 eventos DW por ID con OLTP: cero ausentes en ambas direcciones, cero duplicados y cero diferencias de artículo, tipo, cantidad, fecha, factura u orden. Las huellas de lectura antes/después de ambas fuentes coinciden. No es una garantía criptográfica de toda la BD ni una auditoría de borrados, sino una comprobación focalizada de estabilidad durante las consultas.

## Relación con Ventas y flujo neto

En desarrollo, {overlap['issue_events']} salidas coinciden con {overlap['sales_lines']} líneas de Fact.Sale en {overlap['issue_groups']} grupos fecha–factura–artículo. No hay grupos ausentes, diferencias de cantidad absoluta ni diferencias de número de líneas: cobertura 100% en ambos sentidos. Un objetivo de volumen mensual elevado de salidas sería la misma medida de unidades vendidas para esta población, y se descarta como nuevo experimento.

El flujo neto tiene significado de suma firmada de cambios, pero no de existencias, disponibilidad, rotura de stock ni demanda. En {overlap['net_windows_only_sales']} de 7227 ventanas objetivo+3 meses no hay recepciones ni ajustes; en todas esas ventanas la etiqueta de neto alto equivale exactamente a unidades vendidas bajas respecto de su mediana. D reintroduce principalmente Ventas con sentido invertido. Su mejor equilibrio de clases no lo vuelve una aplicación empresarial independiente.

## Cobertura, ausencia y disponibilidad

Los 36 meses de desarrollo están representados, todos los 219 artículos aparecen en enero de 2013 y tienen movimientos en los 36 meses: 7884 artículo-meses y cero meses sin cualquier movimiento. Solo 521 artículo-meses tienen recepciones y 7363 no las tienen. Después de enero de 2013 solo diez artículos vuelven a recibir; en 2015 hay 108 artículo-meses con recepción, concentrados en nueve artículos. Esta concentración es una limitación sustantiva y no se oculta filtrando por actividad futura.

Los 206143 registros tienen LastEditedWhen igual a TransactionOccurredWhen. Esto respalda disponibilidad por evento en el extracto conservado; LastEditedWhen no es por sí solo un registro de creación ni demuestra inmutabilidad. StockItemTransactions es NON_TEMPORAL_TABLE: no permite descartar retrospectivamente borrados o correcciones nunca conservadas. StockItems sí tiene historia temporal, consultada solo para identidad/unidades anteriores a 2016.

Se admite el supuesto académico explícito de archivo de eventos completo y conservado por artículo a partir de su primera aparición y de cierre del mes anterior. Bajo ese supuesto, un mes cubierto sin tipo 11 vale cero recepciones; un mes no cubierto/desconocido no se convierte en cero. Tener todos los meses y reconciliación exacta respalda el supuesto, pero no certifica cierres operativos. No se requieren snapshots de existencias porque se predicen eventos de recepción, no niveles de stock. Test no se abrió para elegir población ni etiqueta.

Fuentes detalladas: `tipos_y_signos.json`, `calidad_cobertura.json`, `cobertura_articulos.json`, `solapamiento_ventas.json`, archivos de consultas y extracciones acotadas. Cada consulta de eventos usa una frontera estricta anterior a 2016-01-01.
""")
    rows=[]
    for name,c in candidates.items():
        for split,s in c['summary'].items():
            rows.append(f"| {name} | {split} | {s['entities']} | {s['observations']} | {s['positive']} | {s['negative']} | {pct(s['prevalence'])} | {s['positive_entities']} | {s['entities_varying']} | {pct(s['historical']['balanced_accuracy'])} |")
    monthly=[]
    for index,m in enumerate(candidates['A']['monthly']):
        monthly.append('| '+m['month']+' | 219 | '+' | '.join(f"{candidates[k]['monthly'][index]['positive']} / {candidates[k]['monthly'][index]['negative']} ({pct(candidates[k]['monthly'][index]['prevalence'])})" for k in 'ABCD')+' |')
    write('analisis_etiquetas.md',"""
# Análisis de etiquetas de desarrollo y dictamen

Solo train 2013–2014 y validación 2015. No se calcularon etiquetas, prevalencias ni resultados de 2016. Granularidad común: artículo operativo–mes natural; horizonte: siguiente mes, predicción antes de comenzar t, usando tres meses consecutivos cerrados. La elegibilidad no depende de que haya recepción en t ni de supervivencia posterior. Primera aparición por cualquier movimiento conocido antes de t. En este extracto todos aparecen en enero de 2013: train elegible abril de 2013–diciembre de 2014 (21 meses), val enero–diciembre de 2015 (12 meses), 219 artículos en cada mes. No se exige haber recibido recientemente ni se seleccionan solo los artículos con positivos.

## Candidatos y significado

- **A, frecuencia elevada de recepciones:** N(i,t) = número de IDs distintos tipo 11 del artículo en t; y = 1[N(i,t) > mediana(N(i,t-1),N(i,t-2),N(i,t-3))]. Busca anticipar intensidad relativa de trabajo de recepción por artículo; no número de días, unidades, órdenes completas ni necesidad de compra. Cada ID es una recepción registrada, que en desarrollo coincide con un par artículo–orden distinto. No se promete medir carga laboral proporcional ni tiempo de descarga. Historia: conteos, cantidades de recepción previas y calendario. Empate a 0.
- **B, volumen elevado de recepciones:** Q(i,t) = suma de Quantity tipo 11; y = 1[Q(i,t) > mediana(Q(i,t-1),Q(i,t-2),Q(i,t-3))]. Busca planificar intensidad de unidades que llegan, sin mezclar unidades entre artículos. Es interpretable porque el origen expresa unidades base y la unidad se mantiene en desarrollo; no es demanda ni saldo. Misma historia causal; requiere sostener comparabilidad de unidades en futuras extracciones. Empate a 0.
- **C, frecuencia elevada de ajustes:** J(i,t) = número de IDs tipo 12, incluso cantidad cero; y = 1[J(i,t) > mediana(J(i,t-1),J(i,t-2),J(i,t-3))]. Busca anticipar excepciones/recuentos, no deterioro financiero ni pérdidas monetarias. Conteos de ajustes previos, signos históricos y calendario serían causales. Solo 38 eventos en 36 meses, concentrados en meses trimestrales: no hay evidencia suficiente de un proceso recurrente aprendible con esta muestra, aunque hay dos clases agregadas.
- **D, actividad neta:** F(i,t) = suma firmada de Quantity de tipos 10, 11 y 12; y = 1[F(i,t) > mediana(F(i,t-1),F(i,t-2),F(i,t-3))]. Sería incremento relativo de flujo neto mensual, no stock alto. Requiere interpretar todos los tipos y unidades; no incorpora saldo actual ni inventa saldo inicial. Las series de flujo/entradas/salidas previas serían causales. Se descarta como nueva aplicación: en 6594/7227 ventanas es exactamente la etiqueta de ventas bajas y no un fenómeno independiente.

Para cada candidato, el mayoritario es clase 0 obtenida solo de train. Baseline histórico: predecir que el agregado de t será igual al de t-1; por tanto b(i,t) = 1[V(i,t-1) > mediana(V(i,t-1),V(i,t-2),V(i,t-3))], con V=N,Q,J,F. Solo necesita la misma historia de tres meses. No es copiar y(i,t-1), que exigiría el cuarto mes. Ningún baseline usa V(i,t).

## Distribuciones, variación y baseline

| Candidato | Partición | Artículos | Filas | Positivos | Negativos | Prevalencia | Artículos positivos | Con ambas clases | BA histórico |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""+'\n'.join(rows)+"""

Los cuatro candidatos tienen ambas clases en cada partición agregada, pero no necesariamente en cada mes. BA mayoritaria = 50% en train y val; su accuracy elevada en A/B/C es consecuencia de la rareza, no capacidad útil. Los JSON conservan matrices de confusión y accuracy de ambos baselines. En A, el histórico da BA 64,136% en train y 59,749% en val; en B, 92,079% y 85,596%. Estas son referencias causales de desarrollo, no resultados de clasificadores ni de test. No se selecciona una etiqueta por superar o facilitar un baseline.

## Prevalencia mensual completa: positivos / negativos (porcentaje positivo)

| Mes | N por candidato | A | B | C | D |
|---|---:|---|---|---|---|
"""+'\n'.join(monthly)+"""

A tiene meses monoclase: 2013-11, 2014-02, 2015-01 y 2015-11, todos sin positivos; debe informarse BA mensual como no definida cuando falta alguna clase, sin asignar artificialmente 50%. C presenta muchos más meses sin positivos; no se eliminan esos meses para mejorar la evaluación. Las dependencias entre meses y artículos y la concentración de positivos limitan cualquier inferencia estadística. En A, solo diez artículos varían en train y nueve en val; sus 35 positivos de validación no sostienen precisión estadística fina. Los desgloses por identidad están en `exploracion_etiquetas.json`.

## Dictamen

**A: viable bajo supuestos académicos explícitos; recomendado para construcción posterior del dataset.** Se recomienda por identidad verificable del evento, independencia respecto de las líneas vendidas, regla sencilla disponible al corte y valor empresarial de anticipar recepciones registradas. No se recomienda por equilibrio: tiene prevalencia muy baja (1,870% train; 1,332% val), y esta desventaja se acepta y documenta. El experimento sería exploratorio de eventos infrecuentes, sin garantía de mejora sobre el histórico ni de utilidad operativa. Mantener la población completa; no eliminar artículos con meses cero para fabricar equilibrio. La construcción debe respetar el contrato, sin abrir test ni entrenar en esta etapa.

**B: viable bajo los mismos supuestos, alternativa no seleccionada.** Hay comparabilidad de unidades en desarrollo y ambas clases; añade dependencia del significado y conservación de Quantity, menor variación de clase dentro de cada artículo en train y un baseline fuerte. No se rechaza por su BA ni se afirma que sea peor predictivamente. A responde con menos supuestos de unidades a una pregunta distinta de volumen vendido.

**C: condicionado; no recomendado con la fuente actual.** No es matemáticamente monoclase, pero 22/14 positivos train/val y escasa repetición por artículo son insuficientes para recomendar otro clasificador generalizable. Para retomarlo: mayor horizonte de eventos de recuento con tipos y fechas confiables, más ajustes recurrentes o una unidad empresarial diferente justificada antes de mirar test.

**D: no viable como objetivo independiente de Ventas en esta evidencia.** La suma firmada es calculable, pero predomina una transformación de las mismas salidas vendidas. El equilibrio casi 50/50 no compensa esa duplicidad. Si se requiere predecir existencias o roturas, harían falta saldos iniciales certificados, todos los flujos y cierres, reservas y demanda no atendida; los estados actuales no reconstruyen esa historia.

Autorización recomendada: siguiente encargo limitado a construir y verificar causalmente A con controles de integridad; no implica autorización ejecutada de matrices, entrenamiento ni apertura de test. Este encargo se detiene en el dictamen.
""")
    catalog='\n'.join(f"| `{f['name']}` | {f['formula']} | {f['source']} | {f['availability']} |" for f in features)
    write('contrato_datos.md',"""
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
"""+catalog+"""

Las unidades previas son del mismo artículo, ya convertidas a base por el evento; no usar el empaque actual para reinterpretar el pasado. Nombres, IDs de artículo/proveedor/cliente/factura/orden, claves sustitutas, fecha de última actividad futura, estados de compra o finalización actuales, saldo actual, QuantityOnHand, ReorderLevel, TargetStockLevel, mes objetivo y cantidades futuras no son predictores. La cantidad histórica de entrada sí está autorizada; no se autoriza stock retrospectivo inventado.

## Particiones, tipos y preprocesado posterior

Train objetivos 2013–2014, con primer objetivo elegible abril de 2013; val objetivos 2015; test 2016 reservado. Los meses de historia inicial de val pueden proceder de train, y los de test de años previos; eso es historia causal, no mezcla de ajuste. Esperado en desarrollo: 4599 filas train, 2628 val; no se anticipa número de filas/clases de test.

Futuro preprocesado: misma lista y orden para ambos modelos, float64; ajuste único con train. Eliminar columnas constantes o completamente nulas en train y aplicar esa misma máscara a val/test. Historia esencial faltante implica exclusión, no imputación; si se autoriza alguna variable opcional más adelante, mediana train con trazabilidad. Escalado `(x-media_train)/std_train`, ddof=0; ni refit ni selección con val/test. No remuestrear ni ponderar clases en esta versión de contrato. No ejecutar estos pasos ahora.

Futuro paquete verificable: claves/etiquetas separadas de X, cobertura y motivos de exclusión, catálogo ordenado retenido/descartado, parámetros train-only, firmas de archivos y pruebas de causalidad por corte. Construir solo después del dictamen, en un encargo posterior. Congelar toda decisión experimental antes de consultar outcomes de test.
""")
    write('diseno_evaluacion.md',"""
# Diseño de evaluación y reutilización de arquitectura

Objetivo recomendado A, artículo–mes, horizonte de un mes y tres meses previos de historia. Alcance académico bajo supuesto de conservación/cobertura. Este documento no entrena, no elige un checkpoint ni abre test.

## Orden de trabajo posterior

1. Construir y verificar el dataset de desarrollo conforme a `contrato_datos.md`: 4599 filas train y 2628 val esperadas. Verificar tipos, claves, firmas, exclusiones y que ninguna entrada alcance t. La ausencia de recibos cubiertos es cero; historia desconocida se excluye explícitamente.
2. Ajustar preprocesado exclusivamente con train, entrenar las implementaciones propias en un encargo posterior y seleccionar hiperparámetros/checkpoint por BA de validación. Presupuesto, semillas, desempates, umbral logístico inicialmente 0.5 y reglas de parada deben quedar definidos antes del entrenamiento y congelados antes del test. No se prescribe un checkpoint usando estos resultados de viabilidad.
3. Conservar artefactos, parámetros y curvas completas; verificar que reconstruyen predicciones de desarrollo. Las matemáticas de los modelos ya existen: no duplicarlas ni modificar las versiones aprobadas.
4. Congelar objetivo, población, catálogo, preprocesador, modelos, umbral, baselines y protocolo con hashes. Registrar la apertura de test antes de una única extracción posterior. Reservar 2016; la cobertura ya conocida de enero–mayo proviene de la auditoría, no se usa para seleccionar el objetivo. No ajustar decisiones después de mirar sus outcomes.

## Métricas y comparación

BA=(recall+specificity)/2 es primaria, junto con accuracy, precision, recall, specificity, F1 y matriz [[TN,FP],[FN,TP]], clase positiva 1. Reportar soportes y tasa de positivos predichos. Si no se predice ningún positivo, precision/F1 serán 0 con anotación; si falta una clase real, BA será no definida y se anotará el soporte, conservando la fila mensual. No omitir meses monoclase ni calcular promedios de BA omitiéndolos como si fueran BA global. La BA global se obtiene de los conteos agregados. Toda excepción debe resolverse explícitamente en el adaptador de reporte, sin cambiar métricas congeladas de los otros experimentos.

Baseline mayoritario: clase más frecuente de train, empate a 0 (aquí 0), congelada sin recalcular con val/test. Baseline histórico causal para A: pronosticar N(i,t)=N(i,t-1) y clasificar `1[N(i,t-1)>mediana(N(i,t-1),N(i,t-2),N(i,t-3))]`. Usa los mismos tres meses y exactamente las mismas claves que modelos/mayoritario. No se requiere población secundaria de cuatro meses. Copiar la etiqueta del mes anterior sería otro baseline, no este. Las BA de referencia en desarrollo (64,136%/59,749%) no son estimaciones de test.

Evaluación global y por mes, con soporte, prevalencia, confusiones y métricas; errores por artículo y cohorte. Cohortes futuras: visto en train, visto solo en validación y nuevo en test, según primera aparición conocida antes de cada objetivo, sin seleccionar por aparición futura. Un artículo nuevo sin tres meses suficientes queda excluido con motivo; al acumularlos entra con parámetros train congelados. No garantizar que test tenga nuevas identidades evaluables ni ambas clases. Desglosar concentración de positivos en pocas identidades y distinguir muestras dependientes de réplicas independientes.

Accuracy de 98–99% no prueba utilidad para A; hay muy pocos positivos y el mayoritario ya alcanza 98,130%/98,668% en desarrollo. No buscar equilibrio con test, no eliminar artículos por no recibir después ni remuestrear las particiones para elevar puntuaciones. Si los clasificadores no mejoran el histórico, ese es un resultado válido. No demostrar superioridad universal, significación estadística ni valor económico con 35 positivos de validación y solo nueve artículos positivos. Costes de FP/FN y carga de recepción por evento requieren validación empresarial adicional.

## Componentes disponibles y límites de reutilización

| Necesidad | Componente existente | Uso posterior y precaución |
|---|---|---|
| Clasificador lineal | `src/models/perceptron.py`, PerceptronSimple | Dimensión dinámica, sin reimplementar actualización; conservar checkpoint y sesgo |
| Probabilidad binaria | `src/models/logistic_regression.py`, LogisticRegressionManual | Sigmoide, pérdida estable y gradiente ya implementados; congelar umbral |
| Preprocesado | `src/pipeline/preprocesamiento.py`; `src/datamarts/transacciones/preprocesamiento.py` | Reusar reglas train-only; ambos catálogos son específicos de sus experimentos, requieren adaptador de Inventario posterior sin parchear FEATURES globales |
| Métricas | `src/evaluation/metricas.py` | Reusar conteos/métricas; añadir tratamiento explícito de meses monoclase en reporte nuevo, sin modificar los resultados antiguos |
| PCA SVD | `src/visualization/pca.py`, PCA2D | Solo visualización posterior train-only, no cambiar entradas de modelos; no se ejecutó en esta investigación |
| Gráficos | `src/evaluation/graficos.py`, `graficos_logistica.py`, `graficos_finales.py`; `src/visualization/dispersion.py` | Reutilizar estilos y funciones generalizables; separar rutas/etiquetas de Inventario |
| Convergencia | `src/experiments/entrenar_perceptron.py`, `entrenar_logistica.py`, `entrenar_transacciones.py` | Reusar historial, selección de checkpoint y verificación; no ejecutar scripts con rutas de otros experimentos |
| Evaluación reproducible | `src/experiments/evaluar_transacciones.py`, `evaluacion_final.py` | Reutilizar patrón de manifiestos, hashes y registro previo de apertura, con adaptador independiente; no ejecutar sus entradas contra este mart |
| Lectura SQL | `src/db/conexion.py`, Configuracion/Lector/validar_select | SELECT únicamente, autenticación integrada, extracción fija antes de 2016 para diseño |

Los componentes de experimentos no son runners genéricos listos para Inventario: hay que adaptar orquestación/catálogos/rutas en trabajo posterior. No se implementó de nuevo matemática alguna, ni se entrenó, ejecutó PCA o construyó matrices. No se modificó `informe/`, Ventas ni Transacciones. Se recomienda construir A posteriormente, manteniendo todas las restricciones y la reserva de test.
""")
