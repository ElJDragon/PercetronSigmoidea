"""Dictamen financiero de desarrollo; no construye matrices ni ajusta modelos."""
import csv
from .transacciones_viabilidad import BASE, OUT, load
from src.db.auditoria_dw import guardar, sha


def feature_catalog():
    out=[]
    def add(name,formula,source,grain,availability,status,reason):
        out.append(dict(nombre=name,formula=formula,fuente=source,granularidad=grain,disponibilidad=availability,estado=status,justificacion=reason))
    for k in [1,2,3]:
        add(f'cobros_lag_{k}',f'N(c,t-{k})','CustomerTransactions, tipo 3, DISTINCT CustomerTransactionID','Cuenta cliente-mes',
            'Antes de inicio(t), mes anterior cerrado','permitido_contrato_academico','Eventos de pago con identidad y fecha verificadas; sin saldos. Cobertura académica explícita.')
    for name,formula in [('cobros_media_3','(n1+n2+n3)/3'),('cobros_mediana_3','mediana(n1,n2,n3)'),('cobros_std_3','std(n1,n2,n3), ddof=0'),('cobros_cambio_1','n1-n2'),('cobros_tendencia_3','(n1-n3)/2'),('meses_con_cobros_3','SUM 1[nk>0], k=1..3'),('mes_anterior_sin_cobros','1[n1=0]')]:
        add(name,formula,'Agregados de eventos tipo 3','Cuenta cliente-mes','Antes de inicio(t)','permitido_contrato_academico','Hereda identidad, cobertura y disponibilidad de los tres lags; cero solo en libro cubierto.')
    add('variacion_relativa_cobros','(n1-n2)/n2 si n2>0; nulo en otro caso','Agregados de eventos tipo 3','Cuenta cliente-mes','Antes de inicio(t)',
        'permitido_contrato_academico','No dividir entre cero; opcional y acompañada por indicador de base cero.')
    add('base_variacion_cero','1[n2=0]','Agregados de eventos tipo 3','Cuenta cliente-mes','Antes de inicio(t)','permitido_contrato_academico','Depende de cobertura que permita el cero.')
    add('antiguedad_observada_meses','ordinal(t)-ordinal(primer mes de cobro conocido antes de t)','CustomerTransactions tipo 3','Cuenta cliente-mes','Antes de inicio(t)',
        'permitido_contrato_academico','Primera observación, no fecha real de alta; no usar último evento futuro ni supervivencia.')
    for name,formula in [('mes_sin','sin(2*pi*(mes(t)-1)/12)'),('mes_cos','cos(2*pi*(mes(t)-1)/12)'),('dias_mes_objetivo','Días calendario de t')]:
        add(name,formula,'Calendario civil','Cuenta cliente-mes','Conocida antes de inicio(t)','permitido','Calendario conocido; no acredita actividad ni cobertura.')
    add('importe_cobrado_lag_k','-SUM TransactionAmount(c,t-k), solo tipo 3','CustomerTransactions.TransactionAmount','Cuenta cliente-mes',
        'Antes de inicio(t), eventos íntegros y moneda verificada','condicionado_desactivado','Signo y reconocimiento comprobados, pero el esquema extraído no certifica código de moneda o ausencia de mezcla/conversión; fuera del dataset recomendado.')
    add('pagos_proveedor_lag_k','COUNT DISTINCT SupplierTransactionID(s,t-k), tipo 7','SupplierTransactions','Proveedor-mes separado','Antes de inicio(t)',
        'condicionado_otro_objetivo','No mezclar con cuentas cliente; seis proveedores y solo dos series persistentes: no es la población elegida.')
    add('atributos_dimensionales','Atributo en intervalo único [Valid From,Valid To) válido antes de t','Dimensiones relacionadas','Entidad-mes','As-of anterior a t',
        'condicionado_desactivado','No son necesarios para conteos; miembros desconocidos y versiones pueden romper identidad o multiplicar filas.')
    for name,formula,why in [('saldos_y_estados_actuales','OutstandingBalance, IsFinalized, FinalizationDate actuales','No son snapshots de meses anteriores.'),('mes_objetivo','N(c,t), importe(c,t), evento de t como X','Información posterior al instante; N(t) solo para y.'),('identificadores_continuos','Customer Key, Supplier Key, IDs como cantidades','Identificadores se usan para unión/agrupación, no magnitudes.'),('fecha_edicion_como_alta','LastEditedWhen como creación del evento','Una edición no certifica la primera inserción.'),('cualquier_tipo_negativo_como_cobro','SUM/COUNT de importes negativos sin tipo y dominio','El mismo signo describe cobro de cliente o pago a proveedor; crédito/refund tiene otra semántica.')]:
        add(name,formula,'OLTP/DW','Evento/entidad','No autorizable en esa forma','prohibido',why)
    return out


def write(name,text):
    (OUT/name).write_text(text.strip()+'\n',encoding='utf-8',newline='\n')


def exportar():
    d=load('comprobaciones_focalizadas.json');g=load('granularidades.json');cand=load('etiquetas_candidatas.json');profiles=load('perfiles_tipos.json')
    a=cand['customer_business_ledger_academic']['summary'];b=cand['supplier_business_ledger_academic']['summary']
    catalog=feature_catalog();guardar(OUT/'catalogo_caracteristicas.json',catalog)
    assumptions=[
        'TransactionDate representa el día de registro empresarial del pago/cobro; no se afirma una hora auditada de inserción.',
        'El libro de eventos de desarrollo conserva los pagos registrados sin borrados o reclasificaciones retrospectivas relevantes no observados. Los módulos locales y las versiones actuales son evidencia de apoyo, no bitácora exhaustiva.',
        'Las extracciones conciliadas OLTP–DW y presencia de pagos en los 36 meses justifican cobertura académica mensual del libro desde enero de 2013; no certifican conciliación bancaria ni completitud absoluta.',
        'Bajo esa cobertura explícita, ausencia de un pago en un mes posterior a primera observación significa cero eventos registrados, no incapacidad de pagar ni ausencia de ventas.',
        'La disponibilidad retrospectiva de eventos se aproxima por su día empresarial y el control observable de versión. Para todos los pagos seleccionados, última edición coincide con ese día; no se necesita inferir saldos anteriores.'
    ]
    coverage={'status':'justified_academic_business_event_ledger_not_absolute_certification','scope':'Eventos tipos 3/7 con TransactionDate en 2013-01..2015-12; dominios separados',
        'confirmed':['Biyectividad de 86475 transacciones OLTP–DW en desarrollo sin discrepancias de fecha/tipo/importes/estado actual','Tipos de pago presentes en los 36 meses de desarrollo','23338 pagos/cobros con última edición en el propio día empresarial','Módulos locales insertan pagos separados y actualizan saldos/finalizaciones de obligaciones; no se ejecutó SQL de modificación'],
        'assumptions':assumptions,'unknowns':['No hay auditoría exhaustiva de borrados, anulaciones, escrituras desde clientes externos o backdating','No existe snapshot mensual ni bitácora de cada estado; no se necesitan para el objetivo de conteo, sí para saldos','No se certifica latencia ETL histórica o conciliación bancaria'],
        'zero_policy':'Solo business_ledger_academic explícito, después de primera observación y dentro de meses cubiertos; unknown nunca rellena cero',
        'unknown_coverage_sensitivity':'Solo meses con eventos observados en t y tres anteriores; resultado condicionado por actividad de t, no población alternativa aprobada',
        'production_status':'Requiere cierres/latencias y reconciliación operacional antes de uso operativo'}
    guardar(OUT/'cobertura_y_supuestos.json',coverage)
    decision={'recommended_candidate':'A: frecuencia mensual de cobros registrados de cliente superior a mediana reciente',
        'status':'viable_under_explicit_academic_assumptions','granularity':'CustomerTransactions.CustomerID (cuenta Bill To) + mes objetivo',
        'dataset_construction_viable_under_documented_academic_contract':True,'unconditional_or_operational_approval':False,
        'construction_executed':False,'matrices_generated':False,'training_performed':False,'pca_performed':False,
        'test_2016_queried':False,'authorized_future_scope':'Solo candidato A, conteos históricos tipo 3 y calendario; modo académico de cobertura explícito; ninguna matriz en este encargo',
        'development_support':{s:{k:v for k,v in a[s].items() if k not in ['monthly','yearly']} for s in a},
        'candidate_B':{'status':'conditional_secondary_not_selected','reason':'Solo seis proveedores y dos series persistentes; cuatro aportan ceros tras enero. No recomendado como siguiente entrenamiento.'},
        'candidate_C':{'status':'conditioned_currency_verification_pending_not_selected','reason':'Signo, importe y reconocimiento del pago verificados; no hay código monetario/certificación de unidad única en las fuentes consultadas. No se calculan etiquetas de volumen ni se habilitan importes como X.'},
        'assumptions':assumptions,'next_action':'Detenerse tras este dictamen; construir el dataset solo en un encargo posterior con este contrato explícito, sin avanzar automáticamente a Inventario'}
    guardar(OUT/'dictamen.json',decision)
    contract={'status':'academic_event_contract_ready_for_subsequent_construction_not_operationally_certified',
      'identity':'(customer, CustomerTransactions.CustomerID, target_month); source transaction identity=(side,event_id)',
      'population':'Cuentas con al menos un cobro previo y tres meses calendario posteriores a su primera observación; no incluir cuentas por su actividad futura',
      'event_filter':{'side':'customer','TransactionTypeID':3,'TransactionTypeName':'Customer Payment Received'},
      'prediction_time':'inicio del primer día calendario de t, tiempo empresarial local; sin inventar hora de inserción',
      'horizon':'mes calendario t completo [inicio(t),inicio(t+1))',
      'N':'COUNT DISTINCT CustomerTransactionID con tipo 3 y TransactionDate dentro del mes de la cuenta',
      'R':'median(N(c,t-1),N(c,t-2),N(c,t-3))','y':'1[N(c,t)>R(c,t)]','tie_class':0,
      'target_meaning':'Más registros de cobro que su referencia reciente; no más ingresos por ventas, cobro de cada factura, impago ni mayor liquidez monetaria',
      'splits':{'train':['2013-04','2014-12'],'validation':['2015-01','2015-12'],'reserved':'2016 sin lectura de outcomes'},
      'label_cuts_exclusive':{'train':'2015-01-01 00:00:00','validation':'2016-01-01 00:00:00'},
      'availability':'Tres meses completos previos con eventos registrados antes de inicio(t); objetivo completo disponible antes del corte. Las versiones de pago actuales cumplen este control, sin exigir snapshots de saldos.',
      'zero_policy':coverage['zero_policy'],'coverage':coverage,'missing':'Identidad/fecha/tipo o cobertura desconocidos => indeterminado, no cero ni clase negativa por defecto',
      'features':catalog,'dimensions':'No se requieren atributos dimensionales. Recuperar cuenta operacional por evento; claves cero no identifican una cuenta. No joins por nombre ni versionar identidad por PK sustituta.',
      'allowed_future_construction':True,'requires_explicit_academic_coverage_mode':True,'operational_use_certified':False,
      'data_and_models_created_now':False,'decision':decision}
    guardar(OUT/'contrato_datos.json',contract)
    fields=['candidate','coverage','split','month','eligible','positive','negative','positive_proportion','target_zero','reference_zero']
    with (OUT/'distribuciones_mensuales.csv').open('w',encoding='utf-8',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader()
        for name,v in cand.items():
            for split,s in v['summary'].items():
                for row in s['monthly']:w.writerow(dict(candidate=name,coverage=v['coverage'],split=split,**row))
    modules=load('modulos_financieros.json');evidence=[]
    for r in modules:
        text=(r['definition'] or '').replace('\r','');snippets=[]
        if r['module_name']=='Configuration_ApplyDataLoadSimulationProcedures':
            for needle in ['CREATE PROCEDURE DataLoadSimulation.ProcessCustomerPayments','CREATE PROCEDURE DataLoadSimulation.PaySuppliers']:
                i=text.find(needle);j=text.find("END;';",i)
                if i>=0:snippets.append(text[i:j+7] if j>=0 else text[i:i+4000])
        elif r['module_name']=='GetTransactionUpdates':snippets=[text]
        evidence.append({'schema':r['schema_name'],'module':r['module_name'],'executed':False,'snippets':snippets})
    guardar(OUT/'evidencia_semantica.json',{'source_sha256':sha(OUT/'modulos_financieros.json'),'modules':evidence})
    type_names={r['TransactionTypeID']:r['TransactionTypeName'] for r in load('tipos_operacionales.json')}
    type_table='\n'.join(f"| {p['side']} | {p['type_id']}: {type_names[p['type_id']]} | {p['events']} | {p['positive_amount']} | {p['negative_amount']} | {p['zero_amount']} | {p['entities']} |" for p in profiles)
    write('semantica_financiera.md',f'''# Transacciones financieras: semántica y evidencia

## Alcance focalizado

Se leyeron las auditorías existentes antes de consultar. Las nuevas consultas solo inspeccionan eventos de Sales.CustomerTransactions y Purchasing.SupplierTransactions, Fact.Transaction, tipos y medios de pago, y dimensiones relacionadas para identidad/versiones. No se consulta Fact.Sale, no se reauditan otros procesos ni se modifica su contrato. Todas las consultas emitidas son SELECT; las definiciones SQL se leen como texto, nunca se ejecutan. readonly/ApplicationIntent son indicaciones del driver, no una concesión de permisos reducidos.

## Qué representa una fila

Cada fila de Fact.Transaction representa una transacción operacional de **uno de dos dominios**, cliente o proveedor; no una línea de venta, un saldo mensual ni una asignación pago-factura. La identidad correcta es (dominio, ID transacción operacional), no solo el número. En desarrollo hay {d['dw_count']} filas DW: {d['source_counts']['customer']} de cliente y {d['source_counts']['supplier']} de proveedor. El cotejo por ID de origen es uno a uno y exhaustivo para las filas actuales, sin discrepancias de fechas, tipos, importes, documento o estado actual y sin multiplicación. No hay IDs numéricos compartidos entre dominios en esta muestra; aun así se mantienen espacios separados.

| Dominio | Tipo observado | Eventos desarrollo | Importes positivos | Negativos | Cero | Entidades |
|---|---|---:|---:|---:|---:|---:|
{type_table}

El catálogo contiene Customer Invoice (1), Customer Credit Note (2), Customer Payment Received (3), Customer Refund (4), Supplier Invoice (5), Supplier Credit Note (6), Supplier Payment Issued (7), Supplier Refund (8) y Customer Contra (13). Los tipos 9..12 corresponden a movimientos de stock, no eventos de cobro. En las fuentes financieras de desarrollo solo aparecen 1,3,5,7: **no hay notas de crédito, reembolsos, contra ni importes cero observados**. Su ausencia en esta copia no demuestra que sean imposibles. Las notas de crédito ajustan la obligación y los refunds son devoluciones; no se agrupan con pagos ni se les atribuye un signo observado inexistente. Los medios operacionales son Cash, Check, Credit-Card y EFT; los pagos examinados son EFT. Los nombres se verifican en catálogos locales, no se infieren del signo.

## Signos, documentos e importes

Una factura cliente positiva registra una obligación a cobrar; no acredita cobro efectivo. Un cobro cliente tipo 3 negativo reduce la cuenta a cobrar y corresponde a entrada registrada de dinero. Una factura proveedor positiva registra una obligación a pagar; no es ingreso. Un pago proveedor tipo 7 negativo reduce esa obligación y corresponde a salida registrada. Por tanto, **el mismo signo negativo describe flujos de caja opuestos según dominio y tipo**. El análisis mide registros del libro, no una conciliación con extractos bancarios.

Las 61.320 facturas cliente tienen InvoiceID y los 1.817 documentos proveedor tienen PurchaseOrderID. Los 23.018 cobros y 320 pagos tienen documento nulo: es ausencia estructural, no pérdida de identidad. La plantilla de simulación agrupa obligaciones pendientes por cuenta y crea un registro de pago con importe negativo de su suma; no conserva una tabla de aplicación pago-factura. No se atribuye cada cobro a una sola factura ni se inventa una conciliación exacta. En facturas, TransactionAmount=AmountExcludingTax+TaxAmount; en pagos los dos componentes fiscales son cero y el importe de pago sigue siendo negativo: esa igualdad fiscal no se impone a tipos de pago.

TransactionAmount es el importe del evento, no OutstandingBalance. Su signo y reconocimiento se cotejan con OLTP y el código local. Las fuentes consultadas no aportan código monetario ni certifican moneda única/conversiones: no se etiqueta volumen financiero ni se habilitan importes como predictores del contrato recomendado. Nunca se afirma AUD/USD u otra divisa sin evidencia local. Los conteos de eventos no dependen de la moneda.

## Fechas: evento frente a estado

Integration.GetTransactionUpdates asigna Date Key=CAST(TransactionDate AS date) en ambos dominios. TransactionDate de una factura es su día empresarial de registro; para tipo 3/7 es el día del cobro/pago registrado. No se confunde con la fecha de factura subyacente: el pago no lleva InvoiceID ni PurchaseOrderID. FinalizationDate y saldos son estados de la obligación, modificados al liquidar; LastEditedWhen es última modificación, **no nacimiento auditado**. No conocemos la hora exacta de creación de cada evento.

Los 23.338 pagos/cobros tienen última edición el mismo día de TransactionDate y finalización ese mismo día en la copia. Los módulos ProcessCustomerPayments/PaySuppliers insertan un pago con TransactionDate=CAST(@StartingWhen AS date), separadamente de actualizar OutstandingBalance, FinalizationDate y LastEditedWhen de obligaciones. Esa evidencia respalda reconstruir conteos de pagos, sin exigir conocer el saldo anterior de la factura. No equivale a una bitácora completa ni prueba que ningún cliente externo pudo modificar o borrar una transacción.

Ambas tablas de transacción son no temporales y no tienen CDC/change tracking/triggers observados. El código financiero local inspeccionado actualiza estados y no muestra borrados de los eventos; no se auditó el log transaccional ni todas las escrituras externas. Se distingue evidencia textual del mecanismo y prueba de la historia individual. Los snapshots serían necesarios para consultar saldos históricos; **su ausencia no invalida automáticamente el conteo de eventos de pago** con identidad, fecha y cobertura justificadas. Campos de estado quedan prohibidos como X y no deciden la inclusión del evento de pago.

## Identidad y dimensiones

La cuenta de cobro es CustomerTransactions.CustomerID: corresponde a Bill To, no necesariamente al cliente destinatario de una factura. Integration usa COALESCE(Invoice.CustomerID, ct.CustomerID) para Customer, pero ct.CustomerID para Bill To. Se usa el ID operacional de cuenta mediante el evento, sin reagrupar por nombre o clave sustituta. Hay 43.578 transacciones de cliente con Bill To desconocido en DW, incluidos 21.142 cobros; no se eliminan ni se fusionan en una cuenta cero. Todas recuperan su cuenta desde el ID único de transacción OLTP, sin discrepancias en los matches dimensionales conocidos.

Customer tiene 403 IDs incluyendo desconocido y una versión por ID en la ventana dimensional; Supplier tiene hasta tres versiones y un par de intervalos superpuestos para ID 3. Tipo y medio se consultan por claves sustitutas para cotejo y por sus IDs operacionales estables para semántica. No se usan atributos dimensionales como X ni joins de rango arbitrarios. Si se agregaran atributos, exigir exactamente una versión [Valid From,Valid To) válida y conocida al instante de predicción; no elegir clave mayor ni versión futura. La vigencia hasta 9999 no es un período observado.

## Cobertura y diferencia respecto a Ventas

Los cuatro tipos observados están presentes en los 36 meses enero 2013..diciembre 2015, y cada evento de la copia OLTP concilia con DW. No se usó 2016. Las huellas COUNT/CHECKSUM antes/después coinciden; son controles no criptográficos sin aislamiento snapshot, no certificados de completitud histórica.

El objetivo nuevo cuenta cobros asentados por cuenta pagadora, frente al objetivo de Ventas sobre unidades por producto. No se vuelve a etiquetar Fact.Sale: las 61.320 facturas y los 23.018 cobros son eventos distintos con distintos IDs, fechas y grano. Un pago puede cubrir varias obligaciones y no revela unidades vendidas. En WWI la simulación sí vincula los pagos a obligaciones generadas por facturación: no se presume independencia económica ni se presenta como prueba de comportamiento bancario real.

La cobertura es razonable **bajo supuestos académicos explícitos de libro de eventos conservado**, no una certificación absoluta. La conciliación y los 36 meses sustentan ausencia observada en el libro, sin descartar borrados/backdating no registrados. No se impone bloqueo automático por carecer de snapshots; tampoco se convierte una ausencia fuera del alcance o con extracción incompleta en cero. El detalle y riesgos se conservan en cobertura_y_supuestos.json.
''')
    tables=[]
    for name,v in cand.items():
        for split,s in v['summary'].items():
            pct='No estimable' if s['positive_proportion'] is None else f"{s['positive_proportion']*100:.4f}%"
            tables.append(f"| {name} | {split} | {s['eligible']} | {s['positive']} | {s['negative']} | {pct} | {s['entities_switching_class']} | {s['entities_single_class']} |")
    write('analisis_etiquetas.md','''# Transacciones: etiquetas de desarrollo y sensibilidad

## Propuestas y criterio de elección

A: frecuencia de cobros registrados por cuenta cliente-mes. N(c,t)=COUNT DISTINCT CustomerTransactionID, **solo tipo 3**, por CustomerID operacional y TransactionDate del mes. R(c,t)=mediana(N(c,t-1),N(c,t-2),N(c,t-3)); y=1[N(c,t)>R(c,t)]. Empate=0. Se predice al inicio de t para el mes completo. La acción empresarial potencial es anticipar carga de conciliación/gestión de cobros, no impago, ingreso de ventas o liquidez monetaria elevada. No se selecciona por balance: se elige por semántica de eventos, continuidad y variación dentro de cuentas.

B: misma regla con SupplierTransactionID tipo 7 y SupplierID, población proveedor separada. Interpretable como mayor frecuencia de pagos emitidos, pero solo seis proveedores; cuatro tienen actividad únicamente en enero 2013 y solo dos son persistentes. La existencia de dos clases no ofrece por sí sola generalización suficiente. Se mantiene condicionado y no elegido para el primer dataset.

C: mayor importe mensual de cobros o pagos, usando -SUM TransactionAmount dentro de dominio y tipo correctos. El signo y reconocimiento del evento están comprobados, pero falta acreditar moneda/unidad única. No se calcula su prevalencia ni se habilitan importes en el contrato recomendado. No se suman signos de facturas y pagos ni se llama ingreso a toda cantidad positiva. Estado: condicionado a moneda y objetivo empresarial; no viable con evidencia confirmada todavía.

Cliente-tipo-mes con tipo 3 fijo es exactamente A; incorporar tipos de factura o mezclar clientes/proveedores cambia el problema y no crea nuevas observaciones válidas de cobros. No se proponen clasificadores que aprendan el tipo a partir de su signo. No se cambian mediana, horizonte o umbral para equilibrar clases.

## Grano, historia y ceros

Hay 257 cuentas con cobros y 7.709 cuenta-meses con eventos. Desde primera observación hasta diciembre 2015 hay 8.320 casillas posibles y 611 ausencias; nunca se fabrican meses anteriores a primera aparición. Solo 13 cuentas tienen cobros en los 36 meses: se incluye actividad intermitente y primeras apariciones, sin exigir supervivencia futura. En proveedores hay seis identidades, 76 meses-entidad activos y 140 ausencias entre 216 casillas posteriores a primera observación.

Elegibilidad: una cuenta ya observada en cobros, al menos tres meses calendario de historia posterior a su primera observación y meses cubiertos. El primer mes objetivo es tres meses después del primer mes de pago. Se mantienen cuentas cuando dejan de aparecer; no se selecciona por el último evento futuro. Las nuevas cuentas con menos historia quedan indeterminadas. Una cuenta solo facturada pero nunca vista en cobros no pertenece a esta población inicial de pagadores conocidos; no se afirma cubrir todo el universo de clientes.

business_ledger_academic interpreta ausencia dentro del libro de eventos cubierto como cero **eventos registrados**, bajo supuestos documentados. No significa cero ventas ni impago. unknown no rellena ceros: solo etiqueta casillas con actividad observada en t y tres meses anteriores. Esa segunda distribución es sensibilidad censurada por actividad del mes objetivo, no población alternativa de predicción aprobada. Sin cobertura, una ausencia se mantiene indeterminada.

## Conteos y clases

Train usa meses objetivo abril 2013..diciembre 2014 y etiquetas disponibles antes de 2015-01-01. Validación usa 2015 y etiquetas antes de 2016-01-01. En todas las variantes el objetivo debe estar disponible antes del corte, incluso en la sensibilidad sin filtro de edición de X. Los pagos tienen edición en su propio día: el control de disponibilidad observado no elimina ningún caso de los paneles académicos. Se excluyen estados posteriores y no se abren outcomes de 2016.

| Candidato / escenario | Partición | Elegibles | Positivos | Negativos | Positivos % | Entidades con cambio de clase | Entidades de una clase |
|---|---|---:|---:|---:|---:|---:|---:|
'''+'\n'.join(tables)+'''

## Interpretación y sensibilidad

A tiene 4.612 casos train (1.798 positivos, 2.814 negativos; 38,9853%) y 2.943 de validación (1.198 positivos, 1.745 negativos; 40,7068%). Hay 236/252 cuentas elegibles; 233/248 cambian de clase y 3/4 permanecen en una clase dentro de cada partición. No depende exclusivamente de separar identidades activas e inactivas. El escenario unknown reduce a 3.258/2.290 casos y 1.253/930 positivos, con prevalencias 38,4592%/40,6114%; el soporte cambia, pero el balance no nace exclusivamente de rellenar ceros. No se elige ese escenario por su prevalencia ni se interpreta como prueba absoluta de completitud.

En A académico hay 380/182 targets cero y 86/31 referencias cero. Cuando R=0, cualquier cobro del mes es clase 1 por definición; no se usa un percentil nuevo para evitarlo. En B académico hay 126/72 casos, 16/8 positivos y 110/64 negativos; los ceros de cuatro proveedores explican buena parte del soporte. Sin ausencias solo hay 42/24 casos, con 16/8 positivos, todos de dos proveedores. Esta diferencia es una advertencia sobre población y generalización, no motivo para borrar retrospectivamente proveedores inactivos.

etiquetas_candidatas.json incluye distribución mensual, anual y por entidad, cambios de clase, ceros y exclusiones. distribuciones_mensuales.csv permite revisar meses con ambas clases o soporte reducido. No se promedian silenciosamente prevalencias de meses con tamaños distintos. Los números describen un libro retrospectivo bajo los supuestos declarados y no constituyen resultados predictivos. No se construyeron matrices, estimaron baselines, entrenaron modelos o ajustaron umbrales.

## Dictamen por objetivo

- A: **viable bajo supuestos académicos explícitos**, con evidencia local de apoyo; autorizable para una construcción posterior del dataset de conteos y calendario bajo este contrato, sin autorización de uso operativo.
- B: condicionado por soporte y alcance empresarial; no elegido para entrenamiento de este proyecto ahora.
- C: condicionado a unidad monetaria verificable; sin etiquetas calculadas ni importes habilitados.

No hay certificación absoluta de eventos no borrados, backdating o completitud bancaria. Es una limitación del alcance académico, no razón automática para exigir snapshots de saldos que el objetivo no usa. Si se detectan lagunas o correcciones relevantes, esos meses dejan de estar cubiertos y deben permanecer indeterminados o reconstruirse con evidencia. Se detiene la investigación aquí.
''')
    feature_table='\n'.join('| '+' | '.join(str(r[k]) for k in ['nombre','formula','fuente','granularidad','disponibilidad','estado','justificacion'])+' |' for r in catalog)
    assumption_list='\n'.join(f'{i}. {v}' for i,v in enumerate(assumptions,1))
    write('contrato_datos.md','''# Contrato financiero: conteo de cobros por cuenta

**Estado: viable para un experimento retrospectivo académico bajo los supuestos explícitos de este contrato.** No es certificación operativa y no se construye ningún dataset en este encargo. Las variantes proveedor/importe quedan fuera del alcance autorizado. Fuente estructurada: contrato_datos.json y dictamen.json.

## Identidad, objetivo y tiempo

Clave de observación futura: (dominio customer, CustomerTransactions.CustomerID, mes objetivo). ID de evento: (dominio, CustomerTransactionID). La cuenta es la pagadora Bill To operacional, recuperada mediante el ID de transacción cuando DW contiene miembro desconocido. No se agrupa por nombres o claves sustitutas ni se incluyen IDs como magnitudes continuas.

Población: cuentas con un cobro previamente observado y tres meses calendario completos desde su primer mes de cobro, incluyendo meses posteriores sin actividad dentro del alcance cubierto. Primera observación no se presenta como alta real. No se usan últimas apariciones, actividad del target o identidad del catálogo actual para seleccionar filas. Las cuentas nuevas y nunca vistas en cobros quedan fuera de este primer universo, explícitamente.

Instante: inicio del primer día de t en tiempo empresarial local. Horizonte: mes calendario completo [inicio(t),inicio(t+1)). No se inventa hora de inserción: TransactionDate es fecha empresarial diaria. N(c,t)=número de IDs únicos **tipo 3 Customer Payment Received**, sin filtrar por saldo o estado actual. R=mediana de los conteos de t-1,t-2,t-3 consecutivos; y=1[N(c,t)>R]; empate=0. Solo N(t) participa como outcome, nunca como X. La etiqueta significa frecuencia registrada superior a la referencia, no probabilidad de impago, importe cobrado ni unidades vendidas.

Train: objetivos 2013-04..2014-12, outcome conocido antes de 2015-01-01 00:00. Validación: objetivos 2015-01..2015-12, outcome antes de 2016-01-01 00:00. Cortes exclusivos. Test de 2016 reservado, sin consulta ni prevalencia en este diseño. Historia de validación puede usar meses anteriores de 2015 si eran conocidos antes de t; el preprocesador y modelos seguirán congelados tras train.

## Cobertura y disponibilidad

El modo de cobertura deberá registrarse expresamente como business_ledger_academic; no activar ceros por defecto. La ausencia se convierte en cero solo para un mes cubierto del libro y después de primera observación. Cobertura desconocida, extracto fallido, identidad/tipo/fecha faltante o duplicado => error o indeterminado; nunca imputación a clase 0. Los tres lags esenciales no se imputan para fabricar historia. En un mes parcialmente extraído no se cuentan solo las filas disponibles como si fueran el total.

Se exige identidad de evento estable y fecha empresarial confiable dentro de los siguientes supuestos, apoyados por el cotejo y las fechas actuales:

'''+assumption_list+'''

La versión persistida de los pagos tiene última edición el mismo día empresarial; su máximo por agregado es anterior a inicio(t) para lags y anterior al corte para y. Se verifica como control adicional observable, sin llamarlo fecha de alta ni imponerlo a la historia de estados de facturas. No se exige snapshot mensual para contar estos eventos. Si un evento apareciera con versión posterior o se detectara una corrección, conservar el agregado como desconocido o reconstruirlo con evidencia, sin eliminar solo esa fila ni asumir que nunca cambió.

No reconstruir OutstandingBalance o IsFinalized de meses anteriores a partir del último valor. FinalizationDate no es la fecha empresarial del pago para otras clases; no se usa para seleccionar ni para generar X. No se cruza un pago con una factura suponiendo que NULL InvoiceID equivale a dato perdido.

## Catálogo causal

Permitido_contrato_academico requiere cobertura y supuestos anteriores. Permitido refiere a calendario conocido. Condicionado permanece desactivado; prohibido no entra en el futuro dataset.

| Nombre | Fórmula | Fuente | Grano | Disponibilidad | Estado | Justificación |
|---|---|---|---|---|---|---|
'''+feature_table+'''

No se necesitan atributos dimensionales para el objetivo elegido. Si se añadieran después, un join as-of debe devolver exactamente una versión válida y disponible; miembro desconocido, cero o múltiples matches no se resuelven eligiendo una PK mayor o una versión futura. El evento OLTP es la fuente de identidad financiera en este contrato.

## Alcance de la autorización y cierre

Es viable construir posteriormente el dataset académico **solo del candidato A**, con 4.612 observaciones train y 2.943 de validación bajo las reglas documentadas. Una implementación futura debe registrar los supuestos, comprobar nuevamente cobertura/identidad/cortes y conservar sus huellas; estos conteos no son una matriz definitiva ni garantía si la fuente cambia. El modo de cobertura desconocida mantiene los meses sin eventos como indeterminados y no puede presentarse como población completa.

La autorización no cubre importes, saldos, evaluación de test, uso operativo o entrenamiento en este encargo. No se pide certificación absoluta inexistente para un conteo académico justificable, pero el alcance y el riesgo deben acompañar sus resultados. Se preservan Ventas, Pedidos, Compras y el informe. No se avanza automáticamente a Inventario/Movimientos.
''')
    write('diseno_evaluacion.md','''# Evaluación futura de frecuencia de cobros

**Diseño, sin ejecución de modelos, matrices o PCA.** Se aplica al candidato A bajo el contrato académico de eventos y cobertura explícita, no a saldos ni a importes. La elección utiliza exclusivamente desarrollo.

## Protocolo temporal

Predicción al inicio del mes t, horizonte t completo. Train: objetivos abril 2013..diciembre 2014 con etiquetas anteriores a enero 2015. Validación: 2015 con etiquetas anteriores a enero 2016. Reserva: 2016, aún sin lectura. Antes de abrir el test habrá que congelar contrato, catálogo, cobertura, algoritmo, hiperparámetros, umbral y baselines en una fase posterior; esta auditoría no autoriza su apertura.

No hay split aleatorio. Una cuenta puede aparecer en train y validación porque la pregunta es generalización temporal; no se afirma generalización a clientes nuevos. No usar actividad futura para elegir cuentas, ni imputar historia esencial. Una nueva cuenta solo es elegible tras tres meses desde primera observación de cobro. Se deben informar por separado cuentas vistas en train y nuevas elegibles en validación, con sus soportes, sin seleccionarlas por los resultados.

Fit de escala/imputación opcional exclusivamente en train y transform congelado en validación/test. Nunca reentrenar con validación mientras se informa su resultado como selección original. Calendario conocido puede calcularse por mes objetivo; el resto usa únicamente eventos anteriores a inicio(t). Los outcomes se incorporan al cerrar cada mes y cumplir el corte de su partición. No se requieren snapshots de saldos para este conteo.

## Baselines predefinidos

Mayoritaria: clase más frecuente solo en train académico autorizado, empate=0; congelada para validación/test. No se calcula ni evalúa ahora como predictor. Baseline histórico causal: 1[N(c,t-1)>mediana(N(c,t-2),N(c,t-3),N(c,t-4))], equivalente a persistencia de la última clase histórica conocida. Requiere cuarto mes de historia cubierto: se predefine una comparación común en esa intersección para todos los métodos y se informa su reducción de soporte, además del resultado de modelos en el soporte principal. No rellenar el cuarto lag para conservar filas. Nunca usar N(t) para predecirse.

No se optimizan umbrales o hiperparámetros en esta investigación. En una fase posterior, selección por balanced accuracy de validación con reglas de desempate fijadas antes de test; mismos datos y preprocesamiento comparable para perceptrón y logística. La utilidad comercial de anticipar número de registros debe evaluarse sin inventar costes monetarios por FP/FN.

## Métricas y generalización

Principal: balanced accuracy, solo estimable si ambas clases tienen soporte. Secundarias: accuracy, precision, recall, F1 y matriz de confusión [0,1]. Reportar soporte y prevalencia por mes, cuenta y estratos de historia; una alta accuracy por mayoría no demuestra habilidad. Meses/cuentas monoclase conservan sus limitaciones de recall/especificidad/balanced accuracy, sin sustituir valores no estimables por un desempeño aparente.

Average Precision opcional si se decide previamente reportar puntuaciones: probabilidades sigmoid de logística o margen firmado del perceptrón orientado a clase 1; el margen no es probabilidad calibrada. Si no hay positivos en un subconjunto, informar la limitación. No se implementa una métrica nueva ni se calcula AP ahora. Comparar resultados globales y mensuales; no usar test para elegirla después.

El conteo cambia dentro de muchas cuentas, pero las observaciones mensuales no son independientes. Para incertidumbre futura, usar bloques temporales y considerar agrupación por cuenta; no bootstrap i.i.d. ingenuo ni una prueba de generalización bancaria con esta simulación. Evaluar sensibilidad del soporte a cobertura y referencias cero. El escenario unknown está censurado por actividad del target y no sustituye automáticamente la población aprobada. Si falla la cobertura real, declarar indeterminado y revisar el soporte, no fabricar historia.

## Reutilización del proyecto

PerceptronSimple (src/models/perceptron.py), LogisticRegressionManual (src/models/logistic_regression.py), metricas_binarias (src/evaluation/metricas.py) y PCA2D (src/visualization/pca.py) pueden reutilizar sus algoritmos sin cambios. PCA será solo visualización futura, fit en train, sin ejecutar aquí.

PreprocesadorComun incorpora FEATURES/OPTIONAL específicos de Ventas y exige ese catálogo exacto: no es conectable sin adaptar la interfaz a un catálogo financiero. Se reutilizan sus reglas estadísticas mediante un adaptador configurable separado, preservando Ventas y evitando duplicar álgebra. Los gráficos son aprovechables tras adaptar títulos, rutas y metadatos. Los runners de experimentos, rutas, BOUNDS y baselines de Ventas no son universales y no deben ejecutarse sobre estos datos sin orquestación financiera propia.

Las únicas capas nuevas necesarias son extracción de eventos por dominio/tipo, conciliación de identidad, cobertura explícita, agregación mensual, etiqueta y catálogo causal, elegibilidad temporal y orquestación. La entrega actual implementa la investigación y exporta agregados; no genera X/y, preprocesador ajustado, pesos o PCA. La construcción posterior deberá mantener namespace financiero y hashes propios.

## Decisión

Candidato A viable bajo supuestos académicos explícitos; candidato B secundario condicionado por soporte; C pendiente de moneda. No existe certificación operativa absoluta ni se requiere como pretexto para bloquear un conteo retrospectivo justificado. Se detiene aquí, sin entrenamiento, apertura de test o investigación automática de Inventario.
''')


if __name__=='__main__':exportar()
