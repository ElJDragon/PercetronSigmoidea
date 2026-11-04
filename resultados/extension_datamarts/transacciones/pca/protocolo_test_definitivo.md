# Protocolo definitivo de primera apertura de test

## Estado y condiciones de apertura

Reglas congeladas antes de acceder a outcomes de 2016. Esta tarea no abre test. La primera evaluación podrá ejecutarse en un encargo posterior explícito si los controles de integridad, cobertura, cierre y disponibilidad pasan nuevamente. La fuente estructurada y sus referencias a protocolos previos están en protocolo_test_definitivo.json; no se sobrescribe ningún documento anterior.

Antes de abrir, comprobar los hashes de dataset, snapshot de desarrollo, preprocesador, catálogo, modelos, implementaciones y manifiesto actual. Verificar checkpoints 3/399, perceptrón eta=0,01/seed=7, logística eta=0,10/umbral=0,5, pesos, sesgos y orden de 16 columnas. Registrar fecha, alcance y huellas de la primera apertura. No reentrenar con train+val ni seleccionar configuraciones con test.

## Extracción y cobertura

Ventana predefinida: enero-mayo de 2016, establecida a partir de la cobertura ya auditada, no de etiquetas nuevas. Intervalo [2016-01-01,2016-06-01), corte propuesto de conocimiento exclusivo 2016-06-01 00:00. Confirmar cierre de cada mes: objetivo incompleto o no confirmado queda indeterminado; no ampliar la ventana según resultados.

Extraer únicamente cobros tipo 3 Customer Payment Received por SELECT y proyección mínima aprobada: ID único operacional, CustomerID, TransactionDate, tipo, última edición como control de versión y comprobación de signo sin exportar importes. Cotejar ID/fecha/tipo con DW sin duplicar eventos por joins. La cuenta es CustomerTransactions.CustomerID (Bill To), nunca un miembro desconocido DW, un nombre actual o una clave sustituta tratada como magnitud. Guardar snapshot, procedencia, consultas y huellas; no interpretar última edición como alta auditada.

Mantener explícitamente business_ledger_academic y sus supuestos de cobertura/conservación. Cero significa ausencia de eventos registrados en un mes cubierto posterior a primera observación. Falta de cobertura/extracto o mes incompleto significa desconocido, no cero ni clase negativa. No se afirma completitud absoluta, ausencia universal de borrados/backdating o conciliación bancaria.

## Construcción causal y etiqueta

Predicción al inicio de cada mes t. Usar snapshot de desarrollo congelado más eventos de test ya concluidos y conocidos antes de t para formar N(t-1),N(t-2),N(t-3), estadísticas autorizadas, antigüedad observada y calendario. Primera aparición se determina con cobros anteriores al instante; no usar t, supervivencia o última aparición futura para seleccionar cuentas. Nuevas cuentas requieren tres meses de historia. Saldo, finalización, importe y atributos condicionados siguen prohibidos.

Aplicar exclusivamente transform del preprocesador congelado: sin fit, nuevas constantes, reescalado ni selección de columnas. Ambos clasificadores reciben 16 entradas; PCA queda fuera de sus predicciones y solo sirve para visualización. Tras cerrar t, obtener y=1[N(t)>mediana(N(t-1),N(t-2),N(t-3))], empate 0, respetando disponibilidad antes del corte. No usar y o aciertos/errores para decidir elegibilidad.

## Poblaciones y baselines

**Principal:** todas las cuentas-mes elegibles con tres meses anteriores cubiertos y etiqueta válida. Evaluar perceptrón, logística y mayoría en exactamente esas claves. No exigir cuarto lag a los clasificadores ni a la mayoría. Perceptrón usa z>=0; logística sigmoid(z)>=0,5; mayoría fija=0 aprendida de train.

**Secundaria común:** intersección exacta donde además está disponible el baseline histórico contractual 1[N(t-1)>mediana(N(t-2),N(t-3),N(t-4))]. Evaluar los cuatro métodos sobre idénticas claves. Cuando el cuarto mes no existe antes de primera aparición, no está cubierto o no era conocido, dejar histórico indeterminado, sin imputarlo a cero ni eliminar esa fila de la evaluación principal si cumple sus reglas.

Registrar por clave estable y mes: primera aparición, elegibilidad principal, disponibilidad de etiqueta, cobertura, presencia/fecha de conocimiento de cuatro lags, elegibilidad común y motivos separados. Conservar filas excluidas y no sobrescribir predicciones anteriores. No seleccionar observaciones según error de modelo, clase o pago futuro.

## Métricas y cierre

Informar accuracy, BA, precision, recall, F1, especificidad, matriz [0,1], soporte por clase y proporción predicha positiva, globales y mensuales para cada soporte claramente nombrado. AP logística usa scores positivos y grupos de empates, sin interpolación ni papel en selección. Mantener convenciones de denominadores cero y reconocer subconjuntos monoclase. Nunca comparar n de un soporte con métricas de otro como si fueran la misma población.

Desglosar cuentas vistas en train, vistas solo en validación y nuevas elegibles de test cuando el soporte lo permita, sin filtrar por sus outcomes. Reconocer dependencia entre cuentas/meses y límites de la simulación; no inventar significación o costes monetarios. Después de abrir, ningún resultado puede alterar modelos, umbral, etiqueta, horizonte, cobertura, features, soportes o baselines. Un cambio exigiría otro experimento explícito. Conservar predicciones alineadas, exclusiones, hashes y registro de ejecución.
