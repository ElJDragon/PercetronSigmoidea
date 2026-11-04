# Dataset de desarrollo de Inventario: frecuencia de recepciones

Objetivo aprobado: N(i,t) es COUNT de IDs distintos tipo 11; y=1[N(i,t)>mediana(N(i,t-1),N(i,t-2),N(i,t-3))], empate 0. Predicción al comienzo de t. Se usaron 7298 recepciones reales de desarrollo, sin salidas, ajustes, eventos sintéticos ni rebalanceo. No hay campo de anulación en el contrato/origen auditado: no se inventó un filtro de cancelación ni se usaron estados actuales de compras.

## Procedencia y cobertura

Extracción SELECT focalizada de Warehouse.StockItemTransactions anterior a 2016, reconciliada por ID, identidad operativa, tipo, fecha, cantidad y documentos con Fact.Movement. Snapshot compacto en `recepciones_desarrollo.csv.gz`, SQL, parámetros y huellas antes/después en procedencia.json y consultas_sql.json. La primera aparición por cualquier movimiento procede de la auditoría aprobada, sin repetirla ni consultar nuevos outcomes de salida. Las versiones DW se reúnen por WWI Stock Item ID; no se usan nombres, claves sustitutas ni IDs como magnitudes en X.

Modo obligatorio `event_ledger_academic`: conservación del archivo de eventos y cierre mensual supuestos académicamente. Solo meses cubiertos sin tipo 11 valen cero. Sin modo, el pipeline bloquea; estados unknown/incomplete provocan exclusión y registro, nunca imputación de etiquetas. Los tres primeros meses por artículo no son elegibles para predicción. Para nuevos artículos no se rellena historia anterior a su primera aparición. No se reconstruyen existencias y no se consulta Fact.Stock Holding.

## Reconciliación y matrices

| Partición | X | y | Positivos | Negativos | Prevalencia |
|---|---|---|---:|---:|---:|
| Train abril 2013–diciembre 2014 | [4599, 17] | [4599] | 86 | 4513 | 1.869972% |
| Validación enero–diciembre 2015 | [2628, 17] | [2628] | 35 | 2593 | 1.331811% |

219 artículos por partición. Conteos y prevalencias mensuales reconciliados exactamente con exploración A, sin cambiar target ni población. Solo diez artículos tienen positivos en train y nueve en val; las demás filas reales se conservan. Pocos positivos y dependencia por artículo/mes limitan inferencias y utilidad. Baseline histórico causal y mayoritario se conservan por clave; no son predicciones de clasificadores. BA histórica train/val: 0.641360359/0.597493251. En meses monoclase BA no se define. Las métricas previstas siguen siendo BA primaria, accuracy, precision, recall, specificity, F1 y confusiones; no se optimizan umbrales aquí.

## Entradas y preprocesado

17 características aprobadas, mismo orden y fórmulas del catálogo. Dimensión retenida: 17; descartadas por constantes en train: {}. Historia esencial completa, sin imputación; faltantes/infinitos bloquean. Adaptador propio del esquema train-only existente, sin parchear los catálogos globales de otros procesos. Ajuste único con train, media y desviación poblacional (ddof=0), misma máscara y escala en val, sin selección por validación, PCA, remuestreo ni ponderación. X float64, y int8. Medias, escalas, máscara, orden, fórmulas y trazabilidad en preprocesador.json y caracteristicas.json.

## Artefactos, reproducción y reserva

Agregados artículo-mes, exclusiones, primeras apariciones, metadatos, objetivos y entradas crudas están separados; N(i,t) y R(i,t) no se filtran en X. Se exportan claves train/val alineadas con los arrays. Cada fila detalla sus tres meses históricos y última fecha real de evento/edición de historia. El cierre académico cubre meses sin eventos; no se presenta como timestamp técnico de observación.

Desde la raíz:

```powershell
python -m src.datamarts.inventario.construir --origen sql --cobertura event_ledger_academic
python -m src.datamarts.inventario.construir --origen snapshot --cobertura event_ledger_academic
python -m src.datamarts.inventario.construir --verificar --cobertura event_ledger_academic
python -m unittest discover -s tests -v
```

`reproducibilidad.json` contrasta dos exportaciones offline independientes byte a byte y contra los artefactos canónicos. `validaciones.json` recoge causalidad, matrices y preservación; `manifiesto.json` incluye hashes SHA-256 de fuentes y artefactos. Las huellas SQL COUNT/CHECKSUM son comprobaciones focalizadas de estabilidad, no prueba criptográfica de toda la BD ni garantía frente a escrituras externas concurrentes. Todas las consultas propias son SELECT.

`especificacion_test.json` define una construcción futura determinista; el CLI actual no extrae ni construye 2016, X_test o y_test. No se entrenan clasificadores ni se ejecuta PCA experimental. Las pruebas de la suite existente usan fixtures sintéticos; no son entrenamiento de nuevos modelos empresariales. El informe académico y los experimentos anteriores permanecen intactos. Detenerse después de la verificación de desarrollo.

## Resultado de pruebas y discrepancias históricas

30 pruebas focalizadas y 35 controles del dataset satisfactorios; reproducción idéntica de 19 artefactos. Suite completa: 367 pruebas, 358 correctas, 5 fallos y 4 errores. Los nueve casos fallidos verifican sellos previos que incluyen una versión antigua del informe. El contraste de cada diferencia con integridad_antes.json de este encargo demuestra que todas ya existían antes de construir Inventario; las rutas discrepantes son exclusivamente de informe/. No se cambiaron pruebas ni manifiestos anteriores para ocultarlos. Los 847 archivos protegidos permanecen exactamente como al iniciar este encargo. Detalle en suite_existente.json y discrepancias_suite_previas.json. El dataset queda verificado; la suite global no se presenta como aprobada.
