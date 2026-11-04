# Evaluación final de Transacciones: test 2016

## Integridad y apertura

La verificación previa reprodujo validación completa, común y mensual, scores y predicciones, y confirmó modelos, 16 columnas, checkpoints 3/399, umbral logístico 0,5 y preprocesador train-only. Se verificaron manifiestos de dataset, entrenamiento y PCA. apertura_previa.json se escribió antes de la primera consulta de eventos 2016; inicio_consulta_test.json y procedencia_test.json registran la lectura posterior. No se reentrenó, reajustó, reconstruyó desarrollo ni seleccionó parámetros con test. El PCA congelado no interviene en X ni se modifica.

El baseline real es **A, persistencia de la etiqueta anterior**: 1[N(t-1)>median(N(t-2),N(t-3),N(t-4))]. Requiere N(t-1) y N(t-2),N(t-3),N(t-4). No se reemplaza por B, que compara N(t-1) con una mediana que lo incluye y solo necesita tres meses. Se verificó consistencia entre implementación, congelación y protocolo antes de abrir.

## Fuente y población

Ventana congelada enero-mayo 2016, [2016-01-01,2016-06-01), conocimiento de y antes de 2016-06-01 00:00. Extracción focalizada de cobros tipo 3, IDs y CustomerID operacional, TransactionDate, edición como control de versión y comprobación de signo sin exportar importes. Solo SELECT, con huellas antes/después y biyección OLTP-DW. El snapshot gzip usa mtime=0, orden por ID y CSV UTF-8/LF; los hashes y consultas están conservados.

Meses evaluables bajo cobertura académica: ['2016-01', '2016-02', '2016-03', '2016-04', '2016-05']. Cierre por calendario dentro del corte, alcance de la auditoría previa, extracción de intervalo completo conciliada y continuidad mensual apoyan el libro académico; no certifican completitud absoluta, ausencia de borrados/backdating o conciliación bancaria. No se deduce cierre solamente de la fecha del último evento. Meses sin cobertura o incompletos se registran desconocidos, sin fabricar ceros ni extender la ventana por resultados.

Población principal: **1284** filas, 525 positivas y 759 negativas, 260 cuentas. Comparación común: **1276** filas; 8 quedan fuera exclusivamente del histórico. La falta de cuarto mes no reduce la evaluación principal. elegibilidad_test.csv registra también insuficiencia de tres meses, cobertura o disponibilidad; exclusiones_historico.csv identifica las filas primarias sin baseline. Todos los métodos de cada soporte usan idénticas claves, sin selección por error/clase.

Las características de t solo usan tres meses anteriores y calendario conocido; meses de test ya concluidos sirven para meses posteriores. Se reutiliza feature_row del pipeline aprobado, sin tocar el constructor sellado de desarrollo. La primera aparición no incorpora cuentas futuras en períodos anteriores. Se aplica únicamente transform del preprocesador y los modelos congelados sobre 16 columnas float64 finitas. X_test.csv, y_test.csv y metadatos_test.csv se guardan alineados con claves y precisión de 17 cifras; no se crean NPY dentro de resultados, preservando la convención histórica de esa carpeta.

## Métricas globales

| Soporte | Método | n | Accuracy | BA | Precision | Recall | F1 | Especificidad |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | perceptron | 1284 | 0.624611 | 0.662217 | 0.524741 | 0.868571 | 0.654232 | 0.455863 |
| primary | logistic | 1284 | 0.652648 | 0.628090 | 0.589977 | 0.493333 | 0.537344 | 0.762846 |
| primary | majority | 1284 | 0.591121 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| common | perceptron | 1276 | 0.622257 | 0.659798 | 0.523645 | 0.868069 | 0.653237 | 0.451527 |
| common | logistic | 1276 | 0.651254 | 0.627132 | 0.589041 | 0.493308 | 0.536941 | 0.760956 |
| common | majority | 1276 | 0.590125 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| common | historical | 1276 | 0.493730 | 0.475853 | 0.381044 | 0.376673 | 0.378846 | 0.575033 |

Average Precision logística: principal=0.6218208974639627, común=0.6208730791349449. Se usa el algoritmo ya fijado con grupos de scores empatados, sin interpolación ni selección. Las probabilidades no se presentan como calibradas; los márgenes del perceptrón no son probabilidades. metricas_globales.json incluye TN/FP/FN/TP, tasa de falsas alertas FP/(TN+FP), soporte, proporción positiva predicha y convenciones de denominador cero. No se imputa un histórico ausente.

## Generalización temporal y por cuentas

Diferencia BA test menos validación completa: perceptrón=-0.5610 puntos porcentuales; logística=0.7270. comparacion_desarrollo_test.json conserva métricas completas de train, validación y test; las diferencias no autorizan elegir otro checkpoint o umbral. Los soportes de cada período son diferentes: no es una comparación emparejada de las mismas cuentas-mes ni una prueba de significación.

| Mes | n principal | BA perceptrón | BA logística | n común |
|---|---:|---:|---:|---:|
| 2016-01 | 254 | 0.6851945586839607 | 0.6041758937045238 | 252 |
| 2016-02 | 254 | 0.7581672203765226 | 0.647079180509413 | 254 |
| 2016-03 | 257 | 0.6282003479990057 | 0.6334824757643549 | 254 |
| 2016-04 | 259 | 0.6418796720520858 | 0.5910296599951772 | 257 |
| 2016-05 | 260 | 0.581230090664053 | 0.6501470227885322 | 259 |

Se incluyen todos los meses previstos, aun desfavorables o sin casos estimables. Las métricas globales se calculan desde todas las predicciones, no como media simple de porcentajes mensuales; el promedio mensual no ponderado se conserva aparte en resumen_temporal.json. Subconjuntos sin una clase usan las convenciones aprobadas; BA/recall/especificidad pueden no ser estimables.

| Cohorte | Cuentas | n principal | BA perceptrón | BA logística |
|---|---:|---:|---:|---:|
| known_development | 257 | 1279 | 0.660650371568677 | 0.6279321065669077 |
| new_in_test | 3 | 5 | 1.0 | 0.5 |

Se distingue observado en desarrollo de primera aparición en test usando el snapshot anterior, no el resultado futuro. cohortes_detalladas.json separa cuentas elegibles en train, solo validación, observadas en desarrollo sin elegibilidad previa y nuevas de test. Cohortes pequeñas no permiten afirmar estabilidad de generalización. No se borran cuentas sin actividad actual ni se presume alta real a partir de primera observación.

## Errores y convergencia previamente observada

Conteos: {"shared_correct": 600, "shared_error": 244, "disagreement": 440, "perceptron_FP": 413, "perceptron_FN": 69, "logistic_FP": 180, "logistic_FN": 266}. analisis_errores.json y errores_por_observacion.csv conservan FP, FN, aciertos/errores compartidos y desacuerdos, globales y mensuales. Los meses con mayor concentración de errores se identifican por conteos y tasas en métricas mensuales, sin omitirlos ni utilizar esos resultados para cambiar clasificadores o baselines. No se inventan costes monetarios de errores ni se declara mejora estadísticamente significativa.

- perceptron: mayor conteo de errores en 2016-05, 124/260; mayor tasa en 2016-05, 47.69%.
- logistic: mayor conteo de errores en 2016-04, 103/259; mayor tasa en 2016-04, 39.77%.

En la población principal, perceptrón tiene recall=86.86% y tasa de falsas alertas=54.41%; logística tiene recall=49.33% y tasa de falsas alertas=23.72%. El primero detecta más positivos con más FP; la segunda reduce FP y conserva más FN. Son comportamientos de los checkpoints fijados, no una nueva elección por test. La prevalencia positiva test es 40.89%; no se atribuye causalmente el cambio de comportamiento a una variable particular ni se ajusta el umbral para compensarlo.

Se mantienen los historiales anteriores: perceptrón continuaba actualizando a 100 épocas, con checkpoint elegido en 3; logística reducía pérdida train de 0,6931 a 0,6170, alcanzaba el máximo de 400 épocas y no su tolerancia de gradiente, con checkpoint 399. No se afirma convergencia completa ni se generan curvas de entrenamiento sobre test. Los reportes/gráficos previos de convergencia y PCA siguen intactos y pueden consultarse como contexto, sin reinterpretar la proyección 2D como frontera del clasificador.

## Reproducción y límites

```powershell
python -m src.experiments.evaluar_transacciones --preparar
python -m src.experiments.evaluar_transacciones --extraer
python -m src.experiments.evaluar_transacciones --evaluar
python -m unittest discover -s tests -p test_evaluacion_transacciones.py -v
python -m unittest discover -s tests -v
```

Extraer exige el registro previo y no sustituye un snapshot existente. Para reproducciones técnicas, ejecutar solo --evaluar desde el snapshot con hash verificado; no abrir nuevamente SQL ni reinterpretar la búsqueda. El código y el protocolo inicial quedan registrados; correcciones técnicas, si ocurren, deben registrarse sin alterar reglas o configuraciones. Todos los resultados describen el libro académico de WWI y cinco meses de una simulación, no una certificación bancaria o utilidad operacional. Test ya fue observado: cualquier diseño nuevo requeriría un experimento distinto y no podría presentar este período como reserva nunca vista.

Se detiene aquí: sin cambios de pesos, umbrales, escaladores, features, etiquetas, baselines o ventanas; sin modificación de Ventas, Pedidos, Compras o LaTeX, operaciones Git o avance automático a Inventario.
