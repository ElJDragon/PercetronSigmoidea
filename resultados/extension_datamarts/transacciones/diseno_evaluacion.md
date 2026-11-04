# Evaluación futura de frecuencia de cobros

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
