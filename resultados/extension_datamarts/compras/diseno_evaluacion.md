# Compras: diseño de evaluación condicionado

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
