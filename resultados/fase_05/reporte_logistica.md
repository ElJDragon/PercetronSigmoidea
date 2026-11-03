# Regresión logística WWI — Fase 5

## Alcance y datos

Implementación propia NumPy de sigmoidea, pérdida, gradientes y descenso por lote completo. Sin scikit-learn, autodiferenciación, PCA o reconstrucción de datos. Se reutilizaron las matrices de fase 3, sus 20 entradas y los mismos IDs/orden del perceptrón: train (4599, 20), validación (2628, 20). Manifiesto fase 3: `b6ef8d83ec76f2b8d8e8847d3a505874770b1325528598985e36cb8712034879`. Las etiquetas de test no se consultaron ni se generaron sus predicciones; solo se verificaron hashes de integridad mediante los manifiestos aprobados.

## Matemática y estabilidad

Con X(m,d), w(d), b escalar: z=Xw+b; p=sigmoid(z). Para z>=0 se usa 1/(1+exp(-z)); para z<0, exp(z)/(1+exp(z)). Funciona con arreglos multidimensionales finitos, sin overflow de exponenciales extremos y con sigma(0)=0,5.

L=mean(logaddexp(0,z)−y*z), equivalente a entropía cruzada binaria. No se recortan probabilidades ni alteran etiquetas; la media se suma después de dividir por m para reducir overflow. grad_w=Xᵀ(p−y)/m; grad_b=mean(p−y). Se implementa Xᵀ((p−y)/m), algebraicamente idéntico. Actualización: w←w−η grad_w; b←b−η grad_b. Si combinación lineal/gradientes dejan de ser finitos se bloquea la ejecución; no se oculta divergencia. No hay regularización ni optimizadores predefinidos.

## Comprobación previa de gradientes

Diferencias centrales por coordenada con paso h_j=h*max(1,|parametro_j|), para pesos y sesgo no nulos; etiquetas mixtas y entradas positivas/negativas. h=1e-6 en magnitudes moderadas equilibra cancelación y error de truncamiento; h=1e-5 y tolerancia absoluta mayor en logits extremos por la escala de la pérdida. Criterio `allclose`: error <= atol+rtol*|numérico|. El gate matemático corrió antes de abrir WWI; las pruebas añaden dimensiones 1/3/8 y estabilidad hasta logits ±1e300. No se exige diferencias finitas relativas fiables para derivadas prácticamente cero: importa también atol.

| Caso | h | atol | rtol | Máx. error absoluto | Máx. error relativo | Aprobado |
|---|---:|---:|---:|---:|---:|---|
| moderado | 1e-06 | 1e-08 | 1e-05 | 6.352711412e-11 | 4.105428625e-09 | True |
| logits_extremos | 1e-05 | 2e-06 | 1e-05 | 9.549694369e-09 | 2.021260984e-08 | True |

Valores analíticos y numéricos individuales en verificacion_gradientes.json.

## Presupuesto y selección fijados antes de WWI

Tres tasas 0,01/0,05/0,1; 200 épocas por ejecución, lote completo; inicialización cero, sesgo cero, semilla 42. La semilla es metadato reproducible pero no afecta el lote completo con inicio cero. El modelo permite inicio normal reproducible o pesos explícitos, pero no se usaron para ampliar esta búsqueda. Tolerancia desactivada para ejecutar el presupuesto completo; no se afirma convergencia por llegar a 200 épocas.

Checkpoint y tasa se eligen por BA val a umbral 0,5, luego F1, menor época y orden del plan. Después, sobre **un único checkpoint fijo**, se prueba la cuadrícula de fase 2: 0,05,0,10,…,0,95, con desempates BA/F1/cercanía a 0,5/umbral menor. No se evaluó un producto cartesiano de épocas×umbrales. Máximo 600 épocas y 19 umbrales; no se amplió el presupuesto tras mirar resultados. La selección sobre validación genera optimismo y no es estimación final independiente.

| Ejecución | η | Época seleccionada a 0,5 | BA val a 0,5 | Pérdida train al cerrar 200 | Norma gradiente final |
|---|---:|---:|---:|---:|---:|
| eta_001 | 0.01 | 195 | 0.585766 | 0.648477 | 0.094667 |
| eta_005 | 0.05 | 170 | 0.606055 | 0.629514 | 0.023199 |
| eta_010 | 0.1 | 200 | 0.611993 | 0.626954 | 0.011097 |

Modelo seleccionado: **eta_010**, época **200**, tasa **0.1**, umbral final **0.5**. Pesos restaurados a esa época, sin refit train+validación. Historial completo de las tres ejecuciones en CSV; la selección de umbral se conserva íntegra en JSON.

## Pérdida y convergencia

Ejecución seleccionada: pérdida train inicial (época 0) **0.69314718056**, pérdida train al finalizar la ejecución **0.626953985862**. Pérdida train del checkpoint exportado **0.626953985862**. Pérdida val inicial **0.69314718056**, al finalizar la ejecución **0.669574542823**, en el checkpoint **0.669574542823**. Se distingue último estado entrenado del seleccionado.

- eta_001: aumentos de pérdida train mayores que 1e-12: 0; gradiente final 0.0946669.
- eta_005: aumentos de pérdida train mayores que 1e-12: 0; gradiente final 0.0231993.
- eta_010: aumentos de pérdida train mayores que 1e-12: 0; gradiente final 0.0110975.

Norma del gradiente en el checkpoint: 0.0110975. Un gradiente no nulo o pérdida aún decreciente indica que agotar el presupuesto no certifica un óptimo. El criterio de selección es clasificación en validación, no mínima entropía cruzada. La pérdida puede bajar sin que accuracy/BA suban monótonamente. No se imponen resultados objetivo.

## Comparación preliminar de desarrollo

Métricas en fracciones, positiva=ventas del mes superiores a mediana reciente. Todos usan los mismos soportes: train {'0': 2217, '1': 2382}; val {'0': 1299, '1': 1329}. Perceptrón recuperado de fase 4, sin entrenarlo ni modificarlo; sus predicciones reproducen exactamente sus métricas aprobadas. Baseline mayoritario: mayoría de train, empate→0; histórico: 1[Q(t−1)>R(t)], no una comparación de variables estandarizadas con distintas escalas. Para obtenerlo se leyeron solo IDs/mes/lag1/R del dataset aprobado, excluyendo 2016 antes de convertir números y sin leer columnas Q_objetivo/y. Ningún baseline consulta test.

| Modelo | Conjunto | Accuracy | Precision | Recall | F1 | BA |
|---|---|---:|---:|---:|---:|---:|
| logistica | train | 0.658404 | 0.665443 | 0.684719 | 0.674943 | 0.657425 |
| logistica | val | 0.611492 | 0.628120 | 0.568096 | 0.596602 | 0.611993 |
| perceptron | train | 0.646227 | 0.668151 | 0.629723 | 0.648368 | 0.646842 |
| perceptron | val | 0.617199 | 0.606320 | 0.693002 | 0.646770 | 0.616324 |
| mayoritario | train | 0.517939 | 0.517939 | 1.000000 | 0.682424 | 0.500000 |
| mayoritario | val | 0.505708 | 0.505708 | 1.000000 | 0.671721 | 0.500000 |
| historico_ultimo_mes | train | 0.492716 | 0.515399 | 0.344249 | 0.412786 | 0.498241 |
| historico_ultimo_mes | val | 0.490487 | 0.494530 | 0.340105 | 0.403032 | 0.492224 |

Matrices de confusión (filas reales/columnas predichas, orden 0/1):

- logistica: train `[[1397, 820], [751, 1631]]`; val `[[852, 447], [574, 755]]`.
- perceptron: train `[[1472, 745], [882, 1500]]`; val `[[701, 598], [408, 921]]`.
- mayoritario: train `[[0, 2217], [0, 2382]]`; val `[[0, 1299], [0, 1329]]`.
- historico_ultimo_mes: train `[[1446, 771], [1562, 820]]`; val `[[837, 462], [877, 452]]`.

La logística añade entropía cruzada: train 0.626954, val 0.669575. No se calcula una pérdida probabilística ficticia para perceptrón o reglas deterministas. Los resultados permiten comparación descriptiva, no superioridad definitiva ni significación estadística de pequeñas diferencias. Presupuestos/objetivos de optimización son distintos; no se fuerza una comparación por igual número de actualizaciones.

## Parámetros e inferencia

Sesgo: **0.085390789669521322**; umbral: **0.5**. Todos los pesos son float64 y operan sobre características estandarizadas de fase 3, con redundancias: no equivalen a importancia causal independiente.

| Característica | Peso |
|---|---:|
| unidades_lag_1 | 0.0077538429371081241 |
| unidades_lag_2 | 0.021919159372678004 |
| unidades_lag_3 | -0.028874383579176419 |
| unidades_media_3 | 0.00038495335263386433 |
| unidades_mediana_3 | -0.1550018310751814 |
| unidades_std_3 | 0.1603485307302967 |
| cambio_unidades_1 | -0.032390896093326758 |
| variacion_relativa_1 | 0.13404048909634628 |
| tendencia_unidades_3 | 0.083123688544595412 |
| facturas_lag_1 | -0.11308763440928998 |
| facturas_media_3 | -0.52009778670072904 |
| ingresos_lag_1 | -0.037081330837846524 |
| ingresos_media_3 | 0.0073351909665338721 |
| precio_unitario_ponderado_lag_1 | 0.00048668912446919418 |
| clientes_distintos_lag_1 | -0.11718785125889304 |
| unidades_por_factura_lag_1 | 0.028211039550568473 |
| antiguedad_observada_meses | 0.16703575165447174 |
| mes_objetivo_sin | 0.24188638442783236 |
| mes_objetivo_cos | -0.018110885050252778 |
| dias_mes_objetivo | 0.2550458337404401 |

`predict_proba` devuelve un vector (m,) de P(y=1|X); `predict(X)` usa el umbral persistido, y `predict(X, threshold=...)` permite sustituirlo explícitamente. p>=umbral asigna 1. Sigmoidea no garantiza calibración perfecta; evaluar calibración es trabajo posterior. Umbral probabilístico no cambia R ni la definición de y. La serialización JSON conserva precisión float64, nombres, sesgo, configuración y época; sin pickle. La recuperación reproduce probabilidades y métricas de desarrollo exactamente.

## Artefactos y reproducción

Nuevos archivos: src/models/logistic_regression.py, src/experiments/entrenar_logistica.py, src/evaluation/graficos_logistica.py, tests/test_logistica.py y resultados/fase_05/. Se reutilizaron el cargador de desarrollo y las métricas aprobadas, sin cambiar su comportamiento; la visualización reutiliza backend/estilo y agrega un módulo separado para conservar intacta fase 4.

Resultados: modelo_seleccionado.json, pesos.csv, metricas.json, experimentos.json, seleccion_umbral.json, verificacion_gradientes.json, comparacion_desarrollo.json, historiales de tres tasas, confusiones y predicciones/probabilidades train/val, cinco PNG y manifiesto de hashes. Los gráficos de accuracy/BA corresponden a **0,5**, igual que la búsqueda de checkpoint; la confusión corresponde al **umbral final**. Todos proceden de las 20 entradas completas. No hay PCA ni frontera 2D.

```powershell
python -m unittest discover -s tests -p 'test_logistica.py' -v
python -m src.experiments.entrenar_logistica
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Riesgos y siguiente fase

La logística es un modelo lineal en logits; no representa toda interacción posible. Escoger checkpoint, tasa y umbral con la misma validación introduce optimismo. Las filas comparten producto/mes y no son réplicas independientes. Persisten cobertura académica no certificada, identidad por nombre, unidades comerciales por confirmar y ventas observadas que no miden demanda no atendida ni stock. La utilidad empresarial de una subida marginal sobre la mediana no está demostrada.

Para fase 6: congelar modelos, transformaciones, etiqueta y umbrales, y formalizar el protocolo de evaluación final común contra baselines antes de abrir test. Confirmar interpretación de probabilidades y distinguir evaluación de clasificación de calibración. Si se autoriza visualización PCA después, conservarla separada del entrenamiento multidimensional. Esta fase no evalúa test ni avanza automáticamente.
