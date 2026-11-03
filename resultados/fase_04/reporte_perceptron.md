# Perceptrón multidimensional WWI — Fase 4

## Datos y método

Se usaron los arrays de fase 3 sin reconstruir dataset, ajustar transformaciones ni cambiar características: train (4599, 20), validación (2628, 20). Contrato 2.0, manifiesto SHA-256 `b6ef8d83ec76f2b8d8e8847d3a505874770b1325528598985e36cb8712034879`. Pesos asociados al orden validado de las 20 características. El supuesto de completitud académica, identidad por nombre y disponibilidad al cierre permanece vigente.

Implementación propia NumPy: z=wᵀx+b, predicción 1 si z>=0; e=y−predicción; si e≠0, w←w+ηex y b←b+ηe. Etiquetas canónicas 0/1. Una actualización por observación errada. No se optimiza entropía cruzada, no hay probabilidades ni optimizador externo.

## Presupuesto previo y selección

`plan_experimentos.json` se fijó antes del primer entrenamiento: dos órdenes (fijo y barajado), η=0,01, 100 épocas máximas por ejecución, semilla 42 y pesos/sesgo iniciales cero. Una sola tasa: con inicio cero y η positiva constante, cambiar η solo escala la trayectoria en aritmética exacta; flotantes pueden romper esa equivalencia. No se amplió la búsqueda tras observar resultados.

Validación elige máxima balanced accuracy; desempates por F1 positiva, época menor y orden del plan. Las actualizaciones solo usan train. El modelo exportado recupera una copia de los pesos en la época elegida, no selecciona automáticamente el estado final. Sin refit train+val, sin ajuste mediante test.

| Experimento | Época elegida | Accuracy train | Accuracy val | BA val | F1 val |
|---|---:|---:|---:|---:|---:|
| orden_fijo | 80 | 0.619482 | 0.616058 | 0.615082 | 0.648555 |
| orden_barajado | 70 | 0.646227 | 0.617199 | 0.616324 | 0.646770 |

Seleccionado: **orden_barajado**, época **70**. Configuración: `{'learning_rate': 0.01, 'max_epochs': 100, 'seed': 42, 'shuffle': True, 'initial_weights': None, 'initial_bias': 0.0}`.

## Métricas del modelo seleccionado

Las cifras son fracciones, no porcentajes.

| Métrica | Train | Validación |
|---|---:|---:|
| accuracy | 0.646227 | 0.617199 |
| precision | 0.668151 | 0.606320 |
| recall | 0.629723 | 0.693002 |
| f1 | 0.648368 | 0.646770 |
| balanced_accuracy | 0.646842 | 0.616324 |

Soporte train: {'0': 2217, '1': 2382}; validación: {'0': 1299, '1': 1329}. Matrices con filas reales/columnas predichas y orden 0,1: train `[[1472, 745], [882, 1500]]`, val `[[701, 598], [408, 921]]`. JSON conserva precisión completa y convenciones de denominadores cero. No se ejecutó evaluación predictiva de test.

## Errores y convergencia

En la ejecución seleccionada se recorrieron 100 épocas. Errores durante actualizaciones: primera 1973, última 1996; mínimo 1863, máximo 2016. Épocas de cero actualizaciones: ninguna. Hubo aumentos entre épocas: True.

Los errores online corresponden a distintos estados de pesos durante el recorrido; `train_errors_end` y accuracy se calculan con los pesos al cerrar cada época. El historial final es del recorrido completo y no se reescribe tras restaurar la mejor época. La línea de selección de los gráficos identifica el estado exportado. No se exige descenso monótono ni 100 % de exactitud. Si no hay cero errores, esta ejecución no demuestra separabilidad ni prueba su ausencia: el teorema de convergencia requiere separabilidad lineal y no fija 100 épocas como certificado.

## Pesos y sesgo seleccionados

Sesgo: **0.010000000000000004**. Pesos float64 en JSON con ida/vuelta exacta y CSV con 17 cifras significativas:

| Característica escalada | Peso |
|---|---:|
| unidades_lag_1 | 0.086047388811192158 |
| unidades_lag_2 | 0.079559332950772513 |
| unidades_lag_3 | 0.086433806948070374 |
| unidades_media_3 | 0.086820462175113369 |
| unidades_mediana_3 | -0.3720436256352147 |
| unidades_std_3 | -0.0011423824327889114 |
| cambio_unidades_1 | 0.016234371544841163 |
| variacion_relativa_1 | 0.015503476495048884 |
| tendencia_unidades_3 | 0.0022661384069365548 |
| facturas_lag_1 | -0.013416532023155681 |
| facturas_media_3 | -0.074495274545097676 |
| ingresos_lag_1 | -0.086713421543525515 |
| ingresos_media_3 | 0.09642870252787368 |
| precio_unitario_ponderado_lag_1 | 0.14385982435183342 |
| clientes_distintos_lag_1 | -0.048679823470728217 |
| unidades_por_factura_lag_1 | 0.024600708603939501 |
| antiguedad_observada_meses | 0.047891923782994603 |
| mes_objetivo_sin | 0.054229223020853153 |
| mes_objetivo_cos | 0.030914856096277414 |
| dias_mes_objetivo | 0.034500061409917425 |

Estos coeficientes actúan en la escala estandarizada de fase 3, no sobre unidades originales. Hay redundancias entre lags, medias y cambios: magnitud/signo de un peso no equivale a importancia independiente ni efecto causal. Para inferencia, conservar el orden y los parámetros de preprocesamiento previos. Scores de decisión no son probabilidades.

## Artefactos y reproducción

Modelo seguro sin pickle en `modelo_seleccionado.json`; `pesos.csv`, `metricas.json`, confusiones y predicciones train/val con IDs trazables. Historial de ambos experimentos y resumen completo en `experimentos.json`. Los cuatro PNG bajo `graficos/` muestran errores online, accuracy train/val, BA val y confusión val, siempre de las 20 entradas completas del modelo seleccionado. No hay frontera 2D ni PCA.

```powershell
python -m unittest discover -s tests -p 'test_perceptron.py' -v
python -m src.experiments.entrenar_perceptron
python -m unittest discover -s tests -p 'test_*.py' -v
```

El manifiesto vincula hashes de matrices, contrato, plan, código y resultados, con integridad antes/después de fuentes y fases previas. Cargar mediante `PerceptronSimple.load(path)`; `predict(X, feature_names=nombres)` valida orden cuando se proporcionan nombres. La inspección inicial de integridad comprobó forma/tipo/finitud/etiquetas del reservado sin predecir; el cargador del entrenamiento solo abre arrays train y val. La suite histórica incluye pruebas de preparación en destinos temporales, sin sustituir los entregables anteriores.

## Limitaciones y preparación de fase 5

Un clasificador lineal puede ser insuficiente para patrones de ventas y estacionalidad. La validación se usó para elegir entre 200 estados posibles; sus mejores métricas son de selección, no una estimación final independiente. Test permanece reservado. Superar la mediana reciente no mide demanda no atendida, rentabilidad, inventario ni utilidad económica. Ocho productos recientes de test carecen de representación en desarrollo; no se han evaluado aquí.

Para la fase 5, confirmar la biblioteca autorizada para gradientes de la futura regresión logística, mantener exactamente estas matrices/filas y reutilizar las métricas. Su implementación y comparación con test necesitan autorización de la fase correspondiente. No se implementó logística ni PCA; no se avanzó automáticamente.
