# Entrenamiento financiero reproducible

Se entrenan las clases matemáticas propias sin modificarlas, con pesos/sesgo nuevos a cero, 16 columnas congeladas y fit exclusivamente en train. Los checkpoints y la configuración ganadora de cada algoritmo se seleccionan solo con validación completa por BA, F1, menor época y orden predefinido. No hay SQL, reconstrucción del dataset, apertura de test, PCA o cambio del informe. Los supuestos académicos de cobertura y conservación del libro permanecen vigentes: esto mide frecuencia de cobros registrados y no demuestra generalización bancaria o utilidad monetaria.

## Presupuesto y selección

Antes de resultados se fijaron dos perceptrones (eta 0,01, 100 épocas, semillas 42/7) y dos logísticas (eta 0,05/0,10, 400 épocas, semilla 42, inicialización cero, tolerancia de norma de gradiente 1e-6). Sin ampliación adaptativa ni selección de umbral: logística usa 0,5 y perceptrón z>=0. La semilla logística se conserva por trazabilidad pero no cambia pesos cero. Los modelos de Ventas no se cargan ni se transfieren.

| Algoritmo | Ejecución elegida | Learning rate | Semilla | Época |
|---|---|---:|---:|---:|
| perceptron | p_seed7 | 0.01 | 7 | 3 |
| logistic | l_eta010 | 0.1 | 42 | 399 |

## Métricas en soporte completo

| Partición | Método | n | Accuracy | BA | Precision | Recall | F1 | Especificidad |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | perceptron | 4612 | 0.665221 | 0.663005 | 0.560649 | 0.652948 | 0.603289 | 0.673063 |
| train | logistic | 4612 | 0.681483 | 0.619704 | 0.684624 | 0.339266 | 0.453700 | 0.900142 |
| train | majority | 4612 | 0.610147 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| val | perceptron | 2943 | 0.655114 | 0.667827 | 0.557875 | 0.736227 | 0.634761 | 0.599427 |
| val | logistic | 2943 | 0.672443 | 0.620820 | 0.698980 | 0.343072 | 0.460246 | 0.898567 |
| val | majority | 2943 | 0.592932 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |

La mayoría se verifica solo con y_train y es 0. El histórico aprobado requiere cuarto mes: 1[N(t-1)>mediana(N(t-2),N(t-3),N(t-4))]. No se sustituye por 1[N(t-1)>R(t)], que usa otra referencia. No se imputa el cuarto lag ni se mezclan soportes al comparar:

| Partición | Método | n común | Accuracy | BA | F1 |
|---|---|---:|---:|---:|---:|
| train | perceptron | 4376 | 0.666362 | 0.664412 | 0.606257 |
| train | logistic | 4376 | 0.682358 | 0.622115 | 0.458723 |
| train | majority | 4376 | 0.608090 | 0.500000 | 0.000000 |
| train | historical | 4376 | 0.500457 | 0.475345 | 0.360445 |
| val | perceptron | 2927 | 0.655278 | 0.668401 | 0.635609 |
| val | logistic | 2927 | 0.672702 | 0.621145 | 0.461192 |
| val | majority | 2927 | 0.593099 | 0.500000 | 0.000000 |
| val | historical | 2927 | 0.495729 | 0.477882 | 0.381391 |

Soporte perdido por falta de cuarto mes/control de disponibilidad: train=236, val=16. Los cuatro métodos se evalúan en idénticas claves dentro de esa intersección; selección de checkpoints se mantiene en validación completa. métricas.json conserva matrices de confusión [0,1], soporte, proporción de predicciones positivas y convenciones de denominador cero. métricas_mensuales.json desglosa ambos soportes, sin ocultar subconjuntos monoclase.

Average Precision de logística en validación completa: **0.638320**. Se calcula desde probabilidades sigmoid con grupos de scores empatados y sin interpolación; no participa en selección. El margen del perceptrón no se llama probabilidad calibrada. Tampoco se asume calibración perfecta de logística.

## Artefactos y reproducción

runs/ conserva los cuatro historiales completos JSON/CSV, parámetros, métricas seleccionadas y última época, además de cada modelo candidato. modelos/ guarda los dos checkpoints seleccionados y congelacion.json sus hashes, umbrales, nombres y preprocesador. predicciones_desarrollo.csv permite comprobar alineación y la ausencia explícita del histórico sin cuarto lag. Se exportan 12 gráficos reales: curvas de errores/actualizaciones/accuracy/BA y confusión del perceptrón; pérdidas/gradiente/accuracy/BA y confusión de logística; dos comparaciones en soporte común. Las funciones de gráficos existentes se reutilizan; su metadato Software histórico se conserva sin alterar Ventas.

```powershell
python -m src.experiments.entrenar_transacciones --preparar
python -m src.experiments.entrenar_transacciones --entrenar
python -m unittest discover -s tests -p test_entrenamiento_transacciones.py -v
python -m unittest discover -s tests -v
```

Preparar verifica los hashes y fija el plan antes de entrenar; entrenar rechaza ausencia o cambio del presupuesto. No se reescala, reimputa o regenera el dataset. El preflight aplica en memoria el estado ya congelado para comprobar X, sin fit o escritura de matrices. La reproducción de entrenamiento se compara con pesos, historiales y métricas en el mismo entorno numérico registrado, sin añadir nuevas configuraciones al presupuesto.

La validación fue utilizada repetidamente para seleccionar checkpoint: sus resultados son de desarrollo y no estimadores independientes del desempeño final. Comparar con los baselines no certifica significación, valor comercial o independencia entre cuentas/meses. La evaluación futura sigue sellada y no se inicia automáticamente.
