# PCA financiero: representación bidimensional

Se reutiliza PCA2D por SVD NumPy sin modificaciones. Fit solo con X_train 4.612×16 ya estandarizada; no nuevo escalado ni etiquetas. Validación 2.943×16 recibe transform con la misma media y W2. No se cambian las entradas de los clasificadores: conservan 16 dimensiones y sus checkpoints originales. Las proyecciones son artefactos nuevos de visualización, no sustitutos del dataset.

Las proyecciones se guardan en CSV UTF-8 con 17 cifras significativas, claves estables y etiquetas de desarrollo para auditoría gráfica. Su lectura con float_precision=round_trip recupera los valores float64. No se crean matrices .npy en resultados: se respeta la restricción histórica de esa carpeta y se mantienen las matrices originales de clasificadores en su ubicación aprobada.

- PC1: **27.896335%** de varianza de train.
- PC2: **17.125059%**.
- Acumulada en dos PC: **45.021394%**.
- No representada: **54.978606%**.
- Rango numérico observado: 13 de 16, sin retirar columnas ni alterar los modelos.

Varianza muestral ddof=1. Se verifican ortogonalidad, espectro SVD frente a autovalores de covarianza, traza de covarianza, varianza de las proyecciones, suma de ratios y residuo de reconstrucción 2D frente a varianza descartada. Resultados numéricos en comprobaciones_matematicas.json. El rango menor que 16, si ocurre, es compatible con características históricas derivadas y redundantes; no exige cambiar el experimento congelado.

## Componentes y límites de interpretación

| Componente | Característica | Coeficiente |
|---|---|---:|
| PC1 | cobros_media_3 | 0.467396 |
| PC1 | cobros_mediana_3 | 0.458979 |
| PC1 | cobros_lag_1 | 0.424373 |
| PC1 | cobros_lag_2 | 0.411495 |
| PC1 | cobros_lag_3 | 0.407070 |
| PC1 | meses_con_cobros_3 | 0.168534 |
| PC2 | cobros_cambio_1 | 0.535123 |
| PC2 | variacion_relativa_cobros | 0.503452 |
| PC2 | cobros_tendencia_3 | 0.437703 |
| PC2 | mes_anterior_sin_cobros | -0.352326 |
| PC2 | cobros_lag_1 | 0.218738 |
| PC2 | cobros_lag_2 | -0.210995 |

Estos son coeficientes de vectores unitarios en coordenadas estandarizadas, no importancia causal, pesos de los clasificadores ni correlaciones automáticamente. PC1/PC2 resumen combinaciones de frecuencia y variación histórica según los coeficientes anteriores; el signo se fija haciendo positivo el coeficiente de mayor valor absoluto. No se optimiza la separación de clases. Medias, diferencias, tendencias y lags pueden representar varias veces una misma estructura correlacionada; la varianza no equivale a utilidad predictiva.

En este ajuste, PC1 tiene sus mayores coeficientes positivos en media, mediana y los tres lags de cobros: describe principalmente un nivel conjunto de actividad histórica. PC2 contrapone aumentos recientes (cambio, variación relativa y tendencia con coeficientes positivos) a ausencia de cobros el mes anterior (coeficiente negativo): describe variación reciente e intermitencia. Son descripciones de las combinaciones observadas, no factores económicos identificados. Las dos clases se solapan en ambas particiones; los puntos extremos se mantienen visibles y no se recortan para aparentar separación.

Los tres gráficos muestran todos los puntos, transparencia para el solapamiento, mismos límites de ejes en ambas particiones y porcentajes de varianza de entrenamiento. No hay jitter, separación artificial, eliminación de atípicos o frontera 2D presentada como clasificador de 16 entradas. Las etiquetas solo colorean las proyecciones. No se deduce separabilidad o convergencia de una neurona de estos dos componentes; queda fuera de la representación el porcentaje indicado de varianza.

## Congelación y convergencia conservadas

Perceptrón: SHA-256 19aca080e0e38d007dfb6b8e4050c7b371a5b7665e93bcc6a7463aa70b5109b0; checkpoint 3, eta=0,01, seed=7. Logística: SHA-256 11e86463138a284caa4cac90f4ae10fa4448b6cccb5d2a4d8f6c9070d4e8f726; checkpoint 399, eta=0,10, umbral 0,5. Pesos, sesgos y orden de features se verifican en verificacion_modelos.json. Ningún entrenamiento nuevo de clasificadores.

Se leen los historiales aprobados: el perceptrón sigue actualizando al terminar sus 100 épocas y no registra épocas sin errores; logística reduce pérdida train de 0,693147 a aproximadamente 0,616973, termina por 400 épocas y no alcanza norma de gradiente <=1e-6. La pérdida decreciente no certifica convergencia completa. Se conservan curvas y modelos anteriores intactos; no se reemplaza checkpoint por última época.

## Reproducción y siguiente decisión

```powershell
python -m src.experiments.pca_transacciones
python -m unittest discover -s tests -p test_pca_transacciones.py -v
python -m unittest discover -s tests -v
```

El runner solo lee desarrollo y ajusta PCA, no consulta SQL, no carga outcomes de 2016, no reconstruye matrices originales ni entrena clasificadores. Las pruebas unitarias sintéticas de la suite existente son verificaciones de componentes, no nuevas ejecuciones experimentales sobre WWI.

Puede autorizarse la primera evaluación de 2016 en un encargo posterior **bajo el contrato académico de eventos y el protocolo definitivo congelado aquí**, siempre que pasen de nuevo integridad, cobertura/cierre y disponibilidad. No es certificación bancaria/operativa ni completitud absoluta del libro. Esta tarea no abre test ni ejecuta la evaluación final. La población principal nunca se reduce por falta del cuarto lag; el histórico se compara aparte en la intersección exacta.
