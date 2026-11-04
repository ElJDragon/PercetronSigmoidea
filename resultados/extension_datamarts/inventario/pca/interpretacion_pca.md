# PCA bidimensional de Inventario

Se reutiliza PCA2D por SVD NumPy sin modificarla. Fit exclusivamente sobre X_train procesada 4599×17, centrada por su media de train. Validación 2628×17 recibe la misma media y matriz de componentes; no se ajusta un escalador nuevo, no se usan etiquetas en fit y PCA no entra en los clasificadores. Se conservan todos los puntos: negativos transparentes y positivos grandes dibujados después, sin jitter ni submuestreo. Varias observaciones pueden coincidir visualmente.

PC1 explica 55.798494747%, PC2 11.499841939%; acumulada 67.298336686% y fuera del plano 32.701663314%. Rango numérico 14 de 17. Las redundancias entre lags, media, mediana y cambios pueden reducir rango; no se eliminaron entradas de los clasificadores por ello. Ver comprobaciones_matematicas.json: ortogonalidad, autovalores de covarianza, varianza proyectada, suma de ratios y residuo de reconstrucción.

| Componente | Característica | Coeficiente real | Cuadrado del coeficiente |
|---|---|---:|---:|
| PC1 | `recepciones_media_3` | 0.319643656 | 0.102172067 |
| PC1 | `recepciones_lag_1` | 0.319474930 | 0.102064231 |
| PC1 | `recepciones_mediana_3` | 0.318923321 | 0.101712084 |
| PC1 | `recepciones_lag_2` | 0.318652566 | 0.101539458 |
| PC1 | `mes_anterior_sin_recepciones` | -0.317501905 | 0.100807460 |
| PC1 | `recepciones_lag_3` | 0.316185781 | 0.099973448 |
| PC2 | `recepciones_tendencia_3` | 0.601445359 | 0.361736520 |
| PC2 | `recepciones_cambio_1` | 0.560379791 | 0.314025510 |
| PC2 | `recepciones_std_3` | 0.415923721 | 0.172992541 |
| PC2 | `unidades_recibidas_lag_3` | -0.185922522 | 0.034567184 |
| PC2 | `unidades_recibidas_lag_2` | -0.182431621 | 0.033281296 |
| PC2 | `antiguedad_observada_meses` | -0.152749672 | 0.023332462 |

Los nombres y signos de esta tabla definen la interpretación: PC1 combina principalmente recepciones_media_3, recepciones_lag_1, recepciones_mediana_3; PC2, recepciones_tendencia_3, recepciones_cambio_1, recepciones_std_3. En términos empresariales, PC1 resume intensidad habitual de recepciones, en contraste con ausencia de actividad. PC2 recoge principalmente tendencia, cambio reciente y variabilidad de los conteos; describe una dirección de variación, no una predicción de aumentos futuros. Los coeficientes son de autovectores unitarios sobre variables estandarizadas, no efectos causales ni importancias predictivas; su signo se fija por el coeficiente absoluto mayor positivo. Los cuadrados suman uno por componente, no son porcentajes del objetivo.

La agrupación repetida por artículo se cuantifica con centroides y fracción de suma de cuadrados entre artículos: train PC1=0.953408, PC2=0.053344; val PC1=0.992811, PC2=0.716844. Son descriptores geométricos de observaciones repetidas, no evidencia de grupos independientes o separabilidad causal. Los positivos pertenecen a pocos artículos receptores y se muestran junto a sus negativos; el agrupamiento por historia de recepción no garantiza separar aumentos mensuales. La información descartada y los solapamientos impiden deducir el rendimiento en 17 dimensiones de la apariencia 2D.

Parámetros, 17×2 coeficientes y proyecciones por clave se guardan sin tocar X/y. Las etiquetas en CSV se agregan exclusivamente como anotación visual después del ajuste. No hay proyección ni outcomes de test. La reproducibilidad y las perturbaciones de etiquetas se verifican separadamente.
