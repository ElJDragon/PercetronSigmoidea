# Análisis estadístico y generalización — Fase 8

## Método, integridad y alcance

Se verificaron manifiestos/hashes de fases 4–7, datos preparados, fuentes y configuraciones congeladas. Las 1.111 filas de predicciones_alineadas.csv se alinearon con IDs, producto y mes de los metadatos originales; y_real coincide con y_test. Se recalcularon métricas únicamente desde las predicciones guardadas, reproduciendo exactamente las matrices, métricas globales/mensuales/cohortes y acuerdos de fase 7. No se cargan modelos para generar nuevas predicciones, no se ejecuta la evaluación otra vez ni se cambian etiquetas, umbrales, características o PCA.

Se reutilizan resultados train/val guardados y se describen las 20 características existentes de las tres particiones. Sus resúmenes no son parámetros de un nuevo escalador ni selección de variables. Figuras equivalentes (BA global, confusiones, BA mensual, comparación train/val/test) se referencian en resultados/fase_07/graficos; no se duplican.

## Tabla definitiva de test

Soporte real común: clase 0=558, clase 1=553. Sensibilidad=recall positivo, especificidad=TN/(TN+FP); FPR=FP/(TN+FP), FNR=FN/(TP+FN). Proporción positiva=(TP+FP)/N. Denominadores cero mantienen las convenciones de fase 2: tasas no estimables se indican, sin inventar una clase faltante.

| Método | Accuracy | Precision | Sensibilidad | Especificidad | F1 | BA | FPR | FNR | Predicciones positivas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| perceptron | 53.915 % | 52.008 % | 96.022 % | 12.186 % | 67.471 % | 54.104 % | 87.814 % | 3.978 % | 1021/1111 (91.899 %) |
| logistica | 62.286 % | 57.666 % | 91.139 % | 33.692 % | 70.638 % | 62.415 % | 66.308 % | 8.861 % | 874/1111 (78.668 %) |
| mayoritario | 49.775 % | 49.775 % | 100.000 % | 0.000 % | 66.466 % | 50.000 % | 100.000 % | 0.000 % | 1111/1111 (100.000 %) |
| historico | 51.395 % | 51.884 % | 32.369 % | 70.251 % | 39.866 % | 51.310 % | 29.749 % | 67.631 % | 345/1111 (31.053 %) |

| Método | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| perceptron | 68 | 490 | 22 | 531 |
| logistica | 188 | 370 | 49 | 504 |
| mayoritario | 0 | 558 | 0 | 553 |
| historico | 392 | 166 | 374 | 179 |

La logística tiene la mayor BA global observada: 62.415 %, 8.311 puntos por encima del perceptrón, 12.415 sobre mayoritario y 11.106 sobre histórico. Son diferencias descriptivas de esta reserva, no significación ni superioridad universal.

El exceso de positivos es visible: perceptrón 1021 y logística 874 frente a 553 positivos reales. Predecir 1 con frecuencia aumenta la oportunidad de capturar positivos, pero también clasifica muchos negativos como positivos. El caso extremo es mayoritario: sensibilidad 100 %, especificidad 0 %, BA 50 %. Un F1 positivo alto no asegura reconocimiento de ambas clases. Entre los clasificadores, logística ofrece mayor especificidad y BA, aunque conserva 370 falsos positivos y 49 falsos negativos; perceptrón reduce FN a 22 a costa de 490 FP. Histórico tiene mayor especificidad, pero sensibilidad baja; no ofrece el mejor balance global.

## Generalización temporal

Cambios test−validación de BA: perceptrón -7.528 pp; logística +1.216 pp. Todos los cambios train/validación/test para accuracy, precision, recall, specificity, F1 y BA están en analisis_temporal.json. Validación se usó para selección; test es evaluación final ya consumida. No se reajusta el diseño para cambiar estas cifras.

| Conjunto | N | Positivos reales | Pred. positivas perceptrón | Logística | Mayoritario | Histórico |
|---|---:|---:|---:|---:|---:|---:|
| train | 4599 | 51.794 % | 48.815 % | 53.294 % | 100.000 % | 34.594 % |
| val | 2628 | 50.571 % | 57.801 % | 45.738 % | 100.000 % | 34.779 % |
| test | 1111 | 49.775 % | 91.899 % | 78.668 % | 100.000 % | 31.053 % |

| Mes | Positivos reales | Pred. positivas perceptrón | Logística | Mayoritario | Histórico |
|---|---:|---:|---:|---:|---:|
| 2016-01 | 94/219 | 193/219 | 152/219 | 219/219 | 71/219 |
| 2016-02 | 64/219 | 184/219 | 97/219 | 219/219 | 60/219 |
| 2016-03 | 138/219 | 217/219 | 206/219 | 219/219 | 40/219 |
| 2016-04 | 120/227 | 213/227 | 199/227 | 227/227 | 100/227 |
| 2016-05 | 137/227 | 214/227 | 220/227 | 227/227 | 74/227 |

| Mes | Método | BA | Precision | Recall | Specificity | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| 2016-01 | perceptron | 55.740 % | 46.114 % | 94.681 % | 16.800 % | 83.200 % | 5.319 % |
| 2016-01 | logistica | 63.753 % | 52.632 % | 85.106 % | 42.400 % | 57.600 % | 14.894 % |
| 2016-01 | mayoritario | 50.000 % | 42.922 % | 100.000 % | 0.000 % | 100.000 % | 0.000 % |
| 2016-01 | historico | 56.081 % | 52.113 % | 39.362 % | 72.800 % | 27.200 % | 60.638 % |
| 2016-02 | perceptron | 54.667 % | 31.522 % | 90.625 % | 18.710 % | 81.290 % | 9.375 % |
| 2016-02 | logistica | 65.071 % | 43.299 % | 65.625 % | 64.516 % | 35.484 % | 34.375 % |
| 2016-02 | mayoritario | 50.000 % | 29.224 % | 100.000 % | 0.000 % | 100.000 % | 0.000 % |
| 2016-02 | historico | 50.514 % | 30.000 % | 28.125 % | 72.903 % | 27.097 % | 71.875 % |
| 2016-03 | perceptron | 51.235 % | 63.594 % | 100.000 % | 2.469 % | 97.531 % | 0.000 % |
| 2016-03 | logistica | 56.065 % | 66.019 % | 98.551 % | 13.580 % | 86.420 % | 1.449 % |
| 2016-03 | mayoritario | 50.000 % | 63.014 % | 100.000 % | 0.000 % | 100.000 % | 0.000 % |
| 2016-03 | historico | 49.799 % | 62.500 % | 18.116 % | 81.481 % | 18.519 % | 81.884 % |
| 2016-04 | perceptron | 51.238 % | 53.521 % | 95.000 % | 7.477 % | 92.523 % | 5.000 % |
| 2016-04 | logistica | 56.012 % | 56.281 % | 93.333 % | 18.692 % | 81.308 % | 6.667 % |
| 2016-04 | mayoritario | 50.000 % | 52.863 % | 100.000 % | 0.000 % | 100.000 % | 0.000 % |
| 2016-04 | historico | 54.540 % | 58.000 % | 48.333 % | 60.748 % | 39.252 % | 51.667 % |
| 2016-05 | perceptron | 52.620 % | 61.682 % | 96.350 % | 8.889 % | 91.111 % | 3.650 % |
| 2016-05 | logistica | 51.127 % | 60.909 % | 97.810 % | 4.444 % | 95.556 % | 2.190 % |
| 2016-05 | mayoritario | 50.000 % | 60.352 % | 100.000 % | 0.000 % | 100.000 % | 0.000 % |
| 2016-05 | historico | 46.630 % | 55.405 % | 29.927 % | 63.333 % | 36.667 % | 70.073 % |

Se reportan los cinco meses completos, también los difíciles. Predicciones y proporciones de clase se examinan conjuntamente: un agregado no equivale al comportamiento de cada mes.

### Identidad matemática: BA global y promedio mensual

Sea P_t=TP_t+FN_t y N_t=TN_t+FP_t. TPR_global=SUM_t[P_t/SUM(P)]TPR_t; TNR_global=SUM_t[N_t/SUM(N)]TNR_t. Por tanto BA_global=0,5(TPR_global+TNR_global), mientras BA_macro=SUM_t BA_t/5 asigna el mismo peso a cada mes y clase.

BA_global−BA_macro = 0,5 SUM_t[(P_t/SUM(P)−1/5)TPR_t+(N_t/SUM(N)−1/5)TNR_t]. No es la simple ponderación de BA mensual por número total de filas; hay dos distribuciones de pesos por clase. La identidad se verificó numéricamente con atol=1e-14 en BA y 1e-12 en la suma de contribuciones expresada en puntos porcentuales.

En logística: TPR agrupado=0.911392405, TNR agrupado=0.336917563, BA global=62.415 %; promedio mensual=58.406 %. Diferencia=4.009691 pp, explicada exactamente por estas contribuciones:

| Mes | Positivos reales | Negativos reales | Peso TPR | Peso TNR | TPR | TNR | Aporte a diferencia (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2016-01 | 94 | 125 | 0.169982 | 0.224014 | 0.851064 | 0.424000 | -0.768261 |
| 2016-02 | 64 | 155 | 0.115732 | 0.277778 | 0.656250 | 0.645161 | -0.256071 |
| 2016-03 | 138 | 81 | 0.249548 | 0.145161 | 0.985507 | 0.135802 | +2.069130 |
| 2016-04 | 120 | 107 | 0.216998 | 0.191756 | 0.933333 | 0.186916 | +0.716205 |
| 2016-05 | 137 | 90 | 0.247740 | 0.161290 | 0.978102 | 0.044444 | +2.248689 |

No hay contradicción entre 62,42 % global y 58,41 % mensual: son resúmenes distintos. Las fórmulas también se comprobaron para los otros tres métodos.

## Distribuciones observadas y posibles explicaciones

distribuciones_descriptivas.json registra media, desviación poblacional y cuantiles 5/25/50/75/95 de las 20 columnas procesadas en train/val/test, y unidades históricas por mes y cohorte. No se calcula una nueva transformación. Aquí se muestran las seis mayores diferencias de media test−train, ordenadas solo para interpretación, sin retirar o elegir características para los modelos:

| Característica ya preparada | Media val−train / std train | Media test−train / std train |
|---|---:|---:|
| antiguedad_observada_meses | +2.724885 | +4.046562 |
| mes_objetivo_sin | +0.090243 | +0.992743 |
| mes_objetivo_cos | +0.165492 | +0.700490 |
| facturas_media_3 | +0.852867 | +0.485939 |
| clientes_distintos_lag_1 | +0.540260 | +0.276683 |
| facturas_lag_1 | +0.542613 | +0.270225 |

La antigüedad observada crece por definición del tiempo: la diferencia media test−train es 4.046562 desviaciones de train. Los conocidos tienen 36–40 meses observados en test y los nuevos 3–4. Los cambios de seno/coseno del mes reflejan también que test incluye solo enero-mayo, mientras validación cubre un año completo y train 21 meses elegibles: no se comparan calendarios idénticos. La mediana de unidades_lag_1 en test es 192.000 entre conocidos y 3336.000 entre nuevos, de distintas escalas comerciales.

Las diferencias de medias/dispersiones son cambio observado en X y la mezcla de población. No demuestran que haya cambiado P(y|X), ni establecen deriva de concepto, causalidad o un fallo de cobertura. Hipótesis alternativas: extrapolación temporal de una frontera lineal, patrones de ventas cambiantes, muestra breve o problemas no demostrados de identidad/unidades/completitud. Los CSV no permiten discriminar causalmente esas explicaciones. No se modifica ninguna entrada tras este análisis.

## Acuerdos y desacuerdos

Ambos aciertan 554, ambos fallan 374; solo acierta perceptrón 45, solo logística 138. Las cuatro categorías son excluyentes y suman 1.111. Coinciden en 928 y discrepan en 183; acuerdo no equivale a acierto. Los detalles por los cinco meses y las dos cohortes se conservan en acuerdos_desacuerdos.json y en una figura apilada. La diferencia entre aciertos exclusivos contribuye a la diferencia de accuracy, pero no autoriza un ensemble ni inferencia pareada iid ignorando dependencia.

## Productos nuevos: conteos, no solo porcentajes

| Cohorte | Método | N | Soporte 0/1 | TN | FP | FN | TP | BA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| conocidos | perceptron | 1095 | 552/543 | 62 | 490 | 14 | 529 | 54.327 % |
| conocidos | logistica | 1095 | 552/543 | 184 | 368 | 46 | 497 | 62.431 % |
| conocidos | mayoritario | 1095 | 552/543 | 0 | 552 | 0 | 543 | 50.000 % |
| conocidos | historico | 1095 | 552/543 | 389 | 163 | 367 | 176 | 51.442 % |
| nuevos | perceptron | 16 | 6/10 | 6 | 0 | 8 | 2 | 60.000 % |
| nuevos | logistica | 16 | 6/10 | 4 | 2 | 3 | 7 | 68.333 % |
| nuevos | mayoritario | 16 | 6/10 | 0 | 6 | 0 | 10 | 50.000 % |
| nuevos | historico | 16 | 6/10 | 3 | 3 | 7 | 3 | 40.000 % |

Se preservan los 16 casos de ocho nombres nuevos en el global. En perceptrón, precision=100 % procede de solo dos predicciones positivas correctas, con ocho FN: no es éxito general sobre nuevos productos. Logística reconoce siete de diez positivos y cuatro de seis negativos; baseline mayoritario reconoce todos los positivos y ningún negativo. No se puede extrapolar estabilidad con 16 observaciones de dos meses.

## Incertidumbre y sensibilidad agrupada

No se consideran 1.111 réplicas independientes: se repiten productos, se solapan ventanas y existen condiciones compartidas de cada mes. No se presentan p-valores, intervalos de confianza ni bootstrap iid. Solo cinco bloques temporales informan la incertidumbre temporal.

Como exploración sistemática se calcula el agregado reteniendo cuatro meses y omitiendo cada uno por turno. Se reutilizan las mismas predicciones, sin entrenar o elegir omisiones. El resultado oficial sigue siendo el conjunto completo. Estos extremos no constituyen intervalo de confianza ni una nueva evaluación elegida por rendimiento.

| Mes omitido solo para sensibilidad | N retenido | BA perceptrón | BA logística | BA mayoritario | BA histórico |
|---|---:|---:|---:|---:|---:|
| 2016-01 | 892 | 53.575 % | 61.776 % | 50.000 % | 50.226 % |
| 2016-02 | 892 | 53.203 % | 58.157 % | 50.000 % | 51.078 % |
| 2016-03 | 892 | 54.268 % | 62.891 % | 50.000 % | 52.726 % |
| 2016-04 | 884 | 54.804 % | 63.891 % | 50.000 % | 50.225 % |
| 2016-05 | 884 | 54.367 % | 64.129 % | 50.000 % | 52.377 % |

## Interpretación matemática y PCA

Perceptrón: z=wᵀx+b, escalón 1[z>=0], actualización online w←w+η(y−pred)x y b←b+η(y−pred), solo ante error. La ejecución guardada tiene 100 épocas, errores online entre 1863 y 2016, sin época de cero actualizaciones. Pesos del checkpoint 70; online-errors y error al finalizar la época son distintos. Oscilaciones no son prueba automática de implementación errónea ni de no separabilidad.

Logística: p=sigmoid(z), entropía cruzada con logaddexp y gradientes analíticos Xᵀ(p−y)/m y media(p−y), descenso por lote completo. Pérdida train inicial 0.693147181, final 0.626953986; norma gradiente final 0.011097460. La sigmoidea produce probabilidades no necesariamente calibradas. Umbral congelado 0,5: frontera z=0, también un hiperplano en las 20 entradas transformadas. El objetivo suave, lote completo, trayectoria y selección de checkpoint difieren de la actualización por errores del perceptrón; compartir familia de fronteras lineales no obliga a obtener los mismos pesos o resultados. No se retoman épocas ni se recalibra tras test.

PCA existente: PC1=32.988 %, PC2=16.631 %, conservada=49.619 %, no representada=50.381 %. Se reutilizan exclusivamente parámetros/gráficos de fase 6, ajustados en train, sin PCA nuevo sobre test. El solapamiento visual describe esa proyección no supervisada y la pérdida de más de la mitad de la varianza limita su lectura; no demuestra la causa de los errores ni una frontera exacta del clasificador multidimensional. Correlaciones y redundancias impiden interpretar un coeficiente como causa empresarial independiente.

## Reproducibilidad y material académico

Nuevos archivos: src/experiments/analisis_estadistico.py, src/evaluation/graficos_estadisticos.py, tests/test_analisis_estadistico.py y resultados/fase_08/. Las figuras adicionales muestran sensibilidad/especificidad, errores mensuales, predicciones positivas, categorías de acuerdo y cambios de medias de X; no duplican las confusiones ni BA global de fase 7. Este Markdown y interpretacion_empresarial.md aportan conclusiones trasladables a LaTeX; no se genera aún el informe definitivo.

```powershell
python -m src.experiments.analisis_estadistico
python -m unittest discover -s tests -p 'test_analisis_estadistico.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

Manifiesto conserva hashes previos/posteriores de fuentes, matrices, modelos y fases 1–7; validaciones.json registra pruebas, reproducción y revisión visual. El runner no llama predict ni fit de modelos, PCA o preprocesador. Las pruebas históricas ejecutan fixtures sintéticos y preparación temporal aislada sin sustituir artefactos aprobados. No se detectaron inconsistencias demostrables de claves/métricas. Permanecen sin confirmar linaje de identidad, unidad comercial y cierre real. La fase 8 termina con interpretación, sin modificar el experimento ni avanzar a fase 9.
