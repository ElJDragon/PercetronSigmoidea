# Análisis de etiquetas de desarrollo y dictamen

Solo train 2013–2014 y validación 2015. No se calcularon etiquetas, prevalencias ni resultados de 2016. Granularidad común: artículo operativo–mes natural; horizonte: siguiente mes, predicción antes de comenzar t, usando tres meses consecutivos cerrados. La elegibilidad no depende de que haya recepción en t ni de supervivencia posterior. Primera aparición por cualquier movimiento conocido antes de t. En este extracto todos aparecen en enero de 2013: train elegible abril de 2013–diciembre de 2014 (21 meses), val enero–diciembre de 2015 (12 meses), 219 artículos en cada mes. No se exige haber recibido recientemente ni se seleccionan solo los artículos con positivos.

## Candidatos y significado

- **A, frecuencia elevada de recepciones:** N(i,t) = número de IDs distintos tipo 11 del artículo en t; y = 1[N(i,t) > mediana(N(i,t-1),N(i,t-2),N(i,t-3))]. Busca anticipar intensidad relativa de trabajo de recepción por artículo; no número de días, unidades, órdenes completas ni necesidad de compra. Cada ID es una recepción registrada, que en desarrollo coincide con un par artículo–orden distinto. No se promete medir carga laboral proporcional ni tiempo de descarga. Historia: conteos, cantidades de recepción previas y calendario. Empate a 0.
- **B, volumen elevado de recepciones:** Q(i,t) = suma de Quantity tipo 11; y = 1[Q(i,t) > mediana(Q(i,t-1),Q(i,t-2),Q(i,t-3))]. Busca planificar intensidad de unidades que llegan, sin mezclar unidades entre artículos. Es interpretable porque el origen expresa unidades base y la unidad se mantiene en desarrollo; no es demanda ni saldo. Misma historia causal; requiere sostener comparabilidad de unidades en futuras extracciones. Empate a 0.
- **C, frecuencia elevada de ajustes:** J(i,t) = número de IDs tipo 12, incluso cantidad cero; y = 1[J(i,t) > mediana(J(i,t-1),J(i,t-2),J(i,t-3))]. Busca anticipar excepciones/recuentos, no deterioro financiero ni pérdidas monetarias. Conteos de ajustes previos, signos históricos y calendario serían causales. Solo 38 eventos en 36 meses, concentrados en meses trimestrales: no hay evidencia suficiente de un proceso recurrente aprendible con esta muestra, aunque hay dos clases agregadas.
- **D, actividad neta:** F(i,t) = suma firmada de Quantity de tipos 10, 11 y 12; y = 1[F(i,t) > mediana(F(i,t-1),F(i,t-2),F(i,t-3))]. Sería incremento relativo de flujo neto mensual, no stock alto. Requiere interpretar todos los tipos y unidades; no incorpora saldo actual ni inventa saldo inicial. Las series de flujo/entradas/salidas previas serían causales. Se descarta como nueva aplicación: en 6594/7227 ventanas es exactamente la etiqueta de ventas bajas y no un fenómeno independiente.

Para cada candidato, el mayoritario es clase 0 obtenida solo de train. Baseline histórico: predecir que el agregado de t será igual al de t-1; por tanto b(i,t) = 1[V(i,t-1) > mediana(V(i,t-1),V(i,t-2),V(i,t-3))], con V=N,Q,J,F. Solo necesita la misma historia de tres meses. No es copiar y(i,t-1), que exigiría el cuarto mes. Ningún baseline usa V(i,t).

## Distribuciones, variación y baseline

| Candidato | Partición | Artículos | Filas | Positivos | Negativos | Prevalencia | Artículos positivos | Con ambas clases | BA histórico |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | train | 219 | 4599 | 86 | 4513 | 1.870% | 10 | 10 | 64.136% |
| A | val | 219 | 2628 | 35 | 2593 | 1.332% | 9 | 9 | 59.749% |
| B | train | 219 | 4599 | 177 | 4422 | 3.849% | 10 | 4 | 92.079% |
| B | val | 219 | 2628 | 98 | 2530 | 3.729% | 9 | 8 | 85.596% |
| C | train | 219 | 4599 | 22 | 4577 | 0.478% | 22 | 22 | 49.760% |
| C | val | 219 | 2628 | 14 | 2614 | 0.533% | 14 | 14 | 49.732% |
| D | train | 219 | 4599 | 2223 | 2376 | 48.337% | 219 | 213 | 53.219% |
| D | val | 219 | 2628 | 1306 | 1322 | 49.696% | 219 | 218 | 51.191% |

Los cuatro candidatos tienen ambas clases en cada partición agregada, pero no necesariamente en cada mes. BA mayoritaria = 50% en train y val; su accuracy elevada en A/B/C es consecuencia de la rareza, no capacidad útil. Los JSON conservan matrices de confusión y accuracy de ambos baselines. En A, el histórico da BA 64,136% en train y 59,749% en val; en B, 92,079% y 85,596%. Estas son referencias causales de desarrollo, no resultados de clasificadores ni de test. No se selecciona una etiqueta por superar o facilitar un baseline.

## Prevalencia mensual completa: positivos / negativos (porcentaje positivo)

| Mes | N por candidato | A | B | C | D |
|---|---:|---|---|---|---|
| 2013-04 | 219 | 8 / 211 (3.653%) | 8 / 211 (3.653%) | 1 / 218 (0.457%) | 61 / 158 (27.854%) |
| 2013-05 | 219 | 9 / 210 (4.110%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 62 / 157 (28.311%) |
| 2013-06 | 219 | 1 / 218 (0.457%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 123 / 96 (56.164%) |
| 2013-07 | 219 | 7 / 212 (3.196%) | 9 / 210 (4.110%) | 2 / 217 (0.913%) | 92 / 127 (42.009%) |
| 2013-08 | 219 | 2 / 217 (0.913%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 157 / 62 (71.689%) |
| 2013-09 | 219 | 1 / 218 (0.457%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 131 / 88 (59.817%) |
| 2013-10 | 219 | 7 / 212 (3.196%) | 8 / 211 (3.653%) | 3 / 216 (1.370%) | 110 / 109 (50.228%) |
| 2013-11 | 219 | 0 / 219 (0.000%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 113 / 106 (51.598%) |
| 2013-12 | 219 | 1 / 218 (0.457%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 113 / 106 (51.598%) |
| 2014-01 | 219 | 6 / 213 (2.740%) | 8 / 211 (3.653%) | 4 / 215 (1.826%) | 72 / 147 (32.877%) |
| 2014-02 | 219 | 0 / 219 (0.000%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 134 / 85 (61.187%) |
| 2014-03 | 219 | 1 / 218 (0.457%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 112 / 107 (51.142%) |
| 2014-04 | 219 | 6 / 213 (2.740%) | 8 / 211 (3.653%) | 4 / 215 (1.826%) | 83 / 136 (37.900%) |
| 2014-05 | 219 | 5 / 214 (2.283%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 70 / 149 (31.963%) |
| 2014-06 | 219 | 2 / 217 (0.913%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 101 / 118 (46.119%) |
| 2014-07 | 219 | 7 / 212 (3.196%) | 9 / 210 (4.110%) | 5 / 214 (2.283%) | 88 / 131 (40.183%) |
| 2014-08 | 219 | 1 / 218 (0.457%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 158 / 61 (72.146%) |
| 2014-09 | 219 | 5 / 214 (2.283%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 145 / 74 (66.210%) |
| 2014-10 | 219 | 8 / 211 (3.653%) | 9 / 210 (4.110%) | 3 / 216 (1.370%) | 78 / 141 (35.616%) |
| 2014-11 | 219 | 1 / 218 (0.457%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 126 / 93 (57.534%) |
| 2014-12 | 219 | 8 / 211 (3.653%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 94 / 125 (42.922%) |
| 2015-01 | 219 | 0 / 219 (0.000%) | 9 / 210 (4.110%) | 4 / 215 (1.826%) | 107 / 112 (48.858%) |
| 2015-02 | 219 | 1 / 218 (0.457%) | 1 / 218 (0.457%) | 0 / 219 (0.000%) | 114 / 105 (52.055%) |
| 2015-03 | 219 | 6 / 213 (2.740%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 103 / 116 (47.032%) |
| 2015-04 | 219 | 4 / 215 (1.826%) | 9 / 210 (4.110%) | 5 / 214 (2.283%) | 66 / 153 (30.137%) |
| 2015-05 | 219 | 2 / 217 (0.913%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 107 / 112 (48.858%) |
| 2015-06 | 219 | 2 / 217 (0.913%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 108 / 111 (49.315%) |
| 2015-07 | 219 | 7 / 212 (3.196%) | 9 / 210 (4.110%) | 1 / 218 (0.457%) | 75 / 144 (34.247%) |
| 2015-08 | 219 | 1 / 218 (0.457%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 157 / 62 (71.689%) |
| 2015-09 | 219 | 2 / 217 (0.913%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 110 / 109 (50.228%) |
| 2015-10 | 219 | 2 / 217 (0.913%) | 9 / 210 (4.110%) | 4 / 215 (1.826%) | 117 / 102 (53.425%) |
| 2015-11 | 219 | 0 / 219 (0.000%) | 8 / 211 (3.653%) | 0 / 219 (0.000%) | 119 / 100 (54.338%) |
| 2015-12 | 219 | 8 / 211 (3.653%) | 9 / 210 (4.110%) | 0 / 219 (0.000%) | 123 / 96 (56.164%) |

A tiene meses monoclase: 2013-11, 2014-02, 2015-01 y 2015-11, todos sin positivos; debe informarse BA mensual como no definida cuando falta alguna clase, sin asignar artificialmente 50%. C presenta muchos más meses sin positivos; no se eliminan esos meses para mejorar la evaluación. Las dependencias entre meses y artículos y la concentración de positivos limitan cualquier inferencia estadística. En A, solo diez artículos varían en train y nueve en val; sus 35 positivos de validación no sostienen precisión estadística fina. Los desgloses por identidad están en `exploracion_etiquetas.json`.

## Dictamen

**A: viable bajo supuestos académicos explícitos; recomendado para construcción posterior del dataset.** Se recomienda por identidad verificable del evento, independencia respecto de las líneas vendidas, regla sencilla disponible al corte y valor empresarial de anticipar recepciones registradas. No se recomienda por equilibrio: tiene prevalencia muy baja (1,870% train; 1,332% val), y esta desventaja se acepta y documenta. El experimento sería exploratorio de eventos infrecuentes, sin garantía de mejora sobre el histórico ni de utilidad operativa. Mantener la población completa; no eliminar artículos con meses cero para fabricar equilibrio. La construcción debe respetar el contrato, sin abrir test ni entrenar en esta etapa.

**B: viable bajo los mismos supuestos, alternativa no seleccionada.** Hay comparabilidad de unidades en desarrollo y ambas clases; añade dependencia del significado y conservación de Quantity, menor variación de clase dentro de cada artículo en train y un baseline fuerte. No se rechaza por su BA ni se afirma que sea peor predictivamente. A responde con menos supuestos de unidades a una pregunta distinta de volumen vendido.

**C: condicionado; no recomendado con la fuente actual.** No es matemáticamente monoclase, pero 22/14 positivos train/val y escasa repetición por artículo son insuficientes para recomendar otro clasificador generalizable. Para retomarlo: mayor horizonte de eventos de recuento con tipos y fechas confiables, más ajustes recurrentes o una unidad empresarial diferente justificada antes de mirar test.

**D: no viable como objetivo independiente de Ventas en esta evidencia.** La suma firmada es calculable, pero predomina una transformación de las mismas salidas vendidas. El equilibrio casi 50/50 no compensa esa duplicidad. Si se requiere predecir existencias o roturas, harían falta saldos iniciales certificados, todos los flujos y cierres, reservas y demanda no atendida; los estados actuales no reconstruyen esa historia.

Autorización recomendada: siguiente encargo limitado a construir y verificar causalmente A con controles de integridad; no implica autorización ejecutada de matrices, entrenamiento ni apertura de test. Este encargo se detiene en el dictamen.
