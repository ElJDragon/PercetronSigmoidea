# PCA visual y congelación — Fase 6

## Resultado y método

PCA propio NumPy, exclusivamente visual, ajustado con X_train (4.599×20) y aplicado a X_val (2.628×20). Se reutilizó el escalado aprobado sin reestandarizar particiones. Se centra Xc=X−media_train; SVD Xc=UΣVᵀ; W2 son las dos primeras columnas de V, y Z=Xc W2. El clasificador sigue usando X de 20 dimensiones. PCA no recibe y: las etiquetas se consultan únicamente para colorear los gráficos de desarrollo.

Convención de signo: en cada componente, el coeficiente de mayor magnitud se hace positivo; si hay empate, manda el primer índice de entrada. Repetir en el mismo entorno conserva resultados. SVD puede elegir otra base dentro de subespacios con valores singulares iguales o casi iguales en distintas bibliotecas/LAPACK: el signo no resuelve esa ambigüedad rotacional. Se conserva la transformación exacta en JSON para su reutilización.

Autovalores de covarianza: sigma_j²/(m−1), ddof=1. El denominador de las proporciones es la suma de **todas** las varianzas de las 20 características transformadas de train, no solo las dos mayores. Rango numérico: 17, tolerancia eps*max(m,d)*sigma_max=1.77879857e-10. Las columnas constantes pueden tener coeficientes cero; rango menor que dos bloquea la proyección en lugar de inventar un segundo componente informativo.

| Medida | Proporción | Porcentaje |
|---|---:|---:|
| PC1 | 0.329877630223 | 32.987763 % |
| PC2 | 0.166308227605 | 16.630823 % |
| Conservada en 2D | 0.496185857828 | 49.618586 % |
| Descartada | 0.503814142172 | 50.381414 % |

Varianza total de train (ddof=1): 20.0043497173. PCA maximiza varianza conservada, no separación entre clases. Las características de fase 3 contienen redundancias algebraicas y correlaciones: el rango puede ser menor que 20 sin invalidar las dos primeras direcciones.

## Interpretación de coeficientes

Se registran todos los coeficientes firmados y sus cuadrados en coeficientes_componentes.csv. Aquí se muestran los cinco mayores por magnitud en cada componente. El cuadrado es participación en la norma unitaria del vector, no importancia predictiva ni efecto empresarial causal.

| Componente | Característica estandarizada | Coeficiente firmado | Coeficiente² / norma² |
|---|---|---:|---:|
| PC1 | unidades_media_3 | 0.386556480 | 14.9426 % |
| PC1 | unidades_mediana_3 | 0.384643640 | 14.7951 % |
| PC1 | unidades_lag_1 | 0.379973488 | 14.4380 % |
| PC1 | unidades_por_factura_lag_1 | 0.377059706 | 14.2174 % |
| PC1 | unidades_lag_2 | 0.371108446 | 13.7721 % |
| PC2 | facturas_lag_1 | 0.494616346 | 24.4645 % |
| PC2 | clientes_distintos_lag_1 | 0.492434743 | 24.2492 % |
| PC2 | variacion_relativa_1 | 0.367933782 | 13.5375 % |
| PC2 | cambio_unidades_1 | 0.345813511 | 11.9587 % |
| PC2 | tendencia_unidades_3 | 0.300490423 | 9.0294 % |

En esta copia, PC1 resume principalmente el nivel de unidades históricas: media, mediana, lags y unidades por factura tienen coeficientes grandes positivos. PC2 combina frecuencia comercial (facturas y clientes distintos) y cambios recientes de unidades, también positivos en la convención elegida. Sus coeficientes menores negativos incluyen unidades_lag_2 (-0.099616) y unidades_lag_3 (-0.081336); no son categorías de clase. PC1 tiene una contribución negativa menor del precio unitario ponderado (-0.054567). Las dos clases se solapan claramente en las dispersiones: no aparece una división nítida en este plano.

Signo positivo indica aumento de la coordenada con esa variable manteniendo las otras constantes en la combinación; signo negativo indica contribución en sentido opuesto. Esta lectura algebraica no autoriza intervenciones causales y puede carecer de interpretación independiente cuando las variables son derivadas de las mismas ventas. Las características son las entradas ya estandarizadas, no las medidas crudas.

## Gráficos y fidelidad

Se exportaron graficos/pca_train.png, pca_val.png y varianza_explicada.png. Las dispersiones conservan todos los puntos, con transparencia y colores por clase, sin jitter ni selección para aparentar separación. Se usa el mismo rango de ejes para train y val, determinado por ambas proyecciones de desarrollo sin alterar W2. La figura de varianza muestra porcentajes individuales, acumulado de todos los componentes y proporción conservada en 2D. No hay proyección de test, frontera de decisión ni entrenamiento en 2D.

El solapamiento entre clases debe interpretarse como limitación de esta vista no supervisada. Ni separación ni solapamiento permiten concluir por sí solos rendimiento, separabilidad o causalidad del clasificador en las 20 entradas. La pérdida de varianza en 2D puede ocultar señales relevantes para clasificar.

## Modelos congelados

- perceptron: `resultados/fase_04/modelo_seleccionado.json`, SHA-256 `27f168c1ceb304a2a841e14b4ca9ab05313f1182a49e80f776ca58060f93ff82`.
- logistica: `resultados/fase_05/modelo_seleccionado.json`, SHA-256 `dc583e7cd1dbffcd08772141b8f981616898e1589ae8f1cb3697b3b2f84caa59`.

Perceptrón: época 70, η=0,01, semilla 42, regla z>=0. Logística: época 200, η=0,1, semilla 42, probabilidad>=0,5. congelacion_experimental.json registra estados completos (pesos, sesgos, configuración y nombres ordenados), hashes de datos/modelos, parámetros del preprocesador, contratos/particiones y reglas de baselines/cohortes. No se copian/reentrenan modelos ni se cambian umbrales. La verificación contra manifiestos previos detiene el procedimiento ante cualquier contradicción.

Se comprobaron matrices y predicciones train/val antes/después de PCA: iguales. Los modelos no reciben Z. La API PCA no acepta etiquetas y su fit solo se llama con train. El proceso abre arrays de train y val; test solo está referenciado por hashes aprobados y no se consultan sus etiquetas ni se calculan métricas.

## Archivos y reproducción

Nuevos módulos: src/visualization/pca.py, dispersion.py, __init__.py; src/experiments/visualizar_pca.py y tests/test_pca.py. En resultados/fase_06: parametros_pca.json (media, W2, valores singulares, varianzas, nombres, convención), varianza_explicada.json, Z_train.npy/Z_val.npy, metadatos alineados por indice_matriz/observacion_id, coeficientes_componentes.csv, tres gráficos, congelación, protocolo_test.md y manifiesto/validaciones. Usar np.load(..., allow_pickle=False); no hay pickle. PCA2D.load(...).transform(X, feature_names=nombres) aplica el mismo centrado y componentes, sin fit ni otro escalado.

```powershell
python -m src.experiments.visualizar_pca
python -m unittest discover -s tests -p 'test_pca.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

La suite histórica verifica preparación en destinos temporales y entrenamientos sintéticos de control; no reentrena los clasificadores WWI ni sustituye entregables. Las validaciones de esta fase comprueban matemáticas PCA, ortogonalidad, varianza, persistencia, train-only, independencia de etiquetas/test, compatibilidad de la congelación y reproducción de artefactos, con integridad previa/posterior.

## Pendientes y fase 7

Persisten cobertura académica no certificada, identidad por nombre y unidad comercial sin catálogo histórico; el pequeño grupo nuevo y cinco meses de test limitan inferencias. No hay contradicciones detectadas en modelos/manifiestos. El protocolo cerrado especifica evaluación común global/mes/cohorte y prohíbe modificar decisiones después de observar test. Test permanece sin evaluación predictiva. Esta fase termina aquí y no implementa ni ejecuta la fase 7.
