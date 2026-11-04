# Diseño de evaluación y reutilización de arquitectura

Objetivo recomendado A, artículo–mes, horizonte de un mes y tres meses previos de historia. Alcance académico bajo supuesto de conservación/cobertura. Este documento no entrena, no elige un checkpoint ni abre test.

## Orden de trabajo posterior

1. Construir y verificar el dataset de desarrollo conforme a `contrato_datos.md`: 4599 filas train y 2628 val esperadas. Verificar tipos, claves, firmas, exclusiones y que ninguna entrada alcance t. La ausencia de recibos cubiertos es cero; historia desconocida se excluye explícitamente.
2. Ajustar preprocesado exclusivamente con train, entrenar las implementaciones propias en un encargo posterior y seleccionar hiperparámetros/checkpoint por BA de validación. Presupuesto, semillas, desempates, umbral logístico inicialmente 0.5 y reglas de parada deben quedar definidos antes del entrenamiento y congelados antes del test. No se prescribe un checkpoint usando estos resultados de viabilidad.
3. Conservar artefactos, parámetros y curvas completas; verificar que reconstruyen predicciones de desarrollo. Las matemáticas de los modelos ya existen: no duplicarlas ni modificar las versiones aprobadas.
4. Congelar objetivo, población, catálogo, preprocesador, modelos, umbral, baselines y protocolo con hashes. Registrar la apertura de test antes de una única extracción posterior. Reservar 2016; la cobertura ya conocida de enero–mayo proviene de la auditoría, no se usa para seleccionar el objetivo. No ajustar decisiones después de mirar sus outcomes.

## Métricas y comparación

BA=(recall+specificity)/2 es primaria, junto con accuracy, precision, recall, specificity, F1 y matriz [[TN,FP],[FN,TP]], clase positiva 1. Reportar soportes y tasa de positivos predichos. Si no se predice ningún positivo, precision/F1 serán 0 con anotación; si falta una clase real, BA será no definida y se anotará el soporte, conservando la fila mensual. No omitir meses monoclase ni calcular promedios de BA omitiéndolos como si fueran BA global. La BA global se obtiene de los conteos agregados. Toda excepción debe resolverse explícitamente en el adaptador de reporte, sin cambiar métricas congeladas de los otros experimentos.

Baseline mayoritario: clase más frecuente de train, empate a 0 (aquí 0), congelada sin recalcular con val/test. Baseline histórico causal para A: pronosticar N(i,t)=N(i,t-1) y clasificar `1[N(i,t-1)>mediana(N(i,t-1),N(i,t-2),N(i,t-3))]`. Usa los mismos tres meses y exactamente las mismas claves que modelos/mayoritario. No se requiere población secundaria de cuatro meses. Copiar la etiqueta del mes anterior sería otro baseline, no este. Las BA de referencia en desarrollo (64,136%/59,749%) no son estimaciones de test.

Evaluación global y por mes, con soporte, prevalencia, confusiones y métricas; errores por artículo y cohorte. Cohortes futuras: visto en train, visto solo en validación y nuevo en test, según primera aparición conocida antes de cada objetivo, sin seleccionar por aparición futura. Un artículo nuevo sin tres meses suficientes queda excluido con motivo; al acumularlos entra con parámetros train congelados. No garantizar que test tenga nuevas identidades evaluables ni ambas clases. Desglosar concentración de positivos en pocas identidades y distinguir muestras dependientes de réplicas independientes.

Accuracy de 98–99% no prueba utilidad para A; hay muy pocos positivos y el mayoritario ya alcanza 98,130%/98,668% en desarrollo. No buscar equilibrio con test, no eliminar artículos por no recibir después ni remuestrear las particiones para elevar puntuaciones. Si los clasificadores no mejoran el histórico, ese es un resultado válido. No demostrar superioridad universal, significación estadística ni valor económico con 35 positivos de validación y solo nueve artículos positivos. Costes de FP/FN y carga de recepción por evento requieren validación empresarial adicional.

## Componentes disponibles y límites de reutilización

| Necesidad | Componente existente | Uso posterior y precaución |
|---|---|---|
| Clasificador lineal | `src/models/perceptron.py`, PerceptronSimple | Dimensión dinámica, sin reimplementar actualización; conservar checkpoint y sesgo |
| Probabilidad binaria | `src/models/logistic_regression.py`, LogisticRegressionManual | Sigmoide, pérdida estable y gradiente ya implementados; congelar umbral |
| Preprocesado | `src/pipeline/preprocesamiento.py`; `src/datamarts/transacciones/preprocesamiento.py` | Reusar reglas train-only; ambos catálogos son específicos de sus experimentos, requieren adaptador de Inventario posterior sin parchear FEATURES globales |
| Métricas | `src/evaluation/metricas.py` | Reusar conteos/métricas; añadir tratamiento explícito de meses monoclase en reporte nuevo, sin modificar los resultados antiguos |
| PCA SVD | `src/visualization/pca.py`, PCA2D | Solo visualización posterior train-only, no cambiar entradas de modelos; no se ejecutó en esta investigación |
| Gráficos | `src/evaluation/graficos.py`, `graficos_logistica.py`, `graficos_finales.py`; `src/visualization/dispersion.py` | Reutilizar estilos y funciones generalizables; separar rutas/etiquetas de Inventario |
| Convergencia | `src/experiments/entrenar_perceptron.py`, `entrenar_logistica.py`, `entrenar_transacciones.py` | Reusar historial, selección de checkpoint y verificación; no ejecutar scripts con rutas de otros experimentos |
| Evaluación reproducible | `src/experiments/evaluar_transacciones.py`, `evaluacion_final.py` | Reutilizar patrón de manifiestos, hashes y registro previo de apertura, con adaptador independiente; no ejecutar sus entradas contra este mart |
| Lectura SQL | `src/db/conexion.py`, Configuracion/Lector/validar_select | SELECT únicamente, autenticación integrada, extracción fija antes de 2016 para diseño |

Los componentes de experimentos no son runners genéricos listos para Inventario: hay que adaptar orquestación/catálogos/rutas en trabajo posterior. No se implementó de nuevo matemática alguna, ni se entrenó, ejecutó PCA o construyó matrices. No se modificó `informe/`, Ventas ni Transacciones. Se recomienda construir A posteriormente, manteniendo todas las restricciones y la reserva de test.
