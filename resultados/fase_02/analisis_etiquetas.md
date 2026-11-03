# Análisis de etiquetas — Fase 2

## Procedimiento reproducible y separación de prueba

Se reutilizaron los seis entregables de `resultados/auditoria/`. El cálculo adicional está en `src/analizar_etiquetas.py`; ejecutar desde la raíz:

```powershell
python -m src.analizar_etiquetas
python -m unittest discover -s tests -p 'test_*.py' -v
```

El resultado estructurado es `exploracion_etiquetas.json`: contiene método, hash, resúmenes por alternativa/conjunto/producto/mes, empates, sensibilidad de elegibilidad y manifiesto de conservación. No contiene matrices de características ni filas definitivas de etiquetas.

El lector recorre el CSV monolítico por bloques y selecciona fechas anteriores a 2016 **antes de convertir o agregar CantidadVendida**. No consulta distribuciones, etiquetas ni rendimiento de prueba. El contenido de 2016 se atraviesa físicamente como texto para leer el archivo, sin uso estadístico. La prueba unitaria con una cantidad de 2016 deliberadamente no numérica comprueba esta separación. El hash completo se usa solo para integridad.

Se agregaron 7.884 pares producto-mes de enero de 2013 a diciembre de 2015, correspondientes a 219 nombres. Se exigieron tres meses consecutivos completos previos: 4.599 observaciones de entrenamiento (abril de 2013–diciembre de 2014) y 2.628 de validación (enero–diciembre de 2015). Mismo universo para A y B. No hay huecos en desarrollo; el auxiliar deja cualquier hueco como desconocido, sin inventar cero para obtener elegibilidad.

Las ventanas, mediana, corte y comparaciones se establecieron sin utilizar cantidades de 2016. La estabilidad en 2015 se usa como evidencia de diseño, no como evaluación final independiente. Es una única validación de desarrollo; no se hizo búsqueda de ventanas ni de márgenes hasta conseguir un balance deseado.

## A. Umbral global de entrenamiento

Sea E_train el conjunto elegible de entrenamiento según el mes objetivo:

\[
G=\operatorname{mediana}\{Q_{p,t}:(p,t)\in E_{train}\}=182,
\qquad y^A_{p,t}=\mathbf1[Q_{p,t}>G].
\]

G se estima solamente con respuestas mensuales de entrenamiento, una contribución por producto-mes. No se pondera por líneas o importe. Se congela para validación y prueba. Empate Q=G pertenece a 0. La mediana de toda la auditoría (que incluía 2016) **no se utiliza**.

**Interpretación:** venta alta en volumen absoluto, comparada con un producto-mes típico de entrenamiento. Es simple y reproducible, pero suma unidades comerciales heterogéneas. Si las unidades por empaque difieren, 182 unidades no tienen el mismo significado económico entre artículos.

**Riesgos:** recalcular G con validación/prueba, confundir balance global con comparación justa entre productos, o presentar G como si fuera un umbral conocido en cada corte histórico de entrenamiento. A usa todo entrenamiento para definir retrospectivamente sus clases; no es una referencia histórica causal por fila de entrenamiento. Esto está permitido para la alternativa A solicitada, pero limita su interpretación histórica.

## B. Umbral relativo causal por producto, ventana 3

\[
R_{p,t}=\operatorname{mediana}(Q_{p,t-1},Q_{p,t-2},Q_{p,t-3}),
\qquad y^B_{p,t}=\mathbf1[Q_{p,t}>R_{p,t}].
\]

No se calcula una mediana estática del producto mirando todo el histórico ni se usan meses futuros para completar ventanas. R existe antes del mes objetivo; se vuelve a calcular con la misma regla en cada corte. Empates se asignan a 0. No hay epsilon ni redondeo para cambiar clases: Q proviene de sumas enteras de CantidadVendida y la mediana de tres enteros también es entera en estos datos.

**Interpretación:** nivel alto relativo a los tres meses recientes del propio producto. Reduce el dominio de productos de gran volumen sobre la etiqueta y admite productos nuevos tras tres meses. No equivale a crecimiento contra el último mes: un producto puede bajar respecto a t−1 y aun superar la mediana reciente.

**Riesgos:** usar `rolling` sin desplazar un mes, incluir Q objetivo, utilizar relleno retrospectivo de historia, optimizar ventana o margen usando prueba, y mezclar unidades/identidad a lo largo del tiempo. Debe medirse sensibilidad temporal; con tres meses, una recuperación tras un mes débil puede parecer alta sin representar crecimiento sostenido. El umbral no corrige por estacionalidad anual; el modelo puede usar calendario conocido.

## Evidencia comparativa

| Alternativa | Conjunto | n | Clase 1 | Clase 0 | % clase 1 | Empates (clase 0) |
|---|---|---:|---:|---:|---:|---:|
| A global, G=182 | Entrenamiento | 4.599 | 2.299 | 2.300 | 49,9891 | 15 |
| A global, G=182 | Validación | 2.628 | 1.492 | 1.136 | 56,7732 | 10 |
| B mediana móvil 3 | Entrenamiento | 4.599 | 2.382 | 2.217 | 51,7939 | 59 |
| B mediana móvil 3 | Validación | 2.628 | 1.329 | 1.299 | 50,5708 | 37 |

Cambio validación−entrenamiento de proporción positiva: A **+6,7841 puntos porcentuales**; B **−1,2231 puntos**. No son métricas de predicción ni demuestran que B sea más fácil de aprender.

| Dominio por producto | A train | A validación | B train | B validación |
|---|---:|---:|---:|---:|
| Productos siempre clase 1 | 100 | 100 | 0 | 0 |
| Productos siempre clase 0 | 19 | 10 | 0 | 0 |
| Mediana del % positivo por producto | 19,0476 | 41,6667 | 52,3810 | 50,0000 |
| Mínimo–máximo del % positivo por producto | 0–100 | 0–100 | 33,3333–71,4286 | 25–75 |

Todos los productos aportan 21 meses en train y 12 en validación: no hay sobrerrepresentación por número de líneas. A separa gran parte de las entidades por escala habitual: 119 de 219 productos no cambian de clase en entrenamiento. B permite ambas clases dentro de cada producto en desarrollo; no prueba que lo mismo suceda en el futuro.

**Estabilidad temporal:** A oscila entre 45,6621 % y 56,6210 % de positivos por mes de entrenamiento y entre 51,1416 % y 69,8630 % en validación. B oscila entre 27,8539 % y 73,9726 % en entrenamiento, y entre 27,8539 % y 70,7763 % en validación. Por tanto, B es más estable en proporción anual agregada, pero **no** en todos los meses. Hay movimientos compartidos que obligan a evaluar por período y evitar tratar filas como independientes.

| Mes de validación | Positivos B | n | % positivos B |
|---|---:|---:|---:|
| 2015-01 | 111 | 219 | 50,6849 |
| 2015-02 | 96 | 219 | 43,8356 |
| 2015-03 | 121 | 219 | 55,2511 |
| 2015-04 | 155 | 219 | 70,7763 |
| 2015-05 | 109 | 219 | 49,7717 |
| 2015-06 | 111 | 219 | 50,6849 |
| 2015-07 | 150 | 219 | 68,4932 |
| 2015-08 | 61 | 219 | 27,8539 |
| 2015-09 | 110 | 219 | 50,2283 |
| 2015-10 | 106 | 219 | 48,4018 |
| 2015-11 | 101 | 219 | 46,1187 |
| 2015-12 | 98 | 219 | 44,7489 |

El JSON conserva los conteos por producto y todos los meses para ambas alternativas, sin seleccionar ejemplos favorables. No se calcularon distribuciones de clase de 2016.

## Recomendación y decisión

**Recomendar B**, con etiqueta positiva “unidades mensuales superiores a la mediana de los tres meses anteriores”. Su fórmula responde a una comparación dentro del producto, no a diferencias fijas de volumen entre productos; tiene historia mínima corta, es causal por fila y presenta ambas clases en todas las entidades de desarrollo. El balance agregado de validación es evidencia adicional, no el único motivo ni una prueba de utilidad del clasificador.

A sigue siendo matemáticamente válida si el negocio prioriza volumen absoluto; requeriría justificar comparabilidad de unidades y un umbral operativo. No se plantea una tercera etiqueta porque no hace falta para resolver la comparación actual. No se ha probado una barrera porcentual adicional ni un cuantil de prueba.

La regla B queda propuesta como contrato para fase 3, sin nuevos ajustes contra prueba. La utilidad económica requiere aceptar que una diferencia pequeña sea relevante; si el negocio exige un incremento mínimo, habrá que especificarlo externamente y versionar la etiqueta antes de evaluar test. No hay bloqueo matemático para B con los datos actuales; disponibilidad al corte, integridad y semántica de unidades siguen siendo supuestos. La fase 3 no debe volver a escoger entre A/B por balance o rendimiento de 2016.
