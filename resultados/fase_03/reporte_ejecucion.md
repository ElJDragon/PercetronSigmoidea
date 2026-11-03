# Ejecución del pipeline WWI — Fase 3

## Resultado y política

Contrato 2.0. Cobertura seleccionada: **academica**. Supuesto académico explícito de completitud y disponibilidad al cierre; no certificado.
El modo académico asume completos los meses de enero de 2013 a mayo de 2016: no certifica cierres ni latencia del origen. La etiqueta sigue siendo Q_objetivo > mediana(t−1,t−2,t−3), empate 0.

| Partición | Observaciones confirmadas para matrices |
|---|---:|
| train | 4599 |
| val | 2628 |
| test | 1111 |

Calendario producto-mes: 9019 filas desde primera aparición; observaciones de productos reconocidos antes del corte: 8792. Las primeras apariciones no crean predicciones en su propio mes. No se materializan como cero los 288 pares previos a la primera observación de los ocho productos recientes. Las filas no elegibles se conservan con estado y sin y; las pendientes de cobertura no se convierten en clase 0.

## Características y transformación

Catálogo base: 23 numéricas autorizadas, cero categóricas. Matriz final: 20 columnas float64. Etiquetas canónicas int8, 0/1. Las tres razones definidas permiten nulos no esenciales, imputados con mediana de train. Se incorporó `facturas_lag_1_cero` como indicador explícitamente requerido para la razón unidades por factura; `mes_anterior_sin_ventas` también identifica precio sin denominador. No se generan indicadores indiscriminados.

Columnas descartadas exclusivamente por train:

- `base_variacion_cero`: constante_en_train_tras_imputacion.
- `mes_anterior_sin_ventas`: constante_en_train_tras_imputacion.
- `facturas_lag_1_cero`: constante_en_train_tras_imputacion.

No se retiraron columnas por correlación con y, rendimiento o inspección de test. Se conservan redundancias algebraicas entre lags y medias/cambios. Cada parámetro (mediana, media y desviación poblacional ddof=0) proviene solo de train. Las constantes se eliminan sin dividir por cero; no se seleccionó un número fijo de entradas.

Orden final:

1. `unidades_lag_1`
2. `unidades_lag_2`
3. `unidades_lag_3`
4. `unidades_media_3`
5. `unidades_mediana_3`
6. `unidades_std_3`
7. `cambio_unidades_1`
8. `variacion_relativa_1`
9. `tendencia_unidades_3`
10. `facturas_lag_1`
11. `facturas_media_3`
12. `ingresos_lag_1`
13. `ingresos_media_3`
14. `precio_unitario_ponderado_lag_1`
15. `clientes_distintos_lag_1`
16. `unidades_por_factura_lag_1`
17. `antiguedad_observada_meses`
18. `mes_objetivo_sin`
19. `mes_objetivo_cos`
20. `dias_mes_objetivo`

Características condicionadas apagadas: identidad one-hot, Marca/Tamanio/EsRefrigerado, mezclas cliente/territorio, beneficio histórico, ventanas de 12 meses y calendario redundante mes/trimestre. Prohibidas como entradas: Q/y objetivo, resultados monetarios o clientes del horizonte, identificadores directos, variables de inventario inventadas y agregados futuros. R solo se usa como mediana histórica válida; el resto de respuestas permanece separado de X.

## Archivos y uso

En `datos/Procesados/fase_03/`: `producto_mes.csv` contiene agregados y cobertura; `observaciones.csv` contiene estados, claves, Q/R/y y características crudas. Son dos granos diferentes, no copias equivalentes. Los archivos X_train/X_val/X_test.npy y y_train/y_val/y_test.npy son el único formato de matrices/etiquetas. Leer con `numpy.load(path, allow_pickle=False)`. Para reconstruir exactamente las características float64 desde CSV con pandas, usar `read_csv(..., float_precision="round_trip")`; el CSV se escribe con 17 cifras significativas. Metadatos por partición alineados por indice_matriz; claves SHA-256 de la pareja producto-mes, sin usar identidad en X.

`caracteristicas.json` define orden, fórmulas y faltantes; `preprocesador.json` conserva todos los parámetros legibles. Para aplicar nuevos registros base autorizados, usar `PreprocesadorComun.load(...).transform(frame[FEATURES])`, sin fit. No hay vocabulario categórico porque el contrato base no autoriza categóricas. Una transformación cargada valida esquema y finitud; las características históricas se construyen antes con el mismo corte/contrato.

El manifiesto en resultados/fase_03 conserva cobertura, fuente, hashes de contrato/entradas y código, orden/tipos y hashes de derivados. No contiene distribuciones ni métricas de etiquetas test. Las respuestas de test se materializan únicamente para evaluación futura, sin usarlas en fit o selección. Ambos clasificadores deben consumir exactamente estos mismos arrays e índices; conversión bipolar posterior, sin alterar filas.

## Reproducción y verificaciones

Desde la raíz:

```powershell
python -m src.pipeline.construir_dataset --cobertura academica
python -m unittest discover -s tests -p 'test_*.py' -v
```

El modo por defecto es `desconocida`: bloquea matrices si no hay historia/respuestas confirmadas. Para cierres externos usar `--cobertura explicita --archivo-cobertura ruta.json`, con estructura `{"meses":{"2013-01":"completa"},"evidencia":"procedencia del cierre"}`. Meses no declarados quedan desconocidos. El archivo no puede certificar ventas que aún no ocurrieron; quien lo configura debe aportar cierres conocidos al corte. Una ejecución bloqueada no reemplaza resultados de una ejecución académica previa: consultar el manifiesto de la preparación que se vaya a utilizar.

Se verifican hash/esquema/separador contra contrato, sumas de unidades y centavos, facturas distintas, claves, conteos académicos y etiquetas train/val contra fase 2. Los agregados monetarios suman centavos exactos para que el orden de las líneas no cambie su resultado. La historia se desplaza por calendario antes de etiquetar; test anterior puede ser historia de test posterior, sin reajustar transformaciones. El conjunto completo de pruebas incluye perturbación de objetivo/futuro/test, cobertura, fronteras temporales, serialización y reproducción; sus resultados se entregan junto con la ejecución.

## Límites antes de entrenar

No hay modelos ni PCA ni gráficos de entrenamiento. No se modificaron fuentes, referencias ni entregables aprobados. La utilidad económica de superar la mediana no está demostrada. Identidad por nombre, unidades de empaque y disponibilidad al cierre requieren confirmación para uso operacional. Sin metadatos del origen, la completitud es un supuesto académico explícito. Antes de fase de modelos confirmar biblioteca de gradientes autorizada; PCA sigue separado de X y requiere su autorización. Esta ejecución termina aquí, sin avanzar a fase 4.
