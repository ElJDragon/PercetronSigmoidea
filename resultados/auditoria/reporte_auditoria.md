# Auditoría WWI — Fase 1

## Alcance y método

Solo perfilado y análisis exploratorio. No se entrenaron modelos, no se implementaron PCA ni preprocesamiento predictivo, no se construyeron datasets definitivos y no se eliminaron filas. Las recomendaciones no fijan el objetivo.

Lectura CSV completa como texto, sin conversión implícita de nulos; decodificación completa, separador por csv.Sniffer, normalización en memoria según esquema. UTF-8 compatible no identifica de forma única la codificación original. Fechas ISO tratadas como UTC para comparar fechas de calendario; no se infiere una zona comercial real del sufijo Z de OpenRefine.

Comparación por InvoiceLineID único: Decimal exacto en números, fechas ISO normalizadas, nulos explícitos; no se redondea. Espacios de categorías se conservan en la comparación estricta y se diagnostican por separado. No se usa similitud difusa.

IQR (1,5 veces rango intercuartílico) identifica observaciones para revisar, no errores. En IDs, calendario y banderas no debe interpretarse como anomalía de negocio. No se impone un máximo monetario arbitrario. Validaciones de rango son controles exploratorios; devoluciones podrían justificar negativos en otras extracciones.

## Workspace e insumos

Directorios existentes: datos/originales, datos/Procesados, informe, notebooks, referencias/Codigo_anteiror, resultados, src, tests. Se conserva su capitalización. README.md y requirements.txt estaban vacíos. No existían scripts en src ni pruebas en tests. No se accedió al proyecto DeberPerceptron.

Se encontraron ambos CSV, los tres PDF y la nota Markdown. Inventario protegido completo con SHA-256:

| Archivo | SHA-256 |
|---|---|
| `datos\originales\dataset_ventas_wwi.csv` | `9da69cab3fd573f0f36d92cb97f533740bcb73c08b3909885209843ec8990745` |
| `datos\originales\datos_limpios.csv` | `d4f3b3b7b018ee49c515915fe851ff7327d82fb198cbc8af117e8f0e73fff37e` |
| `referencias\Carta_de_Diseño (2).pdf` | `e342b047019d74c2c936b942367b8ba6b1bb9109989b342ad2eb0a299bc4d7c0` |
| `referencias\Codigo_anteiror\EXPLICACION_PERCEPTRON.md` | `486d85abce5871c1fd479fb100ee3998072e8bde9f6302a065017f3cece05881` |
| `referencias\Codigo_anteiror\Perceptron.py` | `1390a68564a9b458b875f3b5eeeae3e8766ff3d818639a3cb4059739d0fb5aab` |
| `referencias\Codigo_anteiror\requirements.txt` | `b63aea5d6238472ccdd07a5fa41c45a7f62b98df8900d8c05344035431186cbd` |
| `referencias\Codigo_anteiror\test_perceptron.py` | `60fbfe0bb03eb26a6c8dbfd8ca4502ff0ce0f173660c2aece82536b610045596` |
| `referencias\Dashboard (1).pdf` | `4708dcea91a45c0fee641c1173073d72eda6e2921a65eaeeb5c2ada13a409102` |
| `referencias\GRaficos con MNIST 4077751154f78080828ff5640e835bc1.md` | `61d073b362cce2b19551df991618c34bc4b4c83cb9b0324ba986fa53dbf3e1d2` |
| `referencias\Openrefine_Cluster (2).pdf` | `8fc6edd6cecb6eb77101002a7caa938e35e9070b99ea533156797d818057360e` |

El informe OpenRefine menciona dataset_ventas_wwi_limpio.csv; no existe con ese nombre. El archivo disponible es datos_limpios.csv. El código de referencia apunta a un CSV sintético y resultados antiguos que no fueron copiados: no son insumos necesarios de WWI. La carpeta real es Codigo_anteiror, no Codigo_anterior.

Del código previo se aprovechan como referencia conceptual validación, CLI, métricas manuales y trazabilidad. No se importa ni ejecuta: fija tres entradas, etiquetas Hombre/Mujer, partición aleatoria y mínimo de 200 épocas, reglas no transferibles automáticamente a este problema.

## Perfil general

| Fuente | Filas | Columnas | Bytes | Separador | Codificación válida |
|---|---:|---:|---:|---|---|
| dataset_ventas_wwi.csv | 228265 | 30 | 59150151 | `;` | utf-8-sig |
| datos_limpios.csv | 228265 | 30 | 59899758 | `;` | utf-8-sig |

Período: 2013-01-01 00:00:00+00:00 a 2016-05-31 00:00:00+00:00; 41 meses, 1069 fechas con ventas. 2016 es parcial (enero-mayo). No hay ventas dominicales en esta extracción; esto no prueba ausencia de demanda.

Las 30 columnas son atributos observados, no 30 predictores aprobados. Hay 14 columnas numéricas, 2 fechas y 14 textos según el esquema auditado. El número final de características será dinámico después de elegir el grano y codificar categorías.

Nombres y tipos completos en diccionario_datos.csv; distribuciones, cardinalidades, estadísticos, nulos y outliers por fuente/columna en perfilado_datos.json.

## Calidad y grano

| Columna | Faltantes | % | No nulos distintos |
|---|---:|---:|---:|
| Marca | 209328 | 91.703941 | 1 |
| Tamanio | 67315 | 29.489847 | 43 |
| GrupoCompra | 84297 | 36.929446 | 2 |

Esos faltantes son NULL literal en dataset_ventas_wwi.csv y vacíos en datos_limpios.csv; no hay otras columnas con ausencia. No convertirlos indiscriminadamente en categorías comerciales ni borrar filas. Marca solo tiene Northwind entre no nulos, pero con ausencia no es constante completa. DescuentoEstandar sí es constante en cero.

Duplicados completos: 0; duplicados de InvoiceLineID: 0. Hay 70510 facturas y de 1 a 5 líneas por factura. Las repeticiones de InvoiceID son legítimas. Una línea no se asocia a múltiples facturas; fecha, cliente, ciudad y vendedor son consistentes dentro de cada factura.

Grano comprobado: una línea de factura identificada por InvoiceLineID. Producto, cliente y ubicación están descritos por nombres, sin StockItemID/CustomerID/CityID; la unicidad real de esas entidades no puede garantizarse con nombres. No asumir que InvoiceID identifica una venta de un solo producto.

Hay 227 nombres de producto, 663 de cliente, 655 ciudades, 10 vendedores y 9 nombres preferidos. Estos últimos no identifican unívocamente al vendedor.

No se detectaron conversiones numéricas inválidas, no finitos, fechas inválidas ni violaciones de los rangos controlados en estos CSV. Las medidas de cantidad, precio, ingresos e impuesto son positivas. Beneficio negativo es válido como resultado, no una violación de rango.

Inconsistencias del calendario: `{'Dia': 0, 'Mes': 0, 'Anio': 0, 'Trimestre': 0, 'NombreDia': 0, 'NombreMes': 0}`. FechaApertura posterior a venta: 0.

## Relaciones monetarias

Se comprobaron CantidadVendida × PrecioUnitario = VentaSinImpuesto y VentaSinImpuesto + Impuesto = VentaTotal en todas las filas de ambas fuentes, sin incumplimientos. Tolerancia absoluta 1e-8, relativa cero: absorbe representación binaria sin tolerar diferencias de un centavo. Error máximo observado: 2,2737367544323206e-13. La comparación ENTRE CSV usa Decimal exacto, sin esta tolerancia.

No se puede verificar Beneficio = venta - costo porque no hay costo histórico. No se puede validar la tasa fiscal de cada línea sin TaxRate. Los importes no demuestran descuentos, devoluciones o causas de pérdida.

| Medida | Mínimo | Máximo | Atípicos IQR |
|---|---:|---:|---:|
| CantidadVendida | 1.0 | 360.0 | 13907 |
| PrecioUnitario | 0.66 | 1899.0 | 24184 |
| VentaSinImpuesto | 2.5 | 18990.0 | 20012 |
| Impuesto | 0.38 | 2848.5 | 19935 |
| VentaTotal | 2.88 | 21838.5 | 19935 |
| Beneficio | -645.0 | 9200.0 | 26845 |

No se truncaron ni excluyeron observaciones. IQR debe revisarse por familias de producto y contexto, sin ajustar futuros umbrales con prueba.

## Comparación de fuentes

Esquema y claves iguales; ninguna clave exclusiva. Diferencias de contenido estrictas: 268. Todas corresponden a espacios periféricos en NombreCliente. Equivalencia estricta: False. Equivalencia aceptando trim textual explícito: True.

Las 268 celdas de nombre cambian de `David Novacek ` a `David Novacek`. El diagnóstico de cardinalidad de comparacion_csv.json comprueba si trim colapsa nombres; no se considera un simple cambio de formato de número/fecha/nulo. Las mismas líneas comerciales y sus medidas se conservan. El informe OpenRefine no documenta expresamente esta transformación de espacios: falta su historial de operaciones para atribuir su origen.

Recomiendo datos_limpios.csv como fuente principal de trabajo: mantiene claves, categorías salvo ese trim y medidas exactas; representa ausencias como vacíos y fechas ISO. Conservar ambos hashes y documentar la decisión de aceptar el trim. No declarar archivos idénticos ni equivalencia estricta.

## Beneficio recalculado

| Clase | Líneas | % |
|---|---:|---:|
| positive | 223639 | 97.973408 |
| negative | 4626 | 2.026592 |
| zero | 0 | 0.000000 |

Exactitud teórica del mayoritario binario no negativo: 97.973408 %. No es un modelo entrenado. Accuracy aislada sería engañosa; proponer recall de negativos, precision, F1, balanced accuracy y matriz de confusión.

Hay 18 productos con pérdidas. Cuatro variantes de Halloween zombie mask concentran 88.932123 % de casos negativos; cada una pierde en todas sus líneas observadas. El top 5 concentra 89.926502 %. Riesgo: memorizar identidad de producto, sin demostrar anticipación de causas de pérdida ni generalización a productos nuevos.

| Producto | Líneas | Negativas | % pérdidas dentro del producto | % de todos los negativos |
|---|---:|---:|---:|---:|
| Halloween zombie mask (Light Brown) XL | 1071 | 1071 | 100.0000 | 23.1518 |
| Halloween zombie mask (Light Brown) S | 1044 | 1044 | 100.0000 | 22.5681 |
| Halloween zombie mask (Light Brown) M | 1017 | 1017 | 100.0000 | 21.9844 |
| Halloween zombie mask (Light Brown) L | 982 | 982 | 100.0000 | 21.2278 |
| USB food flash drive - sushi roll | 1066 | 46 | 4.3152 | 0.9944 |
| USB food flash drive - dessert 10 drive variety pack | 1065 | 43 | 4.0376 | 0.9295 |
| USB food flash drive - shrimp cocktail | 1117 | 41 | 3.6705 | 0.8863 |
| USB food flash drive - chocolate bar | 1068 | 41 | 3.8390 | 0.8863 |
| USB missile launcher (Green) | 1048 | 40 | 3.8168 | 0.8647 |
| USB food flash drive - hot dog | 1040 | 39 | 3.7500 | 0.8431 |
| USB food flash drive - dim sum 10 drive variety pack | 1015 | 38 | 3.7438 | 0.8214 |
| USB food flash drive - donut | 1089 | 38 | 3.4894 | 0.8214 |
| USB food flash drive - banana | 1061 | 36 | 3.3930 | 0.7782 |
| USB rocket launcher (Gray) | 1078 | 35 | 3.2468 | 0.7566 |
| USB food flash drive - pizza slice | 1028 | 31 | 3.0156 | 0.6701 |
| USB food flash drive - hamburger | 1046 | 29 | 2.7725 | 0.6269 |
| USB food flash drive - cookie | 1041 | 28 | 2.6897 | 0.6053 |
| USB food flash drive - fortune cookie | 1072 | 27 | 2.5187 | 0.5837 |

### Distribución temporal del beneficio

| Mes | Líneas | Positivas | Negativas | Cero | % negativas |
|---|---:|---:|---:|---:|---:|
| 2013-01 | 5246 | 5158 | 88 | 0 | 1.6775 |
| 2013-02 | 3707 | 3649 | 58 | 0 | 1.5646 |
| 2013-03 | 5330 | 5226 | 104 | 0 | 1.9512 |
| 2013-04 | 5254 | 5167 | 87 | 0 | 1.6559 |
| 2013-05 | 5617 | 5529 | 88 | 0 | 1.5667 |
| 2013-06 | 5287 | 5179 | 108 | 0 | 2.0427 |
| 2013-07 | 5834 | 5735 | 99 | 0 | 1.6969 |
| 2013-08 | 4767 | 4686 | 81 | 0 | 1.6992 |
| 2013-09 | 5021 | 4926 | 95 | 0 | 1.8921 |
| 2013-10 | 5097 | 5001 | 96 | 0 | 1.8835 |
| 2013-11 | 4899 | 4809 | 90 | 0 | 1.8371 |
| 2013-12 | 4909 | 4815 | 94 | 0 | 1.9149 |
| 2014-01 | 5610 | 5510 | 100 | 0 | 1.7825 |
| 2014-02 | 4768 | 4684 | 84 | 0 | 1.7617 |
| 2014-03 | 5070 | 4963 | 107 | 0 | 2.1105 |
| 2014-04 | 5443 | 5343 | 100 | 0 | 1.8372 |
| 2014-05 | 5930 | 5826 | 104 | 0 | 1.7538 |
| 2014-06 | 5798 | 5692 | 106 | 0 | 1.8282 |
| 2014-07 | 6227 | 6128 | 99 | 0 | 1.5899 |
| 2014-08 | 5059 | 4963 | 96 | 0 | 1.8976 |
| 2014-09 | 5059 | 4963 | 96 | 0 | 1.8976 |
| 2014-10 | 5983 | 5867 | 116 | 0 | 1.9388 |
| 2014-11 | 5070 | 4978 | 92 | 0 | 1.8146 |
| 2014-12 | 5924 | 5809 | 115 | 0 | 1.9413 |
| 2015-01 | 5651 | 5550 | 101 | 0 | 1.7873 |
| 2015-02 | 5406 | 5314 | 92 | 0 | 1.7018 |
| 2015-03 | 5877 | 5762 | 115 | 0 | 1.9568 |
| 2015-04 | 6549 | 6428 | 121 | 0 | 1.8476 |
| 2015-05 | 6000 | 5897 | 103 | 0 | 1.7167 |
| 2015-06 | 6084 | 5969 | 115 | 0 | 1.8902 |
| 2015-07 | 6955 | 6824 | 131 | 0 | 1.8835 |
| 2015-08 | 5321 | 5225 | 96 | 0 | 1.8042 |
| 2015-09 | 6297 | 6179 | 118 | 0 | 1.8739 |
| 2015-10 | 6152 | 6040 | 112 | 0 | 1.8205 |
| 2015-11 | 5625 | 5538 | 87 | 0 | 1.5467 |
| 2015-12 | 5981 | 5870 | 111 | 0 | 1.8559 |
| 2016-01 | 5914 | 5703 | 211 | 0 | 3.5678 |
| 2016-02 | 5196 | 4993 | 203 | 0 | 3.9069 |
| 2016-03 | 6074 | 5887 | 187 | 0 | 3.0787 |
| 2016-04 | 5923 | 5709 | 214 | 0 | 3.6130 |
| 2016-05 | 6351 | 6145 | 206 | 0 | 3.2436 |

## Evidencia para viabilidad

Producto-mes: 9019 pares observados de 9307 en cuadrícula global. Los 288 ausentes preceden a las primeras ventas de ocho productos en enero de 2016; después de su primera observación no hay meses vacíos por producto. 219 productos aparecen desde enero de 2013 y ocho solo tienen cinco meses. No rellenar con cero antes de conocer la vigencia de catálogo.

Cliente-mes: 24029 pares observados; 1957 meses sin compra desde la primera observación hasta el final, sobre 25986 pares elegibles por ese criterio exploratorio. Esta tasa de inactividad no es aún la distribución de etiquetas futuras: depende de elegibilidad, ventana y censura.

Hay solo tres años completos y cinco meses de 2016. Se puede estudiar estacionalidad anual con cautela; observar ventas mensuales no valida su predictibilidad. Las ventas son demanda atendida, potencialmente censurada por quiebres de stock. No hay stock, entradas, salidas, plazos de reposición ni catálogo histórico suficiente para inventario predictivo real.

Ver viabilidad_predictiva.md para comparación A-D y pipeline documental; requisitos_confirmados.md distingue evidencia documental, decisiones del usuario, recomendaciones y pendientes.

## Integridad, reproducción y límites

SHA-256 antes/después iguales para 10 archivos protegidos (dos CSV y ocho referencias). No se modificaron. Versiones: `{'python': '3.12.3', 'pandas': '3.0.2', 'numpy': '2.4.4'}`.

Ejecutar `python src/auditar_datos.py` y `python -m unittest discover -s tests -p 'test_*.py' -v` desde la raíz. Los JSON, diccionario y este informe se regeneran. Los dos documentos de requisitos/viabilidad son análisis documental estático basado en estas evidencias, no una implementación.

Los tres PDF se extrajeron con pdftotext -layout de MiKTeX; texto de todas las páginas disponible. Se revisaron secciones, tablas y pies de figuras. No se renderizaron las capturas incrustadas: no se verificaron sus valores visuales ni fórmulas de Power BI no expuestas en texto. La equivalencia MongoDB es una afirmación documental, no auditada aquí. No hay SQL de extracción, DW conectado ni historial JSON de OpenRefine para verificar linaje y vigencia de dimensiones.
