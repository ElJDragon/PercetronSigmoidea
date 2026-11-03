# Requisitos y arquitectura — Fase 1

## 1. Evidencia documental y alcance de lectura

Contexto: Universidad Técnica de Ambato, FISEI, carrera de Software, Inteligencia de Negocios; docente Ing. Rubén Nogales Portero, Ph.D. Se leyeron las tres referencias PDF mediante extracción completa de texto con `pdftotext -layout` (MiKTeX). La ejecución requirió acceso a la configuración de usuario de MiKTeX fuera del aislamiento. No se instalaron dependencias. No se inspeccionaron visualmente las capturas incrustadas; los números mostrados solo en imágenes no se atribuyen como verificados. Se revisó también la nota Markdown completa.

### Carta de Diseño (`referencias/Carta_de_Diseño (2).pdf`)

- Secciones 2–3: evolución de ventas; productos de mayor ingreso/volumen; clientes y territorios de mayor aporte; disponibilidad de stock y relación ventas-inventario para abastecimiento.
- Secciones 4–5: ventas, clientes e inventario de prioridad alta; compras y cobros de prioridad media. Requiere Quantity/UnitPrice, CustomerID/CustomerCategoryID, QuantityOnHand, TransactionAmount, InvoiceDate/DeliveryCityID, StockItemID/SupplierID. Competencia queda fuera del alcance inicial.
- Secciones 8–9: `FactVentas`, grano línea de factura por producto, cliente, fecha y ubicación. DimTiempo, DimProducto, DimCliente, DimUbicacion y DimVendedor. DimUbicacion sustituye sucursal porque el caso no modela tiendas físicas. Claves propuestas: VentaKey y claves dimensionales; claves operativas StockItemID, CustomerID, CityID, PersonID. Actualización diaria propuesta; horizonte sujeto a copia disponible.
- Sección 7: SUM(Quantity × UnitPrice), SUM(Quantity), ventas por cliente, SUM(QuantityOnHand), relación unidades/stock, número de facturas, SUM(TransactionAmount), ticket promedio y COUNT DISTINCT(StockItemID). La Carta escribe COUNT(InvoiceID): al grano línea, deberá ser COUNT DISTINCT(InvoiceID) para contar facturas sin multiplicarlas. Esta corrección es recomendación técnica, no una transcripción del documento.
- Secciones 10–11: validar nulos, uniones 1:N, rangos, códigos, fechas e histórico; homologar productos, unificar fechas, segmentar clientes, calcular margen cuando exista costo, conservar registros con nulos y calcular importe consistente.
- Secciones 12–14: SQL Server OLTP/DW, SSIS, modelo estrella, SSAS y Power BI propuestos. Validación docente e indicadores aún pendientes en la Carta. No impone una implementación de Machine Learning ni demuestra que se hayan implementado todos los procesos.

Faltan en los CSV stock histórico, compras, cobros, costos, proveedores e IDs de entidades. Las cinco dimensiones tienen atributos descriptivos, pero no sus claves operativas. El alcance deseado del DW no equivale a disponibilidad en esta extracción.

### Dashboard (`referencias/Dashboard (1).pdf`)

Secciones 2.1–2.6 y tabla 1: resumen, evolución temporal, productos, clientes, geografía y vendedores. Indicadores descritos: ingresos sin impuesto, unidades, beneficio, margen porcentual, facturas, clientes, productos y ticket promedio. Productos comparados por volumen, ingresos, beneficio y margen; clientes por categoría/grupo; ciudades y territorios por ingresos/unidades; vendedores por facturas, unidades, ingresos y beneficio.

Se advierte expresamente que 2016 es parcial. GrupoCompra vacío puede representar cliente independiente. Se describen filtros de año/trimestre y navegación. La sección 4.1 declara equivalencia con MongoDB, pero no se aportan aquí sus bases o validaciones ejecutables. No se ha comprobado esa equivalencia ni las fórmulas DAX o los valores de las capturas.

Preguntas predictivas derivadas por análisis técnico: demanda del siguiente mes (evolución/productos), actividad futura de clientes, riesgo de pérdida por operación y, solo con datos adicionales, inventario. No son exigencias predictivas expresas del dashboard.

### OpenRefine (`referencias/Openrefine_Cluster (2).pdf`)

Secciones 1–2: extracción desnormalizada de WWI_Kimball con FactVentas y cinco dimensiones; se declaran 228.265 líneas. Secciones 3–4: GREL `if(value == "NULL", null, value)` en Marca (209.328), Tamanio (67.315), GrupoCompra (84.297). Estas cantidades se corroboraron programáticamente, sin copiarlas como resultado.

Secciones 5–7: unicidad de InvoiceLineID, conversiones numéricas de las seis medidas y fechas Fecha/FechaApertura. Se preservan atípicos y beneficios negativos. Las explicaciones de liquidaciones o ventas por debajo del costo son posibilidades del documento, no causas demostradas por estos CSV.

Sección 8: Fingerprint en CategoriaCliente sin fusiones; Levenshtein radio 1 en Ciudad detecta Premont/Tremont y Marcella/Marcell, cuya fusión se rechaza por representar entidades distintas. Este clustering es limpieza textual, no un clasificador predictivo.

Sección 9: salida declarada `dataset_ventas_wwi_limpio.csv`; el archivo real es `datos_limpios.csv`. Se comprobaron idénticas claves y medidas, pero también 268 cambios de `David Novacek ` a `David Novacek`, no descritos explícitamente. Pedir historial de operaciones antes de atribuir esa transformación a una acción concreta.

Consecuencias para ML: preservar ausencia semántica; no fusionar entidades por similitud; tipificar fechas/números en cada lectura porque CSV no conserva tipos; no eliminar pérdidas ni extremos; no tratar métricas posteriores como información previa. Marca solo tiene un valor no nulo, pero su ausencia puede tener información; DescuentoEstandar es constante cero.

### Nota del docente (`referencias/GRaficos con MNIST 4077751154f78080828ff5640e835bc1.md`)

El texto original menciona MNIST, reducción a dos características para dispersión, dos clases, prohibición de la clase LogisticRegression, entropía cruzada y biblioteca para gradientes. Pide pesos finales, exactitud de entrenamiento/prueba y gráficos de entrenamiento para fases futuras. La nota no precisa biblioteca autorizada, método de reducción, umbral de demanda ni partición temporal. No establece por sí sola hiperparámetros ni autoriza autodiferenciación específica.

## 2. Decisiones arquitectónicas vinculantes del usuario

1. Solo WWI; MNIST es ejemplo conceptual. No descargarlo ni experimentarlo.
2. Perceptrón simple desde cero y regresión logística binaria matemática (sigmoidea, entropía cruzada, gradientes) en otra fase. Prohibida `sklearn.linear_model.LogisticRegression`; no asumir autorización de PyTorch, TensorFlow o JAX.
3. Entrenamiento con todas las características válidas, dimensión dinámica. La selección depende de disponibilidad temporal, nulos, codificación, escalado y prevención de fuga.
4. Perceptrón y logística reciben exactamente las mismas matrices y particiones. Una codificación de etiquetas distinta, si es necesaria, deberá conservar correspondencia explícita entre clases.
5. Proceso A: entrenamiento/evaluación multidimensional. Proceso B independiente: reducción a dos componentes para dispersión. La proyección no sustituye X de los clasificadores.
6. PCA mediante NumPy es candidato, sujeto a restricciones y justificación matemática. Se ajustará solo en entrenamiento y se reutilizará en validación/prueba. No interpretar una frontera 2D como frontera exacta del modelo multidimensional; cualquier corte o aproximación deberá identificarse.
7. Originales y referencias en solo lectura; código nuevo en src, pruebas en tests, auditoría en resultados/auditoria. Conservar estructura y capitalización. informe reservado para entregables posteriores.
8. En fase 1 no implementar modelos, PCA ni pipeline predictivo; no entrenar, producir gráficos de entrenamiento o matrices definitivas. No avanzar sin autorización.

Estas decisiones resuelven la ambigüedad de la nota del docente respecto a MNIST y a usar solo dos variables. Tienen prioridad según la solicitud; no se presentan como nuevas frases de los PDF.

## 3. Recomendaciones técnicas, todavía no requisitos docentes

Usar datos_limpios.csv con aceptación documentada del trim observado; pedir claves operativas e historia dimensional. Priorizar definición de demanda alta del próximo mes por producto; usar evaluación temporal común y baselines. Evaluar precision/recall/F1 por clase y balanced accuracy además de exactitud. Mantener indicadores de faltantes y categorías desconocidas. No reutilizar partición aleatoria, tres entradas fijas, etiquetas ni 200 épocas del ejercicio anterior.

Pipeline futuro: definir población y grano, ventanas, etiqueta, cortes cronológicos, disponibilidad de cada atributo, transformaciones ajustadas con entrenamiento y contrato de nombres de características. Detalle en viabilidad_predictiva.md. Nada de este pipeline se implementó en esta fase.

## 4. Pendientes de confirmación

- Objetivo y utilidad empresarial; definición de alta/baja demanda, horizonte y momento de corte. Ningún objetivo queda seleccionado definitivamente.
- Qué biblioteca de gradientes autoriza el docente y si acepta cálculo manual con NumPy; autorización y justificación de PCA NumPy.
- StockItemID, CustomerID, CityID y demás claves; catálogo/vigencia de producto, historial de cambios dimensionales y corte completo de extracción a mayo de 2016.
- Historial de operaciones OpenRefine y SQL de extracción; interpretación comercial de ausencias y de FechaApertura.
- Costos y condiciones comerciales históricas para pérdida; stock/entradas/salidas/plazos para inventario. Confirmar evaluación y formato académico posterior.

No son bloqueos para terminar la auditoría de fase 1; sí condicionan formulación e implementación posteriores.
