# Dataset financiero de desarrollo

Construcción completada exclusivamente para train y validación. Fuente: Sales.CustomerTransactions, CustomerID operacional, cobros tipo 3 con signo contable confirmado; correspondencia de IDs con Fact.Transaction. No se exportan importes, saldos, documentos, nombres ni atributos de pago. La última edición se guarda como control de versión y no certifica fecha de alta ni completitud absoluta.

| Partición | Observaciones | Positivos | Negativos | Forma X |
|---|---:|---:|---:|---|
| train | 4612 | 1798 | 2814 | [4612, 16] |
| val | 2943 | 1198 | 1745 | [2943, 16] |

El catálogo aprobado aporta 16 columnas de entrada; quedan 16 tras eliminar constantes solo con train. Descartadas: {}. No se fija una dimensión arbitraria ni se añade CustomerID, y, conteo objetivo, importe, estado o atributo dimensional a X. La mediana histórica R sí entra como cobros_mediana_3, porque solo depende del pasado. Se mantienen redundancias exactas autorizadas; no se seleccionan columnas por validación.

X usa float64, y int8 binario. La única característica que puede faltar es variacion_relativa_cobros cuando su denominador histórico es cero: se imputa con la mediana de train y conserva base_variacion_cero. Toda historia esencial debe existir; no se imputa para fabricar elegibilidad. Se ajustan media y desviación poblacional exclusivamente en train; la validación solo recibe transform. preprocesador.json guarda parámetros y orden.

Snapshot cobros_desarrollo.csv.gz: CSV UTF-8/LF, orden de event_id, compresión gzip nivel 9 con mtime=0 y nombre vacío. Sus bytes son reproducibles; procedencia.json separa el instante real de extracción de los artefactos deterministas. No se guarda una copia de todas las transacciones. Las consultas y sus parámetros/huellas se conservan en consultas_sql.json y procedencia.json. El snapshot se contrasta con la proyección de cobros de la auditoría aprobada antes de construir matrices.

Población: meses desde primera observación de cobro, sin usar última aparición o actividad futura. Se exigen tres meses consecutivos previos y control de disponibilidad antes de inicio(t); etiquetas conocidas antes de 2015-01-01 para train y 2016-01-01 para val. El modo business_ledger_academic está activado explícitamente. Solo permite ceros del libro cubierto después de primera observación; no afirma cero ventas, impago o conciliación bancaria. Sin ese modo se bloquea el constructor. Los supuestos y limitaciones están en cobertura_aplicada.json y se conservan los originales sin cambios.

Orden de todas las observaciones: mes, CustomerID. metadatos_train/val.csv contienen identidad, fechas y cortes. etiquetas_train/val.csv contienen por separado y, conteo objetivo y R. caracteristicas_crudas_train/val.csv permiten auditar X antes de transformar, con una columna de ID que se excluye mediante catálogo cerrado. elegibilidad.csv.gz explica también los meses sin historia; cuentas_meses.csv.gz separa actividad observada de ceros por cobertura académica.

El código matemático y los artefactos de Ventas permanecen intactos. El adaptador financiero conserva las reglas numéricas del preprocesador existente, con catálogo propio y sin parchear sus variables globales. No hay entrenamiento ni PCA nuevo. Las pruebas con datos sintéticos son solo fixtures de verificación, no sustituyen eventos empresariales.

## Reproducción

Desde la raíz, primera extracción focalizada local:

```powershell
python -m src.datamarts.transacciones.construir --origen sql --cobertura business_ledger_academic
```

Reproducción sin SQL usando el snapshot y la procedencia ya verificados:

```powershell
python -m src.datamarts.transacciones.construir --origen snapshot --cobertura business_ledger_academic
python -m unittest discover -s tests -p test_transacciones_dataset.py -v
python -m unittest discover -s tests -v
```

No modificar la auditoría para ajustar discrepancias. El pipeline aborta si cambia la proyección aprobada de cobros, el contrato, el catálogo o la reconciliación esperada. Las huellas COUNT/CHECKSUM antes/después son controles acotados no criptográficos, sin aislamiento snapshot ni prueba de inexistencia de escrituras externas; todas las consultas propias son SELECT y no se modifica SQL Server.

## Test reservado

No hay X_test ni y_test, ni consultas a eventos de 2016. especificacion_test.json define su construcción posterior, con preprocesador congelado, historial secuencial y verificación de cierre. El CLI actual solo admite desarrollo y no expone fechas de test. No se avanza automáticamente al entrenamiento o a otro Data Mart.
