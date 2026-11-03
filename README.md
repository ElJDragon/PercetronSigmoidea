# PerceptronReal: clasificación temporal de ventas WWI

Proyecto académico de Inteligencia de Negocios de la Universidad Técnica de Ambato. Primera aplicación desarrollada y evaluada: **Data Mart de Ventas de Wide World Importers**. Implementa desde cero con NumPy un perceptrón simple y una regresión logística binaria; PCA por SVD se utiliza exclusivamente para visualización.

La respuesta indica si las unidades facturadas observadas de un producto en el mes siguiente superarán la mediana de sus tres meses previos. No representa rentabilidad, inventario ni demanda insatisfecha. Ambos clasificadores utilizan las mismas 20 características y particiones temporales.

**La ampliación a los demás Data Marts es una obligación académica pendiente.** No se presentan como implementados ni se confunden dimensiones Cliente o Producto con Data Marts independientes.

## Clonar, instalar y verificar

```powershell
git clone https://github.com/ElJDragon/PercetronSigmoidea.git
Set-Location PercetronSigmoidea
git switch develop
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Entorno verificado: **Windows/PowerShell**, Python **3.12.3**, NumPy **2.4.4**, pandas **3.0.2** y Matplotlib **3.10.8**. Las pruebas usan `unittest` de la biblioteca estándar. No se requieren scikit-learn, autodiferenciación, MNIST ni acceso a bases de datos externas. Los manifiestos históricos usan separadores de ruta de Windows; no se certifica ejecución directa en Linux/macOS. Una adaptación debe hacerse en una versión posterior sin reescribir la evidencia congelada.

El nombre remoto `PercetronSigmoidea` y el del proyecto `PerceptronReal` difieren; no es necesario renombrar la carpeta. Los scripts localizan la raíz por su ubicación. Algunas evidencias históricas conservan rutas absolutas del equipo de origen; no son requisitos de instalación.

## Estado actual

| Método | Balanced accuracy de test |
|---|---:|
| Regresión logística | 62,42 % |
| Perceptrón | 54,10 % |
| Histórico | 51,31 % |
| Mayoritario | 50,00 % |

Train: 4599 observaciones, abril de 2013–diciembre de 2014. Validación: 2628, enero–diciembre de 2015. Test: 1111, enero–mayo de 2016. Se seleccionó únicamente con validación; test ya se consumió y no debe usarse para rediseñar esta versión.

Hay **104 pruebas automatizadas**. El [informe académico PDF](informe/main.pdf) tiene **37 páginas, 21 figuras y 13 tablas**; incluye fundamentos, los 40 pesos y sesgos, convergencia, PCA, resultados y conclusiones parciales de Ventas.

Perceptrón: checkpoint 70 de 100 épocas, tasa 0,01, semilla 42; ninguna época sin actualizaciones. Logística: checkpoint 200, tasa 0,1, umbral 0,5; pérdida train de 0,693147 a 0,626954, descenso estable sin convergencia completa demostrada.

## Arquitectura y artefactos

| Ruta | Contenido |
|---|---|
| `datos/originales/` | Los dos CSV WWI, completos y sin cambios |
| `datos/Procesados/fase_03/` | Matrices X/y, metadatos, agregados, características y preprocesador |
| `referencias/` | Documentos académicos y código anterior; solo lectura |
| `src/pipeline/` | Agregación producto-mes, entradas causales y particiones |
| `src/models/` | Implementaciones propias de perceptrón y logística |
| `src/visualization/` | PCA y dispersión, separados del entrenamiento |
| `src/experiments/` | Experimentos, congelación, evaluación y análisis |
| `src/evaluation/` | Métricas y gráficos |
| `tests/` | Matemática, temporalidad, integridad y reproducibilidad |
| `resultados/` | Auditoría, modelos, historiales, predicciones y manifiestos |
| `informe/` | LaTeX, figuras, PDF y evidencia documental |

Se conserva la capitalización de `datos/Procesados/` y el nombre histórico `referencias/Codigo_anteiror/`; no ejecutar ese código antiguo como experimento WWI.

Modelos y resultados principales:

- `resultados/fase_04/modelo_seleccionado.json`: perceptrón; pesos en `pesos.csv`.
- `resultados/fase_05/modelo_seleccionado.json`: logística; pesos y verificación de gradientes.
- `resultados/fase_06/congelacion_experimental.json`: estados completos, hashes y protocolo.
- `resultados/fase_07/`: evaluación final, baselines, predicciones alineadas y métricas globales/mensuales/cohortes.
- `resultados/fase_08/`: análisis estadístico y empresarial, acuerdos, distribuciones y sensibilidad temporal.

Los gráficos se encuentran en `resultados/fase_04/graficos/` hasta `resultados/fase_08/graficos/`; las imágenes insertadas en el documento están también en `informe/figuras/`. Todos los manifiestos y validaciones necesarios se incluyen.

## Datasets grandes incluidos

No se requiere descarga adicional ni Git LFS. Se incluyen mediante Git normal:

| Archivo | Bytes | MiB aproximados |
|---|---:|---:|
| `datos/originales/dataset_ventas_wwi.csv` | 59150151 | 56,41 |
| `datos/originales/datos_limpios.csv` | 59899758 | 57,12 |

Son extracciones del caso demostrativo WWI. Ambos superan el umbral de advertencia de 50 MiB, pero están por debajo del límite de 100 MiB por archivo de GitHub ([documentación oficial](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)). Se preservan datos originales, matrices y modelos completos. `.gitattributes` impide convertir finales de línea porque los hashes comparan bytes exactos.

## Verificar sin reentrenar

Con el entorno activado, ejecutar desde la raíz:

```powershell
python -m unittest discover -s tests -v
python -c "from src.experiments.analisis_estadistico import verificar_entradas; verificar_entradas(); print('Integridad y metricas verificadas')"
```

La suite utiliza controles sintéticos y preparación en destinos temporales; no sustituye los modelos WWI aprobados. `verificar_entradas()` lee las predicciones guardadas y contrasta hashes y métricas sin entrenamiento o inferencia nueva.

## Reproducción completa

**Usar una copia de trabajo separada destinada a reproducción**, conservando el checkout publicado como referencia. Estos comandos escriben salidas; no son una simple comprobación de instalación:

```powershell
python -m src.auditar_datos
python -m src.pipeline.construir_dataset --cobertura academica
python -m src.experiments.entrenar_perceptron
python -m src.experiments.entrenar_logistica
python -m src.experiments.visualizar_pca
python -m src.experiments.evaluacion_final
python -m src.experiments.analisis_estadistico
```

Leer primero los contratos de `resultados/fase_02/` y los reportes técnicos. El modo académico asume completitud; cobertura desconocida bloquea las matrices. Repetir la evaluación final añade un registro persistente y puede cambiar sus hashes: para verificar esta entrega, utilizar la lectura anterior. No ajustar modelos a partir del test consumido. Diferencias de LAPACK/entorno pueden alterar SVD o figuras; no reemplazar silenciosamente artefactos congelados.

## Compilar el informe

El PDF ya está incluido. Para compilar fuentes con una distribución LaTeX y los paquetes declarados en `main.tex`:

```powershell
Set-Location informe
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

No usa BibTeX/Biber. Consultar [informe/README.md](informe/README.md) para exportación documental y revisión. El verificador documental completo requiere auxiliares de compilación: compilar primero. Se excluyen cachés, logs, páginas de revisión y auxiliares, conservando el PDF, las fuentes, las figuras y la evidencia final.

## Continuación y limitaciones

Leer [CONTINUIDAD_PROYECTO.md](CONTINUIDAD_PROYECTO.md). Se mantienen `main` y `develop`, con trabajo nuevo desde `develop` en ramas `feature/...`. La configuración local de Gitflow no se transfiere al clonar.

La identidad por nombre, unidades comerciales y cierres requieren confirmación operacional. Cinco meses de test y 16 filas de ocho productos nuevos limitan la generalización; las observaciones son dependientes. Hay elevada tasa de falsos positivos. PCA conserva solo 49,62 % de varianza y no es un clasificador. No se atribuyen causalidad, superioridad universal ni costos económicos no observados. La extensión a los demás Data Marts sigue siendo obligatoria.
