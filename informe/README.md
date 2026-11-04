# Informe académico unificado de WWI

Entrada LaTeX: `main.tex`. PDF: `main.pdf`. Esta versión conserva Ventas e incorpora el experimento completo de frecuencia de cobros de Transacciones Financieras, seguido de una comparación académica y conclusiones integradas. Pedidos y Compras son investigaciones de viabilidad, no modelos implementados; Inventario/Movimientos sigue pendiente.

## Contenido

- Las secciones detalladas de Ventas, sus 21 figuras y 13 tablas permanecen intactas. Se actualizaron portada, resumen, alcance, aclaraciones generales de dimensión y conclusiones para incorporar la segunda aplicación.
- `secciones/11_transacciones.tex`: fuente, contrato, 16 características, preparación temporal, modelos/pesos, convergencia, PCA y evaluación principal/común.
- `secciones/12_comparacion_datamarts.tex`: comparación de problemas distintos, compromisos de error y alcance pendiente.
- `generado/transacciones/`: 13 tablas nuevas y macros con cifras estructuradas. `figuras/transacciones/`: 18 copias binarias de gráficos aprobados: ocho curvas de desarrollo, tres de PCA y siete de evaluación final.
- En total: 39 figuras y 26 tablas. El número final de páginas y los controles de compilación constan en `evidencias/transacciones/verificacion_informe.json`.
- `evidencias/transacciones/`: trazabilidad SHA-256, afirmaciones contrastadas, integridad experimental, compilación y revisión visual. `version_previa/` conserva el PDF y las fuentes principales de la versión anterior.

## Reproducción documental desde la raíz

```powershell
python informe/herramientas/generar_transacciones.py
python informe/herramientas/compilar_transacciones.py
python informe/herramientas/verificar_transacciones_informe.py
```

El compilador usa MiKTeX con `--disable-installer`, hasta tres pasadas para resolver índices y referencias, sin instalar paquetes. Necesita acceso a su configuración local de usuario. No hay consultas SQL, entrenamiento, transformación PCA ni recálculo de predicciones para producir el informe. El generador solo lee JSON/CSV aprobados y copia figuras.

El exportador/verificador inicial de Ventas se conserva como herramienta histórica de la versión anterior, con sus conteos originales. No sustituye al verificador actual del informe unificado. Las 313 pruebas experimentales previas se conservan como evidencia histórica; esta edición ejecuta controles documentales específicos y no presenta esa suite como reejecutada.

## Revisión del PDF

Renderizar el PDF final mediante Poppler y revisar páginas, tablas, ecuaciones, referencias y gráficos:

```powershell
pdftoppm -r 75 -png informe/main.pdf informe/revision/transacciones/pagina
pdftotext -layout informe/main.pdf informe/evidencias/transacciones/texto_pdf.txt
```

Los resultados de la inspección visual se registran después del último render, junto con el hash del PDF revisado. La verificación automática no sustituye esa inspección.

## Resultados y límites

Transacciones: 4.612/2.943/1.284 filas train/validación/test y 16 entradas. BA principal test: perceptrón 66,222%, logística 62,809%; la comparación con histórico de cuatro meses usa 1.276 filas y no reemplaza el soporte principal. PCA conserva 45,021% de varianza y no interviene en clasificadores. El perceptrón sigue actualizando y la logística agota 400 épocas sin alcanzar su tolerancia; no se declara convergencia completa.

Ventas conserva sus resultados originales: 20 entradas, mejor BA test logística 62,415%. Los porcentajes de ambos procesos no representan la misma etiqueta o población ni prueban superioridad universal. La cobertura es académica; cinco meses de test, observaciones dependientes y pocas entidades nuevas limitan la generalización. No se cierra el proyecto completo ni se recomienda automatización productiva.
