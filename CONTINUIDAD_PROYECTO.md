# Transferencia y continuidad académica

Ventas está auditado, preparado, entrenado, evaluado y documentado. `informe/main.pdf` reúne la primera aplicación efectivamente realizada. La extensión a los demás Data Marts es **obligatoria y pendiente**. Antes de ampliarla, verificar procesos, hechos, extracciones y disponibilidad real: una dimensión no es automáticamente otro Data Mart.

## Evidencia que debe conservarse

1. Originales y referencias, incluidos los documentos históricos. El informe aclara que los 268 recortes de espacios afectan a `NombreCliente`, aunque un resumen anterior los atribuye a vendedor.
2. Contrato causal: producto-mes, predicción al cierre anterior, tres meses de historia, etiqueta estricta `Q > mediana`, empate 0, elegibilidad y cobertura explícitas.
3. Las 20 columnas ordenadas, parámetros del preprocesador y mismos arrays para comparar clasificadores. Imputación, escalado y PCA ajustados solo con train.
4. Modelos, umbrales, checkpoints, predicciones, manifiestos y registros aprobados. No reemplazar el checkpoint por la última época.
5. Test ya consumido. Cambiar para mejorar sus métricas no produce una evaluación independiente de la misma versión.

El código permite dimensión dinámica: las futuras aplicaciones no tienen por qué usar 20 entradas ni la misma etiqueta. Cada Data Mart requiere problema, contrato temporal y evaluación propios. No mezclar sus datos con el test congelado de Ventas.

## Verificación inicial

Clonar, cambiar a `develop`, instalar dependencias en Python 3.12 y ejecutar las 104 pruebas según el README. La plataforma verificada es Windows/PowerShell; los manifiestos guardan separadores Windows y la portabilidad a otros sistemas requiere una adaptación posterior documentada. Abrir el PDF y ejecutar `verificar_entradas()` para contrastar las predicciones existentes sin reentrenamiento. Los datos están incluidos, sin LFS ni descargas adicionales. Mantener `.gitattributes`: convertir LF/CRLF rompería hashes aunque los CSV parecieran equivalentes.

No ejecutar formateadores masivos sobre fuentes o resultados congelados. Los manifiestos documentales guardan también hashes del código: si se autoriza una versión nueva, debe registrarse como tal, no reemplazarse silenciosamente la evidencia anterior.

## Gitflow

El repositorio ya tenía `main`, `develop` y Gitflow configurados. La publicación inicial conserva el commit raíz y publica ambas ramas con el progreso verificado. No se fuerza ni reemplaza historial remoto.

Un clon transfiere ramas, no la configuración local `gitflow.*`. Si está instalado git-flow, ejecutar `git flow init`, seleccionando `main`, `develop` y los prefijos `feature/`, `bugfix/`, `release/`, `hotfix/`, `support/`. También puede trabajarse con Git estándar:

```powershell
git switch develop
git switch -c feature/nombre-del-trabajo
```

Usar ramas independientes y revisión antes de incorporar trabajo nuevo. Conservar esta versión y crear nuevas rutas de datos/experimentos con sus propios manifiestos. No se necesita renombrar la carpeta clonada `PercetronSigmoidea`: los scripts deducen su raíz.

## Pendientes

- Confirmar ciclo y fecha de entrega actuales. Autores, nivel, asignatura y docente ya están documentados según la evidencia disponible y la solicitud.
- Obtener claves operativas y catálogos con vigencia; confirmar unidades y empaques.
- Certificar cierres, completitud y disponibilidad al corte. La cobertura actual es un supuesto académico.
- Validar relevancia comercial de superar la mediana y costos FP/FN; ventas observadas no equivalen a demanda total.
- Completar obligatoriamente los demás Data Marts con formulación y evaluación efectivamente realizadas.

El perceptrón continúa actualizándose al terminar; logística muestra descenso estable sin convergencia completa demostrada. Cinco meses y observaciones dependientes no permiten superioridad universal. Los cambios de población no prueban por sí solos deriva de concepto. Estas limitaciones deben acompañar cualquier presentación posterior.
