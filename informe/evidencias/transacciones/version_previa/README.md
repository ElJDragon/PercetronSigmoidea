# Informe académico incremental de WWI

Entrada LaTeX: `main.tex`. Entregable compilado: `main.pdf`.

Se documenta exclusivamente la aplicación realizada en el Data Mart de Ventas. La ampliación a los demás Data Marts es una obligación académica pendiente; esta entrega no implementa esas aplicaciones.

## Contenido y organización

- `secciones/`: narrativa académica, fundamentos, metodología, implementaciones, convergencia, PCA, resultados, discusión y anexos.
- `generado/`: 13 tablas extraídas de artefactos aprobados, incluidas las dos tablas de 20 pesos cada una.
- `figuras/`: 21 imágenes insertadas. Se copiaron 16 figuras previas con identidad binaria y se generaron cinco vistas documentales nuevas. La comparación global incluye las cinco métricas exigidas, agregando BA a la vista previa de cuatro métricas. La distribución de clases y el calendario comparten una figura; las actualizaciones del perceptrón coinciden exactamente con los errores online y no requieren otra curva idéntica.
- `referencias.tex`: siete referencias verificables, incluidas las tres referencias académicas locales. Se usa `thebibliography` para compilar sin BibTeX ni Biber.
- `herramientas/generar_informe.py`: exportación de tablas y figuras desde archivos congelados, sin entrenamiento, transformaciones nuevas o predicciones.
- `herramientas/verificar_informe.py`: comprobaciones de integridad, cifras, pesos, curvas, referencias, inclusión de figuras/tablas y reproducción con operaciones de modelos bloqueadas.
- `evidencias/`: trazabilidad con SHA-256, afirmaciones verificadas, dos CSV por época, análisis de convergencia, registro de compilación y verificación final.
- `revision/`: páginas renderizadas para control visual; son material auxiliar.

## Reproducir desde la raíz del proyecto

```powershell
python informe/herramientas/generar_informe.py
Set-Location informe
pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex
Set-Location ..
python informe/herramientas/verificar_informe.py
```

Las tres pasadas resuelven el índice, las listas y las referencias cruzadas. No ejecutar entrenadores, preparación del dataset, PCA ni evaluación final para producir el informe. No se requieren paquetes Python adicionales a los ya usados en el proyecto. Se compiló con pdfLaTeX de MiKTeX, sin instalar automáticamente paquetes LaTeX. En este entorno, los ejecutables MiKTeX requieren acceso a su configuración de usuario fuera del aislamiento del workspace.

Para inspección visual se utilizó `pdftoppm`, incluido en la instalación local:

```powershell
Set-Location informe
pdftoppm -r 75 -png main.pdf revision/final
pdftotext -layout main.pdf evidencias/texto_pdf.txt
```

El verificador reproduce 38 artefactos documentales byte a byte y comprueba que 169 archivos previos permanezcan intactos. La evidencia de la suite anterior de 104 pruebas se conserva en `resultados/fase_08/validaciones.json`; no se presenta como una suite reejecutada en esta tarea documental. La inspección visual humana del agente se registra después de ejecutar el verificador; el script no puede realizarla por sí solo.

## Resultado de convergencia

Perceptrón: recorrido barajado de 100 épocas, checkpoint 70. Errores online entre 1863 y 2016; 49 transiciones crecientes; ninguna época sin actualizaciones; 1996 actualizaciones en la última. El checkpoint y el último estado tienen métricas diferentes. No se demuestra convergencia a cero errores ni se deduce no separabilidad de un presupuesto finito.

Logística: pérdida inicial train 0.6931471805599452 y final 0.6269539858615067; caída absoluta 0.06619319469843843 (9.549659 % de la inicial). No hay subidas estrictas de pérdida train. El último descenso y cambio relativo son 1.2399940257257214e-05; norma final del gradiente 0.01109745983712376. Se detuvo por máximo de 200 épocas, con tolerancia desactivada. Conclusión: descenso estable sin convergencia completa demostrada.

Los cálculos por época y las normas de parámetros disponibles están en los CSV de `evidencias/`; no fueron criterios retrospectivos de parada ni justificaron ampliar épocas o cambiar hiperparámetros.

## Correcciones y limitaciones documentales

- El cotejo estructurado detectó que los 268 recortes de espacios afectan a `NombreCliente`, no a vendedor como indica un resumen anterior. El informe usa `comparacion_csv.json` y documenta la discrepancia; no altera los entregables históricos.
- Se distinguieron las 9019 filas del calendario `producto_mes.csv` de las 8792 observaciones conocidas antes del corte en `observaciones.csv`.
- La compilación inicial encontró ausente `caption.sty`. Se sustituyó por formato nativo de LaTeX, sin instalar paquetes. Se corrigieron desbordamientos de rutas, el destino duplicado de portada, los pies de figura y la extensión del índice de figuras.
- El gráfico de acuerdos muestra agrupaciones mensuales; los detalles por cohorte están en el JSON y la tabla de cohortes, no se atribuyen al gráfico.
- El informe no afirma convergencia completa, significación estadística universal ni causa demostrada de deriva conceptual. No inventa costos empresariales ni resultados de otros Data Marts.

## Información pendiente y ampliación

Los autores Josue Guevara y Jonathan López, nivel Sexto «A» y unidad Profesional constan en las portadas previas. Se conserva esa forma de los nombres. El docente sigue la identificación de la solicitud actual: Ing. Rubén Nogales Portero, Ph.D. Las referencias previas indican ciclo enero–julio de 2026, pero la carta de diseño se fecha el 18/08/2026; no se inventa el ciclo o fecha de entrega de este informe.

Continúan pendientes las claves operativas y catálogos históricos, unidades comerciales, cierres y disponibilidad al corte certificados, contexto comercial y costos FP/FN. La cobertura académica y cinco meses de test no bastan para recomendar uso productivo automático.

Para extender el informe, incorporar nuevos módulos académicos solo cuando cada Data Mart tenga datos, definición de respuesta y evaluación efectivamente realizados. Mantener el protocolo temporal y la trazabilidad propios de cada aplicación, sin identificar dimensiones como Data Marts por su nombre. Esta entrega se detiene con Ventas documentado.
