# Presupuesto fijado antes de entrenamiento

Cuatro ejecuciones: perceptrón eta=0,01, 100 épocas, semillas 42 y 7, permutación por época; logística eta=0,05 y 0,10, 400 épocas, seed=42, pesos cero y lote completo. Todas parten de pesos/sesgo cero; ninguna reutiliza pesos de Ventas. En logística la semilla no altera la inicialización cero. Una única eta del perceptrón evita una búsqueda redundante de escalas con inicialización cero.

Selección por balanced accuracy de validación completa, después F1, época más temprana y orden de experimento. Umbral logístico fijo 0,5; no se realiza búsqueda posterior. Perceptrón conserva z>=0. No se amplía el presupuesto por resultados. Parada logística por norma de gradiente <=1e-6 o 400 épocas; perceptrón ejecuta 100 según su clase actual. AP logística descriptiva, no selección.

El baseline aprobado requiere cuatro meses: 1[N(t-1)>mediana(N(t-2),N(t-3),N(t-4))]. No se sustituye por la fórmula candidata de tres meses del nuevo texto porque no coincide con el diseño aprobado. Se comparan los cuatro métodos en una misma intersección con cuarto lag disponible; se conservan además métricas de modelos y mayoría en todo el soporte. No se imputa historia para el baseline.

No hay acceso SQL, test 2016 o PCA. La configuración JSON es la fuente estructurada del presupuesto y se escribe antes de obtener resultados nuevos.
