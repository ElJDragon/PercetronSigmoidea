# Checkpoints y convergencia, sin nuevas épocas

Perceptrón: checkpoint 30, 100 épocas originales, 6983 actualizaciones, 68 errores online en la última época, cero épocas sin errores. BA de validación seleccionada 0.987466255 frente a última 0.945815658. Continúa actualizándose: no hay convergencia a cero demostrada.

Logística estándar: checkpoint 156 de 400 épocas. Pérdida inicial/final train 0.693147181/0.077311004; gradiente final 0.051118254, superior a 1e-6. Terminación max_epochs, no tolerancia; no convergencia completa. BA val del checkpoint 0.986502121, última 0.652707840.

Se cotejaron historiales JSON y trazas de parámetros originales con el diagnóstico guardado; no se alteraron curvas, pesos, sesgos, checkpoints ni hiperparámetros. No se llamó a entrenamiento.
