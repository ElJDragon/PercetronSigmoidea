# Interpretación empresarial — evidencia de fase 8

## Pregunta que realmente se respondió

Al cierre del mes anterior, clasificar si las unidades facturadas observadas de un producto superarán la mediana de sus tres meses previos. No se predijo rentabilidad, inventario, demanda no atendida, número exacto de unidades futuras ni impacto económico de una acción comercial.

## Respuestas sustentadas para el informe académico

1. **Rendimiento global observado:** logística obtuvo BA 62.415 %, perceptrón 54.104 %, mayoritario 50 % e histórico 51.310 %. La logística encabeza esta comparación descriptiva sobre la misma reserva; no demuestra superioridad universal o significación estadística.
2. **Equilibrio entre clases:** entre los clasificadores, logística conserva sensibilidad 91.139 % y especificidad 33.692 %, frente a 96.022 % y 12.186 % del perceptrón. Ambos alertan en exceso; el histórico reconoce más negativos pero pierde muchos positivos. La BA, matrices y soportes aportan más que accuracy/recall aislados.
3. **Diferencia frente a baselines:** logística supera BA mayoritaria en 12.415 pp e histórica en 11.106 pp; perceptrón las supera en 4.104 y 2.794 pp. Son mejoras observadas, no ahorro o retorno de inversión demostrado.
4. **Errores que limitan utilidad:** perceptrón tiene 490 FP/22 FN y logística 370 FP/49 FN. FP puede motivar una revisión comercial innecesaria y FN omitir una posible subida; no se inventan costos ni se afirma que FP represente sobrestock real. El mayoritario alerta siempre y logra recall 100 % sin reconocer ningún negativo: alta sensibilidad no basta.
5. **Generalización temporal:** BA test−validación del perceptrón es -7.528 pp y de logística +1.216 pp. La BA global logística 62.415 % difiere de su media mensual 58.406 % por ponderaciones de clases. Cinco meses no acreditan estabilidad futura; cambios de X observados no prueban concept drift.
6. **Uso productivo inmediato:** la evidencia no basta para recomendar decisiones automáticas de abastecimiento o comerciales. Hay falsa alarma elevada, solo cinco meses de test y 16 filas nuevas, probabilidades sin calibración validada, cobertura asumida e identidad por nombre. El umbral de alta relativa puede separar diferencias de una unidad sin relevancia económica aprobada. Esto no invalida el ejercicio académico; limita las afirmaciones de utilidad operacional.
7. **Datos adicionales para interpretar:** StockItemID/CustomerID y catálogos con vigencia/renombres; unidad y empaque consistentes; registros de cierre/ingestión que prueben disponibilidad al corte; promociones y precios planificados; stockout, inventario y demanda no atendida; decisiones/acciones registradas y costos de FP/FN validados por el negocio. Es información faltante, no valores deducibles del CSV.

## Hechos frente a hipótesis

Comprobado: diferencias de métricas, distribuciones descriptivas de X, frecuencias de predicción, errores y desacuerdos de los mismos casos. Hipótesis no demostradas: cambio de concepto, causa comercial de los errores, que la antigüedad observada explique por sí sola el exceso de alertas, o que problemas de cobertura/identidad sean la causa del rendimiento. No se confunde una limitación lineal posible con un fallo de implementación demostrado.

No se propone reentrenar, ajustar umbrales o combinar clasificadores sobre este test. Cualquier investigación posterior requiere otra versión y una evaluación independiente. Los resultados finales y todos los artefactos congelados se conservan como evidencia de esta versión. No se avanza automáticamente a fase 9.
