# Convergencia observada


## perceptron: p_seed7

Checkpoint: época 3; última época: 100; terminación: max_epochs.

Errores online inicial/final: 1856/1810; mínimo/máximo: 1768/1917; actualizaciones totales: 183820. Aumentos entre épocas: 52; épocas sin errores: []. Errores train al final de la última época: 1722. Actualizar parámetros al final impide declarar convergencia perfecta. Los errores online y los errores al cierre de época son medidas distintas.

BA train/val seleccionada: 0.663005/0.667827. BA train/val de última época: 0.595835/0.590994. Norma final de pesos: 0.168538179. Se conservan por separado ambas épocas; no se presenta la última como la elegida.


## logistic: l_eta010

Checkpoint: época 399; última época: 400; terminación: max_epochs.

Pérdida train inicial/final: 0.693147181/0.616972884; val inicial/final: 0.693147181/0.631334062. Incrementos de pérdida train (>1e-12): 0; último cambio: -2.64131331e-05; gradiente final: 0.0162425381; tolerancia alcanzada: False. Pérdidas del checkpoint train/val: 0.616999297/0.631359239. Descenso de pérdida no demuestra convergencia completa si no se alcanza el criterio. El gradiente final corresponde a la última época, no necesariamente al checkpoint restaurado.

BA train/val seleccionada: 0.619704/0.620820. BA train/val de última época: 0.619704/0.620402. Norma final de pesos: 0.947973849. Se conservan por separado ambas épocas; no se presenta la última como la elegida.


Los cuatro historiales y sus diagnósticos están en convergencia.json. No se demuestra separabilidad lineal ni falta de ella con estos ensayos; no hay ampliación del presupuesto por oscilaciones o por llegar al máximo de épocas.
