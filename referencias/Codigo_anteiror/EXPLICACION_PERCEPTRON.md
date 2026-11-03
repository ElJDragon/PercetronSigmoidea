# Cómo se construyó el perceptrón

## 1. Del taller a un programa

El taller pide clasificar dos clases mediante tres números, usando una sola neurona. Primero se inspeccionó el CSV para identificar entradas y objetivo; después se implementaron validación, partición, neurona, entrenamiento, evaluación y exportación. La función `main()` conecta esos pasos.

No se usó un clasificador ya construido. pandas lee y organiza tablas, NumPy hace operaciones numéricas y matplotlib dibuja las evidencias.

## 2. Qué representa una fila

La primera fila del CSV contiene:

```text
x1 = -0.515, x2 = -0.695, x3 = -0.874, clase = Mujer
```

El vector de entrada es `x = [-0.515, -0.695, -0.874]`. La etiqueta esperada es `y = -1`. Hombre se representa con `1`. Para todas las filas, `X` es una matriz de tamaño `(5000, 3)` e `y` es un vector de tamaño `(5000,)`.

Los nombres de las tres características no describen atributos reales: son variables sintéticas del ejercicio.

## 3. Carga y preparación: funciones anteriores a la clase

### `cargar_dataset(ruta)`

Lee con `pd.read_csv`, detecta el separador e intenta UTF-8 con soporte de BOM. Si falla esa codificación, intenta Latin-1 e informa el cambio. Imprime cinco filas, dimensiones, tipos, columnas, nulos, duplicados y distribución de clases.

Comprueba que estén exactamente las cuatro columnas esperadas. `pd.to_numeric(..., errors="coerce")` transforma entradas a números; una conversión imposible queda como nulo y provoca un error. `np.isfinite` rechaza infinitos. Los datos inválidos detienen el experimento para que se revise la fuente. Los registros nunca se eliminan silenciosamente.

Luego codifica Hombre=1 y Mujer=-1 y devuelve tres objetos: la tabla original `datos`, la matriz numérica `X` y las etiquetas `y`. Los rangos del original ya están en [-1, 1], por lo que no se normalizan.

### `dividir_datos(y, semilla=42)`

Devuelve índices para seleccionar filas, no un modelo. NumPy mezcla por separado los índices de cada clase y reserva el 20 % para prueba. Luego mezcla los índices resultantes de cada conjunto. En el CSV original quedan 2.000 muestras por clase para entrenar y 500 por clase para probar.

La semilla fija permite repetir la selección con el mismo entorno. La estratificación mantiene la proporción de clases; usar etiquetas para esta selección no implica entrenar con las muestras de prueba. Los índices no se superponen. El modelo solo recibe `X[indices_entrenamiento]` e `y[indices_entrenamiento]` en `fit`.

## 4. La clase `PerceptronSimple`

Una clase reúne el estado de un objeto y las operaciones que lo modifican. Aquí el estado está formado por tres pesos, un sesgo y el historial. `self` hace referencia al objeto concreto que se está usando.

### `__init__(learning_rate=0.01, epochs=200, verbose=True)`

Es el constructor. Se ejecuta al escribir `PerceptronSimple()`. Valida la configuración e inicializa:

```python
self.weights = np.zeros(3, dtype=float)
self.bias = 0.0
self.historial = []
```

`weights[0]`, `weights[1]` y `weights[2]` corresponden a `w1`, `w2`, `w3`. `bias` es `w0`. `learning_rate` es el tamaño de cada corrección; `epochs` indica cuántas veces recorrer el entrenamiento y `verbose` controla los mensajes de progreso. Se exigen al menos 200 épocas y una tasa positiva finita.

### `activacion(z)`

Implementa el escalón bipolar:

```text
f(z) =  1 si z >= 0
f(z) = -1 si z < 0
```

`np.where(np.asarray(z) >= 0, 1, -1)` puede operar sobre un valor o sobre varios. Nunca devuelve cero. Con pesos y sesgo iniciales en cero, `z=0`, por lo que la predicción inicial siempre es Hombre=1.

`@staticmethod` indica que esta operación no necesita acceder al estado de `self`.

### `validar_entradas(X)`

Convierte la entrada a un arreglo numérico y exige una matriz no vacía con exactamente tres columnas y valores finitos. Evita entrenar accidentalmente con una cuarta característica o usar un vector de forma incorrecta. También es estático porque no depende de los pesos.

### `predict(X)`

Predice sin modificar los parámetros. Primero calcula:

```text
z = w1*x1 + w2*x2 + w3*x3 + w0
```

Para todas las filas a la vez, el código equivalente es `X @ self.weights + self.bias`. El operador `@` realiza el producto matricial: cada fila se combina con los tres pesos. Finalmente aplica el escalón a los resultados.

### `fit(X, y)`

Es el entrenamiento. Valida las etiquetas, reinicia pesos y sesgo en cero y recorre todas las épocas. Cada época visita una vez cada fila del conjunto de entrenamiento en el mismo orden.

Para cada ejemplo realiza estos pasos:

1. Extrae explícitamente `x1`, `x2`, `x3`, `w1`, `w2`, `w3` y `w0`.
2. Calcula `z` con la suma ponderada.
3. Obtiene `y_predicha` con la activación.
4. Calcula `error = y_real - y_predicha`.
5. Si hay error, aplica la regla de actualización y suma un error al contador.

Las fórmulas son:

```text
w_nuevo = w_actual + eta * (y_real - y_predicha) * x
b_nuevo = b_actual + eta * (y_real - y_predicha)
```

En Python, `self.weights += self.learning_rate * error * x` corrige los tres pesos simultáneamente. `self.bias += self.learning_rate * error` corrige el sesgo. Como las etiquetas son bipolares, la diferencia puede ser -2, 0 o 2; no debe sustituirse sin explicación por una regla para etiquetas 0/1.

Ejemplo de control, independiente del dataset: `x=[1,1,1]`, `y_real=-1`, pesos y sesgo cero, tasa 0.01. La salida inicial es 1, así que el error es -2. Los pesos pasan a `[-0.02,-0.02,-0.02]` y el sesgo a `-0.02`. La nueva suma es -0.08 y la predicción ya es -1.

Al cerrar cada época se guardan errores durante el recorrido, número de actualizaciones, exactitud con los pesos finales de esa época y valores de los parámetros. Errores y actualizaciones coinciden porque cada clasificación equivocada produce una actualización. Se continúa hasta completar las épocas incluso si ya no hay errores.

`return self` permite construir y entrenar en una expresión, por ejemplo `PerceptronSimple().fit(X, y)`.

## 5. Funciones de entrenamiento y evaluación

### `entrenar(X, y, learning_rate, epochs, verbose)`

Es una función breve que construye la clase y llama a `fit`. Permite reconocer la etapa de entrenamiento dentro del flujo principal. La matemática se encuentra en la clase, sin repetirse aquí.

### `evaluar(y_real, y_predicha)`

Compara etiquetas y calcula los cuatro conteos. La clase positiva es Hombre=1:

| Conteo | Clase real | Predicción |
|---|---|---|
| VP | Hombre | Hombre |
| VN | Mujer | Mujer |
| FP | Mujer | Hombre |
| FN | Hombre | Mujer |

Las comparaciones booleanas de NumPy generan máscaras, y `np.sum` cuenta sus valores verdaderos. No se llama a ninguna función de métricas de scikit-learn.

```text
exactitud = (VP + VN) / total
precisión = VP / (VP + FP)
recall = VP / (VP + FN)
F1 = 2 * precisión * recall / (precisión + recall)
```

Cuando un denominador de precisión, recall o F1 vale cero, se devuelve 0 como convención explícita. La matriz usa filas reales y columnas predichas, ambas en orden Mujer, Hombre: `[[VN, FP], [FN, VP]]`.

### `analizar_convergencia(historial)`

Busca la primera época con cero errores durante el recorrido, verifica cómo terminó y detecta si hubo aumentos y disminuciones entre épocas. Informa también si todos los ejemplos de entrenamiento quedaron bien clasificados al finalizar.

Son dos observaciones distintas: en la primera época hubo dos errores mientras cambiaban los pesos, pero los parámetros al cierre ya clasificaban correctamente todas las muestras de entrenamiento. Por eso la curva de errores empieza en 2 y la de exactitud empieza en 100 %. La primera época completa sin errores es la segunda.

El teorema de convergencia del perceptrón exige separabilidad lineal. Entrenar 200 épocas por sí solo no demuestra convergencia ni separabilidad. En esta ejecución se comprobó un modelo sin errores sobre entrenamiento y prueba; no debe extrapolarse a cualquier dataset.

## 6. Gráficos, exportación y resumen

### `generar_graficos(historial, matriz, carpeta)`

Crea una curva de errores, una de exactitud porcentual y un mapa de calor de prueba con nombres y conteos por celda. `fig.savefig` guarda las figuras a 180 dpi y `plt.close` libera cada figura. El backend `Agg` permite ejecutarlo sin interfaz gráfica.

### `exportar_resultados(...)`

Crea la carpeta de salida y escribe las métricas, historial, predicciones, matriz, resumen y gráficos. Conserva el índice original de cada fila de prueba para poder rastrearla en el CSV. No vuelve a entrenar ni inventa valores. La columna `correcta` indica si la predicción coincide con la etiqueta codificada.

### `crear_resumen(...)`

Construye un texto a partir de los parámetros y métricas obtenidos. Incluye la ecuación aprendida, errores de primera y última época, convergencia, dos conclusiones y una recomendación. No contiene resultados experimentales fijados manualmente.

### `main()`

Usa `argparse` para leer opciones de terminal. Carga los datos, obtiene los índices, entrena con el 80 %, evalúa ambos conjuntos y exporta. Los errores esperados se muestran en español y terminan con código 1, adecuado para indicar que la ejecución no pudo completarse.

El bloque `if __name__ == "__main__": main()` ejecuta el experimento cuando se abre el archivo con Python. Si se importa desde las pruebas, solo define las funciones y la clase, sin iniciar automáticamente un entrenamiento.

## 7. Interpretar los resultados reales

Con semilla 42, tasa 0.01 y 200 épocas se obtuvo:

```text
z = 0.0246*x1 + 0.01178*x2 + 0.02858*x3 + 0
```

Si `z>=0`, se predice Hombre; en caso contrario, Mujer. Cada peso controla cuánto contribuye su característica a la suma. El sesgo desplaza el umbral; que haya terminado en cero es resultado de las actualizaciones, no una restricción que impida aprenderlo.

La exactitud de prueba fue 100 %: VP=500, VN=500, FP=0 y FN=0. Precisión, recall y F1 también fueron 100 %. El aprendizaje necesitó dos actualizaciones y los parámetros quedaron estables desde la primera época; las otras épocas permiten verificar que no aparecen nuevos errores con ese orden.

Esto describe la clasificación de variables sintéticas en esta partición. Para ampliar el experimento se pueden comparar semillas y tasas de aprendizaje conservando una sola neurona y registrando los resultados sin elegir parámetros a partir del conjunto de prueba.
