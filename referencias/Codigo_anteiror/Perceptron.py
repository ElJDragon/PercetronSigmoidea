"""Perceptrón bipolar de tres entradas, implementado desde cero para el taller."""

# 1. IMPORTACIONES Y CONFIGURACIÓN
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Guarda imágenes incluso sin una ventana gráfica disponible.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CSV_PREDETERMINADO = BASE / "dataset_Perceptron_5000_continuo_menos1_a_1.csv"
CARACTERISTICAS = ["Caracteristica_1", "Caracteristica_2", "Caracteristica_3"]
OBJETIVO = "Clase"
NOMBRES = ["Mujer", "Hombre"]  # Orden de filas y columnas de las matrices.


# 2. CARGA Y VALIDACIÓN DEL DATASET
def cargar_dataset(ruta):
    """Lee el esquema del docente y valida sin borrar ni modificar registros."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ValueError(f"No existe el CSV: {ruta}. Colóquelo junto al programa o use --csv RUTA.")
    try:
        datos = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        print("El archivo no es UTF-8; se intentará leer con codificación Latin-1.")
        datos = pd.read_csv(ruta, sep=None, engine="python", encoding="latin-1")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("El CSV está vacío.") from exc

    print("Primeras cinco filas:\n", datos.head().to_string(index=False))
    print("Dimensiones:", datos.shape, "\nTipos:\n", datos.dtypes)
    if set(datos.columns) != set(CARACTERISTICAS + [OBJETIVO]):
        raise ValueError(f"Se esperan exactamente las columnas {CARACTERISTICAS + [OBJETIVO]}.")
    if datos.empty:
        raise ValueError("El CSV no contiene registros.")
    print("Entradas:", CARACTERISTICAS, "| Objetivo:", OBJETIVO)
    nulos = datos.isna().sum()
    duplicados = int(datos.duplicated().sum())
    print("Valores nulos:", nulos.to_dict(), "| Registros duplicados:", duplicados)
    if nulos.any():
        raise ValueError("Hay valores faltantes. Corrija el CSV antes de entrenar; no se eliminaron filas.")
    if duplicados:
        raise ValueError("Hay duplicados. Revise el CSV para evitar copias entre entrenamiento y prueba.")

    entradas = datos[CARACTERISTICAS].apply(pd.to_numeric, errors="coerce")
    if entradas.isna().any().any() or not np.isfinite(entradas.to_numpy(dtype=float)).all():
        raise ValueError("Las tres características deben contener únicamente números finitos.")
    etiquetas = datos[OBJETIVO].astype(str).str.strip().str.casefold()
    if set(etiquetas) == {"hombre", "mujer"}:
        y = etiquetas.map({"hombre": 1, "mujer": -1}).to_numpy(dtype=int)
    else:
        numericas = pd.to_numeric(datos[OBJETIVO], errors="coerce")
        if numericas.isna().any() or set(numericas) != {-1, 1}:
            raise ValueError("Clase debe incluir Hombre/Mujer o los valores -1 y 1.")
        print("Etiquetas numéricas: se asume el contrato del taller, Hombre=1 y Mujer=-1.")
        y = numericas.to_numpy(dtype=int)
    X = entradas.to_numpy(dtype=float)
    for clase, nombre in [(-1, "Mujer"), (1, "Hombre")]:
        cantidad = int(np.sum(y == clase))
        print(f"{nombre} ({clase}): {cantidad} registros ({cantidad / len(y):.2%})")
    print("Rangos de las entradas:\n", entradas.agg(["min", "max"]).to_string())
    print("Se conserva la escala original; no se aplica normalización.")
    return datos, X, y


# 3. PREPARACIÓN DE LOS DATOS
def dividir_datos(y, semilla=42):
    """Devuelve índices disjuntos; estratifica si hay al menos dos filas por clase."""
    rng = np.random.default_rng(semilla)
    clases, cantidades = np.unique(y, return_counts=True)
    if len(y) < 2:
        raise ValueError("Se necesitan al menos dos registros para dividir los datos.")
    if np.all(cantidades >= 2):
        entrenamiento, prueba = [], []
        for clase in clases:
            indices = rng.permutation(np.flatnonzero(y == clase))
            n_prueba = min(len(indices) - 1, max(1, int(round(len(indices) * 0.2))))
            prueba.extend(indices[:n_prueba])
            entrenamiento.extend(indices[n_prueba:])
        print("Partición estratificada, semilla:", semilla)
        return rng.permutation(entrenamiento), rng.permutation(prueba)
    print("No hay suficientes registros por clase: partición aleatoria sin estratificar.")
    indices = rng.permutation(len(y))
    n_prueba = min(len(y) - 1, max(1, int(round(len(y) * 0.2))))
    return indices[n_prueba:], indices[:n_prueba]


# 4. CLASE PERCEPTRON SIMPLE
class PerceptronSimple:
    """Una neurona: z=w1*x1+w2*x2+w3*x3+w0; salida en {-1, 1}."""

    def __init__(self, learning_rate=0.01, epochs=200, verbose=True):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("La tasa de aprendizaje debe ser positiva y finita.")
        if not isinstance(epochs, (int, np.integer)) or epochs < 200:
            raise ValueError("El taller requiere un número entero de épocas de al menos 200.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.weights = np.zeros(3, dtype=float)  # w1, w2, w3
        self.bias = 0.0  # w0
        self.historial = []

    @staticmethod
    def activacion(z):
        """Escalón bipolar: incluso cuando z=0, devuelve la clase positiva."""
        return np.where(np.asarray(z) >= 0, 1, -1)

    @staticmethod
    def validar_entradas(X):
        """Exige una matriz finita de n filas y exactamente tres columnas."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X debe ser una matriz no vacía de tres características finitas.")
        return X

    def predict(self, X):
        """Calcula la combinación lineal y el escalón para cada fila."""
        X = self.validar_entradas(X)
        return self.activacion(X @ self.weights + self.bias)

    def fit(self, X, y):
        """Entrena en orden fijo durante todas las épocas, sin parada anticipada."""
        X = self.validar_entradas(X)
        y = np.asarray(y)
        if y.shape != (len(X),) or not np.isin(y, [-1, 1]).all():
            raise ValueError("y debe tener una etiqueta -1 o 1 por cada fila de X.")
        # Cada llamada a fit empieza un experimento nuevo desde cero.
        self.weights = np.zeros(3, dtype=float)
        self.bias = 0.0
        self.historial = []
        for epoca in range(1, self.epochs + 1):
            errores = 0
            for x, y_real in zip(X, y):
                x1, x2, x3 = x
                w1, w2, w3 = self.weights
                w0 = self.bias
                z = w1 * x1 + w2 * x2 + w3 * x3 + w0
                y_predicha = int(self.activacion(z))
                error = int(y_real) - y_predicha  # En bipolar: -2, 0 o 2.
                if error != 0:
                    # w_nuevo = w_actual + eta*(y_real-y_predicha)*x
                    self.weights += self.learning_rate * error * x
                    # b_nuevo = b_actual + eta*(y_real-y_predicha)
                    self.bias += self.learning_rate * error
                    errores += 1
            # Esta exactitud utiliza los pesos al FINAL de la época.
            exactitud = float(np.mean(self.predict(X) == y))
            self.historial.append({
                "epoca": epoca, "errores": errores, "actualizaciones": errores,
                "exactitud": exactitud, "w1": self.weights[0],
                "w2": self.weights[1], "w3": self.weights[2], "sesgo": self.bias,
            })
            if self.verbose and (epoca == 1 or epoca % 20 == 0):
                print(f"Época {epoca}: errores={errores}, exactitud={exactitud:.2%}")
        return self


# 5. ENTRENAMIENTO
def entrenar(X, y, learning_rate, epochs, verbose):
    """Construye y ajusta el modelo exclusivamente con los datos de entrenamiento."""
    return PerceptronSimple(learning_rate, epochs, verbose).fit(X, y)


# 6. EVALUACIÓN
def evaluar(y_real, y_predicha):
    """Calcula manualmente las métricas; clase positiva: Hombre=1."""
    y_real, y_predicha = np.asarray(y_real), np.asarray(y_predicha)
    if (y_real.ndim != 1 or y_real.size == 0 or y_real.shape != y_predicha.shape
            or not np.isin(y_real, [-1, 1]).all() or not np.isin(y_predicha, [-1, 1]).all()):
        raise ValueError("Se necesitan vectores iguales, no vacíos, de etiquetas -1 y 1.")
    vp = int(np.sum((y_real == 1) & (y_predicha == 1)))
    vn = int(np.sum((y_real == -1) & (y_predicha == -1)))
    fp = int(np.sum((y_real == -1) & (y_predicha == 1)))
    fn = int(np.sum((y_real == 1) & (y_predicha == -1)))
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metricas = {"exactitud": (vp + vn) / len(y_real), "precision": precision,
                "recall": recall, "f1": f1, "VP": vp, "VN": vn, "FP": fp, "FN": fn}
    matriz = pd.DataFrame([[vn, fp], [fn, vp]], index=NOMBRES, columns=NOMBRES)
    matriz.index.name, matriz.columns.name = "Clase real", "Clase predicha"
    return metricas, matriz


def analizar_convergencia(historial):
    """Distingue errores durante las actualizaciones de exactitud al cerrar la época."""
    errores = np.array([fila["errores"] for fila in historial])
    ceros = np.flatnonzero(errores == 0)
    fluctuaron = bool(np.any(np.diff(errores) > 0) and np.any(np.diff(errores) < 0))
    primera = str(int(ceros[0] + 1)) if ceros.size else "ninguna"
    return (
        f"Primera época con cero errores durante el recorrido: {primera}.\n"
        f"Terminó con cero errores durante el recorrido: {'sí' if errores[-1] == 0 else 'no'}.\n"
        f"Los errores aumentaron y disminuyeron entre épocas: {'sí' if fluctuaron else 'no'}.\n"
        f"Clasificó todo el entrenamiento al terminar: "
        f"{'sí' if historial[-1]['exactitud'] == 1 else 'no'}.\n"
        "Un perceptrón no garantiza convergencia a cero errores si los datos no son "
        "linealmente separables. Completar 200 épocas no demuestra convergencia."
    )


# 7. GRÁFICOS
def generar_graficos(historial, matriz, carpeta):
    """Guarda dos curvas de entrenamiento y la matriz de confusión de prueba."""
    for columna, factor, titulo, etiqueta, archivo in [
        ("errores", 1, "Errores durante cada época", "Errores de clasificación", "errores_por_epoca.png"),
        ("exactitud", 100, "Exactitud al finalizar cada época", "Exactitud de entrenamiento (%)", "exactitud_por_epoca.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.plot(historial["epoca"], historial[columna] * factor)
        ax.set(xlabel="Época", ylabel=etiqueta, title=titulo)
        ax.set_xlim(1, int(historial["epoca"].max()))
        if columna == "exactitud":
            ax.set_ylim(0, 101)
            ax.set_yticks(np.arange(0, 101, 20))
        else:
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        fig.savefig(carpeta / archivo, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    mapa = ax.imshow(matriz.to_numpy(), cmap="Blues", vmin=0)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=NOMBRES, yticklabels=NOMBRES,
           xlabel="Clase predicha", ylabel="Clase real", title="Matriz de confusión · prueba")
    for i in range(2):
        for j in range(2):
            valor = int(matriz.iloc[i, j])
            ax.text(j, i, str(valor), ha="center", va="center", fontsize=16,
                    color="white" if valor > matriz.to_numpy().max() / 2 else "black")
    fig.colorbar(mapa, ax=ax, label="Registros")
    fig.savefig(carpeta / "matriz_confusion.png", dpi=180)
    plt.close(fig)


# 8. EXPORTACIÓN DE RESULTADOS
def exportar_resultados(carpeta, modelo, metricas, matriz, datos_prueba, y_real, predichas, resumen):
    """Exporta únicamente resultados obtenidos por este experimento."""
    carpeta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(metricas, orient="index").rename_axis("conjunto").to_csv(
        carpeta / "metricas.csv", encoding="utf-8-sig")
    historial = pd.DataFrame(modelo.historial)
    historial.to_csv(carpeta / "historial_entrenamiento.csv", index=False)
    matriz.to_csv(carpeta / "matriz_confusion.csv", encoding="utf-8-sig")
    salida = datos_prueba.copy()
    salida.insert(0, "indice_original", datos_prueba.index)
    salida["clase_real"] = salida[OBJETIVO]
    salida["clase_real_codificada"] = y_real
    salida["clase_predicha"] = predichas
    salida["nombre_predicho"] = np.where(predichas == 1, "Hombre", "Mujer")
    salida["correcta"] = salida["clase_real_codificada"] == predichas
    salida.to_csv(carpeta / "predicciones.csv", index=False, encoding="utf-8-sig")
    (carpeta / "resumen.txt").write_text(resumen, encoding="utf-8")
    generar_graficos(historial, matriz, carpeta)


def crear_resumen(modelo, metricas, matriz, n_entrenamiento, n_prueba):
    """Construye el resumen con coeficientes, métricas y conclusiones verificables."""
    w1, w2, w3 = modelo.weights
    lineas = [
        "PERCEPTRÓN SIMPLE: RESULTADOS REALES",
        "Pesos iniciales: w1=0, w2=0, w3=0; sesgo inicial w0=0.",
        f"Pesos finales: w1={w1:.12g}, w2={w2:.12g}, w3={w3:.12g}; w0={modelo.bias:.12g}.",
        f"Épocas: {len(modelo.historial)}. Tasa de aprendizaje: {modelo.learning_rate}.",
        f"Registros: entrenamiento={n_entrenamiento}, prueba={n_prueba}. Semilla=42.",
        f"Errores durante primera época: {modelo.historial[0]['errores']}; "
        f"última: {modelo.historial[-1]['errores']}.",
        f"Actualizaciones totales: {sum(f['actualizaciones'] for f in modelo.historial)}.",
        f"Ecuación final: z = ({w1:.12g})*x1 + ({w2:.12g})*x2 + ({w3:.12g})*x3 + ({modelo.bias:.12g})",
        "Salida: Hombre=1 si z>=0; Mujer=-1 si z<0.",
    ]
    for nombre, valores in metricas.items():
        lineas.append(nombre + ": " + ", ".join(f"{k}={valores[k]:.2%}" for k in ["exactitud", "precision", "recall", "f1"]))
    lineas.extend([
        "Matriz de confusión de prueba (filas=reales, columnas=predichas):", matriz.to_string(),
        analizar_convergencia(modelo.historial),
        "Conclusión 1: los pesos ponderan cada entrada y el sesgo desplaza el umbral; "
        "la regla de aprendizaje modifica estos parámetros cuando hay una clasificación errónea.",
        f"Conclusión 2: en esta partición se obtuvo {metricas['prueba']['exactitud']:.2%} de exactitud "
        "de prueba; el resultado describe este dataset sintético y esta partición.",
        "Recomendación: repetir con otras semillas y comparar métricas y errores manteniendo el perceptrón simple.",
        "Las variables sintéticas no representan atributos reales de hombres o mujeres.",
    ])
    return "\n".join(lineas) + "\n"


# 9. FUNCIÓN PRINCIPAL
def main():
    """Coordina lectura, partición, entrenamiento, evaluación y exportación."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PREDETERMINADO)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--salida", type=Path, default=BASE / "resultados")
    parser.add_argument("--silencioso", action="store_true", help="Omite el progreso por época.")
    args = parser.parse_args()
    try:
        datos, X, y = cargar_dataset(args.csv)
        indices_entrenamiento, indices_prueba = dividir_datos(y)
        modelo = entrenar(X[indices_entrenamiento], y[indices_entrenamiento],
                          args.learning_rate, args.epochs, not args.silencioso)
        metricas = {}
        metricas["entrenamiento"], _ = evaluar(y[indices_entrenamiento], modelo.predict(X[indices_entrenamiento]))
        predichas = modelo.predict(X[indices_prueba])
        metricas["prueba"], matriz = evaluar(y[indices_prueba], predichas)
        resumen = crear_resumen(modelo, metricas, matriz, len(indices_entrenamiento), len(indices_prueba))
        exportar_resultados(args.salida, modelo, metricas, matriz, datos.iloc[indices_prueba],
                            y[indices_prueba], predichas, resumen)
        print("\n" + resumen)
        print("Resultados guardados en:", args.salida.resolve())
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    main()

