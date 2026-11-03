"""Gráficos de desarrollo a partir de historial y métricas; no recibe X/test."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


def graficar_perceptron(history, metrics_val, selected_epoch, n_features, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    made = []
    specs = [
        ("errores_entrenamiento.png", [("online_errors", "Errores durante actualizaciones", "#bc4c32")],
         "Errores durante el recorrido de entrenamiento", "Errores / actualizaciones", False),
        ("accuracy_por_epoca.png", [("train_accuracy", "Train al finalizar época", "#236b8e"),
                                    ("val_accuracy", "Validación al finalizar época", "#b86e19")],
         "Exactitud al finalizar cada época", "Accuracy", True),
        ("balanced_accuracy_validacion.png", [("val_balanced_accuracy", "Balanced accuracy de validación", "#367653")],
         "Balanced accuracy de validación", "Balanced accuracy", True),
    ]
    for filename, curves, title, ylabel, percent in specs:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for key, label, color in curves:
            ax.plot(epochs, [h[key] for h in history], label=label, color=color, linewidth=1.4)
        ax.axvline(selected_epoch, color="#4a4057", linestyle="--", linewidth=1, label=f"Época seleccionada: {selected_epoch}")
        ax.set(title=f"{title}\nPerceptrón con {n_features} características", xlabel="Época", ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 1)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        path = output / filename
        fig.savefig(path, dpi=170, metadata={"Software": "PerceptronReal fase 4"})
        plt.close(fig)
        made.append(path)
    cm = np.array(metrics_val["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set(title=f"Matriz de confusión en validación\nModelo seleccionado: época {selected_epoch}",
           xticks=[0, 1], yticks=[0, 1], xticklabels=["0: no superior", "1: superior"],
           yticklabels=["0: no superior", "1: superior"], xlabel="Predicción", ylabel="Clase real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax, label="Observaciones")
    path = output / "matriz_confusion_validacion.png"
    fig.savefig(path, dpi=170, metadata={"Software": "PerceptronReal fase 4"})
    plt.close(fig)
    made.append(path)
    return made
