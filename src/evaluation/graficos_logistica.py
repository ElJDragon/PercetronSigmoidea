"""Gráficos de logística; reutiliza backend/estilo de visualización aprobado."""
from pathlib import Path
import numpy as np
from src.evaluation.graficos import plt, PercentFormatter, MaxNLocator


def graficar_logistica(history, metrics, epoch, threshold, d, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        ("entropia_cruzada.png", "Entropía cruzada binaria", "Pérdida media", [("train_loss","Train"),("val_loss","Validación")], False),
        ("accuracy_por_epoca.png", "Accuracy al umbral 0,5", "Accuracy", [("train_accuracy","Train"),("val_accuracy","Validación")], True),
        ("balanced_accuracy_validacion.png", "Balanced accuracy de validación al umbral 0,5", "Balanced accuracy", [("val_balanced_accuracy","Validación")], True),
        ("norma_gradiente.png", "Norma del gradiente de train al cierre de época", "Norma L2 conjunta de grad(w,b)", [("gradient_norm","Gradiente")], False)]
    for filename, title, ylabel, curves, percent in specs:
        fig, ax = plt.subplots(figsize=(9,5), layout="constrained")
        for (key, label), color in zip(curves,["#236b8e","#b86e19"]):
            ax.plot([h["epoch"] for h in history],[h[key] for h in history],label=label,color=color,linewidth=1.5)
        ax.axvline(epoch,color="#56446a",linestyle="--",label=f"Checkpoint: época {epoch}")
        ax.set(title=f"{title}\nLogística con {d} características",xlabel="Época",ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)
        if percent:
            ax.set_ylim(0,1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        fig.savefig(output/filename,dpi=170,metadata={"Software":"PerceptronReal fase 5"})
        plt.close(fig)
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6,5),layout="constrained")
    im = ax.imshow(cm,cmap="Blues",vmin=0)
    ax.set(title=f"Validación: época {epoch}, umbral {threshold:g}",xlabel="Predicción",ylabel="Clase real",
           xticks=[0,1],yticks=[0,1],xticklabels=["0: no superior","1: superior"],yticklabels=["0: no superior","1: superior"])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=15,color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(im,ax=ax,label="Observaciones")
    fig.savefig(output/"matriz_confusion_validacion.png",dpi=170,metadata={"Software":"PerceptronReal fase 5"})
    plt.close(fig)
