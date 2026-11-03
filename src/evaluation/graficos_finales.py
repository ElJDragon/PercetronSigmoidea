"""Seis gráficos descriptivos de evaluación; no entrenamiento ni PCA."""
from pathlib import Path
import numpy as np
from src.evaluation.graficos import plt, PercentFormatter

NAMES=("perceptron","logistica","mayoritario","historico")
LABELS=("Perceptrón","Logística","Mayoritario","Histórico")
COLORS=("#286b98","#c56025","#467b49","#806192")


def graficar_evaluacion(evaluation,comparison,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(output/name,dpi=175,metadata={"Software":"PerceptronReal fase 7"})
        plt.close(fig)
    def percent(ax):
        ax.set_ylim(0,1.05); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y",alpha=.17)
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    values=[evaluation["global"][n]["balanced_accuracy"] for n in NAMES]
    ax.bar(LABELS,[v if v is not None else 0 for v in values],color=COLORS,width=.6)
    for i,v in enumerate(values): ax.text(i,(v or 0)+.02,"no estimable" if v is None else f"{v*100:.2f} %",ha="center",fontsize=10)
    ax.set(title="Test WWI · balanced accuracy global\nMismas 1.111 observaciones, configuraciones congeladas",ylabel="Balanced accuracy")
    percent(ax); save(fig,"balanced_accuracy_global.png")
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    x=np.arange(4); width=.19
    for i,(name,label,color) in enumerate(zip(NAMES,LABELS,COLORS)):
        values=[evaluation["global"][name][k] for k in ["accuracy","precision","recall","f1"]]
        ax.bar(x+(i-1.5)*width,[v if v is not None else np.nan for v in values],width,label=label,color=color)
    ax.set(xticks=x,xticklabels=["Accuracy","Precision positiva","Recall positivo","F1 positivo"],title="Test WWI · métricas globales de los cuatro métodos",ylabel="Proporción")
    percent(ax); ax.legend(fontsize=9); save(fig,"metricas_globales.png")
    vmax=max(np.max(evaluation["global"][n]["confusion_matrix"]) for n in ["perceptron","logistica"])
    for name,label in zip(NAMES[:2],LABELS[:2]):
        cm=np.asarray(evaluation["global"][name]["confusion_matrix"])
        fig,ax=plt.subplots(figsize=(6,5),layout="constrained")
        im=ax.imshow(cm,cmap="Blues",vmin=0,vmax=vmax)
        ax.set(title=f"{label} · matriz de confusión en test",xlabel="Predicción",ylabel="Clase real",
               xticks=[0,1],yticks=[0,1],xticklabels=["0: no superior","1: superior"],yticklabels=["0: no superior","1: superior"])
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=16,color="white" if cm[i,j]>vmax/2 else "black")
        fig.colorbar(im,ax=ax,label="Observaciones"); save(fig,f"confusion_{name}.png")
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    months=list(evaluation["monthly"])
    for name,label,color in zip(NAMES,LABELS,COLORS):
        values=[evaluation["monthly"][m][name]["balanced_accuracy"] for m in months]
        ax.plot(months,[v if v is not None else np.nan for v in values],marker="o",label=label,color=color,linewidth=1.5)
    ax.set(title="Test WWI · balanced accuracy por mes\nCinco períodos completos bajo supuesto académico",xlabel="Mes objetivo",ylabel="Balanced accuracy")
    percent(ax); ax.legend(fontsize=9); save(fig,"balanced_accuracy_mensual.png")
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    x=np.arange(4)
    for i,(split,label,color) in enumerate([("train","Train","#9bbfd3"),("val","Validación (selección)","#dec19e"),("test","Test (evaluación final)","#4e7c60")]):
        values=[comparison[n][split]["balanced_accuracy"] for n in NAMES]
        ax.bar(x+(i-1)*.25,values,width=.25,label=label,color=color)
    ax.set(xticks=x,xticklabels=LABELS,title="Balanced accuracy · desarrollo y evaluación final",ylabel="Balanced accuracy")
    percent(ax); ax.legend(fontsize=9); save(fig,"comparacion_train_val_test.png")
