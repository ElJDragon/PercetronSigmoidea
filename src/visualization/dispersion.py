"""Dispersión fiel: sin jitter, selección de puntos ni fronteras de decisión."""
from pathlib import Path
import numpy as np
from src.evaluation.graficos import plt, PercentFormatter, MaxNLocator
from src.evaluation.metricas import validar_etiquetas


def graficar_pca(projections, labels, ratios_all, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    combined = np.vstack([projections[s] for s in ["train","val"]])
    lower,upper = combined.min(axis=0),combined.max(axis=0)
    margin = np.maximum((upper-lower)*.04,1e-9)
    limits = [lower-margin,upper+margin]
    titles = {"train":"Entrenamiento WWI · abril 2013–diciembre 2014", "val":"Validación WWI · enero–diciembre 2015"}
    for split in ["train","val"]:
        z = np.asarray(projections[split])
        if z.ndim != 2 or z.shape[1] != 2 or not np.isfinite(z).all():
            raise ValueError("Proyección bidimensional inválida")
        y = validar_etiquetas(labels[split],n=len(z))
        fig,ax = plt.subplots(figsize=(9,6),layout="constrained")
        # Orden intercalado original (producto-mes) para no pintar toda una clase encima.
        colors=np.where(y==0,"#286b98","#c56025")
        ax.scatter(z[:,0],z[:,1],c=colors,s=11,alpha=.28,linewidths=0,rasterized=True)
        for cls,color,description in [(0,"#286b98","0: igual o inferior a referencia"),(1,"#c56025","1: superior a referencia")]:
            ax.scatter([],[],color=color,s=28,label=f"{description} (n={int((y==cls).sum())})")
        ax.set(title=f"{titles[split]}\nPCA ajustado solo en train · proyección visual de 20 entradas",
               xlabel=f"PC1 · {ratios_all[0]*100:.2f} % de varianza train",
               ylabel=f"PC2 · {ratios_all[1]*100:.2f} % de varianza train",
               xlim=(limits[0][0],limits[1][0]),ylim=(limits[0][1],limits[1][1]))
        ax.grid(alpha=.18); ax.legend(fontsize=9,loc="upper right")
        fig.savefig(output/f"pca_{split}.png",dpi=180,metadata={"Software":"PerceptronReal fase 6"})
        plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,5),layout="constrained")
    x = np.arange(1,len(ratios_all)+1)
    ax.bar(x,ratios_all,color=["#236b8e" if i<=2 else "#bac9d3" for i in x],label="Varianza individual")
    ax.plot(x,np.cumsum(ratios_all),color="#aa5d20",marker="o",markersize=3,label="Varianza acumulada")
    retained=float(np.sum(ratios_all[:2]))
    ax.axhline(retained,color="#367653",linestyle="--",label=f"Conservada en 2D: {retained*100:.2f} %")
    ax.set(title="PCA WWI · varianza de entrenamiento",xlabel="Componente principal",ylabel="Proporción de varianza",ylim=(0,1.04))
    ax.set_xticks(x); ax.set_xlim(.4,len(x)+.6); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y",alpha=.18); ax.legend(fontsize=9)
    fig.savefig(output/"varianza_explicada.png",dpi=180,metadata={"Software":"PerceptronReal fase 6"})
    plt.close(fig)
