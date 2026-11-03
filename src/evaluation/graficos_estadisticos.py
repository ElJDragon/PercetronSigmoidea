"""Figuras descriptivas adicionales; no generan predicciones ni reestiman modelos."""
from pathlib import Path
import numpy as np
from src.evaluation.graficos import plt,PercentFormatter

METHODS=("perceptron","logistica","mayoritario","historico")
LABELS=("Perceptrón","Logística","Mayoritario","Histórico")
COLORS=("#286b98","#c56025","#467b49","#806192")


def graficar_analisis(global_metrics,monthly,agreements,distributions,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(output/name,dpi=175,metadata={"Software":"PerceptronReal fase 8"}); plt.close(fig)
    def percent(ax):
        ax.set_ylim(0,1.05); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y",alpha=.18)
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained"); x=np.arange(4)
    ax.bar(x-.18,[global_metrics[n]["recall"] for n in METHODS],width=.36,label="Sensibilidad (TPR)",color="#286b98")
    ax.bar(x+.18,[global_metrics[n]["specificity"] for n in METHODS],width=.36,label="Especificidad (TNR)",color="#c56025")
    ax.set(xticks=x,xticklabels=LABELS,ylabel="Tasa por clase real",title="Test WWI · sensibilidad y especificidad")
    percent(ax); ax.legend(fontsize=9); save(fig,"sensibilidad_especificidad.png")
    months=list(monthly); fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout="constrained")
    for ax,name,label in zip(axes,METHODS[:2],LABELS[:2]):
        ax.plot(months,[monthly[m][name]["false_positive_rate"] for m in months],marker="o",label="FPR: FP / negativos reales",color="#b74d30")
        ax.plot(months,[monthly[m][name]["false_negative_rate"] for m in months],marker="s",label="FNR: FN / positivos reales",color="#286b98")
        ax.set(title=label,ylabel="Tasa de error por clase"); percent(ax); ax.legend(fontsize=9)
    axes[-1].set_xlabel("Mes objetivo"); fig.suptitle("Test WWI · evolución de falsos positivos y negativos")
    save(fig,"tasas_error_mensuales.png")
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    for name,label,color in zip(METHODS,LABELS,COLORS):
        ax.plot(months,[monthly[m][name]["predicted_positive_proportion"] for m in months],marker="o",label=label,color=color)
    ax.plot(months,[monthly[m]["perceptron"]["actual_positive_proportion"] for m in months],color="#242424",linestyle="--",marker="x",label="Proporción positiva real")
    ax.set(title="Test WWI · frecuencia de predicciones positivas",xlabel="Mes objetivo",ylabel="Proporción de observaciones")
    percent(ax); ax.legend(fontsize=9); save(fig,"predicciones_positivas.png")
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained"); bottom=np.zeros(len(months))
    for key,label,color in [("both_correct","Ambos aciertan","#467b49"),("both_wrong","Ambos fallan","#a84842"),
                            ("only_perceptron_correct","Solo perceptrón acierta","#286b98"),("only_logistic_correct","Solo logística acierta","#c58a30")]:
        values=np.array([agreements["monthly"][m][key] for m in months]); ax.bar(months,values,bottom=bottom,label=label,color=color); bottom+=values
    ax.set(title="Test WWI · acuerdos y desacuerdos de los clasificadores",xlabel="Mes objetivo",ylabel="Observaciones")
    ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.01,1)); ax.grid(axis="y",alpha=.15)
    save(fig,"acuerdos_desacuerdos.png")
    names=distributions["feature_names"]
    heat=np.array([[distributions["mean_shifts"][n]["val_minus_train_in_train_std"],distributions["mean_shifts"][n]["test_minus_train_in_train_std"]] for n in names])
    fig,ax=plt.subplots(figsize=(10,9),layout="constrained"); extent=max(float(np.max(np.abs(heat))),1e-9)
    im=ax.imshow(heat,cmap="RdBu_r",vmin=-extent,vmax=extent,aspect="auto")
    ax.set(xticks=[0,1],xticklabels=["Validación − train","Test − train"],yticks=np.arange(len(names)),yticklabels=names,
           title="Cambios descriptivos de medias de X\nUnidades de desviación de train; sin reajustar transformaciones")
    ax.tick_params(axis="y",labelsize=9)
    for i in range(len(names)):
        for j in range(2): ax.text(j,i,f"{heat[i,j]:+.2f}",ha="center",va="center",fontsize=9,color="white" if abs(heat[i,j])>extent*.6 else "black")
    fig.colorbar(im,ax=ax,label="Diferencia de media / std train"); save(fig,"cambios_medias_caracteristicas.png")
    history=distributions["raw_history_by_month"]; months_all=list(history)
    fig,ax=plt.subplots(figsize=(12,5),layout="constrained"); t=np.arange(len(months_all))
    ax.plot(t,[history[m]["lag1_mean_units"] for m in months_all],label="Media de unidades_lag_1",color="#286b98")
    ax.plot(t,[history[m]["lag1_median_units"] for m in months_all],label="Mediana de unidades_lag_1",color="#c56025")
    for month,label in [("2015-01","Inicio validación"),("2016-01","Inicio test")]:
        i=months_all.index(month); ax.axvline(i-.5,linestyle="--",color="#555555",linewidth=1); ax.text(i,max(v["lag1_mean_units"] for v in history.values())*.95,label,rotation=90,va="top",fontsize=8)
    ticks=np.arange(0,len(months_all),4); ax.set(xticks=ticks,xticklabels=[months_all[i] for i in ticks],
        title="Historial existente · unidades del mes previo por producto elegible",xlabel="Mes objetivo",ylabel="Unidades facturadas del mes previo")
    ax.grid(alpha=.18); ax.legend(fontsize=9); save(fig,"unidades_historicas_mensuales.png")
