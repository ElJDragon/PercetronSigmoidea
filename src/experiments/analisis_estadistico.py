"""python -m src.experiments.analisis_estadistico — interpretación sin nuevas predicciones."""
import argparse
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.auditar_datos import BASE,sha256
from src.evaluation.metricas import metricas_binarias
from src.experiments.entrenar_perceptron import read_json,write_json
from src.experiments.evaluacion_final import (METHODS,hashes_anteriores,verificar_congelacion,
                                             evaluar_comun,analizar_errores,metricas_con_conteos)


def hashes_previos(base=BASE):
    base=Path(base); result=hashes_anteriores(base)
    manifest=read_json(base/"resultados/fase_07/manifiesto.json")
    for path in manifest["implementation_sha256"]: result[path]=sha256(base/path)
    for p in sorted((base/"resultados/fase_07").rglob("*")):
        if p.is_file(): result[str(p.relative_to(base))]=sha256(p)
    return result


def verificar_entradas(base=BASE):
    base=Path(base); frozen,freeze_hash=verificar_congelacion(base)
    folder=base/"resultados/fase_07"; manifest=read_json(folder/"manifiesto.json"); validation=read_json(folder/"validaciones.json")
    if not validation["successful"] or validation["phase07_manifest_sha256"]!=sha256(folder/"manifiesto.json") or manifest["freeze_sha256"]!=freeze_hash:
        raise ValueError("Contradicción de manifiesto/validación de fase 7")
    if manifest["integrity"]["before"]!=manifest["integrity"]["after"]:
        raise ValueError("Integridad previa inconsistente")
    for field in ["implementation_sha256","artifacts_sha256"]:
        for path,digest in manifest[field].items():
            if sha256(base/path)!=digest: raise ValueError(f"Artefacto previo alterado: {path}")
    for path,digest in manifest["integrity"]["after"].items():
        if sha256(base/path)!=digest: raise ValueError(f"Archivo protegido alterado: {path}")
    pred=pd.read_csv(folder/"predicciones_alineadas.csv",float_precision="round_trip",keep_default_na=False)
    meta=pd.read_csv(base/"datos/Procesados/fase_03/metadatos_test.csv",keep_default_na=False)
    if pred.observacion_id.tolist()!=meta.observacion_id.tolist() or pred.observacion_id.duplicated().any():
        raise ValueError("Predicciones y observaciones no alineadas")
    for column in ["indice_matriz","producto_provisional","mes_objetivo"]:
        if not pred[column].equals(meta[column]): raise ValueError(f"Metadato incoherente: {column}")
    actual_y=np.load(base/"datos/Procesados/fase_03/y_test.npy",allow_pickle=False)
    if not np.array_equal(pred.y_real.to_numpy(),actual_y): raise ValueError("Etiquetas guardadas no coinciden con test")
    if not np.array_equal(pred.pred_perceptron.to_numpy(),(pred.score_perceptron.to_numpy()>=0).astype(np.int8)):
        raise ValueError("Predicción de perceptrón guardada incoherente")
    if not np.array_equal(pred.pred_logistica.to_numpy(),(pred.probabilidad_logistica.to_numpy()>=.5).astype(np.int8)):
        raise ValueError("Predicción logística guardada incoherente")
    known=pred.producto_provisional.isin(frozen["new_product_rule"]["train_products"])
    if not np.array_equal(pred.cohorte.to_numpy(),np.where(known,"conocidos","nuevos")):
        raise ValueError("Cohortes guardadas no coinciden con regla congelada")
    predictions={name:pred[f"pred_{name}"].to_numpy() for name in METHODS}
    evaluation=evaluar_comun(pred.y_real.to_numpy(),predictions,pred,frozen["new_product_rule"]["train_products"])
    for field,file in [("global","metricas_globales.json"),("monthly","metricas_mensuales.json"),
                       ("cohorts","metricas_cohortes.json"),("cohort_month","metricas_cohorte_mes.json"),
                       ("monthly_unweighted_means","promedios_mensuales.json")]:
        if evaluation[field]!=read_json(folder/file): raise ValueError(f"Métricas no reproducidas: {field}")
    agreement=analizar_errores(pred.y_real.to_numpy(),predictions,pred,frozen["new_product_rule"]["train_products"])
    if agreement!=read_json(folder/"errores_desacuerdos.json"): raise ValueError("Acuerdos no reproducidos")
    return frozen,pred,evaluation,agreement


def derivar_metricas(metrics):
    c=metrics["confusion_matrix"]; tn,fp,fn,tp=c[0][0],c[0][1],c[1][0],c[1][1]
    n=tn+fp+fn+tp
    if n!=metrics["n"] or n<=0: raise ValueError("Matriz no consistente con N")
    return {**metrics,"TN":tn,"FP":fp,"FN":fn,"TP":tp,
            "false_positive_rate":fp/(tn+fp) if tn+fp else None,
            "false_negative_rate":fn/(tp+fn) if tp+fn else None,
            "predicted_positive_count":tp+fp,"predicted_positive_proportion":(tp+fp)/n,
            "actual_positive_proportion":(tp+fn)/n,
            "positive_prediction_excess_pp":((tp+fp)/n-(tp+fn)/n)*100,
            "total_errors":fp+fn,"error_rate":(fp+fn)/n}


def descomponer_ba(monthly,global_metrics):
    """Identidad exacta de BA agrupada; no estima incertidumbre ni efecto causal."""
    result={}; months=list(monthly); j=len(months)
    for name in METHODS:
        rows=[monthly[m][name] for m in months]
        if any(r["recall"] is None or r["specificity"] is None for r in rows):
            result[name]={"estimable":False,"reason":"Hay meses sin ambas clases; la identidad macro completa no se aplica"}
            continue
        positives=sum(r["support"]["1"] for r in rows); negatives=sum(r["support"]["0"] for r in rows)
        terms=[]
        for month,r in zip(months,rows):
            wp,wn=r["support"]["1"]/positives,r["support"]["0"]/negatives
            terms.append({"month":month,"positive_support":r["support"]["1"],"negative_support":r["support"]["0"],
                          "weight_for_recall":wp,"weight_for_specificity":wn,"uniform_month_weight":1/j,
                          "recall":r["recall"],"specificity":r["specificity"],
                          "gap_contribution_pp":.5*((wp-1/j)*r["recall"]+(wn-1/j)*r["specificity"])*100})
        pooled_recall=sum(t["weight_for_recall"]*t["recall"] for t in terms)
        pooled_spec=sum(t["weight_for_specificity"]*t["specificity"] for t in terms)
        pooled=.5*(pooled_recall+pooled_spec); macro=float(np.mean([r["balanced_accuracy"] for r in rows]))
        gap=sum(t["gap_contribution_pp"] for t in terms)
        if not np.isclose(pooled,global_metrics[name]["balanced_accuracy"],atol=1e-14,rtol=0) or not np.isclose(gap,(pooled-macro)*100,atol=1e-12,rtol=0):
            raise AssertionError("No se verifica identidad de ponderaciones por clase")
        result[name]={"estimable":True,"pooled_recall":pooled_recall,"pooled_specificity":pooled_spec,
                      "pooled_BA":pooled,"unweighted_monthly_BA":macro,"gap_pp":gap,"terms":terms}
    return result


def sensibilidad_temporal(pred):
    """Cinco omisiones sistemáticas de bloques mensuales; no intervalos ni selección."""
    results={}
    for month in sorted(pred.mes_objetivo.unique()):
        part=pred.loc[pred.mes_objetivo!=month]
        results[month]={"omitted_month":month,"retained_months":sorted(part.mes_objetivo.unique()),"n":len(part),
                        "methods":{n:metricas_con_conteos(part.y_real.to_numpy(),part[f"pred_{n}"].to_numpy()) for n in METHODS}}
    ranges={}
    for name in METHODS:
        values=[r["methods"][name]["balanced_accuracy"] for r in results.values() if r["methods"][name]["balanced_accuracy"] is not None]
        ranges[name]={"min":min(values) if values else None,"max":max(values) if values else None,"estimable_blocks":len(values)}
    return {"method":"leave-one-calendar-month-out sobre predicciones existentes",
            "interpretation":"sensibilidad descriptiva, no intervalo de confianza, p-valor ni nueva evaluación seleccionada",
            "official_full_test_unchanged":True,"blocks":results,
            "BA_ranges":ranges}


def describir_distribuciones(frozen,base=BASE):
    """Solo estadísticos de las 20 entradas existentes, sin fit/transform/predict."""
    base=Path(base); folder=base/"datos/Procesados/fase_03"; names=frozen["dataset"]["feature_names"]
    stats={}; means={}; arrays={}
    for split in ["train","val","test"]:
        x=np.load(folder/f"X_{split}.npy",allow_pickle=False,mmap_mode="r")
        if x.shape!=(frozen["dataset"]["counts"][split],len(names)) or x.dtype!=np.float64 or not np.isfinite(x).all():
            raise ValueError("Matriz descriptiva incompatible")
        arrays[split]=x
        means[split]=np.mean(x,axis=0)
        quantiles=np.quantile(x,[.05,.25,.5,.75,.95],axis=0)
        stats[split]={name:{"n":len(x),"mean":float(means[split][j]),"std_population":float(np.std(x[:,j],ddof=0)),
                            "q05":float(quantiles[0,j]),"q25":float(quantiles[1,j]),"median":float(quantiles[2,j]),
                            "q75":float(quantiles[3,j]),"q95":float(quantiles[4,j])} for j,name in enumerate(names)}
    std_train=np.std(arrays["train"],axis=0,ddof=0)
    if not (std_train>0).all(): raise ValueError("Entrada procesada constante, revisar contrato")
    shifts={name:{"val_minus_train_in_train_std":float((means["val"][j]-means["train"][j])/std_train[j]),
                  "test_minus_train_in_train_std":float((means["test"][j]-means["train"][j])/std_train[j])} for j,name in enumerate(names)}
    raw=pd.read_csv(folder/"observaciones.csv",usecols=["producto_provisional","mes_objetivo","particion","elegible_para_prediccion","respuesta_disponible",
                                                          "unidades_lag_1","unidades_media_3","antiguedad_observada_meses"],float_precision="round_trip")
    raw=raw.loc[raw.elegible_para_prediccion & raw.respuesta_disponible & raw.particion.isin(["train","val","test"])].copy()
    raw["cohorte"]=np.where(raw.producto_provisional.isin(frozen["new_product_rule"]["train_products"]),"conocidos","nuevos")
    monthly={}
    for month,group in raw.groupby("mes_objetivo",sort=True):
        monthly[month]={"partition":str(group.particion.iloc[0]),"n":len(group),
                        "lag1_mean_units":float(group.unidades_lag_1.mean()),"lag1_median_units":float(group.unidades_lag_1.median()),
                        "trailing3_mean_units":float(group.unidades_media_3.mean())}
    cohorts={}
    for name,g in raw[raw.particion.eq("test")].groupby("cohorte",sort=True):
        cohorts[name]={"n":len(g),"lag1_mean_units":float(g.unidades_lag_1.mean()),"lag1_median_units":float(g.unidades_lag_1.median()),
                       "observed_age_min_months":float(g.antiguedad_observada_meses.min()),"observed_age_max_months":float(g.antiguedad_observada_meses.max())}
    return {"feature_names":names,"feature_statistics":stats,"mean_shifts":shifts,"raw_history_by_month":monthly,"raw_history_test_cohorts":cohorts,
            "note":"Cambios descriptivos de X, no test de deriva conceptual, transformación nueva o criterio de selección"}


def resumen_entrenamientos(base=BASE):
    base=Path(base); p4=read_json(base/"resultados/fase_04/experimentos.json"); p5=read_json(base/"resultados/fase_05/experimentos.json")
    histp=pd.read_csv(base/f"resultados/fase_04/historial_{p4['selected_id']}.csv",float_precision="round_trip")
    histl=pd.read_csv(base/f"resultados/fase_05/historial_{p5['selected_id']}.csv",float_precision="round_trip")
    run=next(r for r in p5["runs"] if r["id"]==p5["selected_id"])
    return {"perceptron":{"selected_epoch":70,"epochs_recorded":len(histp),"online_errors_first":int(histp.online_errors.iloc[0]),
                           "online_errors_last":int(histp.online_errors.iloc[-1]),"online_errors_min":int(histp.online_errors.min()),"online_errors_max":int(histp.online_errors.max()),
                           "epochs_zero_updates":int((histp.updates==0).sum()),"epochs_increased_online_errors":int((histp.online_errors.diff()>0).sum())},
            "logistica":{"selected_epoch":200,"epochs_recorded":len(histl),"initial_train_loss":run["initial_train_loss"],
                         "last_train_loss":float(histl.train_loss.iloc[-1]),"last_val_loss":float(histl.val_loss.iloc[-1]),
                         "last_gradient_norm":float(histl.gradient_norm.iloc[-1]),"train_loss_increases_above_1e_minus12":int((histl.train_loss.diff()>1e-12).sum())}}


def run(base=BASE):
    base=Path(base); before=hashes_previos(base)
    frozen,pred,evaluation,agreements=verificar_entradas(base)
    development=read_json(base/"resultados/fase_07/comparacion_con_desarrollo.json")
    global_derived={n:derivar_metricas(m) for n,m in evaluation["global"].items()}
    monthly={month:{n:derivar_metricas(m) for n,m in rows.items()} for month,rows in evaluation["monthly"].items()}
    stages={n:{s:derivar_metricas(m) for s,m in values.items()} for n,values in development.items()}
    gaps={n:{metric:{"test_minus_validation_pp":(values["test"][metric]-values["val"][metric])*100,
                    "test_minus_train_pp":(values["test"][metric]-values["train"][metric])*100} for metric in ["accuracy","precision","recall","specificity","f1","balanced_accuracy"]} for n,values in stages.items()}
    decomposition=descomponer_ba(evaluation["monthly"],evaluation["global"])
    sensitivity=sensibilidad_temporal(pred)
    distributions=describir_distribuciones(frozen,base)
    training=resumen_entrenamientos(base)
    pca=read_json(base/"resultados/fase_06/varianza_explicada.json")
    out=base/"resultados/fase_08"; out.mkdir(parents=True,exist_ok=True)
    values={"metricas_derivadas.json":global_derived,"analisis_temporal.json":{"monthly":monthly,"development_and_test":stages,"gaps_pp":gaps,"BA_decomposition":decomposition},
            "acuerdos_desacuerdos.json":agreements,"sensibilidad_mensual.json":sensitivity,"distribuciones_descriptivas.json":distributions,
            "cohortes_productos.json":evaluation["cohorts"],"resumen_matematico.json":{"training":training,"PCA_existing":pca,"no_retraining":True}}
    for name,value in values.items(): write_json(out/name,value)
    pd.DataFrame([{"method":n,**{k:m[k] for k in ["n","TN","FP","FN","TP","accuracy","precision","recall","specificity","f1","balanced_accuracy","false_positive_rate","false_negative_rate","predicted_positive_count","predicted_positive_proportion"]}} for n,m in global_derived.items()]).to_csv(out/"tabla_metricas.csv",index=False,float_format="%.17g",lineterminator="\n")
    from src.evaluation.graficos_estadisticos import graficar_analisis
    graficar_analisis(global_derived,monthly,agreements,distributions,out/"graficos")
    (out/"reporte_estadistico.md").write_text(reporte(global_derived,monthly,evaluation["cohorts"],decomposition,sensitivity,distributions,training,pca,agreements,gaps,stages),encoding="utf-8")
    (out/"interpretacion_empresarial.md").write_text(interpretacion(global_derived,evaluation["cohorts"],decomposition,gaps),encoding="utf-8")
    after=hashes_previos(base)
    if before!=after: raise RuntimeError("Se modificó un artefacto aprobado")
    files=[base/p for p in ["src/experiments/analisis_estadistico.py","src/evaluation/graficos_estadisticos.py","tests/test_analisis_estadistico.py"]]
    write_json(out/"manifiesto.json",{"phase":8,"scope":"descriptive post-test analysis of existing predictions; no optimization", "n":len(pred),"methods":list(METHODS),
               "metrics_from_saved_predictions_only":True,"new_predictions_generated":False,"models_refit":False,"threshold_changes":False,"PCA_refit":False,
               "integrity":{"preserved":True,"before":before,"after":after},
               "artifacts_sha256":{str(p.relative_to(base)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"manifiesto.json","validaciones.json"}},
               "implementation_sha256":{str(p.relative_to(base)):sha256(p) for p in files},
               "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":__import__("matplotlib").__version__}})
    print({"n":len(pred),"protected_files":len(before),"logistic_BA_gap_global_monthly_pp":decomposition["logistica"]["gap_pp"],
           "logistic_month_omission_BA_range":sensitivity["BA_ranges"]["logistica"],"largest_test_mean_shifts":sorted(distributions["mean_shifts"],key=lambda k:abs(distributions["mean_shifts"][k]["test_minus_train_in_train_std"]),reverse=True)[:3]})
    return global_derived,decomposition


def reporte(g,monthly,cohorts,decomposition,sensitivity,distribution,training,pca,agreements,gaps,stages):
    f=lambda v:"no estimable" if v is None else f"{100*v:.3f} %"
    table="\n".join(f"| {n} | {f(m['accuracy'])} | {f(m['precision'])} | {f(m['recall'])} | {f(m['specificity'])} | {f(m['f1'])} | {f(m['balanced_accuracy'])} | {f(m['false_positive_rate'])} | {f(m['false_negative_rate'])} | {m['predicted_positive_count']}/{m['n']} ({f(m['predicted_positive_proportion'])}) |" for n,m in g.items())
    counts="\n".join(f"| {n} | {m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} |" for n,m in g.items())
    rows="\n".join(f"| {month} | {m['perceptron']['support']['1']}/{m['perceptron']['n']} | "+" | ".join(f"{m[n]['predicted_positive_count']}/{m[n]['n']}" for n in METHODS)+" |" for month,m in monthly.items())
    temporal="\n".join(f"| {month} | {n} | {f(m['balanced_accuracy'])} | {f(m['precision'])} | {f(m['recall'])} | {f(m['specificity'])} | {f(m['false_positive_rate'])} | {f(m['false_negative_rate'])} |" for month,rowsm in monthly.items() for n,m in rowsm.items())
    d=decomposition["logistica"]
    decomp="\n".join(f"| {r['month']} | {r['positive_support']} | {r['negative_support']} | {r['weight_for_recall']:.6f} | {r['weight_for_specificity']:.6f} | {r['recall']:.6f} | {r['specificity']:.6f} | {r['gap_contribution_pp']:+.6f} |" for r in d["terms"])
    top=sorted(distribution["mean_shifts"],key=lambda n:abs(distribution["mean_shifts"][n]["test_minus_train_in_train_std"]),reverse=True)[:6]
    shifts="\n".join(f"| {n} | {distribution['mean_shifts'][n]['val_minus_train_in_train_std']:+.6f} | {distribution['mean_shifts'][n]['test_minus_train_in_train_std']:+.6f} |" for n in top)
    cohort="\n".join(f"| {c} | {n} | {m['n']} | {m['support']['0']}/{m['support']['1']} | {m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} | {f(m['balanced_accuracy'])} |" for c,values in cohorts.items() for n,m in values["methods"].items())
    sens="\n".join(f"| {month} | {r['n']} | "+" | ".join(f(r["methods"][n]["balanced_accuracy"]) for n in METHODS)+" |" for month,r in sensitivity["blocks"].items())
    agreement=agreements["global"]
    stage_rows="\n".join(f"| {s} | {stages['perceptron'][s]['n']} | {f(stages['perceptron'][s]['actual_positive_proportion'])} | "+" | ".join(f(stages[n][s]['predicted_positive_proportion']) for n in METHODS)+" |" for s in ["train","val","test"])
    return f"""# Análisis estadístico y generalización — Fase 8

## Método, integridad y alcance

Se verificaron manifiestos/hashes de fases 4–7, datos preparados, fuentes y configuraciones congeladas. Las 1.111 filas de predicciones_alineadas.csv se alinearon con IDs, producto y mes de los metadatos originales; y_real coincide con y_test. Se recalcularon métricas únicamente desde las predicciones guardadas, reproduciendo exactamente las matrices, métricas globales/mensuales/cohortes y acuerdos de fase 7. No se cargan modelos para generar nuevas predicciones, no se ejecuta la evaluación otra vez ni se cambian etiquetas, umbrales, características o PCA.

Se reutilizan resultados train/val guardados y se describen las 20 características existentes de las tres particiones. Sus resúmenes no son parámetros de un nuevo escalador ni selección de variables. Figuras equivalentes (BA global, confusiones, BA mensual, comparación train/val/test) se referencian en resultados/fase_07/graficos; no se duplican.

## Tabla definitiva de test

Soporte real común: clase 0={g['logistica']['support']['0']}, clase 1={g['logistica']['support']['1']}. Sensibilidad=recall positivo, especificidad=TN/(TN+FP); FPR=FP/(TN+FP), FNR=FN/(TP+FN). Proporción positiva=(TP+FP)/N. Denominadores cero mantienen las convenciones de fase 2: tasas no estimables se indican, sin inventar una clase faltante.

| Método | Accuracy | Precision | Sensibilidad | Especificidad | F1 | BA | FPR | FNR | Predicciones positivas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{table}

| Método | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
{counts}

La logística tiene la mayor BA global observada: {f(g['logistica']['balanced_accuracy'])}, {100*(g['logistica']['balanced_accuracy']-g['perceptron']['balanced_accuracy']):.3f} puntos por encima del perceptrón, {100*(g['logistica']['balanced_accuracy']-g['mayoritario']['balanced_accuracy']):.3f} sobre mayoritario y {100*(g['logistica']['balanced_accuracy']-g['historico']['balanced_accuracy']):.3f} sobre histórico. Son diferencias descriptivas de esta reserva, no significación ni superioridad universal.

El exceso de positivos es visible: perceptrón {g['perceptron']['predicted_positive_count']} y logística {g['logistica']['predicted_positive_count']} frente a 553 positivos reales. Predecir 1 con frecuencia aumenta la oportunidad de capturar positivos, pero también clasifica muchos negativos como positivos. El caso extremo es mayoritario: sensibilidad 100 %, especificidad 0 %, BA 50 %. Un F1 positivo alto no asegura reconocimiento de ambas clases. Entre los clasificadores, logística ofrece mayor especificidad y BA, aunque conserva 370 falsos positivos y 49 falsos negativos; perceptrón reduce FN a 22 a costa de 490 FP. Histórico tiene mayor especificidad, pero sensibilidad baja; no ofrece el mejor balance global.

## Generalización temporal

Cambios test−validación de BA: perceptrón {gaps['perceptron']['balanced_accuracy']['test_minus_validation_pp']:+.3f} pp; logística {gaps['logistica']['balanced_accuracy']['test_minus_validation_pp']:+.3f} pp. Todos los cambios train/validación/test para accuracy, precision, recall, specificity, F1 y BA están en analisis_temporal.json. Validación se usó para selección; test es evaluación final ya consumida. No se reajusta el diseño para cambiar estas cifras.

| Conjunto | N | Positivos reales | Pred. positivas perceptrón | Logística | Mayoritario | Histórico |
|---|---:|---:|---:|---:|---:|---:|
{stage_rows}

| Mes | Positivos reales | Pred. positivas perceptrón | Logística | Mayoritario | Histórico |
|---|---:|---:|---:|---:|---:|
{rows}

| Mes | Método | BA | Precision | Recall | Specificity | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
{temporal}

Se reportan los cinco meses completos, también los difíciles. Predicciones y proporciones de clase se examinan conjuntamente: un agregado no equivale al comportamiento de cada mes.

### Identidad matemática: BA global y promedio mensual

Sea P_t=TP_t+FN_t y N_t=TN_t+FP_t. TPR_global=SUM_t[P_t/SUM(P)]TPR_t; TNR_global=SUM_t[N_t/SUM(N)]TNR_t. Por tanto BA_global=0,5(TPR_global+TNR_global), mientras BA_macro=SUM_t BA_t/5 asigna el mismo peso a cada mes y clase.

BA_global−BA_macro = 0,5 SUM_t[(P_t/SUM(P)−1/5)TPR_t+(N_t/SUM(N)−1/5)TNR_t]. No es la simple ponderación de BA mensual por número total de filas; hay dos distribuciones de pesos por clase. La identidad se verificó numéricamente con atol=1e-14 en BA y 1e-12 en la suma de contribuciones expresada en puntos porcentuales.

En logística: TPR agrupado={d['pooled_recall']:.9f}, TNR agrupado={d['pooled_specificity']:.9f}, BA global={f(d['pooled_BA'])}; promedio mensual={f(d['unweighted_monthly_BA'])}. Diferencia={d['gap_pp']:.6f} pp, explicada exactamente por estas contribuciones:

| Mes | Positivos reales | Negativos reales | Peso TPR | Peso TNR | TPR | TNR | Aporte a diferencia (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
{decomp}

No hay contradicción entre 62,42 % global y 58,41 % mensual: son resúmenes distintos. Las fórmulas también se comprobaron para los otros tres métodos.

## Distribuciones observadas y posibles explicaciones

distribuciones_descriptivas.json registra media, desviación poblacional y cuantiles 5/25/50/75/95 de las 20 columnas procesadas en train/val/test, y unidades históricas por mes y cohorte. No se calcula una nueva transformación. Aquí se muestran las seis mayores diferencias de media test−train, ordenadas solo para interpretación, sin retirar o elegir características para los modelos:

| Característica ya preparada | Media val−train / std train | Media test−train / std train |
|---|---:|---:|
{shifts}

La antigüedad observada crece por definición del tiempo: la diferencia media test−train es {distribution['mean_shifts']['antiguedad_observada_meses']['test_minus_train_in_train_std']:.6f} desviaciones de train. Los conocidos tienen 36–40 meses observados en test y los nuevos 3–4. Los cambios de seno/coseno del mes reflejan también que test incluye solo enero-mayo, mientras validación cubre un año completo y train 21 meses elegibles: no se comparan calendarios idénticos. La mediana de unidades_lag_1 en test es {distribution['raw_history_test_cohorts']['conocidos']['lag1_median_units']:.3f} entre conocidos y {distribution['raw_history_test_cohorts']['nuevos']['lag1_median_units']:.3f} entre nuevos, de distintas escalas comerciales.

Las diferencias de medias/dispersiones son cambio observado en X y la mezcla de población. No demuestran que haya cambiado P(y|X), ni establecen deriva de concepto, causalidad o un fallo de cobertura. Hipótesis alternativas: extrapolación temporal de una frontera lineal, patrones de ventas cambiantes, muestra breve o problemas no demostrados de identidad/unidades/completitud. Los CSV no permiten discriminar causalmente esas explicaciones. No se modifica ninguna entrada tras este análisis.

## Acuerdos y desacuerdos

Ambos aciertan {agreement['both_correct']}, ambos fallan {agreement['both_wrong']}; solo acierta perceptrón {agreement['only_perceptron_correct']}, solo logística {agreement['only_logistic_correct']}. Las cuatro categorías son excluyentes y suman 1.111. Coinciden en {agreement['agreement']} y discrepan en {agreement['disagreement']}; acuerdo no equivale a acierto. Los detalles por los cinco meses y las dos cohortes se conservan en acuerdos_desacuerdos.json y en una figura apilada. La diferencia entre aciertos exclusivos contribuye a la diferencia de accuracy, pero no autoriza un ensemble ni inferencia pareada iid ignorando dependencia.

## Productos nuevos: conteos, no solo porcentajes

| Cohorte | Método | N | Soporte 0/1 | TN | FP | FN | TP | BA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{cohort}

Se preservan los 16 casos de ocho nombres nuevos en el global. En perceptrón, precision=100 % procede de solo dos predicciones positivas correctas, con ocho FN: no es éxito general sobre nuevos productos. Logística reconoce siete de diez positivos y cuatro de seis negativos; baseline mayoritario reconoce todos los positivos y ningún negativo. No se puede extrapolar estabilidad con 16 observaciones de dos meses.

## Incertidumbre y sensibilidad agrupada

No se consideran 1.111 réplicas independientes: se repiten productos, se solapan ventanas y existen condiciones compartidas de cada mes. No se presentan p-valores, intervalos de confianza ni bootstrap iid. Solo cinco bloques temporales informan la incertidumbre temporal.

Como exploración sistemática se calcula el agregado reteniendo cuatro meses y omitiendo cada uno por turno. Se reutilizan las mismas predicciones, sin entrenar o elegir omisiones. El resultado oficial sigue siendo el conjunto completo. Estos extremos no constituyen intervalo de confianza ni una nueva evaluación elegida por rendimiento.

| Mes omitido solo para sensibilidad | N retenido | BA perceptrón | BA logística | BA mayoritario | BA histórico |
|---|---:|---:|---:|---:|---:|
{sens}

## Interpretación matemática y PCA

Perceptrón: z=wᵀx+b, escalón 1[z>=0], actualización online w←w+η(y−pred)x y b←b+η(y−pred), solo ante error. La ejecución guardada tiene {training['perceptron']['epochs_recorded']} épocas, errores online entre {training['perceptron']['online_errors_min']} y {training['perceptron']['online_errors_max']}, sin época de cero actualizaciones. Pesos del checkpoint 70; online-errors y error al finalizar la época son distintos. Oscilaciones no son prueba automática de implementación errónea ni de no separabilidad.

Logística: p=sigmoid(z), entropía cruzada con logaddexp y gradientes analíticos Xᵀ(p−y)/m y media(p−y), descenso por lote completo. Pérdida train inicial {training['logistica']['initial_train_loss']:.9f}, final {training['logistica']['last_train_loss']:.9f}; norma gradiente final {training['logistica']['last_gradient_norm']:.9f}. La sigmoidea produce probabilidades no necesariamente calibradas. Umbral congelado 0,5: frontera z=0, también un hiperplano en las 20 entradas transformadas. El objetivo suave, lote completo, trayectoria y selección de checkpoint difieren de la actualización por errores del perceptrón; compartir familia de fronteras lineales no obliga a obtener los mismos pesos o resultados. No se retoman épocas ni se recalibra tras test.

PCA existente: PC1={f(pca['PC1'])}, PC2={f(pca['PC2'])}, conservada={f(pca['retained_2D'])}, no representada={f(pca['discarded'])}. Se reutilizan exclusivamente parámetros/gráficos de fase 6, ajustados en train, sin PCA nuevo sobre test. El solapamiento visual describe esa proyección no supervisada y la pérdida de más de la mitad de la varianza limita su lectura; no demuestra la causa de los errores ni una frontera exacta del clasificador multidimensional. Correlaciones y redundancias impiden interpretar un coeficiente como causa empresarial independiente.

## Reproducibilidad y material académico

Nuevos archivos: src/experiments/analisis_estadistico.py, src/evaluation/graficos_estadisticos.py, tests/test_analisis_estadistico.py y resultados/fase_08/. Las figuras adicionales muestran sensibilidad/especificidad, errores mensuales, predicciones positivas, categorías de acuerdo y cambios de medias de X; no duplican las confusiones ni BA global de fase 7. Este Markdown y interpretacion_empresarial.md aportan conclusiones trasladables a LaTeX; no se genera aún el informe definitivo.

```powershell
python -m src.experiments.analisis_estadistico
python -m unittest discover -s tests -p 'test_analisis_estadistico.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

Manifiesto conserva hashes previos/posteriores de fuentes, matrices, modelos y fases 1–7; validaciones.json registra pruebas, reproducción y revisión visual. El runner no llama predict ni fit de modelos, PCA o preprocesador. Las pruebas históricas ejecutan fixtures sintéticos y preparación temporal aislada sin sustituir artefactos aprobados. No se detectaron inconsistencias demostrables de claves/métricas. Permanecen sin confirmar linaje de identidad, unidad comercial y cierre real. La fase 8 termina con interpretación, sin modificar el experimento ni avanzar a fase 9.
"""


def interpretacion(g,cohorts,decomposition,gaps):
    return f"""# Interpretación empresarial — evidencia de fase 8

## Pregunta que realmente se respondió

Al cierre del mes anterior, clasificar si las unidades facturadas observadas de un producto superarán la mediana de sus tres meses previos. No se predijo rentabilidad, inventario, demanda no atendida, número exacto de unidades futuras ni impacto económico de una acción comercial.

## Respuestas sustentadas para el informe académico

1. **Rendimiento global observado:** logística obtuvo BA {g['logistica']['balanced_accuracy']*100:.3f} %, perceptrón {g['perceptron']['balanced_accuracy']*100:.3f} %, mayoritario 50 % e histórico {g['historico']['balanced_accuracy']*100:.3f} %. La logística encabeza esta comparación descriptiva sobre la misma reserva; no demuestra superioridad universal o significación estadística.
2. **Equilibrio entre clases:** entre los clasificadores, logística conserva sensibilidad {g['logistica']['recall']*100:.3f} % y especificidad {g['logistica']['specificity']*100:.3f} %, frente a {g['perceptron']['recall']*100:.3f} % y {g['perceptron']['specificity']*100:.3f} % del perceptrón. Ambos alertan en exceso; el histórico reconoce más negativos pero pierde muchos positivos. La BA, matrices y soportes aportan más que accuracy/recall aislados.
3. **Diferencia frente a baselines:** logística supera BA mayoritaria en {(g['logistica']['balanced_accuracy']-.5)*100:.3f} pp e histórica en {(g['logistica']['balanced_accuracy']-g['historico']['balanced_accuracy'])*100:.3f} pp; perceptrón las supera en {(g['perceptron']['balanced_accuracy']-.5)*100:.3f} y {(g['perceptron']['balanced_accuracy']-g['historico']['balanced_accuracy'])*100:.3f} pp. Son mejoras observadas, no ahorro o retorno de inversión demostrado.
4. **Errores que limitan utilidad:** perceptrón tiene 490 FP/22 FN y logística 370 FP/49 FN. FP puede motivar una revisión comercial innecesaria y FN omitir una posible subida; no se inventan costos ni se afirma que FP represente sobrestock real. El mayoritario alerta siempre y logra recall 100 % sin reconocer ningún negativo: alta sensibilidad no basta.
5. **Generalización temporal:** BA test−validación del perceptrón es {gaps['perceptron']['balanced_accuracy']['test_minus_validation_pp']:+.3f} pp y de logística {gaps['logistica']['balanced_accuracy']['test_minus_validation_pp']:+.3f} pp. La BA global logística {decomposition['logistica']['pooled_BA']*100:.3f} % difiere de su media mensual {decomposition['logistica']['unweighted_monthly_BA']*100:.3f} % por ponderaciones de clases. Cinco meses no acreditan estabilidad futura; cambios de X observados no prueban concept drift.
6. **Uso productivo inmediato:** la evidencia no basta para recomendar decisiones automáticas de abastecimiento o comerciales. Hay falsa alarma elevada, solo cinco meses de test y 16 filas nuevas, probabilidades sin calibración validada, cobertura asumida e identidad por nombre. El umbral de alta relativa puede separar diferencias de una unidad sin relevancia económica aprobada. Esto no invalida el ejercicio académico; limita las afirmaciones de utilidad operacional.
7. **Datos adicionales para interpretar:** StockItemID/CustomerID y catálogos con vigencia/renombres; unidad y empaque consistentes; registros de cierre/ingestión que prueben disponibilidad al corte; promociones y precios planificados; stockout, inventario y demanda no atendida; decisiones/acciones registradas y costos de FP/FN validados por el negocio. Es información faltante, no valores deducibles del CSV.

## Hechos frente a hipótesis

Comprobado: diferencias de métricas, distribuciones descriptivas de X, frecuencias de predicción, errores y desacuerdos de los mismos casos. Hipótesis no demostradas: cambio de concepto, causa comercial de los errores, que la antigüedad observada explique por sí sola el exceso de alertas, o que problemas de cobertura/identidad sean la causa del rendimiento. No se confunde una limitación lineal posible con un fallo de implementación demostrado.

No se propone reentrenar, ajustar umbrales o combinar clasificadores sobre este test. Cualquier investigación posterior requiere otra versión y una evaluación independiente. Los resultados finales y todos los artefactos congelados se conservan como evidencia de esta versión. No se avanza automáticamente a fase 9.
"""


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
