"""python -m src.experiments.evaluacion_final — evaluación congelada, sin fit."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import uuid

import numpy as np
import pandas as pd

from src.auditar_datos import BASE, sha256
from src.evaluation.metricas import metricas_binarias, validar_etiquetas
from src.experiments.entrenar_perceptron import load_development, read_json, write_json
from src.experiments.entrenar_logistica import evaluar_logistica
from src.experiments.visualizar_pca import hashes_protegidos, verificar_manifiestos
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual
from src.pipeline.caracteristicas import observation_id

METHODS=("perceptron","logistica","mayoritario","historico")
METRICS=("accuracy","precision","recall","f1","balanced_accuracy")
EXPECTED_MONTHS={"2016-01":219,"2016-02":219,"2016-03":219,"2016-04":227,"2016-05":227}


def ahora():
    return datetime.now(timezone(timedelta(hours=-5),name="America/Guayaquil")).isoformat(timespec="microseconds")


def evento(path, run_id, stage, **details):
    record={"run_id":run_id,"timestamp":ahora(),"stage":stage,**details}
    with Path(path).open("a",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def hashes_anteriores(base=BASE):
    base=Path(base); result=hashes_protegidos(base)
    previous=read_json(base/"resultados/fase_06/manifiesto.json")
    for relative in previous["implementation_sha256"]:
        result[relative]=sha256(base/relative)
    for p in sorted((base/"resultados/fase_06").rglob("*")):
        if p.is_file(): result[str(p.relative_to(base))]=sha256(p)
    return result


def verificar_congelacion(base=BASE):
    base=Path(base)
    verificar_manifiestos(base)  # Fases 3–5: hashes; no arrays de test.
    folder=base/"resultados/fase_06"
    manifest=read_json(folder/"manifiesto.json")
    validation=read_json(folder/"validaciones.json")
    if not validation["successful"] or validation["phase06_manifest_sha256"]!=sha256(folder/"manifiesto.json"):
        raise ValueError("Integridad de fase 6 no coincide; no abrir respuestas")
    for field in ["artifacts_sha256","implementation_sha256"]:
        for relative,expected in manifest[field].items():
            if sha256(base/relative)!=expected:
                raise ValueError(f"Artefacto congelado alterado: {relative}; no abrir respuestas")
    for relative,expected in manifest["integrity"]["after"].items():
        if sha256(base/relative)!=expected:
            raise ValueError(f"Archivo protegido alterado: {relative}; no abrir respuestas")
    freeze_path=folder/"congelacion_experimental.json"
    if sha256(freeze_path)!=manifest["freeze_sha256"]:
        raise ValueError("Hash de congelación incorrecto; no abrir respuestas")
    frozen=read_json(freeze_path)
    if frozen["freeze_version"]!=1 or frozen["contract_version"]!="2.0" or frozen["evaluation"]["primary_metric"]!="balanced_accuracy" or frozen["evaluation"]["PCA_used_by_classifiers"]:
        raise ValueError("Versión/reglas congeladas incompatibles")
    data=frozen["dataset"]
    if sha256(base/data["phase03_manifest"])!=data["phase03_manifest_sha256"]:
        raise ValueError("Manifiesto de dataset alterado")
    for field in ["files_sha256","contracts_sha256"]:
        for relative,expected in data[field].items():
            if sha256(base/relative)!=expected:
                raise ValueError(f"Datos/contrato alterados: {relative}; no abrir respuestas")
    if sha256(base/data["source"])!=data["source_sha256"]:
        raise ValueError("Fuente original alterada")
    return frozen,sha256(freeze_path)


def cargar_modelos(frozen,base=BASE):
    models={}
    for name,cls in [("perceptron",PerceptronSimple),("logistica",LogisticRegressionManual)]:
        record=frozen["models"][name]; path=Path(base)/record["file"]
        if sha256(path)!=record["sha256"] or read_json(path)!=record["state"]:
            raise ValueError(f"Modelo {name} no coincide con congelación")
        model=cls.load(path)
        if model.feature_names_!=frozen["dataset"]["feature_names"] or len(model.weights_)!=20:
            raise ValueError("Dimensión/orden de pesos incorrecto")
        models[name]=model
    if models["perceptron"].best_epoch_!=70 or models["logistica"].best_epoch_!=200 or models["logistica"].threshold!=.5:
        raise ValueError("Checkpoint/umbral incorrecto")
    return models


def reproducir_validacion(models,dev,frozen):
    reproduced={}
    for name,model in models.items():
        if model.feature_names_!=dev["feature_names"]:
            raise ValueError("Columnas de validación no coinciden")
        if name=="logistica":
            actual=evaluar_logistica(model,dev)
        else:
            actual={s:metricas_binarias(dev[s]["y"],model.predict(dev[s]["X"],feature_names=dev["feature_names"])) for s in ["train","val"]}
        if actual!=frozen["models"][name]["development_metrics"]:
            raise ValueError(f"No se reproduce desarrollo de {name}; no abrir test")
        reproduced[name]=actual
    return reproduced


def validar_metadatos(meta,n,frozen):
    if len(meta)!=n or meta.observacion_id.duplicated().any() or not np.array_equal(meta.indice_matriz,np.arange(n)):
        raise ValueError("Alineación/IDs de test incorrectos")
    if meta.observacion_id.tolist()!=meta.sort_values(["mes_objetivo","producto_provisional"],kind="stable").observacion_id.tolist():
        raise ValueError("Orden de filas incorrecto")
    if not meta.particion.eq("test").all() or not meta.version_contrato.astype(str).eq("2.0").all() or not meta.fuente_sha256.eq(frozen["dataset"]["source_sha256"]).all():
        raise ValueError("Linaje de test incorrecto")
    if not meta.cobertura_objetivo.eq("completa").all():
        raise ValueError("Cobertura no confirmada bajo la política congelada; no excluir casos")
    if meta.mes_objetivo.value_counts().sort_index().to_dict()!=EXPECTED_MONTHS:
        raise ValueError("Conteos mensuales distintos del protocolo; investigar antes de abrir y")
    for row in meta.itertuples():
        period=pd.Period(row.mes_objetivo,freq="M")
        prior="|".join(str(period-j) for j in [1,2,3])
        if row.observacion_id!=observation_id(row.producto_provisional,row.mes_objetivo) or row.corte_exclusivo!=f"{row.mes_objetivo}-01T00:00:00Z" or row.historia_meses!=prior:
            raise ValueError("Clave/corte/ventana de historia incorrectos")
        if period.ordinal-pd.Period(row.primer_mes_observado_conocido,freq="M").ordinal<3:
            raise ValueError("Historia insuficiente en test")
        if any(s not in {"ventas_observadas","mes_completo_sin_ventas"} for s in row.historia_estados.split("|")):
            raise ValueError("Historia desconocida en test")
    train=set(frozen["new_product_rule"]["train_products"])
    known=meta.producto_provisional.isin(train)
    if int(known.sum())!=1095 or int((~known).sum())!=frozen["new_product_rule"]["expected_test_new_rows"] or meta.loc[~known,"producto_provisional"].nunique()!=8 or meta.loc[known,"producto_provisional"].nunique()!=219:
        raise ValueError("Cohortes distintas del protocolo; no excluir productos")
    if not meta.loc[~known,"mes_objetivo"].isin(["2016-04","2016-05"]).all():
        raise ValueError("Meses de productos nuevos incorrectos")


def cargar_historia(meta,base=BASE):
    # Solo predictores aprobados: no se lee Q_objetivo ni y de observaciones.csv.
    columns=["observacion_id","unidades_lag_1","unidades_lag_2","unidades_lag_3","R_referencia"]
    history=pd.read_csv(Path(base)/"datos/Procesados/fase_03/observaciones.csv",usecols=columns,dtype=str,keep_default_na=False)
    if history.observacion_id.duplicated().any(): raise ValueError("Historia con IDs duplicados")
    indexed=history.set_index("observacion_id")
    if not set(meta.observacion_id)<=set(indexed.index): raise ValueError("Faltan IDs históricos")
    selected=indexed.loc[meta.observacion_id].apply(pd.to_numeric,errors="raise")
    if not np.isfinite(selected.to_numpy()).all(): raise ValueError("Historia no finita")
    if not np.array_equal(np.median(selected[["unidades_lag_1","unidades_lag_2","unidades_lag_3"]].to_numpy(),axis=1),selected.R_referencia.to_numpy()):
        raise ValueError("Referencia histórica no coincide con mediana aprobada")
    return selected


def preflight(base=BASE):
    """Toda comprobación crítica ocurre aquí, ANTES de cargar y_test.npy."""
    base=Path(base)
    frozen,freeze_hash=verificar_congelacion(base)
    models=cargar_modelos(frozen,base)
    dev=load_development(base)
    reproduced=reproducir_validacion(models,dev,frozen)
    if dev["manifest_hash"]!=frozen["dataset"]["phase03_manifest_sha256"]:
        raise ValueError("Dataset de desarrollo distinto del congelado")
    train_names=sorted(dev["train"]["meta"].producto_provisional.unique().tolist())
    pos=int((dev["train"]["y"]==1).sum()); neg=int((dev["train"]["y"]==0).sum())
    baseline=frozen["baseline_rules"]
    if train_names!=frozen["new_product_rule"]["train_products"] or pos!=baseline["train_positive"] or neg!=baseline["train_negative"] or int(pos>neg)!=baseline["majority_class"]:
        raise ValueError("Mayoría/cohorte de train no coincide con congelación")
    folder=base/"datos/Procesados/fase_03"
    if read_json(folder/"preprocesador.json")!=frozen["dataset"]["preprocessing_parameters"] or read_json(folder/"caracteristicas.json")["feature_names"]!=dev["feature_names"]:
        raise ValueError("Transformaciones o columnas alteradas")
    x=np.load(folder/"X_test.npy",allow_pickle=False)
    if x.dtype!=np.float64 or x.shape!=(frozen["dataset"]["counts"]["test"],20) or not np.isfinite(x).all():
        raise ValueError("X_test no válido; no abrir y")
    meta=pd.read_csv(folder/"metadatos_test.csv",keep_default_na=False)
    validar_metadatos(meta,len(x),frozen)
    if set(meta.observacion_id)&(set(dev["train"]["meta"].observacion_id)|set(dev["val"]["meta"].observacion_id)):
        raise ValueError("Claves compartidas entre test y desarrollo")
    history=cargar_historia(meta,base)
    verification={"integrity":"approved","freeze_sha256":freeze_hash,"contract_version":"2.0",
                  "n":len(x),"n_features":x.shape[1],"dtype_X":str(x.dtype),"validation_reproduced":reproduced,
                  "models_sha256":{n:r["sha256"] for n,r in frozen["models"].items()},
                  "matrix_sha256":{str(p.relative_to(base)):sha256(p) for p in sorted(folder.glob("*.npy"))},
                  "baseline_majority_verified_against_train":baseline["majority_class"],
                  "cohort_rows":{"conocidos":1095,"nuevos":16},"test_responses_loaded":False}
    return {"frozen":frozen,"freeze_hash":freeze_hash,"models":models,"X":x,"meta":meta,"history":history,"verification":verification}


def cargar_respuestas(base,prepared):
    y=np.load(Path(base)/"datos/Procesados/fase_03/y_test.npy",allow_pickle=False)
    if y.dtype!=np.int8 or y.shape!=(len(prepared["X"]),): raise ValueError("Respuestas de test incompatibles")
    return validar_etiquetas(y,n=len(prepared["X"]))


def predecir(models,x,history,majority,names):
    if majority not in [0,1] or len(history)!=len(x): raise ValueError("Baseline no alineado")
    score=models["perceptron"].decision_function(x,feature_names=names)
    probability=models["logistica"].predict_proba(x,feature_names=names)
    predictions={"perceptron":models["perceptron"].predict(x,feature_names=names),
                 "logistica":models["logistica"].predict(x,feature_names=names),
                 "mayoritario":np.full(len(x),majority,dtype=np.int8),
                 "historico":(history.unidades_lag_1.to_numpy()>history.R_referencia.to_numpy()).astype(np.int8)}
    if not np.array_equal(predictions["perceptron"],(score>=0).astype(np.int8)) or not np.array_equal(predictions["logistica"],(probability>=.5).astype(np.int8)):
        raise ValueError("Reglas de predicción no coinciden con las congeladas")
    return predictions,score,probability


def metricas_con_conteos(y,pred):
    if not len(y): return {"n":0,"metrics":None,"reason":"grupo_sin_observaciones"}
    m=metricas_binarias(y,pred); c=m["confusion_matrix"]
    m.update(TN=c[0][0],FP=c[0][1],FN=c[1][0],TP=c[1][1])
    if m["TN"]+m["FP"]+m["FN"]+m["TP"]!=m["n"]: raise AssertionError("Matriz no suma N")
    return m


def evaluar_comun(y,predictions,meta,train_products):
    y=validar_etiquetas(y,n=len(meta))
    if tuple(predictions)!=METHODS or meta.observacion_id.duplicated().any(): raise ValueError("Métodos/IDs no válidos")
    for name in METHODS: predictions[name]=validar_etiquetas(predictions[name],n=len(y))
    def subset(mask):
        mask=np.asarray(mask,dtype=bool)
        return {n:metricas_con_conteos(y[mask],predictions[n][mask]) for n in METHODS}
    global_metrics=subset(np.ones(len(y),dtype=bool))
    monthly={m:subset(meta.mes_objetivo.eq(m).to_numpy()) for m in sorted(meta.mes_objetivo.unique())}
    known=meta.producto_provisional.isin(set(train_products)).to_numpy()
    masks={"conocidos":known,"nuevos":~known}
    cohorts={c:{"rows":int(mask.sum()),"products":int(meta.loc[mask,"producto_provisional"].nunique()),"methods":subset(mask)} for c,mask in masks.items()}
    cross={c:{month:subset(mask & meta.mes_objetivo.eq(month).to_numpy()) for month in monthly} for c,mask in masks.items()}
    means={}
    for name in METHODS:
        means[name]={}
        for key in METRICS:
            values=[m[name][key] for m in monthly.values() if m[name][key] is not None]
            means[name][key]={"mean":float(np.mean(values)) if values else None,"months_estimable":len(values)}
    return {"global":global_metrics,"monthly":monthly,"monthly_unweighted_means":means,"cohorts":cohorts,"cohort_month":cross}


def analizar_errores(y,predictions,meta,train_products):
    a,b=predictions["perceptron"],predictions["logistica"]
    correct_a=a==y; correct_b=b==y
    def counts(mask):
        mask=np.asarray(mask,dtype=bool)
        return {"n":int(mask.sum()),"agreement":int(((a==b)&mask).sum()),"disagreement":int(((a!=b)&mask).sum()),
                "both_correct":int((correct_a&correct_b&mask).sum()),"both_wrong":int((~correct_a&~correct_b&mask).sum()),
                "only_perceptron_correct":int((correct_a&~correct_b&mask).sum()),
                "only_logistic_correct":int((~correct_a&correct_b&mask).sum())}
    result={"global":counts(np.ones(len(y),dtype=bool)),"monthly":{m:counts(meta.mes_objetivo.eq(m)) for m in sorted(meta.mes_objetivo.unique())},
            "cohorts":{},"errors_by_method":{}}
    known=meta.producto_provisional.isin(set(train_products))
    result["cohorts"]={"conocidos":counts(known),"nuevos":counts(~known)}
    for name,pred in predictions.items():
        result["errors_by_method"][name]={"total":int((pred!=y).sum()),"correct":int((pred==y).sum()),
            "monthly":{m:{"errors":int(((pred!=y)&meta.mes_objetivo.eq(m).to_numpy()).sum()),"n":int(meta.mes_objetivo.eq(m).sum()),
                           "error_rate":float((pred[meta.mes_objetivo.eq(m).to_numpy()]!=y[meta.mes_objetivo.eq(m).to_numpy()]).mean())} for m in sorted(meta.mes_objetivo.unique())}}
    return result


def run(base=BASE,output_dir=None):
    base=Path(base); out=Path(output_dir).resolve() if output_dir is not None else base/"resultados/fase_07"
    if not out.is_relative_to(base): raise ValueError("Destino fuera del workspace")
    out.mkdir(parents=True,exist_ok=True)
    run_id=uuid.uuid4().hex; log=out/"registro_ejecuciones.jsonl"
    evento(log,run_id,"inicio_intento",python=platform.python_version(),numpy=np.__version__,purpose="evaluacion_final_congelada_sin_optimizacion",implementation_sha256=sha256(Path(__file__)))
    opened=False
    try:
        before=hashes_anteriores(base)
        prepared=preflight(base)
        frozen=prepared["frozen"]
        write_json(out/"verificacion_previa.json",prepared["verification"])
        opening=out/"registro_apertura_test.json"
        first=not opening.exists()
        if first:
            record={"run_id":run_id,"timestamp":ahora(),"timezone":"America/Guayaquil (UTC-05:00)",
                    "declaration":"Primera apertura autorizada para evaluación predictiva final; sin optimización posterior",
                    "python":platform.python_version(),"numpy":np.__version__,"freeze_sha256":prepared["freeze_hash"],
                    "models_sha256":prepared["verification"]["models_sha256"],"matrix_sha256":prepared["verification"]["matrix_sha256"],
                    "contract_version":"2.0","primary_metric":"balanced_accuracy","metrics":list(METRICS),
                    "decision_rules":{"perceptron":"z>=0","logistica":"probabilidad>=0.5"},
                    "threshold_logistic":.5,"integrity":"approved_before_test_labels", "validation_reproduced":True}
            # Registro persistido ANTES de cargar las respuestas; no se sobreescribe en repeticiones.
            with opening.open("x",encoding="utf-8") as stream:
                json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
        else:
            if read_json(opening)["freeze_sha256"]!=prepared["freeze_hash"]:
                raise ValueError("La apertura anterior pertenece a otra congelación; detener")
            previous_manifest=out/"manifiesto.json"
            if previous_manifest.exists() and read_json(previous_manifest)["opening_record_sha256"]!=sha256(opening):
                raise ValueError("El registro original de apertura fue alterado; detener")
        evento(log,run_id,"integridad_y_validacion_aprobadas_antes_de_y_test",first_predictive_opening=first,
               freeze_sha256=prepared["freeze_hash"],opening_record_sha256=sha256(opening))
        y=cargar_respuestas(base,prepared); opened=True
        evento(log,run_id,"respuestas_test_abiertas",n=len(y),technical_repeat=not first)
        predictions,score,probability=predecir(prepared["models"],prepared["X"],prepared["history"],frozen["baseline_rules"]["majority_class"],frozen["dataset"]["feature_names"])
        meta=prepared["meta"]
        evaluation=evaluar_comun(y,predictions,meta,frozen["new_product_rule"]["train_products"])
        errors=analizar_errores(y,predictions,meta,frozen["new_product_rule"]["train_products"])
        for key,filename in [("global","metricas_globales.json"),("monthly","metricas_mensuales.json"),("monthly_unweighted_means","promedios_mensuales.json"),
                             ("cohorts","metricas_cohortes.json"),("cohort_month","metricas_cohorte_mes.json")]:
            write_json(out/filename,evaluation[key])
        write_json(out/"errores_desacuerdos.json",errors)
        development=read_json(base/"resultados/fase_05/comparacion_desarrollo.json")
        # Métricas de desarrollo se reutilizan; no hay búsqueda nueva o reentrenamiento.
        comparison={name:{"train":development["historico_ultimo_mes" if name=="historico" else name]["train"],
                          "val":development["historico_ultimo_mes" if name=="historico" else name]["val"],"test":evaluation["global"][name]} for name in METHODS}
        write_json(out/"comparacion_con_desarrollo.json",comparison)
        aligned=meta[["indice_matriz","observacion_id","producto_provisional","mes_objetivo"]].copy()
        aligned["cohorte"]=np.where(meta.producto_provisional.isin(frozen["new_product_rule"]["train_products"]),"conocidos","nuevos")
        aligned["y_real"]=y; aligned["score_perceptron"]=score; aligned["probabilidad_logistica"]=probability
        for name,pred in predictions.items():
            aligned[f"pred_{name}"]=pred
            aligned[f"resultado_{name}"]=np.select([(y==0)&(pred==0),(y==0)&(pred==1),(y==1)&(pred==0),(y==1)&(pred==1)],["TN","FP","FN","TP"],default="ERROR")
        aligned["desacuerdo_modelos"]=predictions["perceptron"]!=predictions["logistica"]
        aligned.to_csv(out/"predicciones_alineadas.csv",index=False,float_format="%.17g",lineterminator="\n")
        for name in METHODS:
            pd.DataFrame(evaluation["global"][name]["confusion_matrix"],index=["real_0","real_1"],columns=["pred_0","pred_1"]).to_csv(out/f"confusion_{name}.csv",lineterminator="\n")
        from src.evaluation.graficos_finales import graficar_evaluacion
        graficar_evaluacion(evaluation,comparison,out/"graficos")
        (out/"reporte_evaluacion_final.md").write_text(reporte(evaluation,errors,comparison,frozen,prepared["freeze_hash"]),encoding="utf-8")
        after=hashes_anteriores(base)
        if before!=after: raise RuntimeError("Un artefacto congelado cambió durante la evaluación")
        implementation=[base/p for p in ["src/experiments/evaluacion_final.py","src/evaluation/graficos_finales.py","tests/test_evaluacion_final.py"]]
        # El log con fecha/hora es acumulativo y no determinista. Los resultados científicos sí son reproducibles.
        artifacts={str(p.relative_to(base)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"manifiesto.json","validaciones.json","registro_ejecuciones.jsonl"}}
        write_json(out/"manifiesto.json",{"phase":7,"freeze_sha256":prepared["freeze_hash"],"n_test":len(y),"n_features":20,
                  "protocol":"resultados/fase_06/protocolo_test.md","protocol_sha256":sha256(base/"resultados/fase_06/protocolo_test.md"),
                  "test_predictive_evaluation_executed":True,"no_parameter_changes":True,
                  "opening_record_sha256":sha256(opening),"execution_log":"registro_ejecuciones.jsonl (append-only)",
                  "primary_metric":"balanced_accuracy","methods":list(METHODS),
                  "integrity":{"preserved":True,"before":before,"after":after},"artifacts_sha256":artifacts,
                  "implementation_sha256":{str(p.relative_to(base)):sha256(p) for p in implementation},
                  "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":__import__("matplotlib").__version__}})
        evento(log,run_id,"evaluacion_completada",n=len(y),protected_files=len(before),parameter_changes=False)
        print(json.dumps({"n":len(y),"global":evaluation["global"],"cohorts":{k:{"rows":v["rows"],"products":v["products"]} for k,v in evaluation["cohorts"].items()},"protected_files":len(before)},ensure_ascii=False))
        return evaluation,errors
    except Exception as exc:
        evento(log,run_id,"intento_fallido",test_labels_opened=opened,error_type=type(exc).__name__,message=str(exc),action="detener; ninguna optimizacion ni alteracion de parametros")
        raise


def reporte(evaluation,errors,comparison,frozen,freeze_hash):
    def fmt(v): return "no estimable" if v is None else f"{v*100:.3f} %"
    table="\n".join(f"| {name} | {m['n']} | {m['support']['0']} | {m['support']['1']} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['balanced_accuracy'])} |" for name,m in evaluation["global"].items())
    months="\n".join(f"| {month} | {rows['perceptron']['n']} | {rows['perceptron']['support']['0']}/{rows['perceptron']['support']['1']} | "+" | ".join(fmt(rows[name]["balanced_accuracy"]) for name in METHODS)+" |" for month,rows in evaluation["monthly"].items())
    cohort="\n".join(f"| {c} | {v['products']} | {name} | {m['n']} | {m['support']['0']}/{m['support']['1']} | {fmt(m['accuracy'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} | {fmt(m['balanced_accuracy'])} |" for c,v in evaluation["cohorts"].items() for name,m in v["methods"].items())
    gaps="\n".join(f"| {name} | {fmt(v['train']['balanced_accuracy'])} | {fmt(v['val']['balanced_accuracy'])} | {fmt(v['test']['balanced_accuracy'])} | {(v['test']['balanced_accuracy']-v['val']['balanced_accuracy'])*100:+.3f} pp |" for name,v in comparison.items())
    counts="\n".join(f"| {name} | {m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} | {m['FP']+m['FN']} |" for name,m in evaluation["global"].items())
    concentrations=[]
    for name,detail in errors["errors_by_method"].items():
        peak=max(detail["monthly"].items(),key=lambda v:v[1]["error_rate"])
        concentrations.append(f"- {name}: mes con mayor tasa de error {peak[0]}, {peak[1]['errors']}/{peak[1]['n']} ({peak[1]['error_rate']*100:.3f} %). Todos los meses se conservan.")
    agreement=errors["global"]
    return f"""# Evaluación predictiva final WWI — Fase 7

## Integridad y apertura

Primera evaluación predictiva de test autorizada por esta fase. Registro previo e inmutable de primera apertura en registro_apertura_test.json (fecha/hora UTC−05:00, versiones, hashes, reglas y métricas); intentos/repeticiones en registro_ejecuciones.jsonl, sin sobrescribir eventos. Antes de abrir y_test se verificaron manifiestos, datos, modelos, transformaciones, contratos y parámetros congelados, y se reprodujeron exactamente las métricas train/val aprobadas. Congelación SHA-256 `{freeze_hash}`; contrato 2.0. Ninguna inconsistencia se corrige reentrenando.

Se evaluaron **1.111 filas** con **20 características completas**, mismas claves/orden para cuatro métodos. X_test float64, y_test int8, finitos y 0/1. No se recalculó el dataset, preprocesamiento o PCA; no hubo fit, selección de nuevos parámetros ni exclusión por rendimiento. La cubierta temporal completa es el supuesto académico aprobado, no certificación del origen.

Reglas: perceptrón z>=0 (checkpoint 70), logística probabilidad>=0,5 (checkpoint 200), mayoritario=1 verificado con 2.382/2.217 etiquetas train, histórico=1[unidades_lag_1>R_referencia] usando solo agregados históricos aprobados (empate→0). La referencia se comprobó con los tres lags, no se construyó con respuestas del objetivo.

## Resultados globales

Principal: balanced accuracy; no se cambió tras ver los resultados. Positiva=ventas del mes superiores a mediana reciente. Los cuatro métodos usan exactamente el mismo soporte real.

| Método | N | Soporte 0 | Soporte 1 | Accuracy | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

| Método | TN | FP | FN | TP | Errores |
|---|---:|---:|---:|---:|---:|
{counts}

En cada método TN+FP+FN+TP=N. Las matrices tienen filas reales/columnas predichas, orden 0/1. Métricas y convenciones de denominador cero se reutilizan del módulo aprobado; no se inventan estimaciones si falta una clase.

## Generalización frente a desarrollo

| Método | BA train | BA validación | BA test final | Test−validación |
|---|---:|---:|---:|---:|
{gaps}

Train/validación se reutilizaron de fase 5 y se verificó la reproducción de checkpoints. Validación participó en selección; sus métricas son optimistas respecto a una evaluación independiente. Test es el resultado final reservado de esta versión, no otra oportunidad para elegir modelo, época o umbral. Diferencias descriptivas no establecen significación ni superioridad universal; los métodos tienen presupuestos/objetivos distintos.

En esta reserva temporal agregada, la logística tiene BA {fmt(evaluation['global']['logistica']['balanced_accuracy'])}, frente a {fmt(evaluation['global']['perceptron']['balanced_accuracy'])} del perceptrón, una diferencia descriptiva de {(evaluation['global']['logistica']['balanced_accuracy']-evaluation['global']['perceptron']['balanced_accuracy'])*100:.3f} puntos. La BA del perceptrón cambia {(comparison['perceptron']['test']['balanced_accuracy']-comparison['perceptron']['val']['balanced_accuracy'])*100:+.3f} puntos respecto a validación; la logística cambia {(comparison['logistica']['test']['balanced_accuracy']-comparison['logistica']['val']['balanced_accuracy'])*100:+.3f}. Son resultados finales observados; no se elige ni ajusta un nuevo modelo con estas diferencias.

## Comportamiento mensual

| Mes objetivo | N | Soporte 0/1 | BA perceptrón | BA logística | BA mayoritario | BA histórico |
|---|---:|---:|---:|---:|---:|---:|
{months}

metricas_mensuales.json contiene también accuracy, precision, recall, F1, matriz y conteos por método/mes; promedios_mensuales.json conserva el promedio **no ponderado** y número de meses estimables por métrica. Las métricas globales se obtienen de todos los conteos conjuntos; no son necesariamente el promedio mensual. Un grupo sin ambas clases tiene BA no estimable y no se elimina. Los cinco meses, incluidos los desfavorables, se reportan completos.

La BA mensual media de logística es {fmt(evaluation['monthly_unweighted_means']['logistica']['balanced_accuracy']['mean'])}, distinta de su BA global {fmt(evaluation['global']['logistica']['balanced_accuracy'])}. La BA global combina las tasas por clase con soportes temporales distintos, no es una media aritmética de las cinco BA. La composición real cambia entre meses (soportes 0/1 en tabla). El resultado mensual de logística va de {fmt(min(v['logistica']['balanced_accuracy'] for v in evaluation['monthly'].values()))} a {fmt(max(v['logistica']['balanced_accuracy'] for v in evaluation['monthly'].values()))}; su resultado agregado no representa igual comportamiento en cada período.

## Cohortes de productos

Conocidos: nombres exactos presentes en la lista de entrenamiento congelada; nuevos: nombres ausentes de esa lista. No se trata de altas comerciales certificadas. Se confirmaron 219 productos conocidos/1.095 filas y ocho nuevos/16 filas (abril-mayo). No se excluyó ningún producto nuevo por sus resultados.

| Cohorte | Productos | Método | N | Soporte 0/1 | Accuracy | Precision | Recall | F1 | BA |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
{cohort}

metricas_cohorte_mes.json conserva el cruce completo, incluyendo grupos vacíos como no estimables. Dieciséis casos no demuestran estabilidad ni capacidad general para productos nuevos; cada error puede cambiar apreciablemente las cifras.

En conocidos, BA de perceptrón {fmt(evaluation['cohorts']['conocidos']['methods']['perceptron']['balanced_accuracy'])} y logística {fmt(evaluation['cohorts']['conocidos']['methods']['logistica']['balanced_accuracy'])}. En nuevos, BA {fmt(evaluation['cohorts']['nuevos']['methods']['perceptron']['balanced_accuracy'])} y {fmt(evaluation['cohorts']['nuevos']['methods']['logistica']['balanced_accuracy'])}, respectivamente. Una precisión alta en esa cohorte puede coexistir con recall bajo; se deben leer los conteos de ambas clases y no extrapolar con N=16.

## Errores y desacuerdos

Los clasificadores coinciden en {agreement['agreement']} casos y discrepan en {agreement['disagreement']}. Ambos aciertan {agreement['both_correct']}, ambos fallan {agreement['both_wrong']}; solo acierta perceptrón en {agreement['only_perceptron_correct']} y solo logística en {agreement['only_logistic_correct']}. Estas últimas cuatro categorías (ambos aciertan, ambos fallan, solo perceptrón acierta, solo logística acierta) son excluyentes y suman N. Errores y desacuerdos por mes/cohorte están en errores_desacuerdos.json; predicciones_alineadas.csv permite rastrear FP/FN/TP/TN por observación y método, con probabilidades logísticas y scores del perceptrón claramente identificados.

El perceptrón predice clase 1 en {evaluation['global']['perceptron']['TP']+evaluation['global']['perceptron']['FP']} casos y la logística en {evaluation['global']['logistica']['TP']+evaluation['global']['logistica']['FP']}, frente a {evaluation['global']['perceptron']['support']['1']} positivos reales. El recall elevado coexiste con falsos positivos abundantes: {evaluation['global']['perceptron']['FP']} y {evaluation['global']['logistica']['FP']}, respectivamente. La especificidad es {fmt(evaluation['global']['perceptron']['specificity'])} y {fmt(evaluation['global']['logistica']['specificity'])}. F1 y recall positivos aislados no reflejan bien esta dificultad para reconocer meses no superiores.

{chr(10).join(concentrations)}

FP significa anticipar ventas superiores sin que se observe ese resultado; FN significa no anticipar una subida realmente observada. No se conoce su coste monetario y no se presume que uno sea más grave. Acuerdo entre modelos no implica acierto. Estos análisis no se utilizaron para modificar pesos, referencias ni umbrales.

No se detectó una contradicción demostrable de construcción, alineación o implementación en las comprobaciones. Los desaciertos son resultados predictivos bajo el contrato y no prueba automática de un bug. La verificación no certifica el linaje comercial ausente ni que el problema sea completamente predecible.

## Gráficos y reproducción

Se exportan seis PNG: BA global, accuracy/precision/recall/F1 globales, confusión perceptrón, confusión logística, BA mensual y comparación BA train/validación/test de los cuatro métodos. Ejes/proporciones identificados, sin intervalos o precisión inferencial no justificada. PCA no interviene.

```powershell
python -m src.experiments.evaluacion_final
python -m unittest discover -s tests -p 'test_evaluacion_final.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

El comando solo verifica/carga artefactos congelados, predice, calcula métricas y exporta. Las repeticiones son técnicas del mismo protocolo, no búsquedas. El registro de primera apertura se conserva y los eventos de cada intento se agregan; sus timestamps no se confunden con reproducibilidad de resultados científicos. Cualquier intento fallido y su motivo quedan registrados, incluso si ocurrió antes de abrir y_test.

## Límites y cierre

Solo cinco meses de prueba, observaciones correlacionadas por producto/mes e historia compartida; no se realizó bootstrap iid ni se atribuye significación a diferencias pequeñas. La distribución de clases puede variar entre meses/cohortes; por ello accuracy y F1 aisladas no reemplazan BA/matriz/soportes. Las probabilidades no se recalibraron ni se consideran perfectamente calibradas.

Persisten identidad por nombre sin StockItemID, cobertura académica no certificada, unidad comercial sin validación externa y ventas observadas distintas de demanda no atendida o inventario. La clase indica superar una mediana reciente, no rentabilidad ni utilidad económica garantizada.

Conclusión de esta versión: los resultados anteriores constituyen la comparación final de los cuatro métodos sobre la misma reserva temporal; se documenta su rendimiento observado, variación temporal y límites sin seleccionar nuevas configuraciones a partir de test. Modelos, umbrales, características, particiones, población y etiqueta quedan intactos después de observarlo. Cualquier investigación posterior requiere otra versión/protocolo y nueva reserva independiente, sin reutilizar este test como validación. No se avanzó a fase 8 ni se propone un reentrenamiento automático.
"""


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
