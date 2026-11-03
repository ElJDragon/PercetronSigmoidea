"""python -m src.experiments.entrenar_logistica — test permanece reservado."""
import argparse
from decimal import Decimal
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.auditar_datos import BASE, sha256
from src.evaluation.metricas import clave_seleccion, metricas_binarias
from src.experiments.entrenar_perceptron import load_development, protected_hashes, read_json, write_json
from src.models.logistic_regression import (LogisticRegressionManual, binary_cross_entropy_logits,
                                           validar_umbral, verificar_gradientes)
from src.models.perceptron import PerceptronSimple


def seleccionar_umbral(y, probabilities, grid):
    """Cuadrícula fase 2 sobre un único checkpoint ya elegido."""
    probabilities=np.asarray(probabilities)
    if probabilities.ndim!=1 or probabilities.dtype.kind not in "iuf" or not np.isfinite(probabilities).all() or not ((probabilities>=0)&(probabilities<=1)).all():
        raise ValueError("Probabilidades finitas entre 0 y 1 requeridas")
    records = []
    for threshold in grid:
        threshold = validar_umbral(threshold)
        metrics = metricas_binarias(y, (probabilities >= threshold).astype(np.int8))
        if metrics["balanced_accuracy"] is None:
            raise ValueError("Validación sin ambas clases: selección bloqueada")
        records.append({"threshold":threshold,"metrics":metrics})
    if not records:
        raise ValueError("Cuadrícula vacía")
    chosen = max(records,key=lambda r:(r["metrics"]["balanced_accuracy"], r["metrics"]["f1"],
                                       -abs(Decimal(str(r["threshold"]))-Decimal("0.5")), -r["threshold"]))
    return chosen["threshold"], records


def evaluar_logistica(model, dev):
    metrics = {}
    for split in ["train","val"]:
        part=dev[split]
        metrics[split]=metricas_binarias(part["y"], model.predict(part["X"],feature_names=dev["feature_names"]))
        metrics[split]["cross_entropy"]=binary_cross_entropy_logits(part["y"],model.decision_function(part["X"]))
    return metrics


def seleccionar_experimentos(dev, plan):
    runs=[]; chosen_model=None; chosen_id=None; best_key=None
    for index, config in enumerate(plan["experiments"]):
        model=LogisticRegressionManual(**{k:config[k] for k in ["learning_rate","max_epochs","seed"]},
                                       initialization=plan["initialization"], initial_bias=plan["initial_bias"],
                                       tolerance=plan["tolerance"],threshold=plan["history_threshold"])
        model.fit(dev["train"]["X"],dev["train"]["y"],dev["val"]["X"],dev["val"]["y"],feature_names=dev["feature_names"])
        metrics=evaluar_logistica(model,dev)
        key=clave_seleccion(metrics["val"],model.best_epoch_,index)
        runs.append({"id":config["id"], "params":model.get_params(),"selected_epoch":model.best_epoch_,
                     "initial_train_loss":model.initial_train_loss_,"initial_val_loss":model.initial_val_loss_,
                     "metrics_at_05":metrics,"history":model.history_,"selected_weights":model.weights_.tolist(),
                     "selected_bias":model.bias_,"converged_by_tolerance":model.converged_})
        if best_key is None or key>best_key:
            best_key,chosen_model,chosen_id=key,model,config["id"]
    if chosen_model is None:
        raise ValueError("Presupuesto de experimentos vacío")
    threshold, threshold_records=seleccionar_umbral(dev["val"]["y"],chosen_model.predict_proba(dev["val"]["X"]),plan["threshold_grid"])
    chosen_model.threshold=threshold
    return chosen_model,chosen_id,runs,threshold_records


def verificar_perceptron(dev, base=BASE):
    folder=Path(base)/"resultados/fase_04"
    manifest=read_json(folder/"manifiesto.json")
    validation=read_json(folder/"validaciones.json")
    if validation["phase04_manifest_sha256"] != sha256(folder/"manifiesto.json") or not validation["successful"]:
        raise ValueError("Validación/hash de perceptrón no coinciden")
    for relative,digest in manifest["artifacts_sha256"].items():
        if sha256(Path(base)/relative)!=digest:
            raise ValueError(f"Resultado aprobado de perceptrón alterado: {relative}")
    if manifest["matrix_hashes_development"] != dev["matrix_hashes"]:
        raise ValueError("Perceptrón y logística no usarían las mismas matrices")
    model=PerceptronSimple.load(folder/"modelo_seleccionado.json")
    if model.feature_names_!=dev["feature_names"]:
        raise ValueError("Características de modelos no alineadas")
    expected=read_json(folder/"metricas.json")
    for split in ["train","val"]:
        computed=metricas_binarias(dev[split]["y"],model.predict(dev[split]["X"],feature_names=dev["feature_names"]))
        if computed!=expected[split]:
            raise ValueError("Perceptrón recuperado no reproduce métricas aprobadas")
        pred=pd.read_csv(folder/f"predicciones_{split}.csv",usecols=["observacion_id"])
        if pred.observacion_id.tolist()!=dev[split]["meta"].observacion_id.tolist():
            raise ValueError("Observaciones de comparación no coinciden")
    return expected


def evaluar_baselines(dev, base=BASE):
    """Lee solo IDs/mes y dos predictores históricos, nunca Q/y de test."""
    source=Path(base)/"datos/Procesados/fase_03/observaciones.csv"
    columns=["observacion_id","mes_objetivo","unidades_lag_1","R_referencia"]
    parts=[]
    for chunk in pd.read_csv(source,usecols=columns,dtype=str,keep_default_na=False,chunksize=2500):
        parts.append(chunk.loc[chunk.mes_objetivo<="2015-12"])
    history=pd.concat(parts).set_index("observacion_id")
    majority=int((dev["train"]["y"]==1).sum()>(dev["train"]["y"]==0).sum())  # empate -> 0
    result={"mayoritario":{},"historico_ultimo_mes":{}}
    for split in ["train","val"]:
        ids=dev[split]["meta"].observacion_id
        aligned=history.loc[ids]
        lag=pd.to_numeric(aligned.unidades_lag_1,errors="raise").to_numpy(np.float64)
        ref=pd.to_numeric(aligned.R_referencia,errors="raise").to_numpy(np.float64)
        if not np.isfinite(lag).all() or not np.isfinite(ref).all():
            raise ValueError("Historia de baseline incompleta")
        result["mayoritario"][split]=metricas_binarias(dev[split]["y"],np.full(len(ids),majority,dtype=np.int8))
        result["historico_ultimo_mes"][split]=metricas_binarias(dev[split]["y"],(lag>ref).astype(np.int8))
    return result


def hashes_previos(base=BASE):
    base=Path(base)
    values=protected_hashes(base)
    previous=read_json(base/"resultados/fase_04/manifiesto.json")
    for relative in previous["implementation_sha256"]:
        values[relative]=sha256(base/relative)
    for p in sorted((base/"resultados/fase_04").rglob("*")):
        if p.is_file(): values[str(p.relative_to(base))]=sha256(p)
    return values


def run(base=BASE):
    base=Path(base); out=base/"resultados/fase_05"
    before=hashes_previos(base)
    plan_path=out/"plan_experimentos.json"; plan_hash=sha256(plan_path); plan=read_json(plan_path)
    # Gate matemático antes de cargar las matrices empresariales.
    gradients=verificar_gradientes()
    write_json(out/"verificacion_gradientes.json",gradients)
    dev=load_development(base)
    perceptron=verificar_perceptron(dev,base)
    model,selected_id,runs,thresholds=seleccionar_experimentos(dev,plan)
    selected=next(r for r in runs if r["id"]==selected_id)
    metrics=evaluar_logistica(model,dev)
    baselines=evaluar_baselines(dev,base)
    comparison={"logistica":metrics,"perceptron":perceptron,**baselines}
    model.save(out/"modelo_seleccionado.json")
    for r in runs:
        pd.DataFrame(r["history"]).to_csv(out/f"historial_{r['id']}.csv",index=False,float_format="%.17g",lineterminator="\n")
    write_json(out/"experimentos.json",{"plan_sha256":plan_hash,"runs":[{k:v for k,v in r.items() if k!="history"} for r in runs],
                                        "selected_id":selected_id,"final_threshold":model.threshold})
    write_json(out/"seleccion_umbral.json",{"checkpoint_fixed_before_threshold_search":True,"selected_epoch":model.best_epoch_,
                                           "selected_experiment":selected_id,"selected_threshold":model.threshold,"grid_results":thresholds})
    write_json(out/"metricas.json",metrics)
    write_json(out/"comparacion_desarrollo.json",comparison)
    pd.DataFrame({"feature":model.feature_names_,"weight":model.weights_}).to_csv(out/"pesos.csv",index=False,float_format="%.17g",lineterminator="\n")
    for split in ["train","val"]:
        meta=dev[split]["meta"][["observacion_id","producto_provisional","mes_objetivo"]].copy()
        meta["y_real"]=dev[split]["y"]
        meta["probabilidad_1"]=model.predict_proba(dev[split]["X"])
        meta["y_pred"]=model.predict(dev[split]["X"])
        meta.to_csv(out/f"predicciones_{split}.csv",index=False,float_format="%.17g",lineterminator="\n")
        pd.DataFrame(metrics[split]["confusion_matrix"],index=["real_0","real_1"],columns=["pred_0","pred_1"]).to_csv(out/f"confusion_{split}.csv",lineterminator="\n")
    recovered=LogisticRegressionManual.load(out/"modelo_seleccionado.json")
    for split in ["train","val"]:
        if not np.array_equal(recovered.predict_proba(dev[split]["X"]),model.predict_proba(dev[split]["X"])):
            raise AssertionError("Serialización cambia probabilidades")
    if evaluar_logistica(recovered,dev)!=metrics:
        raise AssertionError("Serialización cambia métricas")
    from src.evaluation.graficos_logistica import graficar_logistica
    graficar_logistica(selected["history"],metrics["val"],model.best_epoch_,model.threshold,len(model.weights_),out/"graficos")
    (out/"reporte_logistica.md").write_text(report(model,selected,runs,metrics,comparison,gradients,dev),encoding="utf-8")
    after=hashes_previos(base)
    if before!=after or sha256(plan_path)!=plan_hash:
        raise RuntimeError("Se alteró un archivo aprobado o el plan previo")
    implementation=[base/p for p in ["src/models/logistic_regression.py","src/experiments/entrenar_logistica.py",
                                      "src/evaluation/graficos_logistica.py","tests/test_logistica.py"]]
    artifacts={str(p.relative_to(base)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"manifiesto.json","validaciones.json"}}
    write_json(out/"manifiesto.json",{"contract_version":"2.0","phase03_manifest_sha256":dev["manifest_hash"],
                "matrix_hashes_development":dev["matrix_hashes"],"plan_sha256":plan_hash,"selected_id":selected_id,
                "selected_epoch":model.best_epoch_,"selected_threshold":model.threshold,"n_features":len(model.weights_),
                "test_labels_consulted":False,"test_predictions_generated":False,
                "integrity":{"preserved":True,"before":before,"after":after},
                "implementation_sha256":{str(p.relative_to(base)):sha256(p) for p in implementation},
                "artifacts_sha256":artifacts,"environment":{"python":platform.python_version(),"numpy":np.__version__,
                                                              "pandas":pd.__version__,"matplotlib":__import__("matplotlib").__version__}})
    print(json.dumps({"selected_id":selected_id,"epoch":model.best_epoch_,"threshold":model.threshold,"metrics":metrics,
                      "baselines_val":{name:part["val"]["balanced_accuracy"] for name,part in baselines.items()},"protected_files":len(before)},ensure_ascii=False))
    return model,runs


def report(model,selected,runs,metrics,comparison,gradients,dev):
    hist=selected["history"]
    selected_row=next(h for h in hist if h["epoch"]==model.best_epoch_)
    experiments="\n".join(f"| {r['id']} | {r['params']['learning_rate']} | {r['selected_epoch']} | {r['metrics_at_05']['val']['balanced_accuracy']:.6f} | {r['history'][-1]['train_loss']:.6f} | {r['history'][-1]['gradient_norm']:.6f} |" for r in runs)
    comp="\n".join(f"| {name} | {split} | {m[split]['accuracy']:.6f} | {m[split]['precision']:.6f} | {m[split]['recall']:.6f} | {m[split]['f1']:.6f} | {m[split]['balanced_accuracy']:.6f} |" for name,m in comparison.items() for split in ["train","val"])
    weights="\n".join(f"| {name} | {w:.17g} |" for name,w in zip(model.feature_names_,model.weights_))
    grad="\n".join(f"| {name} | {r['h']} | {r['atol']} | {r['rtol']} | {r['max_absolute_error']:.10g} | {r['max_relative_error']:.10g} | {r['passed']} |" for name,r in gradients.items())
    convergence="\n".join(f"- {r['id']}: aumentos de pérdida train mayores que 1e-12: {sum(b['train_loss']>a['train_loss']+1e-12 for a,b in zip(r['history'],r['history'][1:]))}; gradiente final {r['history'][-1]['gradient_norm']:.6g}." for r in runs)
    confusions="\n".join(f"- {name}: train `{m['train']['confusion_matrix']}`; val `{m['val']['confusion_matrix']}`." for name,m in comparison.items())
    return f"""# Regresión logística WWI — Fase 5

## Alcance y datos

Implementación propia NumPy de sigmoidea, pérdida, gradientes y descenso por lote completo. Sin scikit-learn, autodiferenciación, PCA o reconstrucción de datos. Se reutilizaron las matrices de fase 3, sus 20 entradas y los mismos IDs/orden del perceptrón: train {dev['train']['X'].shape}, validación {dev['val']['X'].shape}. Manifiesto fase 3: `{dev['manifest_hash']}`. Las etiquetas de test no se consultaron ni se generaron sus predicciones; solo se verificaron hashes de integridad mediante los manifiestos aprobados.

## Matemática y estabilidad

Con X(m,d), w(d), b escalar: z=Xw+b; p=sigmoid(z). Para z>=0 se usa 1/(1+exp(-z)); para z<0, exp(z)/(1+exp(z)). Funciona con arreglos multidimensionales finitos, sin overflow de exponenciales extremos y con sigma(0)=0,5.

L=mean(logaddexp(0,z)−y*z), equivalente a entropía cruzada binaria. No se recortan probabilidades ni alteran etiquetas; la media se suma después de dividir por m para reducir overflow. grad_w=Xᵀ(p−y)/m; grad_b=mean(p−y). Se implementa Xᵀ((p−y)/m), algebraicamente idéntico. Actualización: w←w−η grad_w; b←b−η grad_b. Si combinación lineal/gradientes dejan de ser finitos se bloquea la ejecución; no se oculta divergencia. No hay regularización ni optimizadores predefinidos.

## Comprobación previa de gradientes

Diferencias centrales por coordenada con paso h_j=h*max(1,|parametro_j|), para pesos y sesgo no nulos; etiquetas mixtas y entradas positivas/negativas. h=1e-6 en magnitudes moderadas equilibra cancelación y error de truncamiento; h=1e-5 y tolerancia absoluta mayor en logits extremos por la escala de la pérdida. Criterio `allclose`: error <= atol+rtol*|numérico|. El gate matemático corrió antes de abrir WWI; las pruebas añaden dimensiones 1/3/8 y estabilidad hasta logits ±1e300. No se exige diferencias finitas relativas fiables para derivadas prácticamente cero: importa también atol.

| Caso | h | atol | rtol | Máx. error absoluto | Máx. error relativo | Aprobado |
|---|---:|---:|---:|---:|---:|---|
{grad}

Valores analíticos y numéricos individuales en verificacion_gradientes.json.

## Presupuesto y selección fijados antes de WWI

Tres tasas 0,01/0,05/0,1; 200 épocas por ejecución, lote completo; inicialización cero, sesgo cero, semilla 42. La semilla es metadato reproducible pero no afecta el lote completo con inicio cero. El modelo permite inicio normal reproducible o pesos explícitos, pero no se usaron para ampliar esta búsqueda. Tolerancia desactivada para ejecutar el presupuesto completo; no se afirma convergencia por llegar a 200 épocas.

Checkpoint y tasa se eligen por BA val a umbral 0,5, luego F1, menor época y orden del plan. Después, sobre **un único checkpoint fijo**, se prueba la cuadrícula de fase 2: 0,05,0,10,…,0,95, con desempates BA/F1/cercanía a 0,5/umbral menor. No se evaluó un producto cartesiano de épocas×umbrales. Máximo 600 épocas y 19 umbrales; no se amplió el presupuesto tras mirar resultados. La selección sobre validación genera optimismo y no es estimación final independiente.

| Ejecución | η | Época seleccionada a 0,5 | BA val a 0,5 | Pérdida train al cerrar 200 | Norma gradiente final |
|---|---:|---:|---:|---:|---:|
{experiments}

Modelo seleccionado: **{selected['id']}**, época **{model.best_epoch_}**, tasa **{model.learning_rate}**, umbral final **{model.threshold}**. Pesos restaurados a esa época, sin refit train+validación. Historial completo de las tres ejecuciones en CSV; la selección de umbral se conserva íntegra en JSON.

## Pérdida y convergencia

Ejecución seleccionada: pérdida train inicial (época 0) **{selected['initial_train_loss']:.12g}**, pérdida train al finalizar la ejecución **{hist[-1]['train_loss']:.12g}**. Pérdida train del checkpoint exportado **{metrics['train']['cross_entropy']:.12g}**. Pérdida val inicial **{selected['initial_val_loss']:.12g}**, al finalizar la ejecución **{hist[-1]['val_loss']:.12g}**, en el checkpoint **{metrics['val']['cross_entropy']:.12g}**. Se distingue último estado entrenado del seleccionado.

{convergence}

Norma del gradiente en el checkpoint: {selected_row['gradient_norm']:.6g}. Un gradiente no nulo o pérdida aún decreciente indica que agotar el presupuesto no certifica un óptimo. El criterio de selección es clasificación en validación, no mínima entropía cruzada. La pérdida puede bajar sin que accuracy/BA suban monótonamente. No se imponen resultados objetivo.

## Comparación preliminar de desarrollo

Métricas en fracciones, positiva=ventas del mes superiores a mediana reciente. Todos usan los mismos soportes: train {metrics['train']['support']}; val {metrics['val']['support']}. Perceptrón recuperado de fase 4, sin entrenarlo ni modificarlo; sus predicciones reproducen exactamente sus métricas aprobadas. Baseline mayoritario: mayoría de train, empate→0; histórico: 1[Q(t−1)>R(t)], no una comparación de variables estandarizadas con distintas escalas. Para obtenerlo se leyeron solo IDs/mes/lag1/R del dataset aprobado, excluyendo 2016 antes de convertir números y sin leer columnas Q_objetivo/y. Ningún baseline consulta test.

| Modelo | Conjunto | Accuracy | Precision | Recall | F1 | BA |
|---|---|---:|---:|---:|---:|---:|
{comp}

Matrices de confusión (filas reales/columnas predichas, orden 0/1):

{confusions}

La logística añade entropía cruzada: train {metrics['train']['cross_entropy']:.6f}, val {metrics['val']['cross_entropy']:.6f}. No se calcula una pérdida probabilística ficticia para perceptrón o reglas deterministas. Los resultados permiten comparación descriptiva, no superioridad definitiva ni significación estadística de pequeñas diferencias. Presupuestos/objetivos de optimización son distintos; no se fuerza una comparación por igual número de actualizaciones.

## Parámetros e inferencia

Sesgo: **{model.bias_:.17g}**; umbral: **{model.threshold:.17g}**. Todos los pesos son float64 y operan sobre características estandarizadas de fase 3, con redundancias: no equivalen a importancia causal independiente.

| Característica | Peso |
|---|---:|
{weights}

`predict_proba` devuelve un vector (m,) de P(y=1|X); `predict(X)` usa el umbral persistido, y `predict(X, threshold=...)` permite sustituirlo explícitamente. p>=umbral asigna 1. Sigmoidea no garantiza calibración perfecta; evaluar calibración es trabajo posterior. Umbral probabilístico no cambia R ni la definición de y. La serialización JSON conserva precisión float64, nombres, sesgo, configuración y época; sin pickle. La recuperación reproduce probabilidades y métricas de desarrollo exactamente.

## Artefactos y reproducción

Nuevos archivos: src/models/logistic_regression.py, src/experiments/entrenar_logistica.py, src/evaluation/graficos_logistica.py, tests/test_logistica.py y resultados/fase_05/. Se reutilizaron el cargador de desarrollo y las métricas aprobadas, sin cambiar su comportamiento; la visualización reutiliza backend/estilo y agrega un módulo separado para conservar intacta fase 4.

Resultados: modelo_seleccionado.json, pesos.csv, metricas.json, experimentos.json, seleccion_umbral.json, verificacion_gradientes.json, comparacion_desarrollo.json, historiales de tres tasas, confusiones y predicciones/probabilidades train/val, cinco PNG y manifiesto de hashes. Los gráficos de accuracy/BA corresponden a **0,5**, igual que la búsqueda de checkpoint; la confusión corresponde al **umbral final**. Todos proceden de las 20 entradas completas. No hay PCA ni frontera 2D.

```powershell
python -m unittest discover -s tests -p 'test_logistica.py' -v
python -m src.experiments.entrenar_logistica
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Riesgos y siguiente fase

La logística es un modelo lineal en logits; no representa toda interacción posible. Escoger checkpoint, tasa y umbral con la misma validación introduce optimismo. Las filas comparten producto/mes y no son réplicas independientes. Persisten cobertura académica no certificada, identidad por nombre, unidades comerciales por confirmar y ventas observadas que no miden demanda no atendida ni stock. La utilidad empresarial de una subida marginal sobre la mediana no está demostrada.

Para fase 6: congelar modelos, transformaciones, etiqueta y umbrales, y formalizar el protocolo de evaluación final común contra baselines antes de abrir test. Confirmar interpretación de probabilidades y distinguir evaluación de clasificación de calibración. Si se autoriza visualización PCA después, conservarla separada del entrenamiento multidimensional. Esta fase no evalúa test ni avanza automáticamente.
"""


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
