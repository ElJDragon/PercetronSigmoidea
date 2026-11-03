"""python -m src.experiments.entrenar_perceptron

Entrenamiento exclusivamente con matrices aprobadas de train y validación.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.auditar_datos import BASE, sha256
from src.evaluation.metricas import clave_seleccion, metricas_binarias
from src.models.perceptron import PerceptronSimple


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_development(base=BASE):
    """Solo abre arrays train/val; hashes verifican la copia reservada sin evaluarla."""
    base = Path(base)
    manifest_path = base / "resultados/fase_03/manifiesto.json"
    manifest = read_json(manifest_path)
    validation = read_json(base / "resultados/fase_03/validaciones.json")
    if manifest["version_contrato"] != "2.0" or validation["manifest_sha256"] != sha256(manifest_path) or not validation["successful"]:
        raise ValueError("Versión/hash/validaciones de fase 3 no coinciden")
    for relative, expected in manifest["artifact_sha256"].items():
        if sha256(base / relative) != expected:
            raise ValueError(f"Artefacto de fase 3 modificado: {relative}")
    folder = base / "datos/Procesados/fase_03"
    names = read_json(folder / "caracteristicas.json")["feature_names"]
    pre = read_json(folder / "preprocesador.json")
    if names != pre["feature_names"] or len(names) != manifest["features_after"] or len(set(names)) != len(names):
        raise ValueError("Orden de características inconsistente")
    dev = {"feature_names": names, "manifest_hash": sha256(manifest_path), "manifest": manifest,
           "matrix_hashes": {s: {"X": sha256(folder/f"X_{s}.npy"), "y": sha256(folder/f"y_{s}.npy")} for s in ["train", "val"]}}
    ids = set()
    for split in ["train", "val"]:
        x = np.load(folder / f"X_{split}.npy", allow_pickle=False)
        y = np.load(folder / f"y_{split}.npy", allow_pickle=False)
        meta = pd.read_csv(folder / f"metadatos_{split}.csv", keep_default_na=False)
        if x.shape != (manifest["counts"][split], len(names)) or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError(f"Matriz {split} con forma/tipo/valores inválidos")
        if y.shape != (len(x),) or y.dtype != np.int8 or not np.isin(y, [0, 1]).all():
            raise ValueError(f"Etiquetas {split} inválidas")
        if len(meta) != len(x) or meta.observacion_id.duplicated().any() or not np.array_equal(meta.indice_matriz, np.arange(len(x))):
            raise ValueError(f"Alineación de observaciones incorrecta: {split}")
        if not meta.particion.eq(split).all() or not meta.version_contrato.astype(str).eq("2.0").all() or not meta.fuente_sha256.eq(manifest["source_sha256"]).all():
            raise ValueError("Linaje/partición de metadatos inconsistente")
        lower, upper = manifest["partitions"][split]
        if not meta.mes_objetivo.between(lower, upper).all():
            raise ValueError("Mes objetivo fuera de partición")
        sorted_meta = meta.sort_values(["mes_objetivo", "producto_provisional"], kind="stable")
        if meta.observacion_id.tolist() != sorted_meta.observacion_id.tolist() or ids.intersection(meta.observacion_id):
            raise ValueError("Orden o claves compartidas entre particiones")
        for r in meta.itertuples():
            key = json.dumps([r.producto_provisional, r.mes_objetivo], ensure_ascii=False, separators=(",", ":"))
            if hashlib.sha256(key.encode("utf-8")).hexdigest() != r.observacion_id:
                raise ValueError("Clave de observación inconsistente")
        counts = {"positive": int((y == 1).sum()), "negative": int((y == 0).sum())}
        if counts != manifest["development_label_counts"][split]:
            raise ValueError("Etiquetas de desarrollo no alineadas con manifiesto")
        ids.update(meta.observacion_id)
        dev[split] = {"X": x, "y": y, "meta": meta}
    return dev


def select_experiments(dev, plan):
    """La API no acepta datos de test; cada actualización usa solo train."""
    runs = []
    selected_model, selected_key, selected_id = None, None, None
    for index, params in enumerate(plan["experiments"]):
        model = PerceptronSimple(**{k: params[k] for k in ["learning_rate", "max_epochs", "seed", "shuffle"]}, initial_bias=plan["initial_bias"])
        model.fit(dev["train"]["X"], dev["train"]["y"], X_val=dev["val"]["X"], y_val=dev["val"]["y"], feature_names=dev["feature_names"])
        metrics = {s: metricas_binarias(dev[s]["y"], model.predict(dev[s]["X"], feature_names=dev["feature_names"])) for s in ["train", "val"]}
        key = clave_seleccion(metrics["val"], model.best_epoch_, index)
        runs.append({"id": params["id"], "params": model.get_params(), "selected_epoch": model.best_epoch_,
                     "metrics": metrics, "history": model.history_, "selected_weights": model.weights_.tolist(), "selected_bias": model.bias_})
        if selected_key is None or key > selected_key:
            selected_key, selected_model, selected_id = key, model, params["id"]
    if selected_model is None:
        raise ValueError("Plan vacío")
    return selected_model, selected_id, runs


def protected_hashes(base):
    protected = []
    for name in ["datos/originales", "referencias", "datos/Procesados/fase_03", "resultados/auditoria", "resultados/fase_02", "resultados/fase_03", "src/pipeline"]:
        protected.extend(p for p in sorted((base / name).rglob("*")) if p.is_file() and "__pycache__" not in p.parts)
    protected.extend(base / p for p in ["src/auditar_datos.py", "src/analizar_etiquetas.py", "tests/test_auditoria.py", "tests/test_etiquetas.py", "tests/test_pipeline.py"])
    return {str(p.relative_to(base)): sha256(p) for p in protected}


def run(base=BASE):
    base = Path(base)
    out = base / "resultados/fase_04"
    plan_path = out / "plan_experimentos.json"
    plan_hash = sha256(plan_path)
    plan = read_json(plan_path)
    before = protected_hashes(base)
    dev = load_development(base)
    model, selected_id, runs = select_experiments(dev, plan)
    chosen = next(r for r in runs if r["id"] == selected_id)
    model.save(out / "modelo_seleccionado.json")
    for r in runs:
        pd.DataFrame(r["history"]).to_csv(out / f"historial_{r['id']}.csv", index=False, float_format="%.17g", lineterminator="\n")
    summary = [{k: v for k, v in r.items() if k != "history"} for r in runs]
    write_json(out / "experimentos.json", {"plan_sha256": plan_hash, "runs": summary, "selected_id": selected_id})
    write_json(out / "metricas.json", chosen["metrics"])
    pd.DataFrame({"feature": model.feature_names_, "weight": model.weights_}).to_csv(out / "pesos.csv", index=False, float_format="%.17g", lineterminator="\n")
    for split in ["train", "val"]:
        matrix = chosen["metrics"][split]["confusion_matrix"]
        pd.DataFrame(matrix, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(out / f"confusion_{split}.csv", lineterminator="\n")
        predictions = dev[split]["meta"][["observacion_id", "producto_provisional", "mes_objetivo"]].copy()
        predictions["y_real"] = dev[split]["y"]
        predictions["y_pred"] = model.predict(dev[split]["X"], feature_names=model.feature_names_)
        predictions["decision_score"] = model.decision_function(dev[split]["X"])
        predictions.to_csv(out / f"predicciones_{split}.csv", index=False, float_format="%.17g", lineterminator="\n")
    # Persistencia verificada antes de gráficos/reportes.
    recovered = PerceptronSimple.load(out / "modelo_seleccionado.json")
    if not np.array_equal(recovered.predict(dev["val"]["X"]), model.predict(dev["val"]["X"])):
        raise AssertionError("La recuperación del modelo cambia las predicciones")
    from src.evaluation.graficos import graficar_perceptron
    graficar_perceptron(chosen["history"], chosen["metrics"]["val"], model.best_epoch_, len(model.weights_), out / "graficos")
    (out / "reporte_perceptron.md").write_text(report(model, chosen, summary, dev), encoding="utf-8")
    after = protected_hashes(base)
    if before != after or sha256(plan_path) != plan_hash:
        raise RuntimeError("Se modificó un archivo protegido o el presupuesto fijado")
    code = [p for folder in ["src/models", "src/evaluation", "src/experiments"] for p in sorted((base / folder).glob("*.py"))]
    code.append(base / "tests/test_perceptron.py")
    artifacts = {str(p.relative_to(base)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"manifiesto.json", "validaciones.json"}}
    write_json(out / "manifiesto.json", {"contract_version": "2.0", "phase03_manifest_sha256": dev["manifest_hash"],
                "matrix_hashes_development": dev["matrix_hashes"], "plan_sha256": plan_hash,
                "selected_id": selected_id, "selected_epoch": model.best_epoch_, "n_features": len(model.weights_),
                "test_evaluated": False, "test_arrays_loaded_by_training_runner": False,
                "integrity": {"preserved": True, "before": before, "after": after},
                "implementation_sha256": {str(p.relative_to(base)): sha256(p) for p in code},
                "artifacts_sha256": artifacts,
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                "matplotlib": __import__("matplotlib").__version__}})
    print(json.dumps({"selected_id": selected_id, "selected_epoch": model.best_epoch_, "bias": model.bias_, "metrics": chosen["metrics"], "protected_files": len(before)}, ensure_ascii=False))
    return model, runs


def report(model, chosen, runs, dev):
    h = chosen["history"]
    zero_epochs = [r["epoch"] for r in h if r["online_errors"] == 0]
    increasing = any(b["online_errors"] > a["online_errors"] for a, b in zip(h, h[1:]))
    table = "\n".join(f"| {r['id']} | {r['selected_epoch']} | {r['metrics']['train']['accuracy']:.6f} | {r['metrics']['val']['accuracy']:.6f} | {r['metrics']['val']['balanced_accuracy']:.6f} | {r['metrics']['val']['f1']:.6f} |" for r in runs)
    weights = "\n".join(f"| {name} | {value:.17g} |" for name, value in zip(model.feature_names_, model.weights_))
    m = chosen["metrics"]
    metrics = "\n".join(f"| {key} | {m['train'][key]:.6f} | {m['val'][key]:.6f} |" for key in ["accuracy", "precision", "recall", "f1", "balanced_accuracy"])
    return f"""# Perceptrón multidimensional WWI — Fase 4

## Datos y método

Se usaron los arrays de fase 3 sin reconstruir dataset, ajustar transformaciones ni cambiar características: train {dev['train']['X'].shape}, validación {dev['val']['X'].shape}. Contrato 2.0, manifiesto SHA-256 `{dev['manifest_hash']}`. Pesos asociados al orden validado de las {len(model.weights_)} características. El supuesto de completitud académica, identidad por nombre y disponibilidad al cierre permanece vigente.

Implementación propia NumPy: z=wᵀx+b, predicción 1 si z>=0; e=y−predicción; si e≠0, w←w+ηex y b←b+ηe. Etiquetas canónicas 0/1. Una actualización por observación errada. No se optimiza entropía cruzada, no hay probabilidades ni optimizador externo.

## Presupuesto previo y selección

`plan_experimentos.json` se fijó antes del primer entrenamiento: dos órdenes (fijo y barajado), η=0,01, 100 épocas máximas por ejecución, semilla 42 y pesos/sesgo iniciales cero. Una sola tasa: con inicio cero y η positiva constante, cambiar η solo escala la trayectoria en aritmética exacta; flotantes pueden romper esa equivalencia. No se amplió la búsqueda tras observar resultados.

Validación elige máxima balanced accuracy; desempates por F1 positiva, época menor y orden del plan. Las actualizaciones solo usan train. El modelo exportado recupera una copia de los pesos en la época elegida, no selecciona automáticamente el estado final. Sin refit train+val, sin ajuste mediante test.

| Experimento | Época elegida | Accuracy train | Accuracy val | BA val | F1 val |
|---|---:|---:|---:|---:|---:|
{table}

Seleccionado: **{chosen['id']}**, época **{model.best_epoch_}**. Configuración: `{model.get_params()}`.

## Métricas del modelo seleccionado

Las cifras son fracciones, no porcentajes.

| Métrica | Train | Validación |
|---|---:|---:|
{metrics}

Soporte train: {m['train']['support']}; validación: {m['val']['support']}. Matrices con filas reales/columnas predichas y orden 0,1: train `{m['train']['confusion_matrix']}`, val `{m['val']['confusion_matrix']}`. JSON conserva precisión completa y convenciones de denominadores cero. No se ejecutó evaluación predictiva de test.

## Errores y convergencia

En la ejecución seleccionada se recorrieron {len(h)} épocas. Errores durante actualizaciones: primera {h[0]['online_errors']}, última {h[-1]['online_errors']}; mínimo {min(r['online_errors'] for r in h)}, máximo {max(r['online_errors'] for r in h)}. Épocas de cero actualizaciones: {zero_epochs or 'ninguna'}. Hubo aumentos entre épocas: {increasing}.

Los errores online corresponden a distintos estados de pesos durante el recorrido; `train_errors_end` y accuracy se calculan con los pesos al cerrar cada época. El historial final es del recorrido completo y no se reescribe tras restaurar la mejor época. La línea de selección de los gráficos identifica el estado exportado. No se exige descenso monótono ni 100 % de exactitud. Si no hay cero errores, esta ejecución no demuestra separabilidad ni prueba su ausencia: el teorema de convergencia requiere separabilidad lineal y no fija 100 épocas como certificado.

## Pesos y sesgo seleccionados

Sesgo: **{model.bias_:.17g}**. Pesos float64 en JSON con ida/vuelta exacta y CSV con 17 cifras significativas:

| Característica escalada | Peso |
|---|---:|
{weights}

Estos coeficientes actúan en la escala estandarizada de fase 3, no sobre unidades originales. Hay redundancias entre lags, medias y cambios: magnitud/signo de un peso no equivale a importancia independiente ni efecto causal. Para inferencia, conservar el orden y los parámetros de preprocesamiento previos. Scores de decisión no son probabilidades.

## Artefactos y reproducción

Modelo seguro sin pickle en `modelo_seleccionado.json`; `pesos.csv`, `metricas.json`, confusiones y predicciones train/val con IDs trazables. Historial de ambos experimentos y resumen completo en `experimentos.json`. Los cuatro PNG bajo `graficos/` muestran errores online, accuracy train/val, BA val y confusión val, siempre de las {len(model.weights_)} entradas completas del modelo seleccionado. No hay frontera 2D ni PCA.

```powershell
python -m unittest discover -s tests -p 'test_perceptron.py' -v
python -m src.experiments.entrenar_perceptron
python -m unittest discover -s tests -p 'test_*.py' -v
```

El manifiesto vincula hashes de matrices, contrato, plan, código y resultados, con integridad antes/después de fuentes y fases previas. Cargar mediante `PerceptronSimple.load(path)`; `predict(X, feature_names=nombres)` valida orden cuando se proporcionan nombres. La inspección inicial de integridad comprobó forma/tipo/finitud/etiquetas del reservado sin predecir; el cargador del entrenamiento solo abre arrays train y val. La suite histórica incluye pruebas de preparación en destinos temporales, sin sustituir los entregables anteriores.

## Limitaciones y preparación de fase 5

Un clasificador lineal puede ser insuficiente para patrones de ventas y estacionalidad. La validación se usó para elegir entre 200 estados posibles; sus mejores métricas son de selección, no una estimación final independiente. Test permanece reservado. Superar la mediana reciente no mide demanda no atendida, rentabilidad, inventario ni utilidad económica. Ocho productos recientes de test carecen de representación en desarrollo; no se han evaluado aquí.

Para la fase 5, confirmar la biblioteca autorizada para gradientes de la futura regresión logística, mantener exactamente estas matrices/filas y reutilizar las métricas. Su implementación y comparación con test necesitan autorización de la fase correspondiente. No se implementó logística ni PCA; no se avanzó automáticamente.
"""


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
