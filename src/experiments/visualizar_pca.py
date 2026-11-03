"""python -m src.experiments.visualizar_pca — PCA visual y congelación, sin test."""
import argparse
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.auditar_datos import BASE, sha256
from src.experiments.entrenar_perceptron import load_development, read_json, write_json
from src.experiments.entrenar_logistica import hashes_previos, verificar_perceptron, evaluar_logistica
from src.models.perceptron import PerceptronSimple
from src.models.logistic_regression import LogisticRegressionManual
from src.visualization.pca import PCA2D
from src.visualization.dispersion import graficar_pca


def hashes_protegidos(base=BASE):
    base=Path(base)
    values=hashes_previos(base)
    previous=read_json(base/"resultados/fase_05/manifiesto.json")
    for relative in previous["implementation_sha256"]:
        values[relative]=sha256(base/relative)
    for p in sorted((base/"resultados/fase_05").rglob("*")):
        if p.is_file(): values[str(p.relative_to(base))]=sha256(p)
    return values


def verificar_manifiestos(base=BASE):
    """Detecta contradicciones antes de crear la congelación. Solo hashes de test."""
    base=Path(base)
    for phase, validation_field in [("fase_03","manifest_sha256"),("fase_04","phase04_manifest_sha256"),("fase_05","phase05_manifest_sha256")]:
        folder=base/"resultados"/phase
        manifest=read_json(folder/"manifiesto.json")
        validations=read_json(folder/"validaciones.json")
        if not validations["successful"] or validations[validation_field]!=sha256(folder/"manifiesto.json"):
            raise ValueError(f"Contradicción en hash/validación de {phase}; congelación detenida")
        if manifest["integrity"]["before"]!=manifest["integrity"]["after"]:
            raise ValueError(f"Contradicción en integridad de {phase}; congelación detenida")
        for field in ["implementation_sha256", "artifact_sha256" if phase=="fase_03" else "artifacts_sha256"]:
            for relative, expected in manifest[field].items():
                if sha256(base/relative)!=expected:
                    raise ValueError(f"Contradicción entre manifiesto y archivo: {relative}; congelación detenida")
        for relative,expected in manifest["integrity"]["after"].items():
            if sha256(base/relative)!=expected:
                raise ValueError(f"Fuente protegida alterada: {relative}; congelación detenida")
    return True


def preparar_congelacion(dev, base=BASE):
    """Carga pesos existentes y comprueba desarrollo; nunca llama fit de modelos."""
    base=Path(base)
    verificar_manifiestos(base)
    names=dev["feature_names"]
    models={"perceptron":PerceptronSimple.load(base/"resultados/fase_04/modelo_seleccionado.json"),
            "logistica":LogisticRegressionManual.load(base/"resultados/fase_05/modelo_seleccionado.json")}
    files={"perceptron":"resultados/fase_04/modelo_seleccionado.json","logistica":"resultados/fase_05/modelo_seleccionado.json"}
    p4=read_json(base/"resultados/fase_04/manifiesto.json")
    p5=read_json(base/"resultados/fase_05/manifiesto.json")
    for manifest in [p4,p5]:
        if manifest["phase03_manifest_sha256"]!=dev["manifest_hash"] or manifest["matrix_hashes_development"]!=dev["matrix_hashes"] or manifest["n_features"]!=len(names):
            raise ValueError("Contradicción de matrices/columnas entre fases; congelación detenida")
    p,l=models["perceptron"],models["logistica"]
    if (p.best_epoch_!=70 or p.best_epoch_!=p4["selected_epoch"] or p.learning_rate!=.01 or p.seed!=42
            or l.best_epoch_!=200 or l.best_epoch_!=p5["selected_epoch"] or l.learning_rate!=.1 or l.seed!=42
            or l.threshold!=.5 or p5["selected_threshold"]!=.5):
        raise ValueError("Configuración real contradice checkpoints aprobados; congelación detenida")
    for name,model in models.items():
        if model.feature_names_!=names or len(model.weights_)!=len(names):
            raise ValueError(f"Orden de características no coincide: {name}")
    metrics={"perceptron":verificar_perceptron(dev,base),"logistica":evaluar_logistica(l,dev)}
    if metrics["logistica"]!=read_json(base/"resultados/fase_05/metricas.json"):
        raise ValueError("Contradicción de métricas logísticas de desarrollo; congelación detenida")
    records={}
    for name,model in models.items():
        records[name]={"file":files[name],"sha256":sha256(base/files[name]),"state":read_json(base/files[name]),
                       "decision_rule":"1 si dot(X,w)+b >= 0; 0 en otro caso" if name=="perceptron" else "1 si sigmoid(dot(X,w)+b) >= 0.5; 0 en otro caso",
                       "development_metrics":metrics[name]}
    datafolder=base/"datos/Procesados/fase_03"
    files_data={str(p.relative_to(base)):sha256(p) for p in sorted(datafolder.iterdir()) if p.is_file()}
    train_names=sorted(dev["train"]["meta"].producto_provisional.unique().tolist())
    positive=int((dev["train"]["y"]==1).sum()); negative=int((dev["train"]["y"]==0).sum())
    freeze={"freeze_version":1,"contract_version":"2.0","models":records,
            "dataset":{"phase03_manifest":"resultados/fase_03/manifiesto.json","phase03_manifest_sha256":dev["manifest_hash"],
                       "feature_names":names,"n_features":len(names),"counts":dev["manifest"]["counts"],
                       "partitions":dev["manifest"]["partitions"],"source":dev["manifest"]["source"],"source_sha256":dev["manifest"]["source_sha256"],
                       "contracts_sha256":{str(p.relative_to(base)):sha256(p) for p in sorted((base/"resultados/fase_02").glob("*.md"))},
                       "label_rule":"1[Q_objetivo > mediana(Q_t-1,Q_t-2,Q_t-3)]; empate 0",
                       "minimum_consecutive_history_months":3,"product_identity":"NombreProducto como cadena exacta",
                       "coverage":dev["manifest"]["coverage"],"files_sha256":files_data,
                       "preprocessing_parameters":read_json(datafolder/"preprocesador.json"),
                       "row_order":"mes_objetivo ascendente, producto_provisional exacto ascendente",
                       "canonical_labels":[0,1]},
            "baseline_rules":{"majority_class":int(positive>negative),"train_positive":positive,"train_negative":negative,
                              "tie_majority":0,"historical":"1[unidades_lag_1 > R_referencia]; empate -> 0"},
            "new_product_rule":{"definition":"nombre provisional exacto no presente en train","train_products":train_names,
                                "expected_test_new_products":8,"expected_test_new_rows":16},
            "evaluation":{"primary_metric":"balanced_accuracy","secondary_metrics":["accuracy","precision","recall","f1","confusion_matrix","support"],
                          "subsets":["global","month","known_products","new_products"],"fit_forbidden":True,
                          "change_after_test_forbidden":["weights","bias","threshold","features","preprocessing","partitions","labels","baseline_rules"],
                          "test_predictive_evaluation_executed":False,"PCA_used_by_classifiers":False}}
    return models,freeze


def run(base=BASE):
    base=Path(base)
    before=hashes_protegidos(base)
    verificar_manifiestos(base)
    dev=load_development(base)  # Este cargador solo abre arrays train/val.
    models,freeze=preparar_congelacion(dev,base)
    untouched={s:dev[s]["X"].copy() for s in ["train","val"]}
    predictions={name:{s:model.predict(dev[s]["X"],feature_names=dev["feature_names"]) for s in ["train","val"]} for name,model in models.items()}
    pca=PCA2D().fit(dev["train"]["X"],feature_names=dev["feature_names"])
    projections={s:pca.transform(dev[s]["X"],feature_names=dev["feature_names"]) for s in ["train","val"]}
    for split in ["train","val"]:
        if not np.array_equal(untouched[split],dev[split]["X"]):
            raise AssertionError("PCA alteró la matriz de clasificación")
        for name,model in models.items():
            if not np.array_equal(predictions[name][split],model.predict(dev[split]["X"],feature_names=dev["feature_names"])):
                raise AssertionError("PCA alteró las predicciones multidimensionales")
    out=base/"resultados/fase_06"; out.mkdir(parents=True,exist_ok=True)
    pca.save(out/"parametros_pca.json")
    ratios=pca.explained_variance_ratio_all_
    variance={"individual_ratios_all":ratios.tolist(),"cumulative_ratios_all":np.cumsum(ratios).tolist(),
              "PC1":float(ratios[0]),"PC2":float(ratios[1]),"retained_2D":float(ratios[:2].sum()),
              "discarded":float(max(0,1-ratios[:2].sum())),"total_train_variance":pca.total_variance_,
              "variance_ddof":1,"numerical_rank":pca.rank_,"fit_partition":"train_only"}
    write_json(out/"varianza_explicada.json",variance)
    for split in ["train","val"]:
        np.save(out/f"Z_{split}.npy",projections[split],allow_pickle=False)
        metadata=dev[split]["meta"][["indice_matriz","observacion_id","producto_provisional","mes_objetivo"]].copy()
        metadata["clase_visual"]=dev[split]["y"]
        metadata.to_csv(out/f"metadatos_{split}.csv",index=False,lineterminator="\n")
    weights=pd.DataFrame({"feature":pca.feature_names_,"PC1_coefficient":pca.components_[:,0],"PC2_coefficient":pca.components_[:,1],
                          "PC1_squared_contribution":pca.components_[:,0]**2,"PC2_squared_contribution":pca.components_[:,1]**2})
    weights.to_csv(out/"coeficientes_componentes.csv",index=False,float_format="%.17g",lineterminator="\n")
    write_json(out/"congelacion_experimental.json",freeze)
    (out/"protocolo_test.md").write_text(protocolo(),encoding="utf-8")
    (out/"reporte_pca.md").write_text(reporte(pca,variance,freeze),encoding="utf-8")
    graficar_pca(projections,{s:dev[s]["y"] for s in ["train","val"]},ratios,out/"graficos")
    recovered=PCA2D.load(out/"parametros_pca.json")
    for split in ["train","val"]:
        if not np.array_equal(projections[split],recovered.transform(dev[split]["X"],feature_names=dev["feature_names"])):
            raise AssertionError("La serialización PCA cambia las proyecciones")
    after=hashes_protegidos(base)
    if before!=after:
        raise RuntimeError("Se modificó un archivo previo; congelación no válida")
    code=[base/p for p in ["src/visualization/__init__.py","src/visualization/pca.py","src/visualization/dispersion.py",
                           "src/experiments/visualizar_pca.py","tests/test_pca.py"]]
    outputs={str(p.relative_to(base)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"manifiesto.json","validaciones.json"}}
    write_json(out/"manifiesto.json",{"phase":6,"fit_partition":"train","test_arrays_loaded_by_phase06":False,
                "test_predictive_evaluation":False,"n_classifier_features":len(dev["feature_names"]),"n_visual_components":2,
                "retained_variance_2d":variance["retained_2D"],"freeze_sha256":sha256(out/"congelacion_experimental.json"),
                "integrity":{"preserved":True,"before":before,"after":after},"artifacts_sha256":outputs,
                "implementation_sha256":{str(p.relative_to(base)):sha256(p) for p in code},
                "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":__import__("matplotlib").__version__}})
    print({"PC1":variance["PC1"],"PC2":variance["PC2"],"retained":variance["retained_2D"],"discarded":variance["discarded"],
           "rank":pca.rank_,"protected_files":len(before),"models":{name:r["sha256"] for name,r in freeze["models"].items()}})
    return pca,freeze


def protocolo():
    return """# Protocolo cerrado de evaluación final — Fase 7

## Estado y alcance

Este documento es especificación: la fase 6 no implementa ni ejecuta predicciones/métricas de test. Su autoridad es congelacion_experimental.json, enlazado mediante SHA-256 en manifiesto.json de fase 6. Esa congelación fija los modelos originales, datos, transformaciones, reglas y cohortes. Ninguna etiqueta o predicción de test se usó para elegir PCA o el protocolo.

## Secuencia obligatoria antes de abrir test

1. Verificar el hash del archivo de congelación contra el manifiesto de fase 6; verificar los hashes de modelos, matrices X/y, metadatos, características, observaciones, preprocesador y contratos contra la congelación. Si cualquier archivo difiere, detener y documentar la discrepancia; no reentrenar ni regenerar datos para forzar coincidencias.
2. Cargar los JSON originales mediante PerceptronSimple.load y LogisticRegressionManual.load, sin llamar fit. Confirmar que estado, pesos, sesgos, 20 nombres/orden, checkpoints y reglas coinciden con la congelación: perceptrón η=0,01, época 70, semilla 42; logística η=0,1, época 200, semilla 42 y umbral 0,5. No elegir umbral nuevo ni reajustar modelos con train+val.
3. Cargar exactamente X_test.npy, y_test.npy y metadatos_test.csv de fase 3, una sola población común. Verificar float64/int8, finitud, etiquetas 0/1, forma 1.111×20 y 1.111, correspondencia de indice_matriz, claves producto×mes únicas y orden por mes/nombre exacto. El modelo no recibe nombres/IDs como predictores. Los 1.111 casos dependen del supuesto académico de cobertura, no de una certificación externa de cierre.
4. Verificar mismo orden de columnas con caracteristicas.json y estado congelado. No aplicar el escalador otra vez: X_test ya está preparado. No seleccionar/excluir columnas ni observaciones por sus resultados. Si se detecta cobertura realmente incompleta, detener la evaluación y registrar la inconsistencia; no escoger períodos por mejorar métricas.

## Reglas fijas de predicción

Perceptrón: z=X_test w+b, clase 1 si z>=0, clase 0 en otro caso. Scores no son probabilidades. Logística: p=sigmoid(X_test w+b), clase 1 si p>=0,5; clase 0 en otro caso. Usar los mismos índices, sin PCA. La etiqueta no cambia: y=1[Q_objetivo>mediana de los tres meses previos], empate→0. No reconstruirla con umbrales estimados sobre test.

Baseline mayoritario: predecir clase 1, fijada por 2.382 positivos y 2.217 negativos de entrenamiento (empate hubiera sido clase 0). No recalcular mayoría de test ni por mes/cohorte. Baseline histórico: 1[unidades_lag_1>R_referencia], empate→0, usando los predictores históricos crudos de observaciones.csv alineados por observacion_id a las mismas filas de metadatos_test. No comparar directamente columnas estandarizadas con medias/escalas diferentes. No usar Q_objetivo ni y_test para construir baseline.

La información histórica de meses previos de test puede estar ya incorporada causalmente en X_test: son pronósticos mensuales de un paso. Enero de 2016 usa octubre-diciembre de 2015; abril puede usar enero-marzo de 2016. Esto no autoriza fit, búsqueda o actualización de parámetros después de mirar los errores. No es un pronóstico de cinco meses emitido en diciembre.

## Métricas y tablas fijadas

Reutilizar metricas_binarias de fase 4. Principal: balanced accuracy; secundarias: accuracy, precision, recall y F1 de clase positiva, matriz de confusión y soporte/proporción por clase. Matriz con filas reales/columnas predichas, orden 0,1: [[TN,FP],[FN,TP]]. Accuracy=(TP+TN)/N; precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2TP/(2TP+FP+FN); BA=(recall+TN/(TN+FP))/2.

Con denominador cero: precision=0 y bandera; recall no estimable sin positivos reales; especificidad no estimable sin negativos reales; BA no estimable si falta una clase real; F1=0 si su denominador es cero, con bandera. N=0 no genera métrica. No sustituir BA no estimable por 0,5 ni borrar grupos sin ambas clases.

Reportar para los cuatro métodos: global (todos los casos), cada uno de los cinco meses ene-may de 2016 y cohortes conocidas/nuevas. Publicar N, soportes, matriz y métricas de cada grupo, incluidos resultados desfavorables. N esperados por mes: 219/219/219/227/227 bajo cobertura académica. El promedio mensual se calcula solo sobre métricas estimables, declarando cuántos meses entran; no sustituye las métricas globales calculadas a partir de conteos sumados.

Productos nuevos: nombre exacto no incluido en la lista train_products congelada; esa lista se obtuvo exclusivamente de train. Esperados ocho productos y 16 filas elegibles (abril-mayo); conocidos 1.095 filas. No significa fecha de alta comercial: solo ausencia del nombre en entrenamiento. No incorporarlos retrospectivamente al vocabulario ni modificar la elegibilidad para evaluar más meses. Reportar por cohorte y, si resulta estimable, cohorte×mes con los mismos métodos; reconocer soportes pequeños.

Comparar descriptivamente los resultados finales con train/validación ya guardados en fases 4/5 y con baselines congelados. Indicar que validación se utilizó para selección y tiene optimismo, mientras test es la evaluación reservada de configuraciones previamente fijadas. No presentar diferencias pequeñas como significativas ni usar test para elegir y volver a evaluar otro checkpoint. No inventar pérdida probabilística del perceptrón/baselines; la BCE logística es un diagnóstico adicional opcional con logits congelados, no criterio para retocar el modelo.

## Cierre y límites

Exportar predicciones de los cuatro métodos con IDs y orden común, probabilidades solo de logística y scores identificados del perceptrón. Guardar métricas JSON global/mes/cohorte, matrices, configuración/hashes y las incidencias de cobertura. Mantener las respuestas separadas de predictores. No sobrescribir los modelos congelados.

Tras observar test queda prohibido cambiar pesos, sesgos, umbrales, características, escaladores/imputadores, particiones, población, definición de etiqueta o baselines dentro de esta evaluación final. Cualquier investigación nueva requerirá otra versión y una nueva reserva de evaluación; no reutilizar este test como validación encubierta.

Solo hay cinco meses y productos/ventanas correlacionados: no asumir 1.111 réplicas independientes ni bootstrap iid de líneas para afirmar significación. Cualquier incertidumbre debe reconocer dependencia temporal y pocos bloques; la cohorte nueva tiene solo 16 observaciones. No se define un coste monetario que no haya sido aprobado. Las probabilidades logísticas no se consideran calibradas por el solo uso de sigmoidea.

PCA está fuera de este proceso: Z_train/Z_val de fase 6 son solo gráficos, no entradas de los clasificadores ni sustitutos de X_test. No usar separación/solapamiento visual para reajustar modelos después de test. Este protocolo queda preparado para la fase 7, que requiere autorización independiente.
"""


def reporte(pca,variance,freeze):
    top=[]
    for j in range(2):
        order=np.argsort(-np.abs(pca.components_[:,j]),kind="stable")[:5]
        top.append("\n".join(f"| PC{j+1} | {pca.feature_names_[i]} | {pca.components_[i,j]:.9f} | {pca.components_[i,j]**2*100:.4f} % |" for i in order))
    models="\n".join(f"- {name}: `{r['file']}`, SHA-256 `{r['sha256']}`." for name,r in freeze["models"].items())
    return f"""# PCA visual y congelación — Fase 6

## Resultado y método

PCA propio NumPy, exclusivamente visual, ajustado con X_train (4.599×20) y aplicado a X_val (2.628×20). Se reutilizó el escalado aprobado sin reestandarizar particiones. Se centra Xc=X−media_train; SVD Xc=UΣVᵀ; W2 son las dos primeras columnas de V, y Z=Xc W2. El clasificador sigue usando X de 20 dimensiones. PCA no recibe y: las etiquetas se consultan únicamente para colorear los gráficos de desarrollo.

Convención de signo: en cada componente, el coeficiente de mayor magnitud se hace positivo; si hay empate, manda el primer índice de entrada. Repetir en el mismo entorno conserva resultados. SVD puede elegir otra base dentro de subespacios con valores singulares iguales o casi iguales en distintas bibliotecas/LAPACK: el signo no resuelve esa ambigüedad rotacional. Se conserva la transformación exacta en JSON para su reutilización.

Autovalores de covarianza: sigma_j²/(m−1), ddof=1. El denominador de las proporciones es la suma de **todas** las varianzas de las 20 características transformadas de train, no solo las dos mayores. Rango numérico: {pca.rank_}, tolerancia eps*max(m,d)*sigma_max={pca.rank_tolerance_:.9g}. Las columnas constantes pueden tener coeficientes cero; rango menor que dos bloquea la proyección en lugar de inventar un segundo componente informativo.

| Medida | Proporción | Porcentaje |
|---|---:|---:|
| PC1 | {variance['PC1']:.12f} | {variance['PC1']*100:.6f} % |
| PC2 | {variance['PC2']:.12f} | {variance['PC2']*100:.6f} % |
| Conservada en 2D | {variance['retained_2D']:.12f} | {variance['retained_2D']*100:.6f} % |
| Descartada | {variance['discarded']:.12f} | {variance['discarded']*100:.6f} % |

Varianza total de train (ddof=1): {pca.total_variance_:.12g}. PCA maximiza varianza conservada, no separación entre clases. Las características de fase 3 contienen redundancias algebraicas y correlaciones: el rango puede ser menor que 20 sin invalidar las dos primeras direcciones.

## Interpretación de coeficientes

Se registran todos los coeficientes firmados y sus cuadrados en coeficientes_componentes.csv. Aquí se muestran los cinco mayores por magnitud en cada componente. El cuadrado es participación en la norma unitaria del vector, no importancia predictiva ni efecto empresarial causal.

| Componente | Característica estandarizada | Coeficiente firmado | Coeficiente² / norma² |
|---|---|---:|---:|
{top[0]}
{top[1]}

En esta copia, PC1 resume principalmente el nivel de unidades históricas: media, mediana, lags y unidades por factura tienen coeficientes grandes positivos. PC2 combina frecuencia comercial (facturas y clientes distintos) y cambios recientes de unidades, también positivos en la convención elegida. Sus coeficientes menores negativos incluyen unidades_lag_2 ({pca.components_[1,1]:.6f}) y unidades_lag_3 ({pca.components_[2,1]:.6f}); no son categorías de clase. PC1 tiene una contribución negativa menor del precio unitario ponderado ({pca.components_[13,0]:.6f}). Las dos clases se solapan claramente en las dispersiones: no aparece una división nítida en este plano.

Signo positivo indica aumento de la coordenada con esa variable manteniendo las otras constantes en la combinación; signo negativo indica contribución en sentido opuesto. Esta lectura algebraica no autoriza intervenciones causales y puede carecer de interpretación independiente cuando las variables son derivadas de las mismas ventas. Las características son las entradas ya estandarizadas, no las medidas crudas.

## Gráficos y fidelidad

Se exportaron graficos/pca_train.png, pca_val.png y varianza_explicada.png. Las dispersiones conservan todos los puntos, con transparencia y colores por clase, sin jitter ni selección para aparentar separación. Se usa el mismo rango de ejes para train y val, determinado por ambas proyecciones de desarrollo sin alterar W2. La figura de varianza muestra porcentajes individuales, acumulado de todos los componentes y proporción conservada en 2D. No hay proyección de test, frontera de decisión ni entrenamiento en 2D.

El solapamiento entre clases debe interpretarse como limitación de esta vista no supervisada. Ni separación ni solapamiento permiten concluir por sí solos rendimiento, separabilidad o causalidad del clasificador en las 20 entradas. La pérdida de varianza en 2D puede ocultar señales relevantes para clasificar.

## Modelos congelados

{models}

Perceptrón: época 70, η=0,01, semilla 42, regla z>=0. Logística: época 200, η=0,1, semilla 42, probabilidad>=0,5. congelacion_experimental.json registra estados completos (pesos, sesgos, configuración y nombres ordenados), hashes de datos/modelos, parámetros del preprocesador, contratos/particiones y reglas de baselines/cohortes. No se copian/reentrenan modelos ni se cambian umbrales. La verificación contra manifiestos previos detiene el procedimiento ante cualquier contradicción.

Se comprobaron matrices y predicciones train/val antes/después de PCA: iguales. Los modelos no reciben Z. La API PCA no acepta etiquetas y su fit solo se llama con train. El proceso abre arrays de train y val; test solo está referenciado por hashes aprobados y no se consultan sus etiquetas ni se calculan métricas.

## Archivos y reproducción

Nuevos módulos: src/visualization/pca.py, dispersion.py, __init__.py; src/experiments/visualizar_pca.py y tests/test_pca.py. En resultados/fase_06: parametros_pca.json (media, W2, valores singulares, varianzas, nombres, convención), varianza_explicada.json, Z_train.npy/Z_val.npy, metadatos alineados por indice_matriz/observacion_id, coeficientes_componentes.csv, tres gráficos, congelación, protocolo_test.md y manifiesto/validaciones. Usar np.load(..., allow_pickle=False); no hay pickle. PCA2D.load(...).transform(X, feature_names=nombres) aplica el mismo centrado y componentes, sin fit ni otro escalado.

```powershell
python -m src.experiments.visualizar_pca
python -m unittest discover -s tests -p 'test_pca.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

La suite histórica verifica preparación en destinos temporales y entrenamientos sintéticos de control; no reentrena los clasificadores WWI ni sustituye entregables. Las validaciones de esta fase comprueban matemáticas PCA, ortogonalidad, varianza, persistencia, train-only, independencia de etiquetas/test, compatibilidad de la congelación y reproducción de artefactos, con integridad previa/posterior.

## Pendientes y fase 7

Persisten cobertura académica no certificada, identidad por nombre y unidad comercial sin catálogo histórico; el pequeño grupo nuevo y cinco meses de test limitan inferencias. No hay contradicciones detectadas en modelos/manifiestos. El protocolo cerrado especifica evaluación común global/mes/cohorte y prohíbe modificar decisiones después de observar test. Test permanece sin evaluación predictiva. Esta fase termina aquí y no implementa ni ejecuta la fase 7.
"""


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
