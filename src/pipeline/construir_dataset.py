"""python -m src.pipeline.construir_dataset --cobertura academica"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from src.auditar_datos import BASE, sha256
from .agregacion import agregar_producto_mes, cargar_fuente
from .caracteristicas import Cobertura, FEATURES, FEATURE_RULES, OPTIONAL, construir_observaciones
from .particiones import BOUNDS, separar
from .preprocesamiento import PreprocesadorComun

CONTRACT_VERSION = "2.0"
SOURCE_HASH = "d4f3b3b7b018ee49c515915fe851ff7327d82fb198cbc8af117e8f0e73fff37e"
META = ["observacion_id", "producto_provisional", "mes_objetivo", "particion", "corte_exclusivo",
        "primer_mes_observado_conocido", "cobertura_objetivo", "origen_cobertura", "estado_mes_objetivo",
        "historia_meses", "historia_estados"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")


def run(coverage, *, output_dir=None, report_dir=None):
    source = BASE / "datos/originales/datos_limpios.csv"
    protected = [p for folder in ["datos/originales", "referencias", "resultados/auditoria", "resultados/fase_02"]
                 for p in sorted((BASE / folder).rglob("*")) if p.is_file()]
    # También impedir modificaciones accidentales a código/pruebas de fases aprobadas.
    protected += [BASE / p for p in ["src/auditar_datos.py", "src/analizar_etiquetas.py", "tests/test_auditoria.py", "tests/test_etiquetas.py"]]
    before = {str(p.relative_to(BASE)): sha256(p) for p in protected}
    audit = json.loads((BASE / "resultados/auditoria/perfilado_datos.json").read_text(encoding="utf-8"))
    previous = json.loads((BASE / "resultados/fase_02/exploracion_etiquetas.json").read_text(encoding="utf-8"))
    if previous["source_sha256"] != SOURCE_HASH:
        raise ValueError("Hash del contrato de fase 2 distinto de la fuente fijada")
    raw, fmt = cargar_fuente(source, SOURCE_HASH, audit["datasets"][source.name]["column_names"])
    agg = agregar_producto_mes(raw)
    calendar, obs = construir_observaciones(agg, coverage)
    sets = separar(obs)
    counts = {name: len(frame) for name, frame in sets.items()}
    expected = {"train": 4599, "val": 2628, "test": 1111}
    if coverage.modo == "academica" and not coverage.meses and counts != expected:
        raise ValueError(f"Conteos difieren del contrato: {counts}; no se exportan matrices")
    # Comparar solo conteos/etiquetas de desarrollo aprobados; no explorar test.
    for name, old in [("train", "train"), ("val", "validation")]:
        if counts[name] == expected[name]:
            expected_positive = previous["B_trailing_3_median"][old]["positive"]
            if int(sets[name].y.sum()) != expected_positive:
                raise ValueError(f"Etiqueta de {name} difiere de fase 2")
    pre = PreprocesadorComun().fit(sets["train"][FEATURES], particion="train")
    matrices = {name: pre.transform(frame[FEATURES]) for name, frame in sets.items()}
    labels = {name: frame.y.to_numpy(dtype=np.int8) for name, frame in sets.items()}
    if not all(np.isin(y, [0, 1]).all() for y in labels.values()):
        raise AssertionError("Etiquetas no canónicas")
    output = Path(output_dir).resolve() if output_dir is not None else BASE / "datos/Procesados/fase_03"
    reports = Path(report_dir).resolve() if report_dir is not None else BASE / "resultados/fase_03"
    if not output.is_relative_to(BASE) or not reports.is_relative_to(BASE):
        raise ValueError("Los destinos deben permanecer dentro del workspace")
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    # No producir arrays vacíos como si se tratara de una preparación aceptada.
    if any(counts[name] == 0 for name in BOUNDS):
        raise ValueError(f"Sin respuestas confirmadas en alguna partición: {counts}; matrices bloqueadas")
    current = {str(p.relative_to(BASE)): sha256(p) for p in protected}
    if current != before:
        raise RuntimeError("Se modificó un archivo protegido; no se exporta")
    write_csv(calendar, output / "producto_mes.csv")
    write_csv(obs, output / "observaciones.csv")
    for name in BOUNDS:
        np.save(output / f"X_{name}.npy", matrices[name], allow_pickle=False)
        np.save(output / f"y_{name}.npy", labels[name], allow_pickle=False)
        metadata = sets[name][META].copy()
        metadata.insert(0, "indice_matriz", np.arange(len(metadata), dtype=np.int64))
        metadata["version_contrato"] = CONTRACT_VERSION
        metadata["fuente_sha256"] = SOURCE_HASH
        write_csv(metadata, output / f"metadatos_{name}.csv")
    pre.save(output / "preprocesador.json")
    write_json(output / "caracteristicas.json", {
        "version_contrato": CONTRACT_VERSION, "input_features": FEATURES,
        "feature_names": pre.estado["feature_names"], "categorical_features": [],
        "rules": {name: {"formula": rule, "availability": "historia_fecha_menor_al_corte_o_calendario_conocido",
                         "missing": "mediana_train_si_no_esencial" if name in OPTIONAL else "historia_esencial_o_calendario_obligatorio"}
                  for name, rule in FEATURE_RULES.items()}, "discarded": pre.estado["discarded"]})
    report_text = report(counts, pre, obs, calendar, coverage)
    (reports / "reporte_ejecucion.md").write_text(report_text, encoding="utf-8")
    after = {str(p.relative_to(BASE)): sha256(p) for p in protected}
    if before != after:
        raise RuntimeError("Cambió un archivo protegido durante la exportación")
    # Hash de implementaciones para reproducir la versión exacta del procedimiento.
    implementation = {str(p.relative_to(BASE)): sha256(p) for p in sorted((BASE / "src/pipeline").glob("*.py"))}
    implementation["tests/test_pipeline.py"] = sha256(BASE / "tests/test_pipeline.py")
    artifacts = {str(p.relative_to(BASE)): sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    artifacts[str((reports / "reporte_ejecucion.md").relative_to(BASE))] = sha256(reports / "reporte_ejecucion.md")
    manifest = {"version_contrato": CONTRACT_VERSION, "source": str(source.relative_to(BASE)),
                "source_sha256": SOURCE_HASH, "format": fmt, "coverage": asdict(coverage),
                "counts": counts, "expected_counts_academic": expected,
                "partitions": BOUNDS, "features_before": len(FEATURES), "features_after": len(pre.estado["feature_names"]),
                "dtype_X": "float64", "dtype_y": "int8", "label_values": [0, 1],
                "row_order": "mes_objetivo ascendente, producto_provisional exacto ascendente",
                "development_label_counts": {name: {"positive": int(sets[name].y.sum()), "negative": int((sets[name].y == 0).sum())} for name in ["train", "val"]},
                "test_label_statistics": "no calculadas ni reportadas para decisiones de desarrollo",
                "integrity": {"preserved": True, "before": before, "after": after},
                "implementation_sha256": implementation, "artifact_sha256": artifacts,
                "environment": {"numpy": np.__version__, "pandas": pd.__version__}}
    write_json(reports / "manifiesto.json", manifest)
    print(json.dumps({"counts": counts, "features_before": len(FEATURES), "features_after": len(pre.estado["feature_names"]),
                      "discarded": pre.estado["discarded"], "protected_files": len(before), "coverage": coverage.modo}, ensure_ascii=False))
    return manifest


def report(counts, pre, obs, calendar, coverage):
    discarded = "\n".join(f"- `{k}`: {v}." for k, v in pre.estado["discarded"].items())
    names = "\n".join(f"{i+1}. `{name}`" for i, name in enumerate(pre.estado["feature_names"]))
    return f"""# Ejecución del pipeline WWI — Fase 3

## Resultado y política

Contrato 2.0. Cobertura seleccionada: **{coverage.modo}**. {coverage.evidencia}.
El modo académico asume completos los meses de enero de 2013 a mayo de 2016: no certifica cierres ni latencia del origen. La etiqueta sigue siendo Q_objetivo > mediana(t−1,t−2,t−3), empate 0.

| Partición | Observaciones confirmadas para matrices |
|---|---:|
| train | {counts['train']} |
| val | {counts['val']} |
| test | {counts['test']} |

Calendario producto-mes: {len(calendar)} filas desde primera aparición; observaciones de productos reconocidos antes del corte: {len(obs)}. Las primeras apariciones no crean predicciones en su propio mes. No se materializan como cero los 288 pares previos a la primera observación de los ocho productos recientes. Las filas no elegibles se conservan con estado y sin y; las pendientes de cobertura no se convierten en clase 0.

## Características y transformación

Catálogo base: {len(FEATURES)} numéricas autorizadas, cero categóricas. Matriz final: {len(pre.estado['feature_names'])} columnas float64. Etiquetas canónicas int8, 0/1. Las tres razones definidas permiten nulos no esenciales, imputados con mediana de train. Se incorporó `facturas_lag_1_cero` como indicador explícitamente requerido para la razón unidades por factura; `mes_anterior_sin_ventas` también identifica precio sin denominador. No se generan indicadores indiscriminados.

Columnas descartadas exclusivamente por train:

{discarded}

No se retiraron columnas por correlación con y, rendimiento o inspección de test. Se conservan redundancias algebraicas entre lags y medias/cambios. Cada parámetro (mediana, media y desviación poblacional ddof=0) proviene solo de train. Las constantes se eliminan sin dividir por cero; no se seleccionó un número fijo de entradas.

Orden final:

{names}

Características condicionadas apagadas: identidad one-hot, Marca/Tamanio/EsRefrigerado, mezclas cliente/territorio, beneficio histórico, ventanas de 12 meses y calendario redundante mes/trimestre. Prohibidas como entradas: Q/y objetivo, resultados monetarios o clientes del horizonte, identificadores directos, variables de inventario inventadas y agregados futuros. R solo se usa como mediana histórica válida; el resto de respuestas permanece separado de X.

## Archivos y uso

En `datos/Procesados/fase_03/`: `producto_mes.csv` contiene agregados y cobertura; `observaciones.csv` contiene estados, claves, Q/R/y y características crudas. Son dos granos diferentes, no copias equivalentes. Los archivos X_train/X_val/X_test.npy y y_train/y_val/y_test.npy son el único formato de matrices/etiquetas. Leer con `numpy.load(path, allow_pickle=False)`. Para reconstruir exactamente las características float64 desde CSV con pandas, usar `read_csv(..., float_precision="round_trip")`; el CSV se escribe con 17 cifras significativas. Metadatos por partición alineados por indice_matriz; claves SHA-256 de la pareja producto-mes, sin usar identidad en X.

`caracteristicas.json` define orden, fórmulas y faltantes; `preprocesador.json` conserva todos los parámetros legibles. Para aplicar nuevos registros base autorizados, usar `PreprocesadorComun.load(...).transform(frame[FEATURES])`, sin fit. No hay vocabulario categórico porque el contrato base no autoriza categóricas. Una transformación cargada valida esquema y finitud; las características históricas se construyen antes con el mismo corte/contrato.

El manifiesto en resultados/fase_03 conserva cobertura, fuente, hashes de contrato/entradas y código, orden/tipos y hashes de derivados. No contiene distribuciones ni métricas de etiquetas test. Las respuestas de test se materializan únicamente para evaluación futura, sin usarlas en fit o selección. Ambos clasificadores deben consumir exactamente estos mismos arrays e índices; conversión bipolar posterior, sin alterar filas.

## Reproducción y verificaciones

Desde la raíz:

```powershell
python -m src.pipeline.construir_dataset --cobertura academica
python -m unittest discover -s tests -p 'test_*.py' -v
```

El modo por defecto es `desconocida`: bloquea matrices si no hay historia/respuestas confirmadas. Para cierres externos usar `--cobertura explicita --archivo-cobertura ruta.json`, con estructura `{{"meses":{{"2013-01":"completa"}},"evidencia":"procedencia del cierre"}}`. Meses no declarados quedan desconocidos. El archivo no puede certificar ventas que aún no ocurrieron; quien lo configura debe aportar cierres conocidos al corte. Una ejecución bloqueada no reemplaza resultados de una ejecución académica previa: consultar el manifiesto de la preparación que se vaya a utilizar.

Se verifican hash/esquema/separador contra contrato, sumas de unidades y centavos, facturas distintas, claves, conteos académicos y etiquetas train/val contra fase 2. Los agregados monetarios suman centavos exactos para que el orden de las líneas no cambie su resultado. La historia se desplaza por calendario antes de etiquetar; test anterior puede ser historia de test posterior, sin reajustar transformaciones. El conjunto completo de pruebas incluye perturbación de objetivo/futuro/test, cobertura, fronteras temporales, serialización y reproducción; sus resultados se entregan junto con la ejecución.

## Límites antes de entrenar

No hay modelos ni PCA ni gráficos de entrenamiento. No se modificaron fuentes, referencias ni entregables aprobados. La utilidad económica de superar la mediana no está demostrada. Identidad por nombre, unidades de empaque y disponibilidad al cierre requieren confirmación para uso operacional. Sin metadatos del origen, la completitud es un supuesto académico explícito. Antes de fase de modelos confirmar biblioteca de gradientes autorizada; PCA sigue separado de X y requiere su autorización. Esta ejecución termina aquí, sin avanzar a fase 4.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cobertura", choices=["academica", "desconocida", "explicita"], default="desconocida")
    parser.add_argument("--archivo-cobertura", type=Path)
    args = parser.parse_args()
    config = {}
    if args.archivo_cobertura:
        config = json.loads(args.archivo_cobertura.read_text(encoding="utf-8"))
    if args.cobertura == "explicita" and not args.archivo_cobertura:
        parser.error("Cobertura explícita requiere archivo con cierres y procedencia")
    coverage = Cobertura(modo=args.cobertura, meses=config.get("meses", {}),
                         evidencia=config.get("evidencia", "Supuesto académico explícito de completitud y disponibilidad al cierre; no certificado" if args.cobertura == "academica" else "Sin certificación de cierres"))
    try:
        run(coverage)
    except (ValueError, AssertionError, RuntimeError) as exc:
        parser.exit(1, f"Preparación bloqueada: {exc}\n")


if __name__ == "__main__":
    main()
