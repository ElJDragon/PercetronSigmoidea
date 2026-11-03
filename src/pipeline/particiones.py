"""Partición únicamente por mes objetivo; sin aleatoriedad."""

BOUNDS = {"train": ("2013-04", "2014-12"), "val": ("2015-01", "2015-12"),
          "test": ("2016-01", "2016-05")}


def particion(mes):
    for name, (start, end) in BOUNDS.items():
        if start <= mes <= end:
            return name
    return "fuera_particion"


def separar(observaciones):
    active = observaciones[observaciones.elegible_para_prediccion & observaciones.respuesta_disponible]
    result = {name: active.loc[active.particion == name].copy() for name in BOUNDS}
    keys = [set(frame.observacion_id) for frame in result.values()]
    if any(keys[i] & keys[j] for i in range(len(keys)) for j in range(i)):
        raise AssertionError("Claves compartidas entre particiones")
    return result
