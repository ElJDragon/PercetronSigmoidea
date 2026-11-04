# Etiquetas de cabeceras: solo desarrollo

Cada observación es un OrderID contado una vez. Todas las consultas de datos filtran OrderDate < 2016-01-01 antes de clasificar; no hay outcomes test. Los conteos se recalcularon desde Sales.Orders, no se copiaron de las líneas. Se conserva el diagnóstico de todas las cabeceras separado de la cohorte empresarial defendible.

## Estados exhaustivos al corte

| Estado | Train | Validación |
|---|---:|---:|
| invalid | 0 | 0 |
| insufficient_followup | 172 | 154 |
| missing_pick_indeterminate | 1217 | 1197 |
| pick_not_observed_by_cut | 349 | 187 |
| row_not_available_by_cut | 0 | 0 |
| within_two_days | 37179 | 21093 |
| after_two_days | 1732 | 698 |
| **Total de documentos** | **40649** | **23329** |

Prioridad: fecha incoherente, seguimiento incompleto, picking nulo, picking posterior/al corte, versión no disponible y clases 0/1. Los motivos no se suman dos veces. El picking nulo es indeterminado: no se cuenta automáticamente como atraso, cancelación ni resultado negativo. Las cabeceras completas en la copia pero observadas después del corte no generan etiquetas disponibles en ese corte.

## Por cohorte y año

| Cohorte | Año | Documentos de origen | Etiquetables | Positivos | Negativos | Prevalencia |
|---|---|---:|---:|---:|---:|---:|
| Todas las cabeceras (solo diagnóstico) | 2013 | 19450 | 18940 | 957 | 17983 | 5.0528 % |
| Todas las cabeceras (solo diagnóstico) | 2014 | 21199 | 19971 | 775 | 19196 | 3.8806 % |
| Todas las cabeceras (solo diagnóstico) | 2015 | 23329 | 21791 | 698 | 21093 | 3.2032 % |
| Original sin backorder | 2013 | 15770 | 15769 | 0 | 15769 | 0.0000 % |
| Original sin backorder | 2014 | 16917 | 16770 | 0 | 16770 | 0.0000 % |
| Original sin backorder | 2015 | 18445 | 18318 | 0 | 18318 | 0.0000 % |
| Original con pendientes trasladados | 2013 | 1808 | 1808 | 3 | 1805 | 0.1659 % |
| Original con pendientes trasladados | 2014 | 2108 | 2095 | 1 | 2094 | 0.0477 % |
| Original con pendientes trasladados | 2015 | 2396 | 2381 | 0 | 2381 | 0.0000 % |
| Derivado terminal | 2013 | 1808 | 1299 | 913 | 386 | 70.2848 % |
| Derivado terminal | 2014 | 2108 | 1048 | 732 | 316 | 69.8473 % |
| Derivado terminal | 2015 | 2396 | 1009 | 654 | 355 | 64.8167 % |
| Derivado que vuelve a trasladar | 2013 | 64 | 64 | 41 | 23 | 64.0625 % |
| Derivado que vuelve a trasladar | 2014 | 66 | 58 | 42 | 16 | 72.4138 % |
| Derivado que vuelve a trasladar | 2015 | 92 | 83 | 44 | 39 | 53.0120 % |

Train incluye 2013–2014 con corte común 2015-01-01, no cortes ficticios al final de cada año. Validación es 2015 con corte 2016-01-01. Las prevalencias son proporciones entre outcomes justificados al corte, con sesgo de selección por observación. Cabeceras pendientes/indeterminadas no desaparecen del inventario de estados.

## Mensual: todas las cabeceras, diagnóstico no aprobado como dataset

| Mes | Documentos | Etiquetables | Positivos | Negativos | Prevalencia |
|---|---:|---:|---:|---:|---:|
| 2013-01 | 1674 | 1664 | 49 | 1615 | 2.9447 % |
| 2013-02 | 1139 | 1120 | 46 | 1074 | 4.1071 % |
| 2013-03 | 1683 | 1656 | 87 | 1569 | 5.2536 % |
| 2013-04 | 1696 | 1663 | 109 | 1554 | 6.5544 % |
| 2013-05 | 1808 | 1756 | 102 | 1654 | 5.8087 % |
| 2013-06 | 1675 | 1637 | 88 | 1549 | 5.3757 % |
| 2013-07 | 1886 | 1844 | 103 | 1741 | 5.5857 % |
| 2013-08 | 1537 | 1484 | 79 | 1405 | 5.3235 % |
| 2013-09 | 1617 | 1553 | 78 | 1475 | 5.0225 % |
| 2013-10 | 1618 | 1559 | 83 | 1476 | 5.3239 % |
| 2013-11 | 1552 | 1494 | 70 | 1424 | 4.6854 % |
| 2013-12 | 1565 | 1510 | 63 | 1447 | 4.1722 % |
| 2014-01 | 1791 | 1728 | 79 | 1649 | 4.5718 % |
| 2014-02 | 1538 | 1490 | 73 | 1417 | 4.8993 % |
| 2014-03 | 1586 | 1533 | 67 | 1466 | 4.3705 % |
| 2014-04 | 1739 | 1665 | 69 | 1596 | 4.1441 % |
| 2014-05 | 1908 | 1820 | 86 | 1734 | 4.7253 % |
| 2014-06 | 1887 | 1800 | 88 | 1712 | 4.8889 % |
| 2014-07 | 2008 | 1917 | 81 | 1836 | 4.2254 % |
| 2014-08 | 1642 | 1550 | 55 | 1495 | 3.5484 % |
| 2014-09 | 1637 | 1547 | 43 | 1504 | 2.7796 % |
| 2014-10 | 1921 | 1798 | 58 | 1740 | 3.2258 % |
| 2014-11 | 1628 | 1513 | 37 | 1476 | 2.4455 % |
| 2014-12 | 1914 | 1610 | 39 | 1571 | 2.4224 % |
| 2015-01 | 1830 | 1748 | 77 | 1671 | 4.4050 % |
| 2015-02 | 1732 | 1655 | 70 | 1585 | 4.2296 % |
| 2015-03 | 1891 | 1795 | 62 | 1733 | 3.4540 % |
| 2015-04 | 2111 | 1993 | 62 | 1931 | 3.1109 % |
| 2015-05 | 1959 | 1842 | 56 | 1786 | 3.0402 % |
| 2015-06 | 1989 | 1878 | 96 | 1782 | 5.1118 % |
| 2015-07 | 2273 | 2142 | 74 | 2068 | 3.4547 % |
| 2015-08 | 1729 | 1620 | 57 | 1563 | 3.5185 % |
| 2015-09 | 2065 | 1929 | 46 | 1883 | 2.3847 % |
| 2015-10 | 1965 | 1830 | 42 | 1788 | 2.2951 % |
| 2015-11 | 1834 | 1698 | 35 | 1663 | 2.0612 % |
| 2015-12 | 1951 | 1661 | 21 | 1640 | 1.2643 % |

## Mensual: originales sin backorder

| Mes | Documentos | Etiquetables | Positivos | Negativos | Prevalencia |
|---|---:|---:|---:|---:|---:|
| 2013-01 | 1303 | 1303 | 0 | 1303 | 0.0000 % |
| 2013-02 | 988 | 988 | 0 | 988 | 0.0000 % |
| 2013-03 | 1408 | 1408 | 0 | 1408 | 0.0000 % |
| 2013-04 | 1350 | 1350 | 0 | 1350 | 0.0000 % |
| 2013-05 | 1445 | 1445 | 0 | 1445 | 0.0000 % |
| 2013-06 | 1384 | 1384 | 0 | 1384 | 0.0000 % |
| 2013-07 | 1533 | 1533 | 0 | 1533 | 0.0000 % |
| 2013-08 | 1249 | 1248 | 0 | 1248 | 0.0000 % |
| 2013-09 | 1290 | 1290 | 0 | 1290 | 0.0000 % |
| 2013-10 | 1302 | 1302 | 0 | 1302 | 0.0000 % |
| 2013-11 | 1251 | 1251 | 0 | 1251 | 0.0000 % |
| 2013-12 | 1267 | 1267 | 0 | 1267 | 0.0000 % |
| 2014-01 | 1464 | 1464 | 0 | 1464 | 0.0000 % |
| 2014-02 | 1239 | 1239 | 0 | 1239 | 0.0000 % |
| 2014-03 | 1316 | 1316 | 0 | 1316 | 0.0000 % |
| 2014-04 | 1407 | 1407 | 0 | 1407 | 0.0000 % |
| 2014-05 | 1498 | 1498 | 0 | 1498 | 0.0000 % |
| 2014-06 | 1492 | 1492 | 0 | 1492 | 0.0000 % |
| 2014-07 | 1607 | 1607 | 0 | 1607 | 0.0000 % |
| 2014-08 | 1307 | 1307 | 0 | 1307 | 0.0000 % |
| 2014-09 | 1313 | 1312 | 0 | 1312 | 0.0000 % |
| 2014-10 | 1499 | 1499 | 0 | 1499 | 0.0000 % |
| 2014-11 | 1276 | 1276 | 0 | 1276 | 0.0000 % |
| 2014-12 | 1499 | 1353 | 0 | 1353 | 0.0000 % |
| 2015-01 | 1449 | 1449 | 0 | 1449 | 0.0000 % |
| 2015-02 | 1391 | 1390 | 0 | 1390 | 0.0000 % |
| 2015-03 | 1517 | 1517 | 0 | 1517 | 0.0000 % |
| 2015-04 | 1696 | 1696 | 0 | 1696 | 0.0000 % |
| 2015-05 | 1552 | 1552 | 0 | 1552 | 0.0000 % |
| 2015-06 | 1527 | 1527 | 0 | 1527 | 0.0000 % |
| 2015-07 | 1794 | 1794 | 0 | 1794 | 0.0000 % |
| 2015-08 | 1359 | 1359 | 0 | 1359 | 0.0000 % |
| 2015-09 | 1626 | 1626 | 0 | 1626 | 0.0000 % |
| 2015-10 | 1566 | 1566 | 0 | 1566 | 0.0000 % |
| 2015-11 | 1428 | 1428 | 0 | 1428 | 0.0000 % |
| 2015-12 | 1540 | 1414 | 0 | 1414 | 0.0000 % |

prevalencias_mensuales.csv incluye todos los tipos de backorder por mes; etiquetas_desarrollo.json contiene además estados, años y particiones. Los períodos sin positivos se conservan. La ausencia de clase 1 en train y validación de la cohorte admisible impide una evaluación binaria útil: no se modifica el horizonte para fabricar eventos ni se balancea con datos sintéticos.

No se interpretan las tasas altas de derivados como oportunidad para entrenar sin resolver su fecha heredada. Los cuatro positivos entre originales que generan backorders tampoco justifican el objetivo de cumplimiento completo y no aportan positivos de validación.
