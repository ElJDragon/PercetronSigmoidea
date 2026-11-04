# Análisis de etiquetas y conciliación

## Reconciliación de la auditoría anterior

Esta tabla **reutiliza únicamente agregados históricos existentes**, de toda la copia, para explicar las cifras recibidas. No se volvió a consultar picking de pedidos fechados en 2016 ni se calcularon nuevas tasas o métricas test. Estos conteos globales no seleccionan diseño, umbral o granularidad.

| Estado histórico mutuamente excluyente | Líneas |
|---|---:|
| picking_dentro_horizonte_observado_maduro | 224591 |
| picking_posterior_observado_maduro | 3154 |
| sin_picking_con_seguimiento_nominal_indeterminado | 3129 |
| sin_seguimiento_nominal_suficiente | 538 |
| inconsistencia_fecha_pick_anterior_a_pedido | 0 |
| **Total** | **231412** |

224591+3154=227745 maduras con picking observado. 227745+3129=230874 con seguimiento nominal de la auditoría. Faltan 538 para 231412: su horizonte nominal no estaba completo. Los 3147 nulos totales se descomponen en 3129 maduros y 18 de esos 538; los otros 520 inmaduros tienen picking observado, pero no se subdividen por clase. No hay una fila adicional de nulos que sumar al total: son atributos de los estados anteriores. La auditoría no detectó picking anterior al pedido; otros motivos de no elegibilidad no se inventan y se evalúan en el contrato temporal de desarrollo.

Tener seguimiento nominal no equivale a prueba de ausencia de picking. Por ello, las 3129 líneas siguen indeterminadas y no se asignan automáticamente a retraso ni se borran del inventario.

## Conteos nuevos al corte de cada partición, solo desarrollo

| Estado exhaustivo por línea | Train | Validación |
|---|---:|---:|
| Fechas inválidas/inconsistentes | 0 | 0 |
| Horizonte incompleto al corte de partición | 582 | 481 |
| Picking nulo, horizonte nominal completo: indeterminado | 1235 | 1223 |
| Picking registrado en/después del corte: no disponible | 349 | 187 |
| Última edición en/después del corte: versión no disponible | 0 | 0 |
| Picking observado hasta día 2 inclusive | 124607 | 70414 |
| Picking observado desde día 3 | 1734 | 698 |
| **Total** | **128507** | **73003** |

Train conserva 126341 etiquetas justificadas: 1734 positivas y 124607 negativas (1.3725 % positivas). Validación conserva 71112: 698 positivas y 70414 negativas (0.9816 %). Se exige picking y versión observables en cada corte, no solo fecha de pedido dentro de la partición. Los conteos son candidatos antes de filtros adicionales de características históricas; no certifican el tamaño de matrices futuras.

La selección de outcomes observados excluye demanda no completada y puede sesgar contra los casos más difíciles. Debe informarse siempre su cobertura junto a métricas futuras y no extrapolar rendimiento a indeterminados. No se interpreta la clase ausente en un mes como cero retrasos de toda la población.

## Año: línea y documento actual separados

| Unidad | Año | Elegibles | Positivos | Negativos | Positivos / elegibles |
|---|---:|---:|---:|---:|---:|
| line | 2013 | 61141 | 958 | 60183 | 1.5669 % |
| line | 2014 | 65200 | 776 | 64424 | 1.1902 % |
| line | 2015 | 71112 | 698 | 70414 | 0.9816 % |
| document_diagnostic | 2013 | 18940 | 957 | 17983 | 5.0528 % |
| document_diagnostic | 2014 | 19971 | 775 | 19196 | 3.8806 % |
| document_diagnostic | 2015 | 21791 | 698 | 21093 | 3.2032 % |

Las tasas por documento dan el mismo peso a cada documento; por línea ponderan documentos con más líneas. No son intercambiables y no se eligió la unidad para mejorar el equilibrio. La recomendación de línea se basa en identidad estable, significado del evento y mutación de composición. Los períodos 2013/2014 usan el corte train de 2015; no se exige un corte separado al final de 2013 porque ambos años integran el mismo train.

En los estados elegibles de desarrollo, 2428 de las 2432 líneas tardías pertenecen hoy a documentos derivados de backorder y cuatro a documentos no derivados. Esto es asociación descriptiva, no prueba de causa; exige revisar linaje y reportar desempeño por familia. El indicador final de backorder sería fuga si se usara como entrada de emisión y permanece prohibido. Los traslados no se interpretan como nuevas demandas independientes.

## Mes a mes: líneas

Pedidos se refiere a documentos actuales del universo del mes; filas etiquetadas pueden representar solo parte de un documento. `line_structure_monthly` y `line_states_monthly` en agregados_desarrollo.json incluyen documentos distintos y productos por mes/estado. Las sumas de documentos entre estados no son un total válido si un documento aparece en varios estados.

| Mes | Líneas de origen | Líneas elegibles | Positivos | Negativos | Prevalencia |
|---|---:|---:|---:|---:|---:|
| 2013-01 | 5281 | 5271 | 50 | 5221 | 0.9486 % |
| 2013-02 | 3726 | 3707 | 46 | 3661 | 1.2409 % |
| 2013-03 | 5389 | 5362 | 87 | 5275 | 1.6225 % |
| 2013-04 | 5314 | 5281 | 109 | 5172 | 2.0640 % |
| 2013-05 | 5699 | 5646 | 102 | 5544 | 1.8066 % |
| 2013-06 | 5338 | 5300 | 88 | 5212 | 1.6604 % |
| 2013-07 | 5896 | 5854 | 103 | 5751 | 1.7595 % |
| 2013-08 | 4819 | 4765 | 79 | 4686 | 1.6579 % |
| 2013-09 | 5099 | 5035 | 78 | 4957 | 1.5492 % |
| 2013-10 | 5171 | 5110 | 83 | 5027 | 1.6243 % |
| 2013-11 | 4957 | 4899 | 70 | 4829 | 1.4289 % |
| 2013-12 | 4966 | 4911 | 63 | 4848 | 1.2828 % |
| 2014-01 | 5682 | 5619 | 79 | 5540 | 1.4059 % |
| 2014-02 | 4830 | 4782 | 73 | 4709 | 1.5266 % |
| 2014-03 | 5118 | 5065 | 67 | 4998 | 1.3228 % |
| 2014-04 | 5513 | 5439 | 69 | 5370 | 1.2686 % |
| 2014-05 | 6027 | 5938 | 86 | 5852 | 1.4483 % |
| 2014-06 | 5888 | 5798 | 88 | 5710 | 1.5178 % |
| 2014-07 | 6310 | 6219 | 82 | 6137 | 1.3185 % |
| 2014-08 | 5130 | 5037 | 55 | 4982 | 1.0919 % |
| 2014-09 | 5117 | 5026 | 43 | 4983 | 0.8556 % |
| 2014-10 | 6077 | 5950 | 58 | 5892 | 0.9748 % |
| 2014-11 | 5151 | 5034 | 37 | 4997 | 0.7350 % |
| 2014-12 | 6009 | 5293 | 39 | 5254 | 0.7368 % |
| 2015-01 | 5735 | 5650 | 77 | 5573 | 1.3628 % |
| 2015-02 | 5478 | 5400 | 70 | 5330 | 1.2963 % |
| 2015-03 | 5947 | 5851 | 62 | 5789 | 1.0596 % |
| 2015-04 | 6641 | 6522 | 62 | 6460 | 0.9506 % |
| 2015-05 | 6103 | 5985 | 56 | 5929 | 0.9357 % |
| 2015-06 | 6198 | 6085 | 96 | 5989 | 1.5776 % |
| 2015-07 | 7066 | 6930 | 74 | 6856 | 1.0678 % |
| 2015-08 | 5411 | 5299 | 57 | 5242 | 1.0757 % |
| 2015-09 | 6392 | 6256 | 46 | 6210 | 0.7353 % |
| 2015-10 | 6240 | 6102 | 42 | 6060 | 0.6883 % |
| 2015-11 | 5719 | 5580 | 35 | 5545 | 0.6272 % |
| 2015-12 | 6073 | 5452 | 21 | 5431 | 0.3852 % |

## Mes a mes: documentos actuales (diagnóstico)

| Mes | Documentos de origen | Documentos etiquetables | Positivos | Negativos | Prevalencia |
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

Hay eventos positivos en cada mes de desarrollo; se dispone de 1734 en train y 698 en validación para una primera evaluación de clases raras. Eso no garantiza precisión estable por producto ni independencia estadística. Deben examinarse soportes por mes y familia. No se generaron datos sintéticos, duplicados de positivos, pesos ni balanceo.

## Censura y decisiones

La tabla de estados tiene prioridad definida: inconsistencia, horizonte incompleto, picking nulo, picking no observado al corte, versión posterior al corte y finalmente clases 0/1. Evita doble conteo. Los nulos actuales no se convierten en evidencia negativa histórica; tampoco se usan picks posteriores al corte de una partición para enriquecer sus clases. El test desde 2016 no se abre. Antes de ETL confirmar aceptación de esta población seleccionada por observación, ancla original heredada y limitación de disponibilidad de la copia.
