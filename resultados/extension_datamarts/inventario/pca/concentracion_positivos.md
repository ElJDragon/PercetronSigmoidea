# Concentración de positivos y alertas

Nueve identidades operativas con 35 positivos de validación: [77, 78, 80, 86, 95, 98, 184, 193, 204]. Son observaciones repetidas, no 35 réplicas independientes ni nueve poblaciones elegidas para el resultado principal. Artículos positivos en val sin antecedente positivo en train: []. El detalle de los 219 artículos, incluidos los 210 siempre negativos en validación, permanece en concentracion_positivos.json.

| Artículo | Pos. train | Pos. val | Neg. val | TP perceptrón | FP perceptrón | TP logística | FP logística |
|---|---:|---:|---:|---:|---:|---:|---:|
| 77 | 9 | 4 | 8 | 4 | 7 | 4 | 8 |
| 78 | 11 | 3 | 9 | 3 | 8 | 3 | 9 |
| 80 | 12 | 4 | 8 | 4 | 8 | 4 | 8 |
| 86 | 7 | 3 | 9 | 3 | 8 | 3 | 9 |
| 95 | 11 | 4 | 8 | 4 | 7 | 4 | 7 |
| 98 | 9 | 4 | 8 | 4 | 8 | 4 | 7 |
| 184 | 6 | 6 | 6 | 6 | 3 | 6 | 5 |
| 193 | 9 | 4 | 8 | 4 | 7 | 4 | 8 |
| 204 | 9 | 3 | 9 | 3 | 9 | 3 | 9 |

| Mes | Positivos val |
|---|---:|
| 2015-01 | 0 |
| 2015-02 | 1 |
| 2015-03 | 6 |
| 2015-04 | 4 |
| 2015-05 | 2 |
| 2015-06 | 2 |
| 2015-07 | 7 |
| 2015-08 | 1 |
| 2015-09 | 2 |
| 2015-10 | 2 |
| 2015-11 | 0 |
| 2015-12 | 8 |

TP compartidos: 35. FP compartidos: 64; FP exclusivos del perceptrón: 1; de logística: 6. Desacuerdos totales: 7; ambos correctos: 2557; ambos incorrectos: 64. Las claves exactas de los errores están registradas en JSON. No se eliminan ni se utilizan para nuevas características o selección.

Los intervalos Wilson descriptivos del entrenamiento no establecen independencia temporal o por artículo. Con tan pocos positivos, meses sin positivos y alertas concentradas, un recall de 100% en esta validación no garantiza recall futuro ni utilidad económica. El diagnóstico de actividad histórica del informe de métricas explica la estructura dominante sin redefinir población ni modificar los baselines congelados.
