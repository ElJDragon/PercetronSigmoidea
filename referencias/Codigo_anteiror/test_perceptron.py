"""Casos de control del algoritmo y auditoría de las evidencias del CSV original."""
import unittest

import numpy as np
import pandas as pd

from Perceptron import BASE, CSV_PREDETERMINADO, PerceptronSimple, dividir_datos, evaluar


class PruebasPerceptron(unittest.TestCase):
    def test_inicializacion_y_activacion(self):
        modelo = PerceptronSimple(verbose=False)
        np.testing.assert_array_equal(modelo.weights, np.zeros(3))
        self.assertEqual(modelo.bias, 0)
        np.testing.assert_array_equal(modelo.activacion([-3, -0.01, 0, 0.01, 5]), [-1, -1, 1, 1, 1])

    def test_actualizacion_controlada_y_200_epocas(self):
        # Este caso artificial comprueba la fórmula, no representa datos del docente.
        modelo = PerceptronSimple(verbose=False).fit([[1, 1, 1]], [-1])
        np.testing.assert_allclose(modelo.weights, [-0.02, -0.02, -0.02])
        self.assertAlmostEqual(modelo.bias, -0.02)
        self.assertEqual(len(modelo.historial), 200)
        self.assertEqual(modelo.historial[0]["actualizaciones"], 1)
        self.assertEqual(sum(f["actualizaciones"] for f in modelo.historial), 1)
        np.testing.assert_array_equal(modelo.predict([[1, 1, 1]]), [-1])

    def test_configuracion_invalida(self):
        for opciones in [{"epochs": 199}, {"epochs": 200.5}, {"learning_rate": 0},
                         {"learning_rate": float("nan")}]:
            with self.assertRaises(ValueError):
                PerceptronSimple(**opciones)
        with self.assertRaises(ValueError):
            PerceptronSimple().fit([[1, 2]], [1])
        with self.assertRaises(ValueError):
            PerceptronSimple().fit([[1, 2, 3]], [0])

    def test_metricas_y_division_cero(self):
        metricas, matriz = evaluar([1, 1, -1, -1], [1, -1, 1, -1])
        np.testing.assert_array_equal(matriz.to_numpy(), [[1, 1], [1, 1]])
        for nombre in ["exactitud", "precision", "recall", "f1"]:
            self.assertEqual(metricas[nombre], 0.5)
        metricas, _ = evaluar([-1, -1], [-1, -1])
        self.assertEqual(metricas["exactitud"], 1)
        for nombre in ["precision", "recall", "f1"]:
            self.assertEqual(metricas[nombre], 0)

    def test_particion_original_sin_fuga(self):
        datos = pd.read_csv(CSV_PREDETERMINADO)
        y = datos.Clase.map({"Hombre": 1, "Mujer": -1}).to_numpy()
        entrenamiento, prueba = dividir_datos(y)
        repeticion = dividir_datos(y)
        np.testing.assert_array_equal(entrenamiento, repeticion[0])
        np.testing.assert_array_equal(prueba, repeticion[1])
        self.assertEqual(len(entrenamiento), 4000)
        self.assertEqual(len(prueba), 1000)
        self.assertEqual(len(np.intersect1d(entrenamiento, prueba)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([entrenamiento, prueba])), np.arange(5000))
        self.assertEqual(np.sum(y[prueba] == 1), 500)
        self.assertEqual(np.sum(y[entrenamiento] == 1), 2000)

    def test_exportaciones_reales(self):
        carpeta = BASE / "resultados"
        esperados = {"metricas.csv", "historial_entrenamiento.csv", "predicciones.csv",
                     "matriz_confusion.csv", "resumen.txt", "errores_por_epoca.png",
                     "exactitud_por_epoca.png", "matriz_confusion.png"}
        self.assertTrue(esperados.issubset({p.name for p in carpeta.iterdir()}))
        historial = pd.read_csv(carpeta / "historial_entrenamiento.csv")
        np.testing.assert_array_equal(historial.epoca, np.arange(1, 201))
        np.testing.assert_array_equal(historial.errores, historial.actualizaciones)
        predicciones = pd.read_csv(carpeta / "predicciones.csv")
        original = pd.read_csv(CSV_PREDETERMINADO)
        y = original.Clase.map({"Hombre": 1, "Mujer": -1}).to_numpy()
        _, prueba = dividir_datos(y)
        np.testing.assert_array_equal(predicciones.indice_original, prueba)
        np.testing.assert_array_equal(predicciones.clase_real, original.iloc[prueba].Clase)
        self.assertEqual(len(predicciones), 1000)
        calculadas, matriz = evaluar(predicciones.clase_real_codificada, predicciones.clase_predicha)
        guardadas = pd.read_csv(carpeta / "metricas.csv", index_col="conjunto")
        for nombre, valor in calculadas.items():
            self.assertAlmostEqual(guardadas.loc["prueba", nombre], valor)
        np.testing.assert_array_equal(pd.read_csv(carpeta / "matriz_confusion.csv", index_col=0), matriz)
        np.testing.assert_array_equal(predicciones.correcta,
                                      predicciones.clase_real_codificada == predicciones.clase_predicha)
        # Los coeficientes del historial deben reproducir las predicciones exportadas.
        ultima = historial.iloc[-1]
        X = original.iloc[prueba][["Caracteristica_1", "Caracteristica_2", "Caracteristica_3"]].to_numpy()
        estimadas = np.where(X @ ultima[["w1", "w2", "w3"]].to_numpy() + ultima.sesgo >= 0, 1, -1)
        np.testing.assert_array_equal(estimadas, predicciones.clase_predicha)
        for archivo in carpeta.glob("*.png"):
            self.assertEqual(archivo.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
