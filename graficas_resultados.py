"""
Script con funciones de evaluación y visualización para un modelo de regresión.

Flujo principal:
- graficar_error_entrenamiento: lee un archivo de texto con errores por época y
  genera una gráfica de la evolución del MSE.
- graficar_real_vs_predicho: muestra un scatter comparando valores reales vs predichos
  con la línea de referencia ideal (y = x).
- calcular_bias: calcula el sesgo promedio de las predicciones (tendencia a sobre/subestimar).
- calcular_varianza_errores: calcula la dispersión de los errores alrededor de su media.
"""

import matplotlib.pyplot as plt
import csv

def graficar_error_entrenamiento(path="errores_por_epoca.txt"):
    
    """
    Lee un archivo de texto con errores por época y grafica la evolución del error (MSE)
    durante el entrenamiento del modelo.

    Parámetros:
        path (str): Ruta del archivo que contiene los errores. 
                    Cada línea debe tener el formato: "época,error".
    """
    
    # Lista para almacenar los números de épocas
    epocas = []

    # Lista para almacenar los errores (MSE)
    errores_train = []
    errores_val = []

    # Abrir archivo y leer línea por línea
    with open(path, "r") as f:
        for line in f:
            valores = line.strip().split(",")
            if len(valores) == 2:
                i, e = valores
                epocas.append(int(i))
                errores_train.append(float(e))
                errores_val.append(None)
            elif len(valores) == 3:
                i, et, ev = valores
                epocas.append(int(i))
                errores_train.append(float(et))
                errores_val.append(float(ev))

    plt.figure(figsize=(8, 4))
    plt.plot(epocas, errores_train, marker='o', linestyle='-', color='blue', label="Entrenamiento")
    if all(e is not None for e in errores_val):
        plt.plot(epocas, errores_val, marker='x', linestyle='--', color='orange', label="Validación")

    plt.title("Evolución del error (MSE) por época")
    plt.xlabel("Época")
    plt.ylabel("Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafica_error_entrenamiento.png")
    plt.show()

def graficar_error_entrenamiento_dual(path="errores_entrenamiento_train_val.txt"):
    """
    Grafica la evolución del MSE para entrenamiento y validación.
    """

    epocas = []
    errores_train = []
    errores_val = []

    with open(path, "r") as f:
        for line in f:
            i, et, ev = line.strip().split(",")
            epocas.append(int(i))
            errores_train.append(float(et))
            errores_val.append(float(ev))

    plt.figure(figsize=(8, 4))
    plt.plot(epocas, errores_train, label="Entrenamiento", color="blue")
    plt.plot(epocas, errores_val, label="Validación", color="orange", linestyle="--")
    plt.title("Error (MSE) por época")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafica_error_entrenamiento.png")
    plt.show()

def graficar_real_vs_predicho(Y_real, Y_predicho, title="Valores reales vs predichos", output_path="grafica_real_vs_predicho.png"):
    """
    Grafica los valores reales contra los valores predichos para visualizar
    el ajuste del modelo.

    Parámetros:
        Y_real (list or array): Valores reales.
        Y_predicho (list or array): Valores predichos por el modelo.
        title (str): Título personalizado para la gráfica.
        output_path (str): Ruta donde se guardará la imagen.
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_real, Y_predicho, color='green', alpha=0.6)
    plt.plot([min(Y_real), max(Y_real)], [min(Y_real), max(Y_real)], 'r--')  # Línea ideal
    plt.title(title)
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def calcular_bias(y_real, y_pred):

    """
    Calcula el sesgo (bias), definido como el promedio de los errores
    entre valores reales y predichos.

    Parámetros:
        y_real (list or array): Valores reales.
        y_pred (list or array): Valores predichos por el modelo.

    Retorna:
        float: Valor promedio del error (bias).
    """
    # Lista de errores
    errores = [yr - yp for yr, yp in zip(y_real, y_pred)]

    # Promedio de los errores
    return sum(errores) / len(errores)

def calcular_varianza_errores(y_real, y_pred):
    
    """
    Calcula la varianza de los errores entre valores reales y predichos.

    Parámetros:
        y_real (list or array): Valores reales.
        y_pred (list or array): Valores predichos por el modelo.

    Retorna:
        float: Varianza de los errores.
    """
    
    # Errores individuales
    errores = [yr - yp for yr, yp in zip(y_real, y_pred)]

    # Media del error
    media_error = sum(errores) / len(errores)

    # Varianza
    varianza = sum((e - media_error) ** 2 for e in errores) / len(errores)
    
    return varianza

def graficar_complejidad_vs_metricas(csv_path="complejidad_modelo.csv"):
    """
    Lee archivo de métricas por número de features y grafica evolución de
    MAE, R2, Bias y Varianza respecto a la complejidad del modelo.
    """

    features = []
    mae_test = []
    mae_val = []
    r2_test = []
    r2_val = []
    bias = []
    varianza = []

    with open(csv_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 7:
                ftrs, mae_t, r2_t, mae_v, r2_v, b, var = parts
                features.append(int(ftrs))
                mae_test.append(float(mae_t))
                r2_test.append(float(r2_t))
                mae_val.append(float(mae_v))
                r2_val.append(float(r2_v))
                bias.append(float(b))
                varianza.append(float(var))

    plt.figure(figsize=(10, 6))
    plt.plot(features, mae_test, marker="o", label="MAE (Test)")
    plt.plot(features, mae_val, marker="o", label="MAE (Val)", linestyle="--")
    plt.plot(features, bias, marker="x", label="Bias")
    plt.plot(features, varianza, marker="x", label="Varianza")
    plt.title("Evolución de errores vs complejidad del modelo")
    plt.xlabel("Número de features")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafica_complejidad_vs_metricas.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(features, r2_test, marker="o", label="R² (Test)")
    plt.plot(features, r2_val, marker="o", label="R² (Val)", linestyle="--")
    plt.title("R² vs Complejidad del modelo")
    plt.xlabel("Número de features")
    plt.ylabel("R²")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafica_complejidad_vs_r2.png")
    plt.show()

def graficar_comparativa_modelos(mae_manual, r2_manual, bias_manual, var_manual,
                                  mae_rf, r2_rf, bias_rf, var_rf):
    """
    Gráfica de barras comparando desempeño de modelo manual vs framework.
    """

    labels = ["MAE", "R²", "Bias", "Varianza"]
    valores_manual = [mae_manual, r2_manual, bias_manual, var_manual]
    valores_rf = [mae_rf, r2_rf, bias_rf, var_rf]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], valores_manual, width, label="Manual")
    plt.bar([i + width/2 for i in x], valores_rf, width, label="Random Forest")
    plt.xticks(x, labels)
    plt.ylabel("Valor")
    plt.title("Comparativa de modelos")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("grafica_comparativa_modelos.png")
    plt.show()

def graficar_feature_importance(importancias):
    """
    Dibuja la importancia de features para un modelo Random Forest.
    """

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importancias)), importancias)
    plt.title("Importancia de Features - Random Forest")
    plt.xlabel("Índice del feature")
    plt.ylabel("Importancia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafica_rf_importancia_features.png")
    plt.show()