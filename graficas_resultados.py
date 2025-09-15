"""
Funciones de visualización y evaluación para modelos de regresión.

Incluye:
- Gráficas de errores por época (entrenamiento, validación, prueba).
- Comparación real vs predicho.
- Análisis de bias y varianza de errores.
- Evolución de métricas por complejidad.
- Comparativa entre múltiples modelos.
- Importancia de features en Random Forest.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ========== GRÁFICAS DE ENTRENAMIENTO ==========
def graficar_error_entrenamiento(path="errores_por_epoca.txt"):
    """
    Gráfica de evolución del error (MSE) por época a partir de archivo TXT.
    """
    epocas, errores_train, errores_val = [], [], []
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

def graficar_error_entrenamiento_dual(txt_path="errores_entrenamiento_train_val_test.txt", output_path="grafica_errores.png"):
    """
    Gráfica del error de entrenamiento, validación y prueba por época.
    """
    epocas, train_errors, val_errors, test_errors = [], [], [], []
    with open(txt_path, "r") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 4:
                epocas.append(int(parts[0]))
                train_errors.append(float(parts[1]))
                val_errors.append(float(parts[2]))
                test_errors.append(float(parts[3]))
    plt.figure(figsize=(10, 6))
    plt.plot(epocas, train_errors, label="Train")
    plt.plot(epocas, val_errors, label="Validation")
    plt.plot(epocas, test_errors, label="Test", linestyle="--")
    plt.xlabel("Época")
    plt.ylabel("Error")
    plt.title("Evolución del Error por Época")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# ========== EVALUACIÓN DEL MODELO ==========
def graficar_real_vs_predicho(Y_real, Y_predicho, title="Valores reales vs predichos", output_path="grafica_real_vs_predicho.png"):
    """
    Gráfico de dispersión: valores reales vs predichos, con línea de referencia.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_real, Y_predicho, color='green', alpha=0.6)
    plt.plot([min(Y_real), max(Y_real)], [min(Y_real), max(Y_real)], 'r--')
    plt.title(title)
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def calcular_bias(y_real, y_pred):
    """
    Calcula el sesgo (bias): promedio de errores (real - predicho).
    """
    errores = [yr - yp for yr, yp in zip(y_real, y_pred)]
    return sum(errores) / len(errores)

def calcular_varianza_errores(y_real, y_pred):
    """
    Calcula la varianza de los errores (dispersión respecto al sesgo).
    """
    errores = [yr - yp for yr, yp in zip(y_real, y_pred)]
    media_error = sum(errores) / len(errores)
    varianza = sum((e - media_error) ** 2 for e in errores) / len(errores)
    return varianza

# ========== MÉTRICAS VS COMPLEJIDAD ==========
def graficar_complejidad_vs_metricas(csv_path="complejidad_modelo.csv"):
    """
    Gráficas de métricas (MAE, R2, Bias, Varianza) vs número de features.
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
    # Error y varianza
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
    # R²
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

# ========== COMPARATIVA ENTRE MODELOS ==========
def graficar_comparativa_modelos_multi(modelos, metricas, nombres):

    """
    Gráfico de barras para comparar métricas entre modelos.
    """
    x = range(len(metricas))
    width = 0.2
    plt.figure(figsize=(10, 6))
    for i, vals in enumerate(modelos):
        plt.bar([p + i*width for p in x], vals, width, label=nombres[i])
    plt.xticks([p + width for p in x], metricas)
    plt.ylabel("Valor")
    plt.title("Comparación de métricas entre modelos")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("grafica_comparativa_modelos_multi.png")
    plt.show()

# ========== IMPORTANCIA DE FEATURES EN RANDOM FOREST ==========
def graficar_feature_importance(importancias):
    """
    Gráfico de barras para visualizar la importancia de los features
    en un modelo Random Forest.
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