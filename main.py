"""
Modelo de regresión lineal múltiple (desde cero) para predecir la esperanza de vida.
Implementado por: Mónica Monserrat Martínez Vásquez (A01710965)
Curso: Inteligencia Artificial Avanzada para Ciencia de Datos I

Este script:
- Carga y normaliza datos.
- Aplica reducción de dimensionalidad con PCA.
- Entrena el modelo con gradiente descendente.
- Evalúa métricas de desempeño.
- Genera gráficas y guarda resultados.

Salidas:
- errores_entrenamiento_train_val_test.txt
- metricas_modelo_manual.csv
- gráfica de resultados
"""
from datos_life_expectancy import cargar_datos_normalizados
from graficas_resultados import (
    calcular_varianza_errores,
    calcular_bias,
    graficar_real_vs_predicho,
    graficar_error_entrenamiento_dual,
    graficar_comparativa_modelos_multi
)
from modelo_regresion import (
    functionHyp,
    MSE,
    MAE,
    Huber_loss,
    Huber_ridge_loss,
    update_huber_ridge,
    MSE_ridge,
    update,
    update_huber,
    split_data,
    desnormalizar,
    calcular_MAE,
    calcular_R2,
)
from pca import pca
import numpy as np

# Hiperparámetros del modelo
PCS = 15
NUM_FEATURES = 21
# Opciones: "MSE", "MSE_L2", "MAE", "HUBER", "HUBER_L2"
LOSS_FUNC = "HUBER_L2"
alpha = 0.005
epoc = 3000
lambda_reg = 0.5
delta = 0.6

# Carga y reducción de dimensionalidad
X_final, Y_final, y_mean, y_std = cargar_datos_normalizados(num_features=NUM_FEATURES)
X_pca, var_exp, cum_var = pca(X_final, n_components=PCS)
X_final = X_pca.tolist()

# División de datos en entrenamiento, validación y prueba
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X_final, Y_final, seed=42)

# Inicialización de parámetros
theta = [0.0] * len(X_train[0])
b = 0.0
errores_train = []
errores_val = []
errores_test = []

# Entrenamiento usando gradiente descendente
for i in range(epoc):
    if LOSS_FUNC == "MSE":
        error_train = MSE(X_train, theta, b, Y_train)
        error_val = MSE(X_val, theta, b, Y_val)
    elif LOSS_FUNC == "MSE_L2":
        error_train = MSE_ridge(X_train, theta, b, Y_train, lambda_reg=lambda_reg)
        error_val = MSE_ridge(X_val, theta, b, Y_val, lambda_reg=lambda_reg)
    elif LOSS_FUNC == "MAE":
        error_train = MAE(X_train, theta, b, Y_train)
        error_val = MAE(X_val, theta, b, Y_val)
    elif LOSS_FUNC == "HUBER":
        error_train = Huber_loss(X_train, theta, b, Y_train, delta=delta)
        error_val = Huber_loss(X_val, theta, b, Y_val, delta=delta)
        error_test = Huber_loss(X_test, theta, b, Y_test, delta=delta)
    elif LOSS_FUNC == "HUBER_L2":
        error_train = Huber_ridge_loss(X_train, theta, b, Y_train, delta=delta, lambda_reg=lambda_reg)
        error_val = Huber_ridge_loss(X_val, theta, b, Y_val, delta=delta, lambda_reg=lambda_reg)
        error_test = Huber_ridge_loss(X_test, theta, b, Y_test, delta=delta, lambda_reg=lambda_reg)

    errores_train.append(error_train)
    errores_val.append(error_val)
    errores_test.append(error_test)

    if error_val < 1e-2:
        break

    # Actualización de parámetros
    if LOSS_FUNC == "HUBER":
        theta, b = update_huber(X_train, theta, b, Y_train, alpha)
    if LOSS_FUNC == "HUBER_L2":
        theta, b = update_huber_ridge(X_train, theta, b, Y_train, alpha, delta=delta, lambda_reg=lambda_reg)
    else:
        theta, b = update(X_train, theta, b, Y_train, alpha)

# Predicciones y desnormalización
Y_pred_real = [desnormalizar(functionHyp(x, theta, b), y_mean, y_std) for x in X_test]
Y_test_real = [desnormalizar(y, y_mean, y_std) for y in Y_test]

Y_train_pred_real = [desnormalizar(functionHyp(x, theta, b), y_mean, y_std) for x in X_train]
Y_train_real = [desnormalizar(y, y_mean, y_std) for y in Y_train]

Y_val_pred_real = [desnormalizar(functionHyp(x, theta, b), y_mean, y_std) for x in X_val]
Y_val_real = [desnormalizar(y, y_mean, y_std) for y in Y_val]

# --- Cálculo de métricas ---
def evaluar(nombre, Y_real, Y_pred):
    mae = calcular_MAE(Y_real, Y_pred)
    r2 = calcular_R2(Y_real, Y_pred)
    bias = calcular_bias(Y_real, Y_pred)
    varianza = calcular_varianza_errores(Y_real, Y_pred)
    print(f"\n {nombre}:")
    print(f"  MAE      = {mae:.4f}")
    print(f"  R²       = {r2:.4f}")
    print(f"  Bias     = {bias:.4f}")
    print(f"  Varianza = {varianza:.4f}")
    return [mae, r2, bias, varianza]

print("\n================ RESULTADOS =================")
train_metrics = evaluar("Entrenamiento (Train)", Y_train_real, Y_train_pred_real)
val_metrics = evaluar("Validación (Validation)", Y_val_real, Y_val_pred_real)
test_metrics = evaluar("Prueba (Test)", Y_test_real, Y_pred_real)

# Varianza explicada por PCA
print(f"\n Varianza explicada por los primeros {PCS} componentes: {sum(var_exp[:PCS]):.2f}%")
print("🔍 Varianza acumulada por componente:")
for i, v in enumerate(cum_var[:PCS], 1):
    print(f"  PC{i}: {v:.2f}%")

# Muestra 5 predicciones
print("\n Ejemplos de predicción (Real vs. Predicho):")
for i in range(5):
    print(f"  Real = {Y_test_real[i]:.2f} | Predicho = {Y_pred_real[i]:.2f}")

# Guardar errores por época
with open("errores_entrenamiento_train_val_test.txt", "w") as f:
    f.write("Epoca,Train,Val,Test\n")
    for i in range(len(errores_train)):
        f.write(f"{i},{errores_train[i]},{errores_val[i]},{errores_test[i]}\n")

# Guardar métricas finales en CSV
with open("metricas_modelo_manual.csv", "w") as f:
    f.write("MAE,R2,Bias,Varianza\n")
    f.write(f"{test_metrics[0]:.4f},{test_metrics[1]:.4f},{test_metrics[2]:.4f},{test_metrics[3]:.4f}")

# Gráficas finales
graficar_error_entrenamiento_dual()
graficar_real_vs_predicho(Y_test_real, Y_pred_real, title="Modelo Huber+L2+PCA: Real vs Predicho", output_path="grafica_huber_l2_pca_real_vs_predicho.png")
graficar_comparativa_modelos_multi(
    modelos=[
        [2.6744, 0.8307, -0.1382, 11.6048],
        [2.6744, 0.8308, -0.1373, 11.5938],
        [2.6308, 0.8322, -0.1886, 11.4854],
        [1.1746, 0.9555, -0.1034, 3.0425]
    ],
    metricas=["MAE", "R2", "Bias", "Varianza"],
    nombres=["Lineal","Ridge","Huber+L2+PCA","Random Forest"]
)