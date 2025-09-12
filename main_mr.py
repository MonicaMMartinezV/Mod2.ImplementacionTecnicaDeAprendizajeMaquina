"""
Script de entrenamiento y evaluación para regresión lineal múltiple (desde cero).

Flujo principal:
- Carga y normaliza datos (X, Y) con un número configurable de features.
- Separa en train/test.
- Entrena con gradiente descendente (funciones MSE y update).
- Desnormaliza predicciones para evaluar en escala real.
- Calcula métricas (MAE, R², varianza de errores).
- Persiste trazas de entrenamiento y curva de error por época.
- Registra la relación complejidad (N° de features) vs desempeño.

Salidas:
- Archivo CSV "complejidad_modelo.csv" con: N_Features, MAE, R2, Bias, Varianza
- Archivo "errores_por_epoca.txt" con el MSE por época (para graficar).
"""

from datos_life_expectancy import cargar_datos_normalizados
from graficas_resultados import (
    calcular_varianza_errores,
    calcular_bias,
    graficar_complejidad_vs_metricas,
)
from modelo_regresion import (
    functionHyp,
    MSE,
    update,
    split_data_val,
    desnormalizar,
    calcular_MAE,
    calcular_R2,
)

def main(num_features):

    """
    Entrena un modelo con 'num_features' columnas de entrada, registra métricas y
    guarda archivos de trazas.
    """

    # Mensaje de inicio (muestra # de features efectivas sin incluir el target)
    print(f"\n--- Entrenando con {num_features-1} features ---")

    # Cargar y normalizar datos (X_final, Y_final) y parámetros de normalización de Y
    X_final, Y_final, y_mean, y_std = cargar_datos_normalizados(num_features=num_features)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data_val(X_final, Y_final)

    # Inicialización de parámetros del modelo
    theta = [0.0] * len(X_train[0])
    b = 0.0
    alpha = 0.01
    epoc = 1000
    errores_train = []
    errores_val = []

    # Bucle de entrenamiento (gradiente descendente)
    for i in range(epoc):
        error_train = MSE(X_train, theta, b, Y_train)
        error_val = MSE(X_val, theta, b, Y_val)

        errores_train.append(error_train)
        errores_val.append(error_val)

        # Criterio de parada temprano si el error ya es pequeño
        if error_val < 1e-2:
            break

        # Actualización de parámetros (un paso de gradiente)
        theta, b = update(X_train, theta, b, Y_train, alpha)

    # Predicciones en validación y prueba (normalizadas)
    Y_val_pred_norm = [functionHyp(x, theta, b) for x in X_val]
    Y_test_pred_norm = [functionHyp(x, theta, b) for x in X_test]
    
    # Desnormalizar predicciones y etiquetas verdaderas para evaluar en escala real
    Y_val_pred = [desnormalizar(y, y_mean, y_std) for y in Y_val_pred_norm]
    Y_val_real = [desnormalizar(y, y_mean, y_std) for y in Y_val]
    Y_test_pred = [desnormalizar(y, y_mean, y_std) for y in Y_test_pred_norm]
    Y_test_real = [desnormalizar(y, y_mean, y_std) for y in Y_test]

    # Métricas en escala real
    mae_test = calcular_MAE(Y_test_real, Y_test_pred)
    r2_test = calcular_R2(Y_test_real, Y_test_pred)
    bias_test = calcular_bias(Y_test_real, Y_test_pred)
    variance_test = calcular_varianza_errores(Y_test_real, Y_test_pred)
    # Métricas para validación
    mae_val = calcular_MAE(Y_val_real, Y_val_pred)
    r2_val = calcular_R2(Y_val_real, Y_val_pred)

    # Reporte de resultados del modelo entrenado
    print(f"MAE Test: {mae_test:.4f} | R² Test: {r2_test:.4f} | Bias: {bias_test:.4f} | Varianza: {variance_test:.4f}")
    print(f"MAE Val: {mae_val:.4f} | R² Val: {r2_val:.4f}")

    # Muestra algunos pares real vs predicho (sanity check)
    for i in range(3):
        print(f"[Test] Real = {Y_test_real[i]:.2f} | Predicho = {Y_test_pred[i]:.2f}")

    # Guardar complejidad vs desempeño
    with open("complejidad_modelo.csv", "a") as f_cme:
        f_cme.write(f"{num_features-1},{mae_test},{r2_test},{mae_val},{r2_val},{bias_test},{variance_test}\n")

# Inicializar archivo CSV de complejidad vs error con encabezados
with open("complejidad_modelo.csv", "w") as f_cme:
    f_cme.write("N_Features,MAE_Test,R2_Test,MAE_Val,R2_Val,Bias,Varianza\n")

# Barrido de complejidad: entrena el modelo con distintos números de features.
# Se pasa 'num_features + 1' a main para que el print muestre 'num_features - 1' (excluyendo target).
for num_features in [2, 4, 8, 12, 16, 20, 22]:
    main(num_features+1)

graficar_complejidad_vs_metricas("complejidad_modelo.csv")
