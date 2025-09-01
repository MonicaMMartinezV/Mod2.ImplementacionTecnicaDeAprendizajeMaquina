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
from graficas_resultados import calcular_varianza_errores
from modelo_regresion import (
    functionHyp,
    MSE,
    update,
    split_data,
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
    
    # Split en entrenamiento y prueba
    X_train, Y_train, X_test, Y_test = split_data(X_final, Y_final)

    # Inicialización de parámetros del modelo
    theta = [0.0] * len(X_train[0])
    b = 0.0
    alpha = 0.01
    epoc = 1000
    errores = []

    # Bucle de entrenamiento (gradiente descendente)
    for i in range(epoc):
        error = MSE(X_train, theta, b, Y_train)
        errores.append(error)

        # Criterio de parada temprano si el error ya es pequeño
        if error < 1e-2:
            break

        # Actualización de parámetros (un paso de gradiente)
        theta, b = update(X_train, theta, b, Y_train, alpha)

    # Predicciones en el conjunto de prueba (aún en escala normalizada)
    Y_pred_norm = [functionHyp(x, theta, b) for x in X_test]
    
    # Desnormalizar predicciones y etiquetas verdaderas para evaluar en escala real
    Y_pred_real = [desnormalizar(y, y_mean, y_std) for y in Y_pred_norm]
    Y_test_real = [desnormalizar(y, y_mean, y_std) for y in Y_test]

    # Métricas en escala real
    mae = calcular_MAE(Y_test_real, Y_pred_real)
    r2 = calcular_R2(Y_test_real, Y_pred_real)
    variance = calcular_varianza_errores(Y_test_real, Y_pred_real)

    # Reporte de resultados del modelo entrenado
    print("Theta final:", theta)
    print("Bias final:", b)
    print(f"MAE (real): {mae:.4f}")
    print(f"R² (real): {r2:.4f}")
    print(f"Varianza: {variance:.4f}")

    # Muestra algunos pares real vs predicho (sanity check)
    for i in range(5):
        print(f"Real = {Y_test_real[i]:.2f} | Predicho = {Y_pred_real[i]:.2f}")

    # Registrar complejidad y desempeño (append) para análisis posterior
    with open("complejidad_modelo.csv", "a") as f_cme:
        f_cme.write(f"{num_features-1},{mae},{r2},{b},{variance}\n")

    # Guardar curva de error por época para graficar después
    with open("errores_por_epoca.txt", "w") as f:
        for i, err in enumerate(errores):
            f.write(f"{i},{err}\n")

# Inicializar archivo CSV de complejidad vs error con encabezados
with open("complejidad_modelo.csv", "w") as f_cme:
    f_cme.write("N_Features,MAE,R2,Bias,Varianza\n")

# Barrido de complejidad: entrena el modelo con distintos números de features.
# Se pasa 'num_features + 1' a main para que el print muestre 'num_features - 1' (excluyendo target).
for num_features in [2, 4, 8, 12, 16, 20, 22]:
    main(num_features+1)
