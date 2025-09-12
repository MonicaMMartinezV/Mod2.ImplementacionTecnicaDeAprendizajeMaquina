"""
Script de entrenamiento y evaluación de un modelo de regresión lineal múltiple (desde cero).

Flujo principal:
- Carga y normaliza los datos de esperanza de vida.
- Separa el conjunto de datos en entrenamiento y prueba.
- Inicializa los coeficientes del modelo y entrena mediante gradiente descendente.
- Calcula las predicciones sobre los datos de prueba.
- Desnormaliza los valores para interpretarlos en su escala original.
- Evalúa el desempeño del modelo usando MAE, R² y varianza de errores.
- Imprime los resultados y muestra una comparación entre valores reales y predichos.
- Guarda la evolución del error por época para su análisis posterior.

Salidas:
- Archivo "errores_por_epoca.txt" con la evolución del MSE durante el entrenamiento.
"""

from datos_life_expectancy import cargar_datos_normalizados
from graficas_resultados import (
    calcular_varianza_errores,
    calcular_bias,
    graficar_error_entrenamiento,
    graficar_real_vs_predicho,
    graficar_error_entrenamiento_dual,
    graficar_complejidad_vs_metricas,
    graficar_comparativa_modelos
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

# Carga X (features) y Y (objetivo), ambos normalizados,
# junto con la media y desviación estándar de la variable objetivo
X_final, Y_final, y_mean, y_std = cargar_datos_normalizados(num_features=21)

# División con validación
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data_val(X_final, Y_final)

# Inicializa los coeficientes theta en 0 (uno por cada feature)
theta = [0.0] * len(X_train[0])

# Inicializa el sesgo (bias)
b = 0.0

# Define tasa de aprendizaje
alpha = 0.01

# Número máximo de iteraciones (épocas)
epoc = 1000

# Lista para almacenar el historial de errores (MSE)
errores_train = []
errores_val = []

# Entrenamiento usando gradiente descendente
for i in range(epoc):
    # Calcula el error cuadrático medio en los datos de entrenamiento
    error_train = MSE(X_train, theta, b, Y_train)
    error_val = MSE(X_val, theta, b, Y_val)

    errores_train.append(error_train)
    errores_val.append(error_val)

    # Parada temprana si el error es suficientemente pequeño
    if error_val < 1e-2:
        break

    # Actualiza los parámetros (theta y bias) con un paso de gradiente
    theta, b = update(X_train, theta, b, Y_train, alpha)

# Realiza predicciones con los datos de prueba (normalizados)
Y_pred_norm = [functionHyp(x, theta, b) for x in X_test]

# Desnormaliza las predicciones para interpretarlas en escala real
Y_pred_real = [desnormalizar(y, y_mean, y_std) for y in Y_pred_norm]

# Desnormaliza los valores reales de prueba para comparar correctamente
Y_test_real = [desnormalizar(y, y_mean, y_std) for y in Y_test]

# Calcula métricas de desempeño en escala real
mae = calcular_MAE(Y_test_real, Y_pred_real)
r2 = calcular_R2(Y_test_real, Y_pred_real)
variance = calcular_varianza_errores(Y_test_real, Y_pred_real)
bias = calcular_bias(Y_test_real, Y_pred_real)

# Imprime resultados clave
print("Theta final:", theta)
print("Bias final:", b)
print(f"MAE (real): {mae:.4f}")
print(f"R² (real): {r2:.4f}")
print(f"Bias: {bias:.4f}")
print(f"Varianza: {variance:.4f}")

# Imprime las primeras 5 predicciones comparadas con los valores reales
for i in range(5):
    print(f"Real = {Y_test_real[i]:.2f} | Predicho = {Y_pred_real[i]:.2f}")

# Escribe los errores por época en un archivo .txt
# (útil para graficar la evolución del MSE)
with open("errores_entrenamiento_train_val.txt", "w") as f:
    for i, (err_train, err_val) in enumerate(zip(errores_train, errores_val)):
        f.write(f"{i},{err_train},{err_val}\n")

with open("metricas_modelo_manual.csv", "w") as f:
    f.write("MAE,R2,Bias,Varianza\n")
    f.write(f"{mae:.4f},{r2:.4f},{bias:.4f},{variance:.4f}")

graficar_error_entrenamiento_dual()
graficar_real_vs_predicho(Y_test_real, Y_pred_real)