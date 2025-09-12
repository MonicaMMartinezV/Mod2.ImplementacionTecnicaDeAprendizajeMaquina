"""
Implementación de un modelo Random Forest para regresión de esperanza de vida
usando scikit-learn, con evaluación sobre train, validación y test.

Este archivo es autónomo y compatible con el mismo pipeline que el modelo
manual, pero con salidas independientes.

Salidas:
- Métricas (MAE, R2, Bias, Varianza)
- Gráficas PNG (error y real vs predicho)
"""

from datos_life_expectancy import cargar_datos_normalizados
from graficas_resultados import (
    calcular_bias,
    calcular_varianza_errores,
    graficar_real_vs_predicho,
    graficar_feature_importance,
    graficar_comparativa_modelos
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from modelo_regresion import split_data_val
import csv

def main_framework(num_features=20, seed=42):
    X_final, Y_final, y_mean, y_std = cargar_datos_normalizados(num_features=num_features)
    
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data_val(X_final, Y_final, val_ratio=0.2, test_ratio=0.2, seed=seed)

    random_forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=seed,
        n_jobs=-1
    )

    random_forest.fit(X_train, Y_train)

    # --- PREDICCIONES ---
    Y_val_pred = random_forest.predict(X_val)
    Y_test_pred = random_forest.predict(X_test)

    # Desnormalizar
    Y_val_pred_real = [y * y_std + y_mean for y in Y_val_pred]
    Y_val_real = [y * y_std + y_mean for y in Y_val]
    Y_test_pred_real = [y * y_std + y_mean for y in Y_test_pred]
    Y_test_real = [y * y_std + y_mean for y in Y_test]

    # --- MÉTRICAS ---
    mae = mean_absolute_error(Y_test_real, Y_test_pred_real)
    r2 = r2_score(Y_test_real, Y_test_pred_real)
    bias = calcular_bias(Y_test_real, Y_test_pred_real)
    varianza = calcular_varianza_errores(Y_test_real, Y_test_pred_real)
    mae_val = mean_absolute_error(Y_val_real, Y_val_pred_real)
    r2_val = r2_score(Y_val_real, Y_val_pred_real)
    

    print("\n--- Resultados Random Forest ---")
    print(f"MAE (test): {mae:.4f}")
    print(f"R²  (test): {r2:.4f}")
    print(f"MAE (val):  {mae_val:.4f}")
    print(f"R²  (val):  {r2_val:.4f}")
    print(f"Bias:       {bias:.4f}")
    print(f"Varianza:   {varianza:.4f}")

    with open("metricas_modelo_framework.csv", "w") as f:
        f.write("MAE,R2,Bias,Varianza\n")
        f.write(f"{mae:.4f},{r2:.4f},{bias:.4f},{varianza:.4f}")

    graficar_real_vs_predicho(
        Y_test_real,
        Y_test_pred_real,
        title="Random Forest: Real vs Predicho",
        output_path="grafica_rf_real_vs_predicho.png"
    )

    graficar_feature_importance(random_forest.feature_importances_)

    with open("metricas_modelo_manual.csv", "r") as f:
            next(f)
            linea = f.readline()
            mae_m, r2_m, bias_m, var_m = map(float, linea.strip().split(","))

    graficar_comparativa_modelos(
        mae_m, r2_m, bias_m, var_m,
        mae, r2, bias, varianza
        )

if __name__ == "__main__":
    main_framework()
