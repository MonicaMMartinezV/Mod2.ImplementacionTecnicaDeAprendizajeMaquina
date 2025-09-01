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
    errores = []

    # Abrir archivo y leer línea por línea
    with open(path, "r") as f:
        for line in f:
            i, e = line.strip().split(",")
            epocas.append(int(i))
            errores.append(float(e))

    # Crear la gráfica de error por época
    plt.figure(figsize=(8, 4))
    plt.plot(epocas, errores, marker='o', linestyle='-', color='blue')
    plt.title("Evolución del error (MSE) durante el entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Error (MSE)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafica_error_entrenamiento.png")
    plt.show()

def graficar_real_vs_predicho(Y_real, Y_predicho):

    """
    Grafica los valores reales contra los valores predichos para visualizar
    el ajuste del modelo.

    Parámetros:
        Y_real (list or array): Valores reales.
        Y_predicho (list or array): Valores predichos por el modelo.
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_real, Y_predicho, color='green', alpha=0.6)
    plt.plot([min(Y_real), max(Y_real)], [min(Y_real), max(Y_real)], 'r--')  # Línea ideal
    plt.title("Valores reales vs predichos")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafica_real_vs_predicho.png")
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

if __name__ == "__main__":
    # Importar resultados desde el script principal (main.py)
    from main import Y_test_real, Y_pred_real, b

    # Graficar evolución del error durante el entrenamiento
    graficar_error_entrenamiento()

    # Graficar valores reales contra valores predichos
    graficar_real_vs_predicho(Y_test_real, Y_pred_real)
