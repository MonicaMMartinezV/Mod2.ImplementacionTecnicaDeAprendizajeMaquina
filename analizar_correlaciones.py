"""
Script para calcular y visualizar la matriz de correlación de un dataset.

Flujo principal:
- Lee un archivo CSV con datos (por defecto "Life Expectancy Data.csv").
- Limpia espacios en los nombres de columnas.
- Selecciona únicamente variables numéricas.
- Calcula la matriz de correlación entre todas las variables numéricas.
- Imprime en consola la correlación de cada variable respecto a la variable objetivo.
- Genera y guarda un heatmap de la matriz de correlación como "matriz_correlacion.png".
"""
import pandas as pd
import matplotlib.pyplot as plt

def graficar_matriz_correlacion(
    path="Life Expectancy Data.csv", target="Life expectancy"
):
    """
    Lee un archivo CSV, calcula la matriz de correlación de las variables numéricas
    y la grafica como un mapa de calor. Además, imprime las correlaciones ordenadas
    respecto a una variable objetivo.
    """

    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(path)

    # Eliminar espacios en blanco en los nombres de las columnas
    df.columns = df.columns.str.strip()

    # Seleccionar únicamente las columnas numéricas
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Calcular la matriz de correlación
    corr = numeric_df.corr()

    # Mostrar correlaciones ordenadas con respecto a la variable objetivo
    print("\nCorrelación con la variable objetivo:")
    print(corr[target].sort_values(ascending=False))

    # Crear la figura para graficar la matriz de correlación
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap="coolwarm", interpolation="none")
    plt.title("Matriz de correlación")
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("matrizcorrelacion.png")
    plt.show()

if __name__ == "__main__":
    graficar_matriz_correlacion()
