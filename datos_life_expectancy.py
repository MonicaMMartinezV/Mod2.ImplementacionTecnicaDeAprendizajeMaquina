"""
Script para cargar y normalizar datos desde un archivo CSV de esperanza de vida.

Flujo principal:
- Lee el archivo CSV (por defecto "Life Expectancy Data.csv").
- Limpia los nombres de las columnas eliminando espacios.
- Elimina columnas no numéricas irrelevantes ("Country", "Status").
- Elimina filas con valores nulos.
- Selecciona un número definido de variables predictoras (num_features).
- Calcula la media y desviación estándar de cada feature, normalizando sus valores.
- Normaliza también la variable objetivo ("Life expectancy").

Salidas:
- X_final: matriz de características normalizadas (lista de listas).
- Y_final: lista de valores objetivo normalizados.
- y_mean, y_std: parámetros de normalización de la variable objetivo.
"""

import pandas as pd

def cargar_datos_normalizados(path="Life Expectancy Data.csv", num_features=23):
    
    """
    Carga datos desde un CSV, selecciona un número de características (features),
    elimina columnas irrelevantes y normaliza tanto las variables predictoras
    como la variable objetivo ("Life expectancy").

    Parámetros:
        path (str): ruta al archivo CSV.
        num_features (int): número de características a incluir en el modelo.

    Retorna:
        X_final (list[list[float]]): matriz con features normalizadas.
        Y_final (list[float]): lista con la variable objetivo normalizada.
        y_mean (float): media original de la variable objetivo.
        y_std (float): desviación estándar original de la variable objetivo.
    """
    
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(path)

    # Eliminar espacios en blanco en los nombres de las columnas
    df.columns = df.columns.str.strip()

    # Quitar columnas categóricas que no sirven para el análisis numérico
    df = df.drop(columns=["Country", "Status"])
    
    # Eliminar filas con valores faltantes (NaN)
    df = df.dropna()

    # Definir variable objetivo
    target = "Life expectancy"

    # Seleccionar primeras 'num_features' columnas como features,
    # excluyendo la variable objetivo si estuviera entre ellas
    features = [col for col in df.columns[:num_features] if col != target]
    
    # Esto es en caso de que solo quiera 20 features específicas
    #features = [
    #    'Schooling', 'Income composition of resources', 'BMI', 'Diphtheria', 'Polio',
    #    'GDP', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Total expenditure',
    #    'Year', 'Population', 'Measles', 'infant deaths', 'under-five deaths',
    #    'thinness 5-9 years', 'thinness  1-19 years', 'HIV/AIDS', 'Adult Mortality'
    #]

    # Seleccionar las columnas de entrada (X) y los valores de salida (Y)
    X_raw = df[features]
    Y_raw = df[target].tolist()

    # Aquí se guardarán las columnas normalizadas
    X_norm = []

    # Normalización manual de cada feature: (x - media) / desviación estándar
    for col in features:
        col_values = X_raw[col].tolist()
        mean = sum(col_values) / len(col_values)
        std = (sum((x - mean) ** 2 for x in col_values) / len(col_values)) ** 0.5
        norm_col = [(x - mean) / std for x in col_values]
        X_norm.append(norm_col)

    # Reestructurar X_norm para obtener lista de listas:
    # cada fila representa una observación con todas sus features
    X_final = [
        [X_norm[j][i] for j in range(len(features))] for i in range(len(X_norm[0]))
    ]

    # Calcular media y desviación estándar de la variable objetivo
    y_mean = sum(Y_raw) / len(Y_raw)
    y_std = (sum((y - y_mean) ** 2 for y in Y_raw) / len(Y_raw)) ** 0.5
    
    # Normalizar Y
    Y_final = [(y - y_mean) / y_std for y in Y_raw]

    # Retornar matriz de features, lista objetivo normalizada
    # y parámetros de normalización de Y
    return X_final, Y_final, y_mean, y_std
