"""
Funciones base para la implementación desde cero de un modelo de regresión lineal múltiple.

Incluye:
- Función de hipótesis (predicción).
- Función de costo (MSE).
- Algoritmo de entrenamiento (gradiente descendente).
- Separación de datos en entrenamiento y prueba.
- Desnormalización de valores.
- Cálculo de métricas de evaluación (MAE y R²).

Estas funciones son utilizadas para entrenar, evaluar y analizar modelos de predicción de esperanza de vida
en un conjunto de datos numéricos previamente normalizado.

Salidas:
- Coeficientes ajustados (theta), bias optimizado (b), predicciones y métricas.
"""

def functionHyp(x, theta, b):
    
    """
    Función de hipótesis: calcula la predicción de y dada una entrada x,
    los coeficientes theta y el sesgo b.
    """

    y = 0.0
    for i in range(len(theta)):
        y += theta[i] * x[i]
    y += b
    return y


def MSE(data, theta, b, Y):
    
    """
    Calcula el error cuadrático medio (Mean Squared Error)
    entre las predicciones y los valores reales.
    """

    m = len(data)
    cost = 0
    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        cost += (pred - Y[i]) ** 2
    return cost / m


def update(data, theta, b, Y, alpha):

    """
    Realiza un paso de gradiente descendente para actualizar
    los coeficientes theta y el bias b.
    """

    m = len(data)
    n = len(theta)

    # Gradientes para cada theta
    grad = [0.0 for _ in range(n)]

    # Calcular el gradiente para cada coeficiente theta
    for j in range(n):
        for i in range(m):
            pred = functionHyp(data[i], theta, b)
            error = pred - Y[i]
            # Derivada parcial respecto a theta_j
            grad[j] += error * data[i][j]

    # Actualizar cada coeficiente theta con su gradiente
    for j in range(n):
        theta[j] -= (alpha / m) * grad[j]

    # Calcular y actualizar el gradiente del bias (b)
    grad_b = 0.0
    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        error = pred - Y[i]
        grad_b += error

    # Actualización del bias
    b -= (alpha / m) * grad_b

    return theta, b


def split_data(X, Y, test_ratio=0.2, seed=42):
    
    """
    Divide los datos en conjuntos de entrenamiento y prueba según un test_ratio.

    Parámetros:
        X (list): matriz de características.
        Y (list): lista de valores objetivo.
        test_ratio (float): proporción de datos para prueba.
        seed (int): semilla para reproducibilidad.

    Retorna:
        X_train, Y_train, X_test, Y_test
    """

    import random

    # Para que el resultado sea reproducible
    random.seed(seed)
    indices = list(range(len(X)))

    # Mezcla aleatoriamente los índices
    random.shuffle(indices)

    # Tamaño del conjunto de prueba
    test_size = int(len(X) * test_ratio)

    # Índices para prueba
    test_idx = indices[:test_size]

    # Índices para entrenamiento
    train_idx = indices[test_size:]

    # Separar los datos según los índices    
    X_train = [X[i] for i in train_idx]
    Y_train = [Y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    Y_test = [Y[i] for i in test_idx]
    return X_train, Y_train, X_test, Y_test


def desnormalizar(val, mean, std):
    
    """
    Convierte un valor normalizado a su escala original.
    """

    return val * std + mean


def calcular_MAE(y_real, y_pred):

    """
    Calcula el error absoluto medio entre valores reales y predichos.
    """
    
    return sum(abs(yr - yp) for yr, yp in zip(y_real, y_pred)) / len(y_real)


def calcular_R2(y_real, y_pred):

    """
    Calcula el coeficiente de determinación R².

    R² mide qué proporción de la variabilidad en los datos reales
    es explicada por el modelo.
    """

    mean_y = sum(y_real) / len(y_real)

    # Suma total de cuadrados (total variability)
    ss_total = sum((y - mean_y) ** 2 for y in y_real)

    # Suma de los errores al cuadrado (variabilidad no explicada)
    ss_res = sum((yr - yp) ** 2 for yr, yp in zip(y_real, y_pred))
    
    return 1 - (ss_res / ss_total)
