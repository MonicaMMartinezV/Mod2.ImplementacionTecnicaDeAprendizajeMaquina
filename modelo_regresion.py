"""
Funciones para regresión lineal múltiple implementada desde cero.

Incluye:
- Función de hipótesis (predicción lineal).
- Funciones de pérdida: MSE, MAE, Huber, Ridge.
- Optimizadores: gradiente descendente, con o sin regularización.
- Evaluación: MAE, R².
- Utilidades: desnormalización, split de datos.
"""

# ========== FUNCIÓN DE HIPÓTESIS ==========
def functionHyp(x, theta, b):
    """Calcula y = θᵀx + b"""
    y = 0.0
    for i in range(len(theta)):
        y += theta[i] * x[i]
    y += b
    return y

# ========== FUNCIONES DE COSTO ==========
def MSE(data, theta, b, Y):
    """Mean Squared Error"""
    m = len(data)
    cost = 0
    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        cost += (pred - Y[i]) ** 2
    return cost / m

def MAE(data, theta, b, Y):
    """Mean Absolute Error"""
    m = len(data)
    cost = 0
    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        cost += abs(pred - Y[i])
    return cost / m

def MSE_ridge(data, theta, b, Y, lambda_reg):
    """MSE con regularización L2 (Ridge)"""
    m = len(data)
    mse = MSE(data, theta, b, Y)
    ridge_penalty = (lambda_reg / (2 * m)) * sum(t ** 2 for t in theta)
    return mse + ridge_penalty


def Huber_loss(data, theta, b, Y, delta=1.0):
    """Huber Loss (robusta a outliers)"""
    m = len(data)
    loss = 0
    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        error = pred - Y[i]
        if abs(error) <= delta:
            loss += 0.5 * (error ** 2)
        else:
            loss += delta * (abs(error) - 0.5 * delta)
    return loss / m


def Huber_ridge_loss(data, theta, b, Y, delta=1.0, lambda_reg=0.1):
    """Huber Loss + regularización L2 (Ridge)"""
    m = len(data)
    loss = 0
    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        error = pred - Y[i]
        if abs(error) <= delta:
            loss += 0.5 * (error ** 2)
        else:
            loss += delta * (abs(error) - 0.5 * delta)
    ridge_penalty = (lambda_reg / (2 * m)) * sum(t ** 2 for t in theta)
    return (loss / m) + ridge_penalty

# ========== ACTUALIZACIÓN DE PARÁMETROS ==========
def update(data, theta, b, Y, alpha):
    """Gradient Descent básico (MSE)"""
    m = len(data)
    n = len(theta)
    grad = [0.0 for _ in range(n)]

    for j in range(n):
        for i in range(m):
            pred = functionHyp(data[i], theta, b)
            error = pred - Y[i]
            grad[j] += error * data[i][j]

    for j in range(n):
        theta[j] -= (alpha / m) * grad[j]

    grad_b = 0.0
    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        error = pred - Y[i]
        grad_b += error

    b -= (alpha / m) * grad_b

    return theta, b

def update_huber(data, theta, b, Y, alpha, delta=1.0):
    """Gradient Descent con Huber Loss"""
    m = len(data)
    n = len(theta)

    grad = [0.0 for _ in range(n)]
    grad_b = 0.0

    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        error = pred - Y[i]

        if abs(error) <= delta:
            factor = error
        else:
            factor = delta * (1 if error > 0 else -1)

        for j in range(n):
            grad[j] += factor * data[i][j]
        grad_b += factor

    for j in range(n):
        theta[j] -= (alpha / m) * grad[j]
    b -= (alpha / m) * grad_b

    return theta, b

def update_ridge(data, theta, b, Y, alpha, lambda_reg):
    """Gradient Descent con regularización L2 (Ridge)"""
    m = len(data)
    n = len(theta)
    grad = [0.0 for _ in range(n)]

    for j in range(n):
        for i in range(m):
            pred = functionHyp(data[i], theta, b)
            error = pred - Y[i]
            grad[j] += error * data[i][j]

        grad[j] += lambda_reg * theta[j]

    for j in range(n):
        theta[j] -= (alpha / m) * grad[j]

    grad_b = sum(functionHyp(data[i], theta, b) - Y[i] for i in range(m))
    b -= (alpha / m) * grad_b

    return theta, b

def update_huber_ridge(data, theta, b, Y, alpha, delta=1.0, lambda_reg=0.1):
    """Gradient Descent con Huber Loss + Ridge"""
    m = len(data)
    n = len(theta)

    grad = [0.0 for _ in range(n)]
    grad_b = 0.0

    for i in range(m):
        pred = functionHyp(data[i], theta, b)
        error = pred - Y[i]

        if abs(error) <= delta:
            factor = error
        else:
            factor = delta * (1 if error > 0 else -1)

        for j in range(n):
            grad[j] += factor * data[i][j]
        grad_b += factor

    for j in range(n):
        theta[j] -= (alpha / m) * (grad[j] + lambda_reg * theta[j])

    b -= (alpha / m) * grad_b
    return theta, b

# ========== SEPARACION DE DATOS ==========
def split_data(X, Y, val_ratio=0.2, test_ratio=0.2, seed=42):
    """Divide el dataset en entrenamiento, validación y prueba"""
    import random
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)

    test_size = int(len(X) * test_ratio)
    val_size = int(len(X) * val_ratio)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    X_train = [X[i] for i in train_idx]
    Y_train = [Y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    Y_val = [Y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    Y_test = [Y[i] for i in test_idx]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

# ========== DESNORMALIZACIÓN ==========
def desnormalizar(val, mean, std):
    """Convierte valor normalizado a escala original"""
    return val * std + mean

# ========== MÉTRICAS DE EVALUACIÓN ==========
def calcular_MAE(y_real, y_pred):
    """Error absoluto medio"""    
    return sum(abs(yr - yp) for yr, yp in zip(y_real, y_pred)) / len(y_real)

def calcular_R2(y_real, y_pred):
    """Coeficiente de determinación R²"""
    mean_y = sum(y_real) / len(y_real)

    ss_total = sum((y - mean_y) ** 2 for y in y_real)

    ss_res = sum((yr - yp) ** 2 for yr, yp in zip(y_real, y_pred))
    
    return 1 - (ss_res / ss_total)
