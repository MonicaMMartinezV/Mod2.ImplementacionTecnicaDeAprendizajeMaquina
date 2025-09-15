"""
Implementación de Análisis de Componentes Principales (PCA) desde cero.

Este módulo permite reducir la dimensionalidad de una matriz de características
mediante descomposición en componentes principales, calculando también la
varianza explicada y acumulada.

Flujo:
- Estandarización de datos.
- Cálculo de matriz de covarianza.
- Eigendecomposition (valores y vectores propios).
- Ordenamiento por varianza.
- Proyección a subespacio reducido.
"""
import numpy as np

def standardize_data(X):
    """
    Estandariza la matriz X: cada columna con media 0 y desviación estándar 1.
    """
    rows, columns = X.shape
    standardizedArray = np.zeros(shape=(rows, columns))
    for col in range(columns):
        mean = np.mean(X[:, col])
        std = np.std(X[:, col])
        standardizedArray[:, col] = (X[:, col] - mean) / std
    return standardizedArray

def pca(X, n_components):
    """
    Aplica PCA siguiendo el tutorial: covarianza → eigendecomposition → proyección.
    """
    X_np = np.array(X)

    # 1. Estandarizar (aunque ya normalizaste, incluimos para seguir al profe)
    X_std = standardize_data(X_np)

    # 2. Matriz de covarianza
    covariance_matrix = np.cov(X_std.T)

    # 3. Eigenvalores y eigenvectores
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # 4. Orden descendente
    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # 5. Varianza explicada
    variance_explained = [(i / sum(eigen_values)) * 100 for i in eigen_values]
    cumulative_variance = np.cumsum(variance_explained)

    # 6. Proyección de datos en los n_components
    projection_matrix = eigen_vectors[:, :n_components]
    X_pca = X_std.dot(projection_matrix)

    return X_pca, variance_explained, cumulative_variance
