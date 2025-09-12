# Modelo para predicción de la esperanza de vida

Este proyecto implementa dos modelos de regresión para predecir la esperanza de vida (Life Expectancy) a partir de factores de salud, educación, inmunización y economía, utilizando el dataset [*Life Expectancy Data*](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/code) de la OMS y la ONU.

1. **Modelo manual:** Regresión lineal múltiple implementada desde cero, sin frameworks de machine learning.
2. **Modelo con framework:** Regresión mediante Random Forest usando `scikit-learn`.

Ambos modelos son evaluados con métricas como MAE, R², bias y varianza de errores. Además, se comparan gráficamente para identificar su grado de ajuste (underfit, overfit, fit), bias y varianza.

## Requisitos

Para ejecutar el proyecto necesitas Python 3.x y las siguientes librerías:

```bash
pandas
numpy
matplotlib
scikit-learn
````

Puedes instalarlas con:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Archivos principales

* `main.py`: Entrena y evalúa el modelo manual.
* `modelo_regresion_framework.py`: Entrena y evalúa el modelo con Random Forest (scikit-learn).
* `main_mr.py`: Evalúa el modelo manual variando el número de features (complejidad del modelo).
* `graficas_resultados.py`: Contiene todas las funciones de graficación (errores, métricas, comparativas, etc.).
* `datos_life_expectancy.py`: Funciones para cargar y normalizar el dataset.
* `modelo_regresion.py`: Implementación del algoritmo de regresión lineal múltiple desde cero.

## Ejecución

1. Clona el repositorio. Descarga y ubica el archivo `Life Expectancy Data.csv` en la carpeta raíz.

   Puedes obtener el dataset desde:
   [*Life Expectancy Data – Kaggle*](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

2. Para ejecutar el modelo **manual**:

```bash
python main.py
```

3. Para ejecutar el modelo **con framework (Random Forest)**:

```bash
python modelo_regresion_framework.py
```

4. Para evaluar la evolución del desempeño del modelo manual según el número de features:

```bash
python main_mr.py
```

Esto generará el archivo `complejidad_modelo.csv` y sus gráficas asociadas.

5. Para visualizar las correlaciones entre variables:

```bash
python analizar_correlaciones.py
```

6. Las gráficas se generan automáticamente al ejecutar cada modelo:

   * Evolución del error por época
   * Comparación entre valores reales y predichos
   * Comparativa entre modelos
   * Importancia de características (modelo con framework)
   * Curva de complejidad del modelo

## Salidas generadas

* `errores_entrenamiento_train_val.txt`: Error de validación por época del modelo manual.
* `complejidad_modelo.csv`: Métricas por número de features (modelo manual).
* `metricas_modelo_manual.csv` y `metricas_modelo_framework.csv`: Métricas generales de ambos modelos.
* Gráficas PNG: todas las gráficas generadas automáticamente (se guardan en el directorio raíz).

## Evaluación de desempeño

El proyecto también incluye un análisis completo del grado de ajuste de ambos modelos (bias, varianza, overfitting/underfitting) y el impacto del número de características en el rendimiento.

Este análisis está soportado por métricas cuantitativas y gráficas, y se encuentra documentado en el reporte correspondiente.
