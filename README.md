# Modelos Lineales y No Lineales para la Predicción de la Esperanza de Vida: Implementación Manual y Comparativa con Random Forest

Este proyecto implementa dos modelos de regresión para predecir la esperanza de vida (Life Expectancy) a partir de factores de salud, educación, inmunización y economía, utilizando el dataset [*Life Expectancy Data*](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/code) de la OMS y la ONU.

## Modelos implementados

1. **Modelo manual base:** Regresión lineal múltiple desde cero, optimizada con gradiente descendente y evaluada con MAE, R², bias y varianza.
2. **Modelo manual mejorado (L2):** Incluye regularización Ridge (L2).
3. **Modelo manual mejorado (Huber + L2 + PCA):** Incluye función de pérdida robusta, regularización L2 y reducción de dimensionalidad con PCA.
4. **Modelo con framework (Random Forest):** Usando `scikit-learn`, permite capturar relaciones no lineales y mejorar precisión y generalización.

Todos los modelos se evaluaron tanto en conjunto de prueba como de validación, considerando criterios como bias, varianza, y nivel de ajuste (underfit / fit / overfit).

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
* `analizar_correlaciones.py`: Visualiza las correlaciones entre variables para análisis previo.
* `pca.py`: Implementación de PCA

## Ejecución

1. Clona el repositorio. Descarga y ubica el archivo `Life Expectancy Data.csv` en la carpeta raíz.

   Puedes obtener el dataset desde:
   [*Life Expectancy Data – Kaggle*](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

2. Para ejecutar el modelo manual con mejoras más actuales (Huber + L2 + PCA):

```bash
python main.py
```

3. Para ejecutar el análisis de complejidad del modelo (variando el número de variables):

```bash
python main_mr.py
```

4. Para ejecutar el modelo con framework (Random Forest):

```bash
python modelo_regresion_framework.py
```

5. Para visualizar las correlaciones entre variables:

```bash
python analizar_correlaciones.py
```

6. Las gráficas se generan automáticamente al ejecutar cada modelo:

* Evolución del error por época (entrenamiento, validación, test)
* Comparación entre valores reales y predichos
* Comparativa de métricas entre modelos
* Metricas de complejidad del modelo (R², MAE, Bias, Varianza vs #features)
* Importancia de variables (Random Forest)
* Comparativas finales entre todas las versiones

## Salidas generadas

* `errores_entrenamiento_train_val_test.txt`: errores por época del modelo manual.
* `complejidad_modelo.csv`: métrica por número de variables (main\_mr.py).
* `metricas_modelo_manual.csv`: métricas para modelos desde cero.
* `metricas_modelo_framework.csv`: métricas para Random Forest.
* Gráficas PNG en el directorio raíz.

## Evaluación de desempeño

El proyecto también incluye un análisis integral de los aspectos clave del comportamiento de los modelos. Se evaluaron el bias y la varianza para comprender tanto la tendencia sistemática de las predicciones como la estabilidad de los errores. Asimismo, se diagnosticó el nivel de ajuste del modelo, identificando si presentaba underfitting, overfitting o un ajuste adecuado. Se analizó el impacto de la regularización L2 (Ridge) sobre la estabilidad de los coeficientes, la reducción de errores y la mejora en la capacidad de generalización. Además, se exploraron los efectos del uso de PCA y la función de pérdida Huber, observando su capacidad para hacer el modelo más robusto frente a outliers y más eficiente computacionalmente al reducir la dimensionalidad del problema. Finalmente, se realizó una comparación exhaustiva entre todos los modelos desarrollados, resaltando cuál tuvo el mejor desempeño en términos de precisión, estabilidad y generalización. Todo esto se complementa con observaciones adicionales que enriquecen la comprensión y aplicación práctica de los modelos implementados.