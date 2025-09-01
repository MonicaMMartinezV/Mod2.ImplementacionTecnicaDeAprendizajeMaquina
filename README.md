# Modelo para predicción de la esperanza de vida

Este proyecto implementa un modelo de regresión lineal múltiple desde cero, sin el uso de frameworks de machine learning (como scikit-learn, TensorFlow o PyTorch). El objetivo es predecir la esperanza de vida (Life Expectancy) a partir de factores de salud, educación, inmunización y economía, utilizando el dataset [*Life Expectancy Data*](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/code) de la OMS y la ONU.

El modelo fue optimizado mediante gradiente descendiente por lotes y evaluado con métricas como MAE, R², bias y varianza de errores.

## Requisitos

Para ejecutar el proyecto necesitas Python 3.x y las siguientes librerías:

```bash
pandas
matplotlib
```

Puedes instalarlas con:

```bash
pip install pandas matplotlib
```

## Ejecución

1. Clona el repositorio y ubica el dataset `Life Expectancy Data.csv` en la carpeta raíz.
2. Ejecuta el archivo principal para entrenar el modelo con todas las variables:

```bash
python main.py
```

3. Si deseas analizar el desempeño con diferentes cantidades de features:

```bash
python main_mr.py
```

4. Los resultados (MAE, R², bias, varianza) se guardarán en consola y en archivos auxiliares (`errores_por_epoca.txt`, `complejidad_modelo_v_error.csv`).
5. En caso de que no tengas el dataset descargalo en el siguiente link [*Life Expectancy Data*](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/code) y sigue las instrucciones ed ejecución.
