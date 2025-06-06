# Módulo de Análisis Estadístico en Python

Este es un módulo de Python que proporciona herramientas estadísticas básicas y avanzadas para análisis de datos, generación de distribuciones, ajuste de modelos de regresión lineal y logística, y evaluación de supuestos y predicciones.

## Descripción

El módulo contiene varias clases orientadas a la estadística descriptiva, generación de datos simulados, y ajuste de modelos de regresión lineal (simple y múltiple) y logística.

## Características

- **Clase `AnalisisDescriptivo`**  
  Herramientas para análisis exploratorio:  
  - Cálculo de media, mediana, desviación estándar, cuartiles.  
  - Estimación de densidad con distintos núcleos (gaussiano, uniforme, triangular, cuadrático).  
  - Generación y evaluación de histogramas.  
  - Gráfico QQ para evaluar normalidad.

- **Clase `GeneradoraDeDatos`**  
  Generación de datos simulados:  
  - Distribuciones normales, uniformes y mezcla BS.  
  - Funciones de densidad teórica correspondientes.

- **Clase `RegresionLineal`**  
  Ajuste general de regresión lineal:  
  - Ajuste de modelo, obtención de residuos y valores ajustados.  
  - Evaluación de supuestos de normalidad y homocedasticidad.  
  - Cálculo de $R^2$, $R^2$ ajustado, intervalos de confianza y p-valores.  
  - Gráfico de dispersión y residuos.

- **Clase `RegresionLinealSimple`**  
  Extiende `RegresionLineal` para una variable predictora:  
  - Estimación directa de coeficientes por mínimos cuadrados.  
  - Gráfico con recta estimada.  
  - Predicciones puntuales e intervalos de confianza/predicción.

- **Clase `RegresionLinealMultiple`**  
  Extiende `RegresionLineal` para múltiples regresores:  
  - Predicción para nuevos valores.  
  - Visualización de resumen del modelo.

- **Clase `RegresionLogistica`**  
  Ajuste de modelos logísticos binarios:  
  - División train/test aleatoria.  
  - Ajuste del modelo con `statsmodels`.  
  - Predicciones, matriz de confusión, sensibilidad y especificidad.  
  - Curva ROC y cálculo de AUC.  
  - Evaluación e interpretación del desempeño del clasificador.

## Instalación

Este módulo no requiere instalación específica. Solo asegurate de tener instaladas las siguientes librerías:

 numpy ,pandas ,matplotlib ,seaborn ,scipy, statsmodels ,scikit-learn.

Guardá el archivo modulo.py en tu directorio de trabajo y luego podés importarlo desde cualquier script de Pyhton:

from modulo import AnalisisDescriptivo, GeneradoraDeDatos, RegresionLinealSimple.


## Ejemplos de usos:

# Generar datos
gen = GeneradoraDeDatos(n=100)
datos = gen.generar_datos_dist_norm(media=50, desvio=10)

# Análisis descriptivo
ad = AnalisisDescriptivo(datos)
print(ad.resumen_numerico())
ad.miqqplot()

# Regresión lineal simple
x = np.linspace(0, 10, 100)
y = 3 * x + np.random.normal(0, 1, 100)

modelo = RegresionLinealSimple(x, y)
modelo.ajustar_modelo()
modelo.graf_scatter_recta()
print("Betas:", modelo.estimacion_betas())
print("R cuadrado:", modelo.r_cuadrado())

## Ejemplo de uso regresion Logitsica:

# Crear un dataset simulado
np.random.seed(42)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(2, 1, n)
X = pd.DataFrame({'x1': x1, 'x2': x2})

# Generar variable respuesta binaria (dependiente de x1 y x2)
log_odds = 1.5 * x1 - 2 * x2
prob = 1 / (1 + np.exp(-log_odds))
y = np.random.binomial(1, prob)
y = pd.Series(y)

# Combinar en un solo DataFrame
data = pd.concat([X, y.rename("y")], axis=1)

# Inicializar la clase
modelo_logit = RegresionLogistica(data)

# Separar en train/test
test, train = modelo_logit.separar_data_train_test(seed=0, ptje_test=0.2)

# Variables predictoras y respuesta
x_train = train[['x1', 'x2']]
y_train = train['y']
x_test = test[['x1', 'x2']]
y_test = test['y']

# Ajustar el modelo
modelo_logit.ajustar_modelo(x_train, y_train, x_test, y_test)

# Evaluación
modelo_logit.modelo_resumen()
print("Parámetros:", modelo_logit.parametros_modelo())

# Matriz de confusión y métricas
print("Matriz de confusión:", modelo_logit.matriz_confusion())
print("Sensibilidad y especificidad:", modelo_logit.especif_sensib())

# Curva ROC
modelo_logit.curva_ROC()

# Clasificación de un nuevo caso
x_nuevo = [0.5, 1.0]
y_pred = modelo_logit.predict_y(x_nuevo)
print("Predicción binaria para nuevo x:", y_pred)