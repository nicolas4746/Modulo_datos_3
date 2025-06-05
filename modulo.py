import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import norm, uniform, stats
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

class AnalisisDescriptivo:
    def __init__(self, datos):
        """Inicializa la clase con los datos de ingresos."""
        self.datos = np.array(datos)

    def kernel_gaussiano(self, u):
        """Calcula el valor del núcleo gaussiano dado un valor u."""
        valor_kernel_gaussiano = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)
        return valor_kernel_gaussiano

    def kernel_uniforme(self, u):
        """Calcula el valor del núcleo uniforme dado un valor u."""
        valor_kernel_uniforme = 1  if -1/2 <= u <= 1/2 else 0
        return valor_kernel_uniforme

    def kernel_cuadratico(self, u):
        """Calcula el valor del núcleo cuadrático de cuadratico dado valor u."""
        if -1 <= u <= 1:
            return 0.75 * (1 - u**2)
        else:
            return 0

    def kernel_triangular(self, u):
        """Calcula el valor del núcleo triangular dado un valor u."""
        if -1 <= u <= 0:
            return 1 + u
        elif 0 < u <= 1:
            return 1 - u
        else:
            return 0

    def densidad_cauchy(self, x):
        """Estimación de la densidad utilizando el núcleo de Cauchy."""
        #
    def densidad_nucleo(self, h, kernel, x):
        """Estimación de la densidad utilizando el kernel especificado."""
        n = len(self.datos)
        density = np.zeros(len(x))

        for j in range(len(x)):
            val = x[j]
            count = 0

            for i in range(n):
                u = (val - self.datos[i]) / h
                if kernel == 'uniforme':
                    count += self.kernel_uniforme(u)
                elif kernel == 'gaussiano':
                    count += self.kernel_gaussiano(u)
                elif kernel == 'cuadratico':
                    count += self.kernel_cuadratico(u)
                elif kernel == 'triangular':
                    count += self.kernel_triangular(u)

            density[j] = count / (n * h)
        return density

    def genera_histograma(self, h):
        """Calcula la densidad del histograma para un ancho de bin h."""
        bins = np.arange(min(self.datos) - h / 2, max(self.datos) + h, h)
        histograma = np.zeros(len(bins) - 1)

        for i in self.datos:
            for j in range(len(bins) - 1):
                if bins[j] <= i < bins[j + 1]:
                    histograma[j] += 1
                    break

        frecuencia_relativa = histograma / len(self.datos)
        densidad = frecuencia_relativa / h
        return densidad, bins

    def evalua_histograma(self, h, x):
        """Evalúa la densidad del histograma en los puntos x dados un ancho de bin h """
        densidad, bins = self.genera_histograma(h)
        estim_hist = np.zeros(len(x))

        # Iterar sobre los puntos x
        for idx in range(len(x)):
            i = x[idx]
            intervalo_idx = -1
            for j in range(len(bins) - 1):
                if bins[j] <= i < bins[j + 1]:
                    intervalo_idx = j
                    break

            if intervalo_idx != -1:
                estim_hist[idx] = densidad[intervalo_idx]

        return estim_hist

    def calculo_de_media(self):
        return np.mean(self.datos)

    def calculo_de_mediana(self):
        return np.median(self.datos)

    def calculo_de_desvio_estandar(self):
        return np.std(self.datos)

    def calculo_de_cuartiles(self):
        q1 = np.percentile(self.datos, 25)
        q2 = np.percentile(self.datos, 50)
        q3 = np.percentile(self.datos, 75)
        return [q1, q2, q3]

    def resumen_numerico(self):
        return {
            'Media': self.calculo_de_media(),
            'Mediana': self.calculo_de_mediana(),
            'Desvio': self.calculo_de_desvio_estandar(),
            'Cuartiles': self.calculo_de_cuartiles(),
            'Mínimo': np.min(self.datos),
            'Máximo': np.max(self.datos)
        }


class GeneradoraDeDatos:
    def __init__(self, n):
        self.n = n

    def generar_datos_dist_norm(self, media, desvio):
        return np.random.normal(media, desvio, self.n)

    def generar_datos_dist_unif(self, media, desvio):
        return np.random.uniform(media, desvio, self.n)

    def generar_datos_BS(self):
        u = np.random.uniform(size=(self.n,))
        y = u.copy()
        ind = np.where(u > 0.5)[0]
        y[ind] = np.random.normal(0, 1, size=len(ind))
        for j in range(5):
            ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
            y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))
        return y
    def teorica_norm(self, x, media, desvio):
        p = norm.pdf(x, media, desvio)
        return p

    def teorica_unif(a, b, nume_puntos = 1000):
        x = np.linspace(a-1 , b+1 , nume_puntos)
        y = uniform.pdf(x, loc= a, scale= b-a)
        return x,y

    def teorica_BS(self, x):
        term1 = (1 / 2) * norm.pdf(x, loc=0, scale=1)
        term2 = (1 / 10) * sum(norm.pdf(x, loc=j / 2 - 1, scale=1 / 10) for j in range(5))
        p = term1 + term2
        return p


class Regresion:
  pass


class RegresionLineal(Regresion):
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.x_bar = np.mean(self.x)
        self.y_bar = np.mean(self.y)
        self.b1 = self.calcular_pendiente()
        self.b0 = self.calcular_ordenada_origen()

    def calcular_pendiente(self):
        """Calcula la pendiente de la recta de regresión"""
        return np.sum((self.x - self.x_bar) * (self.y - self.y_bar)) / np.sum((self.x - self.x_bar) ** 2)

    def calcular_ordenada_origen(self):
        """Calcula la ordenada al origen de la recta de regresión"""
        return self.y_bar - self.b1 * self.x_bar

    def predecir(self, x):
        """Devuelve el valor predicho de y para un valor o arreglo de x"""
        return self.b0 + self.b1 * np.array(x)

    def residuos(self):
        """Devuelve los residuos del modelo"""
        return self.y - self.predecir(self.x)

    def graficar_dispersion(self):
        """Grafica los puntos originales y la recta de regresión"""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.x, y=self.y, label='Datos observados')
        plt.plot(self.x, self.predecir(self.x), color='red', label='Recta de regresión')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gráfico de dispersión con recta de regresión')
        plt.legend()
        plt.grid(True)
        plt.show()

    def graficar_residuos(self):
        """Grafica los residuos del modelo"""
        plt.figure(figsize=(8, 6))
        sns.residplot(x=self.x, y=self.y, lowess=True, line_kws={'color': 'red'})
        plt.xlabel('x')
        plt.ylabel('Residuos')
        plt.title('Gráfico de residuos')
        plt.axhline(0, color='black', linestyle='--')
        plt.grid(True)
        plt.show()

    def coeficiente_correlacion(self):
        """Calcula y devuelve el coeficiente de correlación de Pearson"""
        r, _ = pearsonr(self.x, self.y)
        return r

    def graficar_qqplot_residuos(self):
        """Genera un Q-Q plot de los residuos para evaluar la normalidad"""
        residuos = self.residuos()
        stats.probplot(residuos, dist="norm", plot=plt)
        plt.title("Q-Q Plot de los residuos")
        plt.grid(True)
        plt.show()

    def ajustar_con_statsmodels(self):
        """Ajusta el modelo utilizando statsmodels y muestra el resumen"""
        X = sm.add_constant(self.x)
        modelo = sm.OLS(self.y, X).fit()
        print(modelo.summary())
        return modelo

    def intervalos_confianza_prediccion(self, x_nuevos, alpha=0.05):
        """
        Calcula intervalos de confianza y predicción para nuevos valores de x
        Parámetros:
        - x_nuevos: lista o array de nuevos valores de x
        - alpha: nivel de significancia (default 0.05 = 95% confianza)
        Devuelve:
        - Un DataFrame con las predicciones y los intervalos
        """
        X = sm.add_constant(self.x)
        modelo = sm.OLS(self.y, X).fit()
        X_nuevos = sm.add_constant(np.array(x_nuevos))
        predicciones = modelo.get_prediction(X_nuevos)
        resumen = predicciones.summary_frame(alpha=alpha)
        return resumen

    def r_cuadrado(self):
        """Calcula y devuelve el coeficiente de determinación R²"""
        ss_total = np.sum((self.y - self.y_bar) ** 2)
        ss_res = np.sum((self.y - self.predecir(self.x)) ** 2)
        return 1 - ss_res / ss_total

    def r_cuadrado_ajustado(self):
        """Calcula y devuelve el R² ajustado"""
        r2 = self.r_cuadrado()
        return 1 - (1 - r2) * (self.n - 1) / (self.n - 2)

class RegresionLogistica(Regresion):
    def __init__(self, X, y, porcentaje_train=0.7):
        """
        Ajusta el modelo de regresión logística usando statsmodels.

        Parámetros:
        - X: matriz de características (numpy array o DataFrame)
        - y: vector de etiquetas (numpy array o Series)
        - porcentaje_train: proporción de los datos usados para el entrenamiento
        """
        self.X = sm.add_constant(np.array(X))  # Agrega intercepto
        self.y = np.array(y)
        self.porcentaje_train = porcentaje_train

        # Separar datos en train y test
        n_train = int(len(self.y) * porcentaje_train)
        self.X_train = self.X[:n_train]
        self.y_train = self.y[:n_train]
        self.X_test = self.X[n_train:]
        self.y_test = self.y[n_train:]

        # Ajustar modelo
        self.modelo = sm.Logit(self.y_train, self.X_train).fit(disp=False)

        # Guardar estadísticas del modelo
        self.betas = self.modelo.params
        self.errores_std = self.modelo.bse
        self.t_obs = self.modelo.tvalues
        self.p_valores = self.modelo.pvalues

    def resumen(self):
        """Imprime los coeficientes, errores estándar, t_obs y p-valores."""
        resumen = pd.DataFrame({
            "Beta": self.betas,
            "Error estándar": self.errores_std,
            "t_obs": self.t_obs,
            "p_valor": self.p_valores
        })
        print(resumen)

    def predecir(self, X_nuevos, umbral=0.5):
        """
        Realiza predicciones sobre nuevos datos usando un umbral.

        Parámetros:
        - X_nuevos: datos nuevos (sin columna de 1s)
        - umbral: valor de corte para clasificar

        Devuelve:
        - array con predicciones (0 o 1)
        """
        X_nuevos = sm.add_constant(np.array(X_nuevos))
        probas = self.modelo.predict(X_nuevos)
        return (probas >= umbral).astype(int)

    def evaluar_test(self, umbral=0.5):
        """
        Calcula métricas en los datos de test: matriz de confusión, sensibilidad, especificidad y error total.
        """
        pred = self.predecir(self.X_test[:, 1:], umbral)  # quitar columna del intercepto
        cm = confusion_matrix(self.y_test, pred)

        TN, FP, FN, TP = cm.ravel()
        sensibilidad = TP / (TP + FN)
        especificidad = TN / (TN + FP)
        error_total = (FP + FN) / len(self.y_test)

        print("Matriz de confusión:")
        print(cm)
        print(f"Sensibilidad (TPR): {sensibilidad:.4f}")
        print(f"Especificidad (TNR): {especificidad:.4f}")
        print(f"Error total: {error_total:.4f}")

    def curva_roc(self):
        """Grafica la curva ROC y calcula el área bajo la curva (AUC)."""
        probas = self.modelo.predict(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, probas)
        auc = roc_auc_score(self.y_test, probas)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR (1 - especificidad)")
        plt.ylabel("TPR (sensibilidad)")
        plt.title("Curva ROC")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Evaluación según AUC
        print(f"Área bajo la curva (AUC): {auc:.4f}")
        if auc >= 0.9:
            interpretacion = "Excelente"
        elif auc >= 0.8:
            interpretacion = "Muy bueno"
        elif auc >= 0.7:
            interpretacion = "Bueno"
        elif auc >= 0.6:
            interpretacion = "Regular"
        else:
            interpretacion = "Pobre"
        print(f"Evaluación del clasificador: {interpretacion}")