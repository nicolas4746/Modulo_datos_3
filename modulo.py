import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import pearsonr
import pandas as pd
from scipy.stats import norm, uniform
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from scipy.stats import shapiro
import random
from sklearn.metrics import auc

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
    def miqqplot(self):
        x_ord = np.sort(self.datos)
        n = len(self.datos)
        media = np.mean(x_ord)
        desvio = np.std(x_ord, ddof=1)
        cuantiles_muestrales = (x_ord - media) / desvio

        probabilidades = np.arange(1, n+1) / (n+1)
        cuantiles_teoricos = norm.ppf(probabilidades)

        sm.qqplot(self.datos, line='45')

        plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
        plt.xlabel('Cuantiles teóricos')
        plt.ylabel('Cuantiles muestrales')
        plt.plot(cuantiles_teoricos,cuantiles_teoricos , linestyle='-', color='red')
        plt.show()


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

    def teorica_unif(self,a, b, nume_puntos = 1000):
        x = np.linspace(a-1 , b+1 , nume_puntos)
        y = uniform.pdf(x, loc= a, scale= b-a)
        return x,y

    def teorica_BS(self, x):
        term1 = (1 / 2) * norm.pdf(x, loc=0, scale=1)
        term2 = (1 / 10) * sum(norm.pdf(x, loc=j / 2 - 1, scale=1 / 10) for j in range(5))
        p = term1 + term2
        return p


class RegresionLineal:
    """Clase que permite el ajuste de un modelo de Regresion Lineal, que puede
        ser Regresion Lineal Simple y Regresion Lineal Multiple.
        Ambas clases depende de esta clase general.
    """
    def __init__(self, x, y):
        # x = variables predictora/s
        # y = variable respuesta
        self.x = x
        self.y = y
        self.resultado = None

    def ajustar_modelo(self):
        """Se ajuta el modelo de Regresión.
        """
        # se arma la matriz de diseño agregando la columna de unos
        X = sm.add_constant(self.x)
        # se estima el modelo de regresión lineal
        modelo = sm.OLS(self.y, X)
        self.resultado = modelo.fit()
        return self.resultado

    def parametros_modelo(self):
        """ Retorna las estimaciones de los betas
            del modelo.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          parametros = self.resultado.params
          return parametros

    def ajustado_y(self):
        """Calula el valor predicho a partir del modelo ajustado
          de regresion lineal.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          y_pred = self.resultado.fittedvalues
          return y_pred

    def residuos(self):
        """ Calcula los residuos de entre los valores reales (y)
            y los valores dados por la recta d eminimos cuadrados (y_sombrero)
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          residuos = self.resultado.resid
          return residuos

    def estim_varianza_del_error(self):
        """Calcula la estimacion de la varianza del error
        """
        n = len(self.x)
        resid = self.residuos()
        var = np.sum( resid**2 ) / (n-2)
        return var

    def r_cuadrado(self):
        """ Calcula (R^2) el coeficiente de determinación y, es una medida de la
            proporción de la variabilidad que explica el modelo ajustado.
            valores de  R^2  cercanos a 1 son valores deseables para una buena
            calidad del ajuste.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          r_squared = self.resultado.rsquared
          return r_squared

    def r_ajustado(self):
        """Calcula el R² ajustado, es una corrección de  R^2  para permitir
          la comparación de modelos con distinta cantidad de regresoras.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          adjusted_r_squared = self.resultado.rsquared_adj
          return adjusted_r_squared

    def supuesto_normalidad(self):
      """Se verifica el supuesto de normalidad de los residuos, de manera
         grafica usando qqplot y de manera analítica usando shapiro test, usando
         el p-valor.
      """
      residuo = self.residuos()
      # grafica:
      rg = AnalisisDescriptivo(residuo)
      #miqqplot(residuo)
      rg.miqqplot()

      # test de normalidad:
      stat, p_valor1 = shapiro(residuo)
      print("\nValor p normalidad:", p_valor1)

    def supuestos_homocedasticidad(self):
      """Se verifica el supuesto de homocedasticidad de los residuos, de
          manera grafica y analítica por medio del p-valor.
      """
      # Homocedasticidad grafico
      predichos = self.ajustado_y()
      residuo = self.residuos()

      plt.scatter(predichos, residuo, marker="o", c='blue', s=30)
      plt.axhline(y=0, color='r', linestyle='--')  # Línea horizontal en y=0 para facilitar la visualización de los residuos
      plt.xlabel('Valores predichos')
      plt.ylabel('Residuos')
      plt.title('Gráfico de Residuos vs. Valores Predichos')
      plt.show()
      # Homocedasticidad test:
      X = sm.add_constant(self.x) # matriz de diseño
      bp_test = het_breuschpagan(residuo, X)# X es la matriz de diseño
      bp_value = bp_test[1]
      print("\nValor p Homocedasticidad:", bp_value)

    def int_confianza_betas(self, alfa):
        """funcion inicial: int_confianza_beta1(alfa, beta_1, var_estimada, t_crit)
        Calcula el intervalor de confianza para beta_1, a partir de un alfa (nivel
        de significacion) dado.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          IC = self.resultado.conf_int(alpha = alfa)
          print(f"Los Intervalos de confianza para los estimadores de beta son: {IC}")

    def p_valor_betas(self, b_i=0, i=1):
        """Es una funcion que retorna el p-valor de un test de hipotesis:
                          H_0: beta_i = k vs H_1 beta_i != k
            b_i: es el numero k sobre el cual se quiere hacer el test. Por
            default es 0.
            i: es el indice del beta que se quiere testear, es un natural i
            (i = 0, ..., n). Por default es 1.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          res = self.resultado
          SE_est = res.bse
          coef_xi = res.params[i]
          # valor de t observado:
          t_obs = (coef_xi - b_i)/SE_est[i]

          # el pvalor:
          X = res.model.exog # para recuperar la matriz de diseño del modelo
          grados_libertad = len(X[:, i]) - 2
          p_valor = 2 * stats.t.sf(abs(t_obs), df = grados_libertad)

          return p_valor

    def resumen_grafico(self, z):
        """ Grafico de dispersion de una variable cuantitativa predictora vs
            respuesta.
            z es la variable cuantitativa predictora que se quiere graficar.
        """
        plt.scatter(z, self.y, marker="o", c='blue', s=30)
        plt.xlabel('Variable Predictora')
        plt.ylabel('Variable Respuesta')
        plt.title('Gráfico de Dispersion: Var.Predict. vs. Var. Respuesta')
        plt.show()


class RegresionLinealSimple(RegresionLineal):
    """Clase regresion Lineal Simple, hereda funciones de Regresion Lineal.
    """
    def __init__(self, x, y):
        super().__init__(x, y)

    def estimacion_betas(self):
        """Retorna una tupla (b_0, b_1) de los estimadores de beta_0 y beta_1,
            usando minimos cuadrados.
        """
        x_media = np.mean(self.x)
        y_media = np.mean(self.y)
        numerador = np.sum((self.x - x_media) * (self.y - y_media))
        denominador = np.sum((self.x - x_media)**2)
        b_1 = numerador / denominador
        b_0 = y_media - b_1 * x_media
        return (b_0, b_1)

    def graf_scatter_recta(self):
        """Grafica los puntos de la variable predictora vs vaiable de respuesta.
          Ademas grafica la recta de minimos cuadrados.
        """
        # el regresor lineal:
        b_0, b_1 = self.estimacion_betas()
        y_pred = b_0 + b_1 * self.x
        # el grafico:
        plt.scatter(self.x, self.y, marker="o", c='blue', label='Datos', s=30)
        plt.plot(self.x, y_pred, linestyle='--', color='red', label='Recta estimada')
        plt.legend()
        plt.xlabel("variable predictora")
        plt.ylabel("variable respuesta")
        plt.title('')
        plt.show()

    def y_predict_x_new(self, x_new):
        """Retorna el valor de un y_predicho del modelo de regresion
          a partir de un nuevo valor de x: x_new.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")

        else:
          res = self.resultado
          # Crear la matriz de diseño con el nuevo punto de predicción
          X_new = sm.add_constant(np.array([[ x_new]]))
          prediccion = res.predict(X_new)
          return prediccion

    def t_obs_b1(self, b1=0):
        """Funcion que calcula del t observado para determinar el sgte. test
                      H_0: beta_1 = 0 vs H_1: beta_1 != 0
          Donde b es el valor que toma beta_1, por defecto es 0, porque se evalua
          el test de arriba.
          Pero b=1 si se evaluara el test: H_0: beta_1 = 1 vs H_1: beta_1 != 1.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          res = self.resultado
          coef_x = res.params[1]
          # error estándar estimado para el estimador de los betas
          SE_est = res.bse

          # valor de t observado:
          t_obs = (coef_x - b1) / SE_est[1] # SE_est[1] es el error estandar
                                          # estimado para el estimador beta_1
          return t_obs

    def reg_rechazo_b1(self, alfa):
        """Funcion que muestra la region de rechazo para la hipotesis nula H_0
          a favor  de aceptar la hipotesis alternativa H_1.
        """
        ## Completar
        #alfa = 0.05
        grados_libertad = len(self.x) - 2
        t_crit = stats.t.ppf(  1 - (alfa/2), df = grados_libertad )
        print(f"(-inf, {-t_crit}) U ({t_crit}, inf)")

    def p_valor_beta(self, b1=0):
        """calcula el p-valor para evaluar el test de hipotesis:
                      H_0: beta_1 = 0 vs H_1: beta_1 != 0
        """
        t_observado = self.t_obs_b1(b1)
        grados_libertad = len(self.x)  - 2
        p_valor = 2 * stats.t.sf(abs(t_observado), df = grados_libertad)
        return p_valor

    def int_confianza_betas(self, alfa):
        """funcion inicial: int_confianza_beta1(alfa, beta_1, var_estimada, t_crit)
        Calcula el intervalor de confianza para beta_1, a partir de un alfa (nivel
        de significacion) dado.
        """

        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          IC = self.resultado.conf_int()
          print(f"Intervalo de confianza para beta_1 es: {IC[1]}")

    def int_prediccion_y(self, metodo, x_new, alfa):
        """Calcula el intervalo de predccion de una Y, a partir de una x_new,
          usando los metodos:
          Metodo 1:  Construir un intervalo de confianza para el valor esperado de
                    Y para un valor particular de  X , por ejemplo  x0 :  E(Y|X=x0)

          Metodo 2: Construir un intervalo de predicción de  Y  para un valor
                    particular de  X , por ejemplo  x0 :  Y0 .
          se obtiene un intervalo de confianza/prediccion de nivel (1-alfa)
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          res = self.resultado
          # Crear la matriz de diseño con el nuevo punto de predicción:
          X_new = sm.add_constant(np.array([[1, x_new]]))
          prediccion = res.get_prediction(X_new)

          if metodo == 1:
            return prediccion.conf_int(alpha= alfa, obs = False)

          elif  metodo == 2:
            return prediccion.conf_int(obs=True , alpha = alfa)

class RegresionLinealMultiple(RegresionLineal):
    """Clase quepermite ajustar, predcir un modelo de Regresion Lineal
        Multiple.
    """
    def __init__(self, x, y):
          super().__init__(x, y)

    def y_predict_x_new(self, x_new):
        """Retorna el valor de un y_predicho del modelo de regresion
          a partir de un nuevo valor de x.
          x_new debe ser una LISTA con los valores qeu toma la variable
          explicativa para predecir.
        """
        if self.resultado is None:
            print("Falta ajustar el modelo, usar ajustar_modelo()")
            return None

        else:
            res = self.resultado
            # Se usa sm.add_constant para manejar el intercepto automáticamente
            X_new_const = sm.add_constant(np.array(x_new).reshape(1, -1)) # Reshape para asegurar que sea 2D
            prediccion = res.predict(X_new_const)[0]
            return prediccion
        
    def resumen_modelo(self):
        """Imprime el summary() del modelo ajustado.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")

        else:
          res = self.resultado
          print(res.summary())


class RegresionLogistica:
    """Clase que ajusta un modelo de Regresion Logistica.
       Requerimiento: Las variables categoricas sean codificadas antes de ajustar
       el modelo.
       Data es una base de datos con variables que sean cuantitativas.
    """

    def __init__(self, data):
        self.data = data
        self.data_train = None
        self.data_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.resultado = None

    def separar_data_train_test(self, seed = 10, ptje_test = 0.20):
        """Funcion que separa data, de manera aleatoria, en set de train y test.
           seed: es la semilla, por default es 10.
           ptje_test: Default= 0.20. Es el valor entre 0 y 1, es la proporcion
           que se quiere dejar para el conjunto test, tomado de self.data.
        """
        random.seed(seed)
        cant_filas_extraer = int(self.data.shape[0] * ptje_test)
        # Crear un vector de números aleatorios entre 0 y len(data)
        cuales = random.sample(range( int(self.data.shape[0]) ), cant_filas_extraer)
        # datos train:
        self.data_train = self.data.drop(cuales)
        # datos test:
        self.data_test = self.data.iloc[cuales]

        return self.data_test, self.data_train

    def ajustar_modelo(self, x_train, y_train, x_test, y_test):
        """Se ajusta el modelo de Regresión Logistica.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # se arma la matriz de diseño agregando la columna de unos
        X = sm.add_constant(self.x_train)
        # se estima el modelo de Regresión Logistica
        modelo = sm.Logit(self.y_train, X)  # esto es lo nuevo
        self.resultado = modelo.fit()
        return self.resultado

    def parametros_modelo(self):
        """ Retorna las estimaciones de los betas
            del modelo.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          parametros = self.resultado.params
          return parametros

    def ajustados_y(self, prob=0.5):
        """Calula el valor predicho a partir del modelo ajustado
          de regresion lineal.
          prob: es el umbral de probabilidad sobre el cual se considera
          para formar el y_ajustado.
          Retorna una tupla: y_ajustado_binary, ajust_y_prob
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          X_test = sm.add_constant(self.x_test)
          ajust_y_prob = self.resultado.predict(X_test) #probabilidad del "y" ajustado
          # lo que sigue es el "y" ajustado como binario, de acuerdo al "prob" usado:
          y_ajustado_binary = [1 if k >= prob else 0 for k in ajust_y_prob]

          return y_ajustado_binary, ajust_y_prob

    def matriz_confusion(self, prob=0.5):
        """Retorna la Matriz de Confusión:
                      |tp     fp|
                      |fn     tn|
          por medio de una lista de la forma [tp, fp, fn, tn]
          prob es la probabilidad.
        """
        if self.resultado is not None:
          y_ajustado = self.ajustados_y(prob)[0]
          n = len(self.y_test)
          tp = 0 # true positive
          tn = 0 # true negative
          fp = 0 # false positive
          fn = 0 # falso negativo
          for i in range(n):
            if y_ajustado[i] == self.y_test.iloc[i]:
              if y_ajustado[i] == 1:
                tp = tp + 1
              else:
                tn = tn + 1
            else:
              if y_ajustado[i] == 1 and self.y_test.iloc[i] == 0:
                fp = fp + 1
              else:
                fn = fn + 1
          #print(f" {tp}   {fp} \n {fn}   {tn}")
          return [tp, fp, fn, tn]
        else:
          print("Correr primero ajustados_y()")

    def especif_sensib(self, prob=0.5):
        """Retorna una lista con la sensibilidad y especifisidad del modelo
           Regresión Logística, como sigue: [sensibilidad, especificidad].
           prob: es el umbral de probabilidad sobre el cual se determina si una
           respuesta es 0 o 1. Default = 0.5
        """
        matrix_conf = self.matriz_confusion(prob)
        tp = matrix_conf[0]
        fp = matrix_conf[1]
        fn = matrix_conf[2]
        tn = matrix_conf[3]

        sensibilidad = tp / (tp + fn)
        especificidad = tn / (fp + tn)

        #print(f"La sensibilidad del modelo es: {sensibilidad}")
        #print(f"La especificidad del modelo es: {especificidad}")
        return [sensibilidad, especificidad]

    def curva_ROC(self, prob=0.5):
        """prob: es el umbral de probabilidad sobre el cual se determina si una
           respuesta es 0 o 1. Default = 0.5
        """
        if self.resultado is not None:
          grid = np.linspace(0, 1, 100)
          l1 = [] # 1-especificidad
          l2 = [] # sensibilidad
          prediccion = self.ajustados_y(prob)[1]
          for j in grid:
            y_pred_binary = [1 if k >= j else 0 for k in prediccion]
            metrica = self.especif_sensib(j)
            especificidad = metrica[1]
            sensibilidad = metrica[0]
            l1.append(1-especificidad)
            l2.append(sensibilidad)

          plt.plot()
          plt.plot(l1,l2, linestyle='-', color='red', label='Curva ROC')
          plt.legend()
          plt.xlabel("1-especificidad")
          plt.ylabel("sensibilidad")
          plt.title('Curva ROC')
          plt.show()

        else:
          print("Falta ajustar el modelo, usar ajustar_modelo()")

    def predict_y(self, x_new, prob=0.5):
        """x_new es una lista, con el/los valor/es que se quiere predecir.
          La funcion retorna el valor de predicciòn para un x_new.
          prob: es el umbral de probabilidad sobre el cual se determina si una
           respuesta es 0 o 1. Default = 0.5
        """
        X_new = x_new.copy()
        X_new.insert(0, 1)
        aux = np.dot(self.resultado.params, X_new)
        pred = np.exp(aux)/(1 + np.exp(aux))
        if pred >= prob:
          y_pred_bin = 1
        else:
          y_pred_bin = 0

        return y_pred_bin

    def auc(self,prob=0.5):
        """Imprime la evaluacion del clasificador, teniendo en cuenta la tabla
            dada en teoría.
        """
        if self.resultado is not None:
          grid = np.linspace(0, 1, 100)
          especificidad_list = []
          sensibilidad_list = []
          for k in grid:
            metrica = self.especif_sensib(k)
            especificidad = metrica[1]
            sensibilidad = metrica[0]
            especificidad_list.append(1-especificidad)
            sensibilidad_list.append(sensibilidad)

          roc_auc = auc(1-np.array(especificidad_list), sensibilidad_list)
          #print("AUC:", roc_auc)
          if 0.90 < roc_auc <= 1:
            print(f"El clasificador es Excelente, {roc_auc}")
          elif 0.80 < roc_auc <= 0.90:
            print(f"El clasificador es Bueno, {roc_auc}")
          elif 0.70 < roc_auc <= 0.80:
            print(f"El clasificador es Regular, {roc_auc}")
          elif 0.60 < roc_auc <= 0.70:
            print(f"El clasificador es Pobre, {roc_auc}")
          elif 0.50 < roc_auc <= 0.60:
            print(f"El clasificador es Fallido, {roc_auc}")
          else:
            print(f"El clasificador Muy Malo, {roc_auc}")

        else:
          print("Falta ajustar el modelo, usar ajustar_modelo()")

    def modelo_resumen(self):
        """Imprime el summary del modelo ajustado
        """
        if self.resultado is not None:
          print(self.resultado.summary())
        else:
          print("Falta ajustar el modelo, usar ajustar_modelo()")

    def p_valor_betas(self, b_i=0, i=1):
        """Es una funcion que retorna el p-valor de un test de hipotesis:
                          H_0: beta_i = k vs H_1 beta_i != k
            b_i: es el numero k sobre el cual se quiere hacer el test. Por
            default es 0.
            i: es el indice del beta que se quiere testear, es un natural i
            (i = 0, ..., n). Por default es 1.
        """
        if self.resultado is None:
          print("Falta ajustar el modelo, usar ajustar_modelo()")
        else:
          res = self.resultado
          SE_est = res.bse
          coef_xi = res.params[i]
          # valor de t observado:
          t_obs = (coef_xi - b_i)/SE_est[i]

          # el pvalor:
          X = res.model.exog # para recuperar la matriz de diseño del modelo
          grados_libertad = len(X[:, i]) - 2
          p_valor = 2 * stats.t.sf(abs(t_obs), df = grados_libertad)

          print(f"p-valor: {p_valor}")
          print(f"t_observado: {t_obs}")

    