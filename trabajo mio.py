"""
Optimización usando gradiente descendente - Regresión polinomial
-----------------------------------------------------------------------------------------

En este laboratio se estimarán los parámetros óptimos de un modelo de regresión 
polinomial de grado `n`.

"""


def pregunta_01():
    """
    Complete el código presentado a continuación.
    """
    # Importe pandas
    import pandas as pd

    # Importe PolynomialFeatures
    from sklearn.preprocessing import PolynomialFeatures 

    # Cargue el dataset `data.csv`
    data = pd.read_csv("data.csv")

    # Cree un objeto de tipo `PolynomialFeatures` con grado `2`
    poly = PolynomialFeatures(degree=2)

    # Transforme la columna `x` del dataset `data` usando el objeto `poly`
    x_poly = poly.fit_transform(data[["x"]])

    # Retorne x y y
    return x_poly, data.y


def pregunta_02():

    # Importe numpy
    import numpy as np

    x_poly, y = pregunta_01()

    # Fije la tasa de aprendizaje en 0.0001 y el número de iteraciones en 1000
    learning_rate = 0.0001
    n_iterations = 1000

    # Defina el parámetro inicial `params` como un arreglo de tamaño 3 con ceros
    params = np.zeros(x_poly.shape[1])#(___.shape[1])
    for _ in range(n_iterations):

        # Compute el pronóstico con los parámetros actuales
        y_pred = np.polyval(params, x_poly[:,1])#np.___(___, ___) = array

        # Calcule el error
        error =  y - y_pred  #___ - ___

        # Calcule el gradiente
        w2 = -2*sum(error)
        w1 = -2*np.sum([error*x_poly[:,1]])
        w0 = w2
        gradient = np.array([w2, w1, w0])

        # Actualice los parámetros
        params = params - (learning_rate * gradient)

    return params

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv("data.csv")
plt.plot(data.x,data.y,"ro")
xx = np.linspace(-4,4,100)
plt.plot(xx,np.polyval(pregunta_02(),xx))
print(pregunta_02())