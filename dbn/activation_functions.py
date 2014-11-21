

import numpy as np


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
# noinspection PyDocstring
def sigmoid(x):
    return np.tanh(x)


# derivative of our sigmoid function, in terms of the output (i.e. y)
# noinspection PyDocstring
def dsigmoid(y):
    return 1.0 - y**2


# our logistic function
# noinspection PyDocstring
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# derivative of our logistic function, in terms of the output (i.e. y)
# noinspection PyDocstring
def dlogistic(y):
    return logistic(y) * (1 - logistic(y))


func = sigmoid
dfunc = dsigmoid
# func = logistic
# dfunc = dlogistic
