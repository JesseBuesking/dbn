

from dbn.backprop_nn import BackPropNN


class BackPropAutoEncoder(object):
    """
    An Autoencoder built on top of the BackPropNN class.

    http://en.wikipedia.org/wiki/Autoencoder
    """

    def __init__(self, ni, h, lr=0.1, m=0.1):
        self.nn = BackPropNN(ni, ni, h, lr, m)

    def predict(self, values):
        """
        Makes a prediction for some input.

        :param values: some input values
        """
        return self.nn.get_activations(values)

    def test(self, patterns):
        """
        Tests a set of data against the current data, printing some useful
        information and returning some of it.

        :param patterns: a collection of inputs to test the model against
        """

        p = []
        for values, targets in patterns:
            p.append((values, values))

        return self.nn.test(p)

    def train(self, data, iterations=1000):
        """
        Trains the model on the input data.

        :param data: a collection of inputs to train the model on,
         in the form (input values, target values)
        :param iterations: the number of iterations to train for
        """

        d = []
        for values, targets in data:
            d.append((values, values))

        return self.nn.train(d, iterations)
