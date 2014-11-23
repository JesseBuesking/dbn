"""
Back-Propagation Neural Networks

Written in Python.  See http://www.python.org/
Placed in the public domain.
Neil Schemenauer <nas@arctrix.com>
numpy + cleanup: Jesse Buesking
"""


import numpy as np
import activation_functions as af


# noinspection PyDocstring
class BackPropNN(object):

    def __init__(self, ni, no, h, lr=0.1, m=0.1):
        # number of input nodes
        # +1 for bias node
        self.ni = ni + 1
        # learning rate
        self.lr = lr
        # momentum
        self.m = m
        # number of output nodes
        self.no = no

        # hidden layer node counts
        self.h = h
        # total number of hidden layers
        self.nhl = len(self.h)

        # ----------------------------------------
        # create weights
        # ----------------------------------------

        self.w = []
        self.w.append(0.1 * np.random.randn(self.ni, self.h[0]))
        # for hidden layer(s)
        for i in range(1, self.nhl):
            self.w.append(0.1 * np.random.randn(self.h[i-1], self.h[i]))
        # for output layer
        self.w.append(0.1 * np.random.randn(self.h[self.nhl-1], self.no))

        # ----------------------------------------
        # change in weights for momentum
        # ----------------------------------------

        self.c = []
        self.c.append(np.zeros(shape=(self.ni, self.h[0])))
        # for hidden layer(s)
        for i in range(1, self.nhl):
            self.c.append(np.zeros(shape=(self.h[i-1], self.h[i])))
        # for output layer
        self.c.append(np.zeros(shape=(self.h[self.nhl-1], self.no)))

    def get_activations(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # ----------------------------------------
        # for input layer
        # ----------------------------------------

        ai = np.array(inputs)
        ai = np.insert(ai, 0, 1)

        # ----------------------------------------
        # for hidden layer(s)
        # ----------------------------------------

        a = []
        for i in range(self.nhl):
            a.append(np.ones(self.h[i]))

        # for output layer
        a.append(np.ones(self.no))

        # ----------------------------------------
        # calculate activations
        # ----------------------------------------

        s = np.dot(ai, self.w[0])
        a[0] = af.func(s)
        for i in range(self.nhl):
            s = np.dot(a[i], self.w[i+1])
            a[i + 1] = af.func(s)

        return a

    def back_propagate(self, inputs, targets, a):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # ----------------------------------------
        # input activations
        # ----------------------------------------

        ai = np.array(inputs)
        ai = np.insert(ai, 0, 1)

        # ----------------------------------------
        # calculate errors at output layer
        # ----------------------------------------

        deltas = []
        error = targets - a[self.nhl]
        d = af.dfunc(a[self.nhl]) * error
        deltas.append(d)

        # ----------------------------------------
        # calculate errors at hidden layer(s)
        # ----------------------------------------

        for i in range(self.nhl, 0, -1):
            error = np.dot(d, self.w[i].T)
            d = af.dfunc(a[i-1]) * error
            deltas.append(d)

        # ----------------------------------------
        # update weights for hidden layer(s)
        # ----------------------------------------

        for i in range(self.nhl, 0, -1):
            change = np.outer(deltas.pop(0), a[i - 1])
            change = change.T
            self.w[i] = \
                self.w[i] + \
                np.multiply(self.lr, change) + \
                np.multiply(self.m, self.c[i])
            self.c[i] = change

        # ----------------------------------------
        # update weights for the input layer
        # ----------------------------------------

        change = np.outer(deltas.pop(0), ai)
        change = change.T
        self.w[0] = \
            self.w[0] + \
            np.multiply(self.lr, change) + \
            np.multiply(self.m, self.c[0])
        self.c[0] = change

        # ----------------------------------------
        # calculate current error
        # ----------------------------------------

        # sum of squares error
        # noinspection PyUnresolvedReferences
        sse = ((targets-a[self.nhl])**2).sum()

        return np.multiply(0.5, sse).sum()

    def predict(self, values):
        a = self.get_activations(values)

        # get activations for outputs
        a = a[self.nhl]

        return a

    def test(self, patterns):
        error = 0.0
        bin_correct = 0

        precision = np.get_printoptions()['precision']
        np.set_printoptions(precision=7, suppress=True)

        print('')
        for values, targets in patterns:
            a = self.predict(values)

            # sum of squares error
            # noinspection PyUnresolvedReferences
            sse = ((np.array(targets)-a)**2).sum()
            error += sse

            # noinspection PyTypeChecker
            u_bin = a >= .5
            bin_targets = np.array(targets) >= .5
            bin_correct += np.all(u_bin == bin_targets)
            print('{} -> {} = {}'.format(values, targets, a))

        print('')
        print('correct: {}/{}'.format(bin_correct, len(patterns)))
        print('overall error: {}'.format(error))

        np.set_printoptions(precision=precision, suppress=False)
        return bin_correct, len(patterns), error

    def train(self, data, iterations=1000):

        status_iter = iterations / 10

        for i in range(iterations):
            error = 0.0
            for values, targets in data:
                a = self.get_activations(values)
                error += self.back_propagate(values, targets, a)

            if i % status_iter == 0:
                print('{:05d}: error {:.08f}'.format(i, error))
