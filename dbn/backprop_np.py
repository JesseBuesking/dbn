"""
Back-Propagation Neural Networks

Written in Python.  See http://www.python.org/
Placed in the public domain.
Neil Schemenauer <nas@arctrix.com>
"""
import random

import numpy as np
import time
from sklearn import preprocessing
import activation_functions as af
from dbn import dataset


np.random.seed(0)


# noinspection PyDocstring
class NN(object):

    def __init__(self, ni, no, h):
        # number of input, hidden, and output nodes
        # +1 for bias node
        self.ni = ni + 1
        self.no = no

        self.h = h
        self.nhl = len(h)

        # activations for nodes
        self.a = []
        # for hidden layer(s)
        for i in range(self.nhl):
            self.a.append(np.ones(self.h[i]))
        # for output layer
        self.a.append(np.ones(self.no))

        # create weights
        self.w = []
        # for hidden layer(s)
        for i in range(self.nhl):
            if i == 0:
                self.w.append(0.1 * np.random.randn(self.ni, self.h[i]))
            else:
                self.w.append(0.1 * np.random.randn(self.h[i-1], self.h[i]))
        # for output layer
        self.w.append(0.1 * np.random.randn(self.h[self.nhl-1], self.no))

        # last change in weights for momentum
        self.c = []
        # for hidden layer(s)
        for i in range(self.nhl):
            if i == 0:
                self.c.append(np.zeros(shape=(self.ni, self.h[i])))
            else:
                self.c.append(np.zeros(shape=(self.h[i-1], self.h[i])))
        # for output layer
        self.c.append(np.zeros(shape=(self.h[self.nhl-1], self.no)))

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        ai = np.array(inputs)
        ai = np.insert(ai, ai.shape[0], 1)

        # hidden + output activations
        s = np.dot(ai, self.w[0])
        self.a[0] = af.func(s)
        for i in range(self.nhl):
            s = np.dot(self.a[i], self.w[i+1])
            self.a[i + 1] = af.func(s)

        return self.a[self.nhl]

    def backPropagate(self, inputs, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # input activations
        ai = np.array(inputs)
        ai = np.insert(ai, ai.shape[0], 1)

        deltas = []
        # calculate error terms for output
        error = targets - self.a[self.nhl]
        d = af.dfunc(self.a[self.nhl]) * error
        deltas.append(d)

        # calculate error terms for hidden
        for i in range(self.nhl, 0, -1):
            error = np.dot(d, self.w[i].T)
            d = af.dfunc(self.a[i-1]) * error
            deltas.append(d)

        # update hidden weights
        for i in range(self.nhl, 0, -1):
            change = np.outer(deltas.pop(0), self.a[i - 1])
            change = change.T
            self.w[i] = \
                self.w[i] + \
                np.multiply(N, change) + \
                np.multiply(M, self.c[i])
            self.c[i] = change

        # update input weights
        change = np.outer(deltas.pop(0), ai)
        change = change.T
        self.w[0] = \
            self.w[0] + \
            np.multiply(N, change) + \
            np.multiply(M, self.c[0])
        self.c[0] = change

        # calculate error
        # noinspection PyUnresolvedReferences
        error = np.multiply(0.5, ((targets-self.a[self.nhl])**2).sum())
        return error

    def test(self, patterns):
        error = 0.0
        bin_correct = 0
        for p in patterns:
            u = self.update(p[0])
            u_bin = []
            for i in u:
                if i < 0.5:
                    u_bin.append(0)
                else:
                    u_bin.append(1)
            # noinspection PyUnresolvedReferences
            error += ((np.array(p[1])-np.array(u))**2).sum()
            # noinspection PyTypeChecker
            z = u_bin == p[1]
            if isinstance(z, bool):
                bin_correct += u_bin == p[1]
            else:
                # noinspection PyTypeChecker
                bin_correct += all(u_bin == p[1])
            print(p[0], '-> {} was {}'.format(p[1], u))

        print('')
        print('correct: {}/{}'.format(bin_correct, len(patterns)))
        print('overall error: {}'.format(error))

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        print_iter = iterations / 10

        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = (error + self.backPropagate(inputs, targets, N, M))
            if i % print_iter == 0:
                print('error %-.8f' % error)


# noinspection PyDocstring
def run_iris():
    pat = []
    iris = dataset.iris()
    binzer = preprocessing.LabelBinarizer().fit(iris.target)
    scaled = preprocessing.MinMaxScaler().fit(iris.data)
    for i in range(iris.data.shape[0]):
        pat.append([
            scaled.transform(iris.data[i]),
            binzer.transform([iris.target[i]])[0]]
        )
    random.shuffle(pat)
    l = int((len(pat)*.8))
    train, test = pat[:l], pat[l:]

    n = NN(4, binzer.classes_.shape[0], [3, 3])

    print('amt: {}'.format(iris.data.shape[0]))

    # create a network with two input, two hidden, and one output nodes
    # n = NN(2, 1, [2, 2])
    # train it with some patterns
    start = time.clock()
    n.train(train, N=0.001, iterations=1000)
    print('elapsed: {}'.format(time.clock() - start))
    # test it
    n.test(test)


# noinspection PyDocstring
def run_xor():
    # Teach network XOR function
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]
    train = pat
    test = pat

    n = NN(2, 1, [2])

    # create a network with two input, two hidden, and one output nodes
    # n = NN(2, 1, [2, 2])
    # train it with some patterns
    start = time.clock()
    n.train(train, N=0.001, iterations=1000)
    print('elapsed: {}'.format(time.clock() - start))
    # test it
    n.test(test)


# noinspection PyDocstring
def demo():
    run_iris()
    # run_xor()


if __name__ == '__main__':
    demo()
