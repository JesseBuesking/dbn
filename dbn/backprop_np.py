"""
Back-Propagation Neural Networks

Written in Python.  See http://www.python.org/
Placed in the public domain.
Neil Schemenauer <nas@arctrix.com>
"""


import numpy as np
import time
import activation_functions as af


np.random.seed(0)


# noinspection PyDocstring
class NN(object):

    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        # +1 for bias node
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = np.ones(self.ni)
        self.ah = np.ones(self.nh)
        self.ao = np.ones(self.no)

        # create weights

        # uncomment to match the regular backprop.py algorithm
        # self.wi = makeMatrix(self.ni, self.nh)
        # self.wo = makeMatrix(self.nh, self.no)
        # for i in range(self.ni):
        #     for j in range(self.nh):
        #         self.wi[i][j] = rand(-0.2, 0.2)
        # for j in range(self.nh):
        #     for k in range(self.no):
        #         self.wo[j][k] = rand(-2.0, 2.0)
        # self.wi = np.array(self.wi)
        # self.wo = np.array(self.wo)
        self.wi = 0.1 * np.random.randn(self.ni, self.nh)
        self.wo = 0.1 * np.random.randn(self.nh, self.no)

        # last change in weights for momentum
        self.ci = np.zeros(shape=(self.ni, self.nh))
        self.co = np.zeros(shape=(self.nh, self.no))

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        s = np.dot(self.ai, self.wi)
        self.ah = af.func(s)

        # output activations
        s = np.dot(self.ah, self.wo)
        self.ao = af.func(s)

        return self.ao

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        targets = np.array(targets)

        # calculate error terms for output
        error = targets - self.ao
        output_deltas = af.dfunc(self.ao) * error

        # calculate error terms for hidden
        error = np.dot(output_deltas, self.wo.T)
        hidden_deltas = af.dfunc(self.ah) * error

        # update output weights

        # uncomment for SPEEDUP
        change = np.outer(output_deltas, self.ah)
        # change = output_deltas.ravel()[:, np.newaxis] * \
        #     self.ah.ravel()[np.newaxis, :]

        change = change.T
        self.wo = self.wo + np.multiply(N, change) + np.multiply(M, self.co)
        self.co = change

        # update input weights

        # uncomment for SPEEDUP
        change = np.outer(hidden_deltas, self.ai)
        # change = hidden_deltas.ravel()[:, np.newaxis] * \
        #     self.ai.ravel()[np.newaxis, :]

        change = change.T
        self.wi = self.wi + np.multiply(N, change) + np.multiply(M, self.ci)
        self.ci = change

        # calculate error
        # noinspection PyUnresolvedReferences
        error = np.multiply(0.5, ((targets-self.ao)**2).sum())
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

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
                error = (error + self.backPropagate(targets, N, M))
            if i % print_iter == 0:
                print('error %-.8f' % error)


# noinspection PyDocstring
def demo():
    # Teach network XOR function
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    start = time.clock()
    n.train(pat, N=0.1, iterations=10000)
    print('elapsed: {}'.format(time.clock() - start))
    # test it
    n.test(pat)


if __name__ == '__main__':
    demo()
