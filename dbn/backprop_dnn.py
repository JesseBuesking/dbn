""" Deep Back-Propagation Neural Network """


import numpy as np


# noinspection PyDocstring
class BackPropDNN(object):

    def __init__(self, rbms, nn):
        self.rbms = rbms
        self.nn = nn

    def predict(self, values):
        return self.nn.predict(values)

    def test(self, patterns):
        return self.nn.test(patterns)

    def train(self, unlabeled, labeled):
        d = unlabeled

        for rbm in self.rbms:
            rbm.train(d)
            new_d = []
            for row in d:
                row = np.array([row])
                new_d.append(rbm.hidden_probs(row)[0])
            d = new_d

        for i in range(len(self.rbms)):
            rbm = self.rbms[i]
            self.nn.w[i] = rbm.w[0]

        self.nn.train(labeled)
