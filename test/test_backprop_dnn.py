

import numpy as np
import unittest
import time
from sklearn import preprocessing
from dbn import dataset
from dbn.backprop_dnn import BackPropDNN
from dbn.backprop_nn import BackPropNN


# noinspection PyDocstring
from dbn.rbm import RBM


class BackPropDNNTests(unittest.TestCase):

    def test_iris(self):
        np.random.seed(0)

        pat = []
        iris = dataset.iris()
        binzer = preprocessing.LabelBinarizer().fit(iris.target)
        scaled = preprocessing.MinMaxScaler().fit(iris.data)
        for i in range(iris.data.shape[0]):
            pat.append([
                scaled.transform(iris.data[i]),
                binzer.transform([iris.target[i]])[0]]
            )

        l = int((len(pat)*.8))
        train, test = pat[:l], pat[l:]
        unlabeled = []
        for values, _ in pat:
            unlabeled.append(values)

        n = BackPropDNN(
            [
                RBM(4, 3, iterations=1000, lr=0.01, m=0.01),
                RBM(3, 3, iterations=100, lr=0.01, m=0.01),
            ],
            BackPropNN(
                4, binzer.classes_.shape[0], [3, 3],
                iterations=100, lr=0, m=0.1
            )
        )

        print('input rows: {}'.format(iris.data.shape[0]))
        print('')

        start = time.clock()
        n.train(unlabeled, train)

        print('')
        print('elapsed: {}'.format(time.clock() - start))

        # test it
        correct, total, error = n.test(test)
        percent_correct = correct / float(total)
        self.assertTrue(
            correct / float(total) >= .9,
            'expected min accuracy of .9, got {:.2f}%'.format(percent_correct)
        )
        self.assertEqual(0.76574192351657244, error)
        self.assertEqual(30, correct)
