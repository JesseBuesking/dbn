

import numpy as np
import random
import unittest
import time
from sklearn import preprocessing
from dbn import dataset
from dbn.backprop_nn import BackPropNN


# noinspection PyDocstring
class BackPropNNTests(unittest.TestCase):

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
        random.seed(0)
        random.shuffle(pat)
        l = int((len(pat)*.8))
        train, test = pat[:l], pat[l:]

        n = BackPropNN(4, binzer.classes_.shape[0], [3, 3], lr=0.001)

        print('input rows: {}'.format(iris.data.shape[0]))
        print('')

        start = time.clock()
        n.train(train, iterations=1000)

        print('')
        print('elapsed: {}'.format(time.clock() - start))

        # test it
        correct, total, error = n.test(test)
        percent_correct = correct / float(total)
        self.assertTrue(
            correct / float(total) >= .9,
            'expected min accuracy of .9, got {:.2f}%'.format(percent_correct)
        )
        self.assertEqual(1.5186660231263243, error)
        self.assertEqual(28, correct)

    def test_xor(self):
        """ Teach network XOR function. """
        np.random.seed(0)

        pat = [
            [[0, 0], [0]],
            [[0, 1], [1]],
            [[1, 0], [1]],
            [[1, 1], [0]]
        ]
        train = pat
        test = pat

        n = BackPropNN(2, 1, [2], lr=0.001)

        print('input rows: {}'.format(len(pat)))
        print('')

        start = time.clock()
        n.train(train, iterations=1000)

        print('')
        print('elapsed: {}'.format(time.clock() - start))

        # test it
        correct, total, error = n.test(test)
        percent_correct = correct / float(total)
        self.assertEqual(
            correct / float(total),
            1.0,
            'expected a perfect score, got {:.2f}%'.format(percent_correct)
        )

        self.assertEqual(0.0099941253524785407, error)
        self.assertEqual(4, correct)
