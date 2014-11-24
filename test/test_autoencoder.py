

import unittest
import numpy as np
import time
from sklearn import preprocessing
from dbn import dataset
from dbn.backprop_autoencoder import BackPropAutoEncoder


class AutoEncoderTests(unittest.TestCase):

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

        it = 1000
        n = BackPropAutoEncoder(4, [3], it, lr=0.001)

        print('input rows: {}'.format(iris.data.shape[0]))
        print('')

        start = time.clock()
        n.train(train)

        print('')
        print('elapsed: {}'.format(time.clock() - start))

        # test it
        _, total, error = n.test(test)
        self.assertEqual(0.42991140461790117, error)

