#
#
# import random
# from sklearn import preprocessing
# from dbn import dataset
#
#
# class DNN(object):
#     pass
#
#
# # noinspection PyDocstring
# def run_iris():
#     pat = []
#     iris = dataset.iris()
#     binzer = preprocessing.LabelBinarizer().fit(iris.target)
#     scaled = preprocessing.MinMaxScaler().fit(iris.data)
#     for i in range(iris.data.shape[0]):
#         pat.append([
#             scaled.transform(iris.data[i]),
#             binzer.transform([iris.target[i]])[0]]
#         )
#     random.shuffle(pat)
#     l = int((len(pat)*.8))
#     train, test = pat[:l], pat[l:]
#
#     n = NN(4, binzer.classes_.shape[0], [3, 3])
#     # n = NN(4, 4, [3])
#
#     print('amt: {}'.format(iris.data.shape[0]))
#
#     # create a network with two input, two hidden, and one output nodes
#     # n = NN(2, 1, [2, 2])
#     # train it with some patterns
#     start = time.clock()
#     n.train(train, N=0.001, iterations=1000)
#     print('elapsed: {}'.format(time.clock() - start))
#     # test it
#     n.test(test)
#
#
# def demo():
#     run_iris()
#
#
# if __name__ == '__main__':
#     demo()
