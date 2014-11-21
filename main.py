

import numpy as np
# from dbn.dbn import DBN
from dbn.jrbm import JRBM
# from dbn.rbm import test_rbm


def predict_user(values, verbose=False):
    user = np.array([values])
    print('')
    print('users preferred movies:')
    for i in range(user.shape[1]):
        if user.T[i] == 1:
            print('  {}'.format(titles[i + 1]))
    print('')
    predictions = r.run_visible(user)
    rex = r.run_hidden(predictions)

    if verbose:
        print(rex)

    print('recommendations:')
    for i in range(rex.shape[1]):
        if rex.T[i] == 1:
            print('  {}'.format(titles[i + 1]))

    if verbose:
        print(predictions)

    # drop bias column
    weights = r.weights[:, 1:]
    scores = np.dot(weights, predictions.T)
    # drop bias row
    scores = scores[1:]

    res = []

    if verbose:
        print(scores)

    for i in range(scores.shape[0]):
        res.append((scores[i], titles[i + 1]))

    res.sort(key=lambda x: x[0], reverse=True)
    print('')
    for i in res:
        print('{:<15}: {}'.format(i[1], i[0]))

        # test_rbm()
        # print('building the model')
        # numpy_rng = numpy.random.RandomState(123)
        # dbn = DBN(
        #     numpy_rng=numpy_rng,
        #     n_ins=28*28,
        #     hidden_layer_sizes=[1000,1000,1000],
        #     n_outs=10
        # )


if __name__ == '__main__':
    r = JRBM(
        num_visible=6,
        num_hidden=2,
        learning_rate=0.1
    )
    training_data = [
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0]
    ]

    titles = [
        'Bias Unit',
        'Harry Potter',
        'Avatar',
        'LOTR 3',
        'Gladiator',
        'Titanic',
        'Glitter'
    ]

    r.train(training_data, max_epochs=5000)

    verbose = False
    if verbose:
        print('')
        for idx in range(r.weights.shape[0]):
            print('{:<15}: {}'.format(titles[idx], r.weights[idx]))

    predict_user([0, 0, 0, 1, 1, 0], verbose)

    predict_user([1, 0, 0, 0, 0, 0])
