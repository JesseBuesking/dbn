

import numpy as np
from dbn.rbm import RBM


# noinspection PyDocstring
def predict_user(titles, values, verbose=False):
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
            print('  {}'.format(titles[i+1]))

    if verbose:
        print(predictions)

    # drop bias column
    weights = r.w[0][:, 1:]
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


if __name__ == '__main__':
    r = RBM(
        ni=6,
        nh=2,
        lr=0.1
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
    r = RBM(
        ni=6,
        nh=2,
        lr=0.1
    )

    r.train(training_data, iterations=10000)

    verbose = False
    if verbose:
        print('')
        for idx in range(r.w[0].shape[0]):
            print('{:<15}: {}'.format(titles[idx], r.w[0][idx]))

    predict_user(titles, [0, 0, 0, 1, 1, 0], verbose)
    predict_user(titles, [1, 0, 0, 0, 0, 0], verbose)
