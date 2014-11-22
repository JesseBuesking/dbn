"""
Restricted Boltzmann Machine
"""


from __future__ import print_function
import numpy as np
import activation_functions as af


np.random.seed(1)


# noinspection PyDocstring
class RBM(object):

    def __init__(self, ni, nh, lr=0.1, m=0.1):
        # number of input nodes
        # +1 for bias node
        self.ni = ni + 1
        # learning rate
        self.lr = lr
        # momentum
        self.m = m

        # hidden layer node counts
        self.h = [nh]
        # total number of hidden layers
        self.nhl = len(self.h)

        # initialize a weight matrix, of dimensions (ni x num_hidden), using a
        # Gaussian distribution with mean 0 and standard deviation 0.1

        # ----------------------------------------
        # create weights
        # ----------------------------------------

        self.w = []
        self.w.append(0.1 * np.random.randn(self.ni, self.h[0]))

        # insert weights for the bias units into the first row and first column
        # self.weights = np.insert(self.weights, 0, 0, axis=1)

        # example of what self.weights could be
        # [[ 0.          0.          0.        ]
        #  [ 0.          0.0555121   0.01520456]
        #  [ 0.          0.09536031 -0.11957774]
        #  [ 0.          0.0223247  -0.08246847]
        #  [ 0.          0.0031644   0.24317695]
        #  [ 0.          0.14754101  0.09360807]
        #  [ 0.         -0.05690645  0.04741315]]

    def train(self, data, iterations=1000, batch_size=10):
        """
        Train the machine.

        Parameters
        ----------
        data: a matrix where each row is a training example consisting of the
         states of visible units
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        def forward(d):
            activations = np.dot(d, self.w[0])
            probs = af.func(activations)
            associations = np.dot(d.T, probs)
            return probs, associations

        def back(s):
            activations = np.dot(s, self.w[0].T)
            probs = af.func(activations)
            # Fix the bias unit.
            probs[:, 0] = 1
            associations = np.dot(s.T, probs)
            return probs, associations

        status_iter = iterations / 10

        for i in range(iterations):

            nvp = None
            c = 0
            while c < data.shape[0]:
                d = data[c:c+batch_size]
                c += batch_size

                # ----------------------------------------
                # FORWARD: positive CD / reality phase
                # ----------------------------------------

                # clamp to the data and sample from the hidden units
                #
                # note that we're using the activation *probabilities* of the
                # hidden states, not the hidden states themselves, when
                # computing associations. we could also use the states; see
                # section 3 of Hinton's "A Practical Guide to Training
                # Restricted Boltzmann Machines" for more
                pos_hidden_probs, pos_associations = forward(d)

                pos_hidden_states = pos_hidden_probs > np.random.rand(
                    d.shape[0], self.h[0]
                )

                # ----------------------------------------
                # BACK: negative CD / daydreaming phase
                # ----------------------------------------

                # reconstruct the visible units and sample again from the hidden
                # units
                neg_visible_probs, _ = back(pos_hidden_states)

                # ----------------------------------------
                # FORWARD AGAIN
                # ----------------------------------------

                # note, again, that we're using the activation *probabilities*
                # when computing associations, not the states themselves
                _, neg_associations = forward(neg_visible_probs)

                # ----------------------------------------
                # LEARN
                # ----------------------------------------

                updates = self.lr * (
                    (pos_associations - neg_associations) / d.shape[0]
                )
                updates = self.lr * updates

                # ----------------------------------------
                # IMPROVE MODEL
                # ----------------------------------------

                # update weights
                self.w[0] += updates

                if nvp is None:
                    nvp = neg_visible_probs
                else:
                    nvp = np.concatenate((nvp, neg_visible_probs), axis=0)

            # ----------------------------------------
            # STATUS
            # ----------------------------------------

            if i % status_iter == 0:
                error = np.sum((data - nvp) ** 2)
                print('{:05d}: error {:.08f}'.format(i, error))

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have
        been learned), run the network on a set of visible units, to get a
        sample of the hidden units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible
         units.

        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units
         activated from the visible
        units in the data matrix passed in.
        """
        return self._run(data, self.w[0], self.h[0])

    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have
        been learned), run the network on a set of hidden units, to get a sample
        of the visible units.

        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden
         units.

        Returns
        -------
        visible_states: A matrix where each row consists of the visible units
        activated from the hidden units in the data matrix passed in.
        """
        return self._run(data, self.w[0].T, self.ni)

    def _run(self, data, matrix, num):
        # create a matrix, where each row is to be the hidden units (plus a bias
        # unit) sampled from a training example
        states = np.ones((data.shape[0], num))

        # insert bias units of 1 into the first column of data
        data = np.insert(data, 0, 1, axis=1)

        # calculate the activations of the hidden units
        activations = np.dot(data, matrix)
        # calculate the probabilities of turning the hidden units on
        probs = af.func(activations)
        # turn the hidden units on with their specified probabilities
        states[:, :] = probs > np.random.rand(data.shape[0], num)

        # always fix the bias unit to 1
        # states[:, 0] = 1

        # ignore the bias units
        states = states[:, 1:]
        return states

    def daydream(self, num_samples):
        """
        Randomly initialize the visible units once, and start running
        alternating Gibbs sampling steps (where each step consists of updating
        all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step. Note that we only
        initialize the network *once*, so these samples are correlated.

        Returns
        -------
        samples: A matrix, where each row is a sample of the visible units
         produced while the network was daydreaming.
        """

        # Create a matrix, where each row is to be a sample of of the visible
        # units (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.ni + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.ni)

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i-1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.w[0])
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = af.func(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.h[0] + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.w[0].T)
            visible_probs = af.func(visible_activations)
            visible_states = visible_probs > np.random.rand(
                self.ni + 1
            )
            samples[i, :] = visible_states

        # Ignore the bias units (the first column), since they're always set to
        # 1.
        return samples[:, 1:]
