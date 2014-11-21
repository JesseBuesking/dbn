"""
JRBM.
"""


from __future__ import print_function
import numpy as np
import activation_functions as af


# noinspection PyDocstring
class JRBM(object):

    def __init__(self, num_visible, num_hidden, learning_rate=0.1):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate

        # Initialize a weight matrix, of dimensions (num_visible x num_hidden),
        # using a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis=0)
        self.weights = np.insert(self.weights, 0, 0, axis=1)

        # example of what self.weights could be
        # [[ 0.          0.          0.        ]
        #  [ 0.          0.0555121   0.01520456]
        #  [ 0.          0.09536031 -0.11957774]
        #  [ 0.          0.0223247  -0.08246847]
        #  [ 0.          0.0031644   0.24317695]
        #  [ 0.          0.14754101  0.09360807]
        #  [ 0.         -0.05690645  0.04741315]]

    def train(self, data, max_epochs=1000, verbose=False):
        """
        Train the machine.

        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the
         states of visible units.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        num_examples = data.shape[0]

        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis=1)

        for epoch in range(max_epochs):
            # ----------------------------------------
            # FORWARD
            # ----------------------------------------

            # Clamp to the data and sample from the hidden units.
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)
            pos_hidden_probs = af.func(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(
                num_examples, self.num_hidden + 1
            )
            # Note that we're using the activation *probabilities* of the hidden
            # states, not the hidden states themselves, when computing
            # associations. We could also use the states; see section 3 of
            # Hinton's "A Practical Guide to Training Restricted Boltzmann
            # Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)
            if verbose:
                print('weights')
                print(self.weights)
                print('data')
                print(data)
                print('pos')
                print(pos_hidden_activations)
                print(pos_hidden_probs)
                print(pos_hidden_states)
                print(pos_associations)

            # ----------------------------------------
            # BACK
            # ----------------------------------------

            # Reconstruct the visible units and sample again from the hidden
            # units. This is the "negative CD phase", aka the daydreaming phase.
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = af.func(neg_visible_activations)
            # Fix the bias unit.
            neg_visible_probs[:, 0] = 1

            # ----------------------------------------
            # FORWARD AGAIN
            # ----------------------------------------

            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = af.func(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when
            # computing associations, not the states themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            if verbose:
                print('neg')
                print(neg_visible_activations)
                print(neg_visible_probs)
                print(neg_hidden_activations)
                print(neg_hidden_probs)
                print(neg_associations)

            # ----------------------------------------
            # LEARN
            # ----------------------------------------

            updates = self.learning_rate * (
                (pos_associations - neg_associations) / num_examples
            )
            updates = self.learning_rate * updates
            if verbose:
                print('updates')
                print(updates)

            # ----------------------------------------
            # FIX MODEL
            # ----------------------------------------

            # Update weights.
            self.weights += updates

            # ----------------------------------------
            # PROGRESS
            # ----------------------------------------

            error = np.sum((data - neg_visible_probs) ** 2)
            print("Epoch %s: error is %s" % (epoch, error))

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
        return self._run(data, self.weights, self.num_hidden)

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
        return self._run(data, self.weights.T, self.num_visible)

    def _run(self, data, matrix, num):
        num_examples = data.shape[0]

        # Create a matrix, where each row is to be the hidden units (plus a bias
        # unit) sampled from a training example.
        states = np.ones((num_examples, num + 1))

        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis=1)

        # Calculate the activations of the hidden units.
        activations = np.dot(data, matrix)
        # Calculate the probabilities of turning the hidden units on.
        probs = af.func(activations)
        # Turn the hidden units on with their specified probabilities.
        states[:, :] = probs > np.random.rand(num_examples, num + 1)
        # Always fix the bias unit to 1.
        # states[:, 0] = 1

        # Ignore the bias units.
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
        samples = np.ones((num_samples, self.num_visible + 1))

        # Take the first sample from a uniform distribution.
        samples[0, 1:] = np.random.rand(self.num_visible)

        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i-1, :]

            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = af.func(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1

            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = af.func(visible_activations)
            visible_states = visible_probs > np.random.rand(
                self.num_visible + 1
            )
            samples[i, :] = visible_states

        # Ignore the bias units (the first column), since they're always set to
        # 1.
        return samples[:, 1:]
