#
#
# from theano import RandomStreams
# import theano.tensor as T
# from theano.tensor.tests.mlp_test import HiddenLayer
#
#
# class DBN(object):
#
#     def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
#                  hidden_layer_sizes=[500,500], n_outs=10):
#
#         self.sigmoid_layers = []
#         self.rbm_layers = []
#         self.params = []
#         self.n_layers = len(hidden_layer_sizes)
#
#         assert self.n_layers > 0
#
#         if not theano_rng:
#             theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
#
#         self.x = T.matrix('x')
#         self.y = T.ivector('y')
#         for i in xrange(self.n_layers):
#             # construct the sigmoidal layer
#
#             # the size of the input is either the number of hidden
#             # units of the layer below or the input size if we are on
#             # the first layer
#             if i == 0:
#                 input_size = n_ins
#             else:
#                 input_size = hidden_layers_sizes[i - 1]
#
#             # the input to this layer is either the activation of the
#             # hidden layer below or the input of the DBN if you are on
#             # the first layer
#             if i == 0:
#                 layer_input = self.x
#             else:
#                 layer_input = self.sigmoid_layers[-1].output
#
#             sigmoid_layer = HiddenLayer(rng=numpy_rng,
#                                         input=layer_input,
#                                         n_in=input_size,
#                                         n_out=hidden_layer_sizes[i],
#                                         activation=T.nnet.sigmoid)
#
#             # add the layer to our list of layers
#             self.sigmoid_layers.append(sigmoid_layer)
#
#             # its arguably a philosophical question...  but we are
#             # going to only declare that the parameters of the
#             # sigmoid_layers are parameters of the DBN. The visible
#             # biases in the RBM are parameters of those RBMs, but not
#             # of the DBN.
#             self.params.extend(sigmoid_layer.params)
#
#             # Construct an RBM that shared weights with this layer
#             rbm_layer = RBM(numpy_rng=numpy_rng,
#                             theano_rng=theano_rng,
#                             input=layer_input,
#                             n_visible=input_size,
#                             n_hidden=hidden_layer_sizes[i],
#                             W=sigmoid_layer.W,
#                             hbias=sigmoid_layer.b)
#             self.rbm_layers.append(rbm_layer)
