import numpy as np
from simple_two_layer_backprop import SimpleTwoLayerBackprop


def g(p):

    """
    The function used in the example.
    """

    return 1. + np.sin(0.25 * np.pi * p)


def get_training_set_hagan_test_1(interpolate=False):

    """
    This function returns the training data to reproduce
    the Hagan example on p. 11-15.
    """

    x = np.array(1.).reshape(1,1)

    return (x, g(x))


def logsig(x):

    return 1. / (1. + np.exp(-x))

def dlogsig(x):
    
    e = np.exp(-x)
    d = (1. + e)
    return e / (d * d)

def purelin(x):
    return x

def dpurelin(x):
    return 1.


kwargs = dict()
kwargs['input_dim'] = 1
kwargs['layer1_neuron_count'] = 2
kwargs['layer2_neuron_count'] = 1
kwargs['learning_rate'] = 0.1

kwargs['layer1_transfer_function'] = logsig
kwargs['layer2_transfer_function'] = purelin

kwargs['layer1_transfer_function_derivative'] = dlogsig
kwargs['layer2_transfer_function_derivative'] = dpurelin

W1 = np.array([[-0.27], [-0.41]])
b1vec = np.array([[-0.48], [-0.13]])
W2 = np.array([[0.09, -0.17]])
b2vec = np.array([[0.48]])

kwargs['layer1_initial_weights'] = [W1, b1vec]
kwargs['layer2_initial_weights'] = [W2, b2vec]

(V, y) = get_training_set_hagan_test_1()
kwargs['training_data'] = (V, y)

# Instantiate backprop with init values
sp = SimpleTwoLayerBackprop(** kwargs)
sp.train_step()

# Check values
error_count = 0
if not int(round(sp.W1[0][0] * 1000)) == -265:
    error_count += 1
if not int(round(sp.W1[1][0] * 1000)) == -420:
    error_count += 1
if not int(round(sp.b1vec[0][0] * 1000)) == -475:
    error_count += 1
if not int(round(sp.b1vec[1][0] * 1000)) == -140:
    error_count += 1
if not int(round(sp.W2[0][0] * 1000)) == 171:
    error_count += 1
if not int(round(sp.W2[0][1] * 10000)) == -772:
    error_count += 1
if not int(round(sp.b2vec[0][0] * 1000)) == 732:
    error_count += 1


if error_count == 0:
    print('SUCCESS')
else:
    print('ERROR (error_count = {})'.format(error_count))
