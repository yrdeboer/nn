import numpy as np
import matplotlib.pyplot as plt
import utils as nn_utils
from nets.levenberg_marquard_backprop import LevenbergMarquardBackprop
from utils import print_dbg

"""
This script uses a simple 2-layer net with multi-dim
net output (n2) to verify the Levenberg-Marquard algo.
"""


def get_data_sets():

    """
    This function returns the training data set.
    It also returns a validation set for consistenty,
    but that is left empty.
    """

    # Input shape = (R, Q) = (1, 8)
    training_input = np.array([[0., 1., 2., 3., 4., 5., 6., 7.]])

    # Target shape = (S2, Q) = (3, 8)
    training_target = np.array(
        [[0., 0., 0., 0., 1., 1., 1., 1.],
         [0., 0., 1., 1., 0., 0., 1., 1.],
         [0., 1., 0., 1., 0., 1., 0., 1.]])

    validation_input = np.zeros((1, 1))
    validation_target = np.zeros((1, 1))

    return training_input, training_target, validation_input, validation_target


def get_debug_weights():

    W1 = np.array([[-0.55324921],
                   [0.50472091]])
    b1 = np.array([[0.4749701],
                   [-0.27889807]])
    W2 = np.array([[0.26564146, -0.69353998],
                   [0.39402367, 0.19349045],
                   [0.31790691, -0.83271261]])
    b2 = np.array([[0.28713551],
                   [-0.91427618],
                   [0.39205414]])

    return (W1, b1, W2, b2)


def normalise_vector(vec):

    """
    Project on range [-1, 1]
    """

    min = np.min(vec)
    max = np.max(vec)

    if max == min:
        raise ValueError(
            'Cannot normalise vector, max equals min')

    return -1. + 2. * (vec - min) / (max - min)


(train_input, train_target, val_inp, val_tar) = get_data_sets()

print('N_train = {} N_val = {}'.format(train_input.shape[1], val_inp.shape[1]))

kwargs = dict()
kwargs['training_data'] = (train_input, train_target)
kwargs['input_dim'] = train_input.shape[0]
kwargs['layer1_neuron_count'] = 2
kwargs['layer2_neuron_count'] = 3

kwargs['layer1_transfer_function'] = nn_utils.purelin
kwargs['layer2_transfer_function'] = nn_utils.tansig
kwargs['layer1_transfer_function_derivative'] = nn_utils.dpurelin
kwargs['layer2_transfer_function_derivative'] = nn_utils.dtansig

(W1, b1vec, W2, b2vec) = get_debug_weights()
kwargs['layer1_initial_weights'] = (W1, b1vec)
kwargs['layer2_initial_weights'] = (W2, b2vec)

# Instantiate backprop with init values
sp = LevenbergMarquardBackprop(** kwargs)

print_dbg('Weights:')
sp.print_weights()

iteration_count = 100
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]


# Interactive plotting of the mean squared error
plt.subplot(1, 1, 1)
plt.axis([1, 10. * iteration_count, 1e-7, 10.])
plt.yscale('log')
plt.xscale('log')
plt.ion()

for i in range(1, iteration_count):

    converged = sp.train_step()

    if i in plot_points or converged:

        rms_trn = sp.rms

        print('It={:6} rms_trn={:.5f}'.format(i, rms_trn)),
        print('g_norm={:.3f} conv={}'.format(sp.g_norm, converged))

        plt.subplot(2, 1, 1)
        plt.scatter(i, rms_trn, c='b')

        # Dont' remove this, needed for
        # plotting, sucks, yeah :(
        plt.pause(.000001)

        if converged:
            break

plt.savefig('binary_simple_test_lb.png')
