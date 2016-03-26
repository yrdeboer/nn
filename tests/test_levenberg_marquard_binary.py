import numpy as np
import matplotlib.pyplot as plt
import utils as nn_utils
from nets.levenberg_marquard_backprop import LevenbergMarquardBackprop


np.set_printoptions(linewidth=1000)


"""
This script uses a simple 2-layer net with multi-dim
net output (n2) to verify the Levenberg-Marquard algo.
"""

DEBUG = True


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

    training_input = training_target
    
    validation_input = np.zeros((1, 1))
    validation_target = np.zeros((1, 1))

    return training_input, training_target, validation_input, validation_target


def get_debug_weights():

    W1 = np.array([[-0.55324921, 0.30472091, -0.25384921],
                   [-0.2492100,  0.04720913, -0.35324921],
                   [-0.65324921, 0.50472091, 0.95324921]])
    b1 = np.array([[0.4749701],
                   [-0.27889807],
                   [-0.69353998]])
    W2 = np.array([[-0.32492100, 0.20472091, 0.75324921],
                   [-0.4921000, -0.20472091, 0.4324921],
                   [-0.5324921, 0.90472091, 0.64921]])
    b2 = np.array([[0.4749701],
                   [-0.27889807],
                   [-0.69353998]])

    return (W1, b1, W2, b2)


(train_input, train_target, val_inp, val_tar) = get_data_sets()

# print('N_train = {} N_val = {}'.format(train_input.shape[1], val_inp.shape[1]))

kwargs = dict()
kwargs['training_data'] = (train_input, train_target)
kwargs['input_dim'] = train_input.shape[0]
kwargs['layer1_neuron_count'] = 3
kwargs['layer2_neuron_count'] = 3

kwargs['layer1_transfer_function'] = nn_utils.purelin
kwargs['layer2_transfer_function'] = nn_utils.tansig
kwargs['layer1_transfer_function_derivative'] = nn_utils.dpurelin
kwargs['layer2_transfer_function_derivative'] = nn_utils.dtansig

if DEBUG:
    (W1, b1vec, W2, b2vec) = get_debug_weights()
    kwargs['layer1_initial_weights'] = (W1, b1vec)
    kwargs['layer2_initial_weights'] = (W2, b2vec)

# Instantiate backprop with init values
sp = LevenbergMarquardBackprop(** kwargs)

iteration_count = 18
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

for i in range(1, iteration_count):

    converged = sp.train_step()

    if i in plot_points or converged:

        rms_trn = sp.get_rms_error()

        # print('It={:6} rms_trn={:.5f}'.format(i, rms_trn)),
        # print('g_norm={:.3f} conv={}'.format(sp.g_norm, converged))

        if converged:
            break

plt.savefig('binary_simple_test_lb.png')

# print('V=\n{}'.format(train_input))
# print('y=\n{}'.format(train_target))
# print('y^=\n{}'.format(sp.get_response(train_target)))
# print('error\n{}'.format(train_target - sp.get_response(train_target)))

success = np.array_equal(
    np.round(1000 * sp.get_response(
        train_input)), train_target * 1000)

test_name = 'Levenberg-Marquard ("binary")'

if success:
    print('SUCCESS ({})'.format(test_name))
else:
    print('ERROR   ({})'.format(test_name))
