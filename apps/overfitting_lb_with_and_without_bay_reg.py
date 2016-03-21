import numpy as np
import matplotlib.pyplot as plt
from nets.levenberg_marquard_backprop import LevenbergMarquardBackprop
import utils as nn_utils
import sys


def g(p):

    """
    The function used in the example (equation 11.57)
    """

    return 1. + np.sin(0.5 * np.pi * p)

NOISE = [-0.08230345, -0.08804563, -0.11719107, -0.03871356, 0.02517687,
         0.11848636, 0.23875976, -0.06373871, -0.00112604, 0.21156093,
         -0.3530613,
         -0.247158, -0.15701409, 0.46783705, 0.02523148, -0.23911402,
         0.04711678,
         0.10378847, -0.37651379, -0.07710619, -0.03495549, 0.38485966,
         -0.38275189,
         -0.25700503,
         0.23190758, 0.03926231, 0.62861338, -0.13706857, 0.10880528,
         -0.0454776,
         0.10889065, 0.30228765, -0.39040886, 0.04372521, 0.23434904,
         0.05687156,
         0.24948046, 0.02746811, 0.20458933, -0.42425311, -0.19365129]


def get_training_set(interpolate, add_noise):

    """
    This function returns the training data, also
    used to reproduce
    the Hagan example on p. 11-15.

    Shapes must be:
      X = input data (Nrows, Ncols)

      where Nrows is the input dimension (1) and
      Ncols is the point count for training.

      y = target data (1, Ncols)

    If interpolate is set to True, then thrice
    as many points are provided on the same range.
    This is to see how the network interpolates.

    If add_noise is set to True, then some random
    gaussian noise is added to g(x).
    """

    step = 0.1
    if interpolate:
        step = step / 3.

    x = np.arange(-2., 2.1, step)
    y = g(x)
    if add_noise:
        y += NOISE
    N = len(x)

    return (x.reshape(1, N), y.reshape(1, N))


(train_input, train_target) = get_training_set(False, True)

S1 = 20

kwargs = dict()
kwargs['training_data'] = (train_input, train_target)
kwargs['input_dim'] = train_input.shape[0]
kwargs['layer1_neuron_count'] = S1
kwargs['layer2_neuron_count'] = 1


kwargs['layer1_transfer_function'] = nn_utils.logsig
kwargs['layer2_transfer_function'] = nn_utils.purelin

kwargs['layer1_transfer_function_derivative'] = nn_utils.dlogsig
kwargs['layer2_transfer_function_derivative'] = nn_utils.dpurelin

# W1, b1vec, W2, b2vec = get_fixed_test_weights()

# kwargs['layer1_initial_weights'] = (W1, b1vec)
# kwargs['layer2_initial_weights'] = (W2, b2vec)

# Instantiate backprop with init values
sp = LevenbergMarquardBackprop(** kwargs)

iteration_count = 1000
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

# Interactive plotting of the mean squared error
plt.subplot(2, 1, 1)
plt.axis([1, 10. * iteration_count, 1e-6, 10.])
plt.yscale('log')
plt.xscale('log')
plt.ion()

x_g = np.arange(-2., 2., .01)
y_g = g(x_g)

print('Initial weights:')
sp.print_weights()
print('Initial x:\n{}'.format(np.transpose(sp.weights_to_x())))

print('Initial response:')
print(sp.get_response(train_input))
print('Initial rms:')
print(sp.get_rms_error())
print('--')

for i in range(1, iteration_count):

    converged = sp.train_step()

    if i < 25 or i in plot_points or converged:

        rms = sp.rms

        print(
            'Iteration: {:5} rms: {:.8f} g_norm: {:.6f} converged: {}'.format(
                i, rms, sp.g_norm, converged))
        sys.stdout.flush()

        plt.subplot(2, 1, 1)
        plt.scatter(i, rms, c='b')

        plt.subplot(2, 1, 2)
        plt.cla()

        plt.plot(x_g, y_g)

        x, _ = get_training_set(True, False)
        y = sp.get_response(x)
        plt.scatter(x[0], np.transpose(y), c='b')

        plt.scatter(train_input[0], train_target[0], c='r', marker='s')

        plt.show()
        plt.pause(.00001)

    if converged:
        break

plt.savefig('hagan_sinus_fit_slow_lb.png')
