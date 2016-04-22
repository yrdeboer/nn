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

    step = 0.05
    if interpolate:
        step = step / 3.

    x = np.arange(-2., 2.1, step)
    y = g(x)
    if add_noise:

        y += np.random.normal(0.0, 0.1, y.shape)
    N = len(x)

    return (x.reshape(1, N), y.reshape(1, N))


def get_data_sets():

    """
    This function returns three data sets in a 6-tuple:

      (train_inp, train_tar, val_inp, val_tar, test_inp, test_tar)

    """

    interpolate = False
    add_noise = True
    (all_data_inp, all_data_tar) = get_training_set(interpolate,
                                                    add_noise)


    return nn_utils.all_data_to_data_sets(all_data_inp,
                                          all_data_tar,
                                          0.7,
                                          0.15)


def plot_data_sets(data_6_tup):

    (train_inp, train_tar, val_inp, val_tar, test_inp, test_tar) = data_6_tup

    plt.plot(train_inp[0], train_tar[0], 'r^')
    plt.plot(val_inp[0], val_tar[0], 'gs')
    plt.plot(test_inp[0], test_tar[0], 'bD')

    plt.title('N_tr={} N_val={} N_test={}'.format(
        train_inp.shape[1],
        val_inp.shape[1],
        test_inp.shape[1]))

    plt.ion()
    plt.show()
    plt.ioff()


np.random.seed(1)
    
data_6_tup = get_data_sets()

plot_data_sets(data_6_tup)

(train_inp, train_tar, val_inp, val_tar, test_inp, test_tar) = data_6_tup

S1 = 20

kwargs = dict()
kwargs['training_data'] = (train_inp, train_tar)
kwargs['input_dim'] = train_inp.shape[0]
kwargs['layer1_neuron_count'] = S1
kwargs['layer2_neuron_count'] = 1

kwargs['layer1_transfer_function'] = nn_utils.logsig
kwargs['layer2_transfer_function'] = nn_utils.purelin

kwargs['layer1_transfer_function_derivative'] = nn_utils.dlogsig
kwargs['layer2_transfer_function_derivative'] = nn_utils.dpurelin

# W1, b1vec, W2, b2vec = nn_utils.get_fixed_test_weights(S1)
# kwargs['layer1_initial_weights'] = (W1, b1vec)
# kwargs['layer2_initial_weights'] = (W2, b2vec)

# Instantiate backprop with init values
sp = LevenbergMarquardBackprop(** kwargs)

iteration_count = 10
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

# Interactive plotting of the mean squared error
plt.ion()

plt.subplot(3, 1, 1)
plt.title(r'RMS: training (blue) validation (green)')
plt.axis([1, 10. * iteration_count, 1e-7, 100.])
plt.yscale('log')
plt.xscale('log')

plt.subplot(3, 1, 2)
plt.title(r'Training fit')

plt.subplot(3, 1, 3)
plt.title(r'Early stopping fit')

x_g = np.arange(-2., 2., .01)
y_g = g(x_g)

# print('Initial weights:')
# sp.print_weights()
# print('Initial x:\n{}'.format(np.transpose(sp.weights_to_x())))

# print('Initial rms:')
# print(sp.get_rms_error())
# print('--')

rms_val_min = np.finfo('float64').max
weights_val_min = sp.weights_to_x()
updated_val_min = True

for i in range(1, iteration_count):

    converged = sp.train_step()

    if i < 25 or i == iteration_count-1 or i in plot_points or converged:

        rms_train = nn_utils.get_rms_error(train_inp, train_tar, sp)
        rms_val = nn_utils.get_rms_error(val_inp, val_tar, sp)

        if rms_val < rms_val_min:
            print('Updated rms_val_min to {}'.format(rms_val_min))
            rms_val_min = rms_val
            weights_val_min = sp.weights_to_x()
            updated_val_min = True

        print(
            'Iteration: {:5} rms_train: {:.8f} rms_val={:.8f} g_norm: {:.6f} converged: {}'.format(
                i, rms_train, rms_val, sp.g_norm, converged))
        sys.stdout.flush()

        # Training vs. validation rms (error)a
        plt.subplot(3, 1, 1)
        plt.scatter(i, rms_train, c='b')
        plt.scatter(i, rms_val, c='g')

        # Curve showing target/underlying function
        plt.subplot(3, 1, 2)
        plt.cla()
        plt.plot(x_g, y_g)

        # Interpolating points, to see live response updates
        x, _ = get_training_set(True, False)
        y = sp.get_response(x)
        plt.scatter(x[0], np.transpose(y), c='b')
        plt.scatter(train_inp[0], train_tar[0], c='r', marker='s')

        if updated_val_min:
            plt.subplot(3, 1, 3)
            plt.cla()

            plt.plot(x_g, y_g)
            plt.scatter(x[0], np.transpose(y), c='b')
            plt.scatter(val_inp[0], val_tar[0], c='g', marker='D')
            updated_val_min = False

        plt.show()
        plt.pause(.0000001)

    if converged:
        break

plt.savefig('hagan_sinus_early_stopping.png')
