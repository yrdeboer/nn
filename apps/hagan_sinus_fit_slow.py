import numpy as np
import matplotlib.pyplot as plt
from nets.simple_two_layer_backprop import SimpleTwoLayerBackprop
import utils as nn_utils

def g(p):

    """
    The function used in the example (equation 11.57)
    """

    return 1. + np.sin(.5 * np.pi * p)


def get_training_set(interpolate=False):

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
    """

    step = 1.0
    if interpolate:
        step = step / 3.

    x = np.arange(-2., 2.1, step)
    y = g(x)
    N = len(x)

    return (x.reshape(1, N), y.reshape(1, N))


(train_input, train_target) = get_training_set()

S1 = 3

kwargs = dict()
kwargs['training_data'] = (train_input, train_target)
kwargs['input_dim'] = train_input.shape[0]
kwargs['layer1_neuron_count'] = S1
kwargs['layer2_neuron_count'] = 1
kwargs['learning_rate'] = 0.01

kwargs['layer1_transfer_function'] = nn_utils.logsig
kwargs['layer2_transfer_function'] = nn_utils.purelin

kwargs['layer1_transfer_function_derivative'] = nn_utils.dlogsig
kwargs['layer2_transfer_function_derivative'] = nn_utils.dpurelin

W1, b1vec, W2, b2vec = nn_utils.get_fixed_test_weights(S1)

kwargs['layer1_initial_weights'] = (W1, b1vec)
kwargs['layer2_initial_weights'] = (W2, b2vec)


# Instantiate backprop with init values
sp = SimpleTwoLayerBackprop(** kwargs)



iteration_count = 1000000
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

# Interactive plotting of the mean squared error
plt.subplot(2,1,1)
plt.axis([1, 10. * iteration_count, 1e-5, 10.])
plt.yscale('log')
plt.xscale('log')
plt.ion()
plt.show()

print('Initial weights:')
sp.print_weights()
print('Initial response:')
print(sp.get_response(train_input))

rms = nn_utils.get_rms_error(train_input, train_target, sp)
print('Initial rms: {}'.format(rms))

for i in range(1, iteration_count):

    sp.train_step()

    if i in plot_points:
    # if True:

        rms = nn_utils.get_rms_error(train_input, train_target, sp)

        print('Iteration: {:5} rms: {:.6f}'.format(i, rms[0,0]))

        plt.subplot(2,1,1)        
        plt.scatter(i, rms, c='b')
        plt.draw()

        plt.subplot(2,1,2)        
        plt.cla()
        x, _ = get_training_set(True)
        y = sp.get_response(x)
        plt.scatter(x[0], np.transpose(y), c='b')

        plt.scatter(train_input[0], train_target[0], c='r')
        plt.draw()

plt.savefig('hagan_sinus_fit_slow.png')
