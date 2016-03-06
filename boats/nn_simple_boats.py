import numpy as np
import matplotlib.pyplot as plt

# Don't forget to add the parent directory to PYTHONPATH shell variable
from simple_two_layer_backprop import SimpleTwoLayerBackprop


DATA_DIR_CR = '/home/ytsboe/data/boats/computer_readable'

feature_names = np.load('{}/feature_names.npy'.format(DATA_DIR_CR))
builder_names = np.load('{}/builder_names.npy'.format(DATA_DIR_CR))
input_data = np.load('{}/input_data.npy'.format(DATA_DIR_CR))
target_data = np.load('{}/target_data.npy'.format(DATA_DIR_CR))


def get_training_data():

    """
    This function returns the training data.
    """

    return input_data, target_data


def tansig(x):

    return np.tanh(x)


def dtansig(x):
    
    tanh = np.tanh(x)
    return 1. - tanh*tanh


def purelin(x):
    return x


def dpurelin(x):
    return 1.

def get_error(train_in, target_in, sp):

    """
    This function obtains the error, given
    the neural net and train/target data pairs.

    The mean squared error is used.

    Args:
      train_in:  Input data, columns are input vectors
      target_in: Target data, columns are target values
      sp:        Instance of trained SimpleTwoLayerBackprop

    Returns:
      A float, the mean squared error.
    """

    N = len(target_in[0])  # column count
    if N == 0.:
        raise ValueError('No input data, cannot calculate N')

    mse = 0.
    for i in range(N):

        pvec = train_in[:, [i]]
        pnet = sp.get_response(pvec)

        diff = target_in[:, [i]] - pnet
        mse += diff * diff

    return mse / float(N)


R = len(feature_names) + len(builder_names)
S1 = 5
S2 = 1

kwargs = dict()
kwargs['input_dim'] = R
kwargs['layer1_neuron_count'] = S1
kwargs['layer2_neuron_count'] = S2
kwargs['learning_rate'] = 0.00001

kwargs['layer1_transfer_function'] = tansig
kwargs['layer2_transfer_function'] = purelin

kwargs['layer1_transfer_function_derivative'] = dtansig
kwargs['layer2_transfer_function_derivative'] = dpurelin

mult_factor = .5
W1 = (np.random.random_sample((S1, R)) - 0.5).reshape(S1, R) * mult_factor
b1vec = (np.random.random_sample((S1)) - 0.5).reshape((S1, 1)) * mult_factor
W2 = (np.random.random_sample(S1) - 0.5).reshape(1, S1) * mult_factor
b2vec = (np.random.random_sample(S2).reshape((S2, 1)) - 0.5) * mult_factor

kwargs['layer1_initial_weights'] = [W1, b1vec]
kwargs['layer2_initial_weights'] = [W2, b2vec]

train_in, target_in = get_training_data()
kwargs['training_data'] = (train_in, target_in)


# Instantiate backprop with init values
sp = SimpleTwoLayerBackprop(** kwargs)

iteration_count = 100000
print_interval =     50

# Interactive plotting of the mean squared error
plt.axis([1, 10. * iteration_count, .00001, 10.])
plt.yscale('log')
plt.xscale('log')
plt.ion()
plt.show()

for i in range(1, iteration_count):

    sp.train_step()

    interval = print_interval * (1. + int(round(np.log(i))))
    if i % interval == 0:

        mse = get_error(train_in, target_in, sp)
        print('Iteration: {} error: {}'.format(i, mse))

        plt.scatter(i, mse)
        plt.draw()


