import numpy as np
import matplotlib.pyplot as plt

# Don't forget to add the parent directory to PYTHONPATH shell variable
from simple_two_layer_backprop import SimpleTwoLayerBackprop


DATA_DIR_CR = '/home/ytsboe/data/boats/computer_readable'

feature_names = np.load('{}/feature_names.npy'.format(DATA_DIR_CR))
builder_names = np.load('{}/builder_names.npy'.format(DATA_DIR_CR))

def get_training_data():

    """
    This function returns the training and test data.
    """

    input_data = np.load('{}/input_data.npy'.format(DATA_DIR_CR))
    target_data = np.load('{}/target_data.npy'.format(DATA_DIR_CR))

    # Use some ratio

    N = len(target_data[0])  # col count
    N_train = int(round(.15 * N))
    N_test = N - N_train

    if not N_test + N_train == N:
        raise ValueError('N_test + N_train not equal to N')

    test_input = input_data[:, range(0, N_train)]
    test_target = target_data[:, range(0, N_train)]
    train_input = input_data[:, range(N_train, N)]
    train_target = target_data[:, range(N_train, N)]

    return train_input, train_target, test_input, test_target


def tansig(x):

    return np.tanh(x)


def dtansig(x):
    
    tanh = np.tanh(x)
    return 1. - tanh*tanh


def purelin(x):
    return x


def dpurelin(x):
    return 1.

def get_error(dat_inp, dat_tar, sp):

    """
    This function obtains the error, given
    the neural net and train/target data pairs.

    The mean squared error is usually used.

    Args:
      dat_inp:  Input data, columns are input vectors
      dat_tar: Target data, columns are target values
      sp:        Instance of trained SimpleTwoLayerBackprop

    Returns:
      A float, some error.
    """

    N = len(dat_tar[0])  # column count
    if N == 0.:
        raise ValueError('No input data, cannot calculate N')

    mse = 0.
    for i in range(N):

        pvec = dat_inp[:, [i]]
        pnet = sp.get_response(pvec)

        diff = dat_tar[:, [i]] - pnet
        mse += diff * diff
        # mse += np.abs(diff)

    return mse / float(N)


def plot_distribution(dat_inp, dat_tar, sp):

    N = len(dat_tar[0])  # column count
    if N == 0.:
        raise ValueError('No input data, cannot calculate N')

    diff = np.zeros(N)
    mse = 0.
    for i in range(N):

        pvec = dat_inp[:, [i]]
        pnet = sp.get_response(pvec)

        diff[i] = dat_tar[:, [i]] - pnet

    print('Average diff = {} sd = {}'.format(
        np.mean(diff),
        np.std(diff)))
    
    plt.hist(diff, 50)

    diffpng = 'diff.png'
    print('Saving diff to {}'.format(diffpng))
    plt.savefig(diffpng)
    plt.close()

R = len(feature_names) + len(builder_names)
S1 = 100
S2 = 1

kwargs = dict()
kwargs['input_dim'] = R
kwargs['layer1_neuron_count'] = S1
kwargs['layer2_neuron_count'] = S2
kwargs['learning_rate'] = 0.0001

print('S1 = {} alpha = {}'.format(S1, kwargs['learning_rate']))

kwargs['layer1_transfer_function'] = tansig
kwargs['layer2_transfer_function'] = purelin

kwargs['layer1_transfer_function_derivative'] = dtansig
kwargs['layer2_transfer_function_derivative'] = dpurelin

mult_factor = .1
W1 = (np.random.random_sample((S1, R)) - 0.5).reshape(S1, R) * mult_factor
b1vec = (np.random.random_sample((S1)) - 0.5).reshape((S1, 1)) * mult_factor
W2 = (np.random.random_sample(S1) - 0.5).reshape(1, S1) * mult_factor
b2vec = (np.random.random_sample(S2).reshape((S2, 1)) - 0.5) * mult_factor

kwargs['layer1_initial_weights'] = [W1, b1vec]
kwargs['layer2_initial_weights'] = [W2, b2vec]

train_input, train_target, test_input, test_target  = get_training_data()
kwargs['training_data'] = (train_input, train_target)


# Instantiate backprop with init values
sp = SimpleTwoLayerBackprop(** kwargs)

iteration_count = 10000000
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

# Interactive plotting of the mean squared error
plt.axis([1, 10. * iteration_count, .01, 10.])
plt.yscale('log')
plt.xscale('log')
plt.ion()
plt.show()

for i in range(1, iteration_count):

    sp.train_step()

    if i in plot_points:

        mse_train = get_error(train_input, train_target, sp)
        mse_test = get_error(test_input, test_target, sp)

        print('Iteration: {} error train: {} error test: {}'.format(i, mse_train, mse_test))

        plt.scatter(i, mse_train, c='blue')
        plt.scatter(i, mse_test, c='red')
        plt.draw()

msespng = 'mses.png'
print('Saving mses plot to {}'.format(msespng)) 
plt.savefig(msespng, format='png')
plt.close()

sp.print_weights()

plot_distribution(test_input, test_target, sp)
