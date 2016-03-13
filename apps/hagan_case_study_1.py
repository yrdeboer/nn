import os
import numpy as np
import matplotlib.pyplot as plt
import utils as nn_utils
from nets.simple_two_layer_backprop import SimpleTwoLayerBackprop

PYTHONPATH = os.environ.get('PYTHONPATH', None)
if not PYTHONPATH:
    raise ValueError('PYTHONPATH not set')

DATA_FILE_P = '{}/hagan_case_study_data/ball_p.txt'.format(PYTHONPATH)
DATA_FILE_T = '{}/hagan_case_study_data/ball_t.txt'.format(PYTHONPATH)


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


def get_data_sets():

    """
    Returns the normalised training data in a 2-tuple as follows:

    (V, y) with V.shape = and y.shape = 
    
    """

    tvec = np.loadtxt(DATA_FILE_T)
    P = np.loadtxt(DATA_FILE_P)

    Ncol = tvec.shape[0]
    
    v1vec = normalise_vector(P[0])
    v2vec =normalise_vector(P[1])
    y = normalise_vector(tvec)

    V = np.zeros((2, Ncol))
    V[0] = v1vec
    V[1] = v2vec
    y = y.reshape(1, Ncol)

    i_trn = 0
    trn_inp = np.zeros((2, Ncol))
    trn_tar = np.zeros((1, Ncol))
    i_val = 0
    val_inp = np.zeros((2, Ncol))
    val_tar = np.zeros((1, Ncol))

    for i in range(Ncol):

        if np.random.random() < .7:
            trn_inp[:, [i_trn]] = V[:, [i]]
            trn_tar[:, [i_trn]] = y[:, [i]]
            i_trn += 1

        else:
            val_inp[:, [i_val]] = V[:, [i]]
            val_tar[:, [i_val]] = y[:, [i]]
            i_val += 1

    trn_inp = trn_inp[:, range(i_trn)]
    trn_tar = trn_tar[:, range(i_trn)]
    val_inp = val_inp[:, range(i_val)]
    val_tar = val_tar[:, range(i_val)]

    if not trn_inp.shape[1] + val_inp.shape[1] == Ncol:
        raise ValueError('Sum of training and validation columns not correct')

    return (trn_inp, trn_tar, val_inp, val_tar)


def plot_data():

    # Figure 18.4 from book, plots v1 and v2 (67 columns of V) in the y-range: [-1, 1]

    (V1, y1, V2, y2) = get_data_sets()

    plt.plot(y1[0], V1[0], 'ro', y1[0], V1[1], 'bo')
    plt.plot(y2[0], V2[0], 'r*', y2[0], V2[1], 'b*')
    plt.show()


def get_network_response(W1, b1vec, W2, b2vec, V, y):

    yhat = np.zeros(len(y))
    for i in range(len(y)):
        pvec = V[:, [i]]
        tvec = np.array([[y[i]]])
        a1vec = logsig(np.dot(W1, pvec) + b1vec)
        a2vec = np.dot(W2, a1vec) + b2vec
        yhat[i] = a2vec

    return yhat


def plot(W1, b1vec, W2, b2vec, V, y, ssevecx, ssvecy):

    print('\nplot_net')

    # print('W1 = {} W1.shape = {}'.format(W1, W1.shape))
    # print('b1vec = {} b1vec.shape = {}'.format(b1vec, b1vec.shape))
    # print('W2 = {} W2.shape = {}'.format(W2, W2.shape))
    # print('b2vec = {} b2vec.shape = {}'.format(b2vec, b2vec.shape))

    yhat = get_network_response(W1, b1vec, W2, b2vec, V, y)

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(y, yhat, '.')

    ax2 = fig.add_subplot(2,1,2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.plot(ssevecx, ssvecy, '.')
    
    plt.show()


def get_sse(W1, b1vec, W2, b2vec, V, y):

    """
    The sum squared error is calculated
    from the current network weights and
    a data set of inputs p and targets t.
    """

    yhat = get_network_response(W1, b1vec, W2, b2vec, V, y)

    return np.sum(np.power(y - yhat, 2))



    # Plot stats
    plot(W1, b1vec, W2, b2vec, V, y, ssevecx, ssevecy)

# backprop()

# plot_data()



(train_input, train_target, val_inp, val_tar) = get_data_sets()

kwargs = dict()
kwargs['training_data'] = (train_input, train_target)
kwargs['input_dim'] = train_input.shape[0]
kwargs['layer1_neuron_count'] = 3
kwargs['layer2_neuron_count'] = 1
kwargs['learning_rate'] = 0.01

kwargs['layer1_transfer_function'] = nn_utils.tansig
kwargs['layer2_transfer_function'] = nn_utils.purelin
kwargs['layer1_transfer_function_derivative'] = nn_utils.dtansig
kwargs['layer2_transfer_function_derivative'] = nn_utils.dpurelin

# Instantiate backprop with init values
sp = SimpleTwoLayerBackprop(** kwargs)

iteration_count = 100000
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

# Interactive plotting of the mean squared error
plt.subplot(2,1,1)
plt.axis([1, 10. * iteration_count, 1e-5, 10.])
plt.yscale('log')
plt.xscale('log')
plt.ion()
plt.show()

for i in range(1, iteration_count):

    sp.train_step()

    if i in plot_points:

        rms_trn = nn_utils.get_rms_error(train_input, train_target, sp)
        rms_val = nn_utils.get_rms_error(val_inp, val_tar, sp)


        print('Iteration: {} rms_trn: {} rms_val'.format(i, rms_trn, rms_val))

        plt.subplot(2,1,1)
        plt.scatter(i, rms_trn, c='b')
        plt.scatter(i, rms_val, c='r')
        plt.draw()

        plt.subplot(2,1,2)
        plt.cla()
        plt.scatter(train_target[0], sp.get_response(train_input), c='b')
        plt.scatter(0.5 + val_tar[0],  sp.get_response(val_inp), c='r')
        plt.draw()

plt.savefig('hagan_case_study_1.png')
