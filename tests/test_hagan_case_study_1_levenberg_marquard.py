import os
import numpy as np
import matplotlib.pyplot as plt
import utils as nn_utils
from nets.levenberg_marquard_backprop import LevenbergMarquardBackprop
from utils import print_dbg

DEBUG = True

DATA_FILE_P = 'hagan_case_study_data/ball_p.txt'
DATA_FILE_T = 'hagan_case_study_data/ball_t.txt'

def get_debug_weights():

    W1 = np.array([[-0.5324921, 0.472091],
                   [0.47403968, -0.52325356],
                   [-0.0596892, 0.28947488],
                   [0.23209849, 0.86365438],
                   [-0.728602, -0.93854]])
    b1 = np.array([[ -0.8749701] ,  [0.889807],  [0.1427618],  [0.9205414],  [0.3271261]])
    W2 = np.array([[ -0.26564146, 0.69353998, -0.39402367, 0.9349045, -0.1790691]])
    b2 = np.array([[ -0.28713551]])

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


def get_data_sets():

    """
    Returns the normalised training data in a 2-tuple as follows:

    (V, y) with V.shape (R, Q) = and y.shape = (S2, Q)

    For this case study:
      R = 2
      Q = number of points selected, may vary as we divide the
          data in training/validation/test sets etc.
      S2= neuron count of (final) layer 2
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

    if DEBUG:
        for i in [3,25,43,55]:
            trn_inp[:, [i_trn]] = V[:, [i]]
            trn_tar[:, [i_trn]] = y[:, [i]]
            i_trn += 1

        for i in [11]:
            val_inp[:, [i_val]] = V[:, [i]]
            val_tar[:, [i_val]] = y[:, [i]]
            i_val += 1

    else:

        for i in range(Ncol):
    
            if np.random.random() < .85:
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

    # if not trn_inp.shape[1] + val_inp.shape[1] == Ncol:
    #     raise ValueError('Sum of training and validation columns not correct')

    return (trn_inp, trn_tar, val_inp, val_tar)


def plot_data():

    # Figure 18.4 from book, plots v1 and v2 (67 columns of V) in the y-range: [-1, 1]

    (V1, y1, V2, y2) = get_data_sets()

    plt.scatter(y1[0], V1[0], c='b', marker='o')
    plt.scatter(y2[0], V2[0], c='r', marker='o')

    plt.scatter(y1[0], V1[1], c='b', marker='v')
    plt.scatter(y2[0], V2[1], c='r', marker='v')

    plt.show()


def weights_as_expected(sp):

    W1 = np.array([[-0.56356913, 0.54366279],
                   [0.55272437, -0.72401577],
                   [-0.26784775, 0.36279903],
                   [0.45406544, 1.20365471],
                   [-0.92410466, -0.86705573]])
    b1vec = np.array([[-0.76825938],
                      [0.60609654],
                      [0.30372701],
                      [0.63358324],
                      [0.30658882]])
    W2 = np.array([[-0.11766191,
                    0.55124574,
                    -0.5632504,
                    0.76741378,
                    -0.51184742]])
    b2vec = np.array([[-0.78043624]])

    mult = 1000000
    a = np.array_equal(np.round(mult * sp.W1), np.round(mult * W1))
    b = np.array_equal(np.round(mult * sp.b1vec), np.round(mult * b1vec))
    c = np.array_equal(np.round(mult * sp.W2), np.round(mult * W2))
    d = np.array_equal(np.round(mult * sp.b2vec), np.round(mult * b2vec))
    
    return a & b & c & d

(train_input, train_target, val_inp, val_tar) = get_data_sets()

if not DEBUG:
    print('N_train = {} N_val = {}'.format(train_input.shape[1], val_inp.shape[1]))

kwargs = dict()
kwargs['training_data'] = (train_input, train_target)
kwargs['input_dim'] = train_input.shape[0]
kwargs['layer1_neuron_count'] = 5
kwargs['layer2_neuron_count'] = 1

kwargs['layer1_transfer_function'] = nn_utils.tansig
kwargs['layer2_transfer_function'] = nn_utils.purelin
kwargs['layer1_transfer_function_derivative'] = nn_utils.dtansig
kwargs['layer2_transfer_function_derivative'] = nn_utils.dpurelin

if DEBUG:
    (W1, b1vec, W2, b2vec) = get_debug_weights()
    kwargs['layer1_initial_weights'] = (W1, b1vec)
    kwargs['layer2_initial_weights'] = (W2, b2vec)

# Instantiate backprop with init values
sp = LevenbergMarquardBackprop(** kwargs)

if not DEBUG:
    print_dbg('Weights:')
    sp.print_weights()

iteration_count = 1000
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

if not DEBUG:

    # Interactive plotting of the mean squared error
    plt.subplot(2,1,1)
    plt.axis([1, 10. * iteration_count, 1e-7, 10.])
    plt.yscale('log')
    plt.xscale('log')
    plt.ion()

for i in range(1, iteration_count):

    converged = sp.train_step()

    if i < 25 or i in plot_points or converged:

        if not DEBUG:
        
            rms_trn = sp.rms
            rms_val = nn_utils.get_rms_error(val_inp, val_tar, sp)
    
            print('It={:6} rms_trn={:.5f}  rms_val={:.5f} '.format(
                i, rms_trn, rms_val[0, 0]), end='')
            print('g_norm={:.3f} conv={}'.format(sp.g_norm, converged))

            plt.subplot(2, 1, 1)
            plt.scatter(i, rms_trn, c='b')
            plt.scatter(i, rms_val, c='r')

            plt.subplot(2, 1, 2)
            plt.cla()
            plt.scatter(train_target[0], sp.get_response(train_input), c='b')
            plt.scatter(0.5 + val_tar[0],  sp.get_response(val_inp), c='r')
            plt.show()

            # Dont' remove this, needed
            # for plotting, sucks, yeah.
            plt.pause(.000001)  

        if converged:
            break

if DEBUG:

    test_name = 'Levenberg-Marquard Hagan Case study 1 -- light sensors'

    if weights_as_expected(sp):
        print('SUCCESS ({})'.format(test_name))
    else:
        print('ERROR   ({})'.format(test_name))

else:
    plt.savefig('hagan_case_study_1_lb.png')
