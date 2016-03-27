import numpy as np
import matplotlib.pyplot as plt

# Don't forget to add the parent directory to PYTHONPATH shell variable
from nets.simple_two_layer_backprop import SimpleTwoLayerBackprop


DATA_DIR_CR = '/home/ytsboe/data/boats/computer_readable'

feature_names = np.load('{}/feature_names.npy'.format(DATA_DIR_CR))
builder_names = np.load('{}/builder_names.npy'.format(DATA_DIR_CR))
file_paths = np.load('{}/file_paths.npy'.format(DATA_DIR_CR))


def get_data_sets(fixed=False):

    """
    This function returns the training, validation and test data.

    If fixed is True, then there will be no random fetching of
    data for the data sets.
    """

    # Fetch all data
    input_data = np.load('{}/input_data.npy'.format(DATA_DIR_CR))
    target_data = np.load('{}/target_data.npy'.format(DATA_DIR_CR))

    # Division fraction training, validation and testing data
    frac_tr = 0.7
    frac_val = 0.15
    frac_tst = 1. - frac_tr - frac_val 

    # Prepare various data set containers
    (Nrow, Ncol) = input_data.shape
    tr_inp = np.zeros((Nrow, Ncol))
    tr_tar = np.zeros((1, Ncol))
    val_inp = np.zeros((Nrow, Ncol))
    val_tar = np.zeros((1, Ncol))
    tst_inp = np.zeros((Nrow, Ncol))
    tst_tar = np.zeros((1, Ncol))

    if fixed:
        fac = 1. / Ncol
        ran_sample = [fac * x for x in range(0, Ncol)]
    else:
        ran_sample = np.random.random_sample(Ncol)

    # Distribute data randomly
    i_tr = i_val = i_tst = 0
    for i in range(Ncol):

        inp = input_data[:, [i]]
        tar = target_data[:, [i]]
        ran = ran_sample[i]

        if ran < frac_tr:
            tr_inp[:, [i_tr]] = inp
            tr_tar[:, [i_tr]] = tar
            i_tr += 1

        elif ran < 1. - frac_tst:

            val_inp[:, [i_val]] = inp
            val_tar[:, [i_val]] = tar
            i_val += 1

        else:
            tst_inp[:, [i_tst]] = inp
            tst_tar[:, [i_tst]] = tar
            i_tst += 1
            
    if not i_tr + i_val + i_tst == Ncol:
        raise ValueError('Sum of index fractions not well')

    # Cut down to actual sizes
    tr_inp = tr_inp[:, range(0, i_tr)]
    tr_tar = tr_tar[:, range(0, i_tr)]
    val_inp = val_inp[:, range(0, i_val)]
    val_tar = val_tar[:, range(0, i_val)]
    tst_inp = tst_inp[:, range(0, i_tst)]
    tst_tar = tst_tar[:, range(0, i_tst)]

    if not tr_inp.shape[1] + val_inp.shape[1] + tst_inp.shape[1] == Ncol:
        raise ValueError('Sum of data set sizes not well')

    tot = np.sum(input_data[1])
    tr = np.sum(tr_inp[1])
    val = np.sum(val_inp[1])
    tst = np.sum(tst_inp[1])
    if tot - tr - val - tst > 1e-13:
        raise ValueError('Sum of lengths not well')

    tot = np.sum(target_data[0])
    tr = np.sum(tr_tar[0])
    val = np.sum(val_tar[0])
    tst = np.sum(tst_tar[0])
    diff = tot - tr - val - tst
    if diff > 1e-11:
        raise ValueError('Diff of prices not well: {}'.format(diff))

    return tr_inp, tr_tar, val_inp, val_tar, tst_inp, tst_tar


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
      sp:        Instance of a net, should have
                 a function "get_network_response"

    Returns:
      A float, some error.
    """

    Ncol = dat_tar.shape[1]
    if Ncol == 0.:
        raise ValueError('No input data, cannot calculate column count')

    mse = 0.
    for i in range(Ncol):

        pvec = dat_inp[:, [i]]
        pnet = sp.get_response(pvec)

        diff = dat_tar[:, [i]] - pnet
        mse += diff * diff
        # mse += np.abs(diff)

    return mse / float(Ncol)


def plot_error_distributions(tr_inp,
                             tr_tar,
                             val_inp,
                             val_tar,
                             tst_inp,
                             tst_tar,
                             sp):


    i_plot = 1

    for tup in [(tr_inp, tr_tar), (val_inp, val_tar), (tst_inp, tst_tar)]:

        inp = tup[0]
        tar = tup[1]

        N = inp.shape[1]  # column count
        if N == 0.:
            raise ValueError('No input data, cannot calculate N')

        diff = np.zeros(N)
        mse = 0.
        for i in range(N):

            pvec = inp[:, [i]]
            pnet = sp.get_response(pvec)

            diff[i] = tar[:, [i]] - pnet

            if diff[i] > 1.0:
                print('diff (i_plot={}) = {} file = {}'.format(
                    i_plot,
                    diff[i],
                    file_paths[i]))


        print('N={}: Average diff = {} sd = {}'.format(
            N,
            np.mean(diff),
            np.std(diff)))
    
        plt.subplot(3, 1, i_plot)
        i_plot += 1
        plt.hist(diff, N / 20)

    diffpng = 'error_distributions.png'
    print('Saving error distributions to {}'.format(diffpng))
    plt.savefig(diffpng)
    plt.close()

R = len(feature_names) + len(builder_names)
S1 = 10
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

train_input, train_target, val_input, val_target, test_input, test_target  = get_data_sets()
kwargs['training_data'] = (train_input, train_target)

print('Training set size:   {}'.format(train_input.shape[1]))
print('Validation set size: {}'.format(val_input.shape[1]))
print('Test set size: {}'.format(test_input.shape[1]))


# Instantiate backprop with init values
sp = SimpleTwoLayerBackprop(** kwargs)

iteration_count = 500
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

# Interactive plotting of the mean squared error
plt.axis([1, 10. * iteration_count, .01, 10.])
plt.yscale('log')
plt.xscale('log')
plt.ion()
plt.show()

# import ipdb; ipdb.set_trace()

error_train = np.array([[-1. ]])
error_val = np.array([[-1. ]])
error_tst = np.array([[-1. ]])

for i in range(1, iteration_count):

    sp.train_step()

    if i in plot_points:

        error_train = get_error(train_input, train_target, sp)
        error_val = get_error(val_input, val_target, sp)
        error_tst = get_error(test_input, test_target, sp)

        print('Iteration: {} errors: train {}, val {}, tst {}'.format(
            i,
            error_train,
            error_val,
            error_tst))

        plt.scatter(i, error_train, c='blue')
        plt.scatter(i, error_val, c='red')
        plt.scatter(i, error_tst, c='green')
        plt.draw()

msespng = 'mses.png'
print('Saving mses plot to {}'.format(msespng)) 
plt.title('Errors tr: {0:.3f} val: {1:.3f} tst: {2:.3f}'.format(
    error_train[0,0],
    error_val[0,0],
    error_tst[0,0]))
plt.savefig(msespng, format='png')
plt.close()

sp.print_weights()

plot_error_distributions(train_input,
                         train_target,
                         val_input,
                         val_target,
                         test_input,
                         test_target,
                         sp)
