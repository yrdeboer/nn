import sys
import numpy as np
import matplotlib.pyplot as plt
import utils as nn_utils

# Don't forget to add the parent directory to PYTHONPATH shell variable
from nets.levenberg_marquard_backprop import LevenbergMarquardBackprop

DATA_DIR_CR = 'computer_readable_data'

feature_names = np.load('{}/feature_names.npy'.format(DATA_DIR_CR))
builder_names = np.load('{}/builder_names.npy'.format(DATA_DIR_CR))
file_paths = np.load('{}/file_paths.npy'.format(DATA_DIR_CR))


def get_data_sets(fixed=False):

    """
    This function fetches the training data from disk and uses
    a utils function to randomly select a training, validation
    and test set.
    """

    input_data = np.load('{}/input_data.npy'.format(DATA_DIR_CR))
    target_data = np.load('{}/target_data.npy'.format(DATA_DIR_CR))

    return nn_utils.all_data_to_data_sets(input_data,
                                          target_data,
                                          0.7,
                                          0.15,
                                          fixed)


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
S1 = 20
S2 = 1

kwargs = dict()
kwargs['input_dim'] = R
kwargs['layer1_neuron_count'] = S1
kwargs['layer2_neuron_count'] = S2

print('S1 = {}'.format(S1))

kwargs['layer1_transfer_function'] = nn_utils.logsig
kwargs['layer2_transfer_function'] = nn_utils.purelin

kwargs['layer1_transfer_function_derivative'] = nn_utils.dlogsig
kwargs['layer2_transfer_function_derivative'] = nn_utils.dpurelin

train_inp, train_tar, val_inp, val_tar, test_inp, test_tar  = get_data_sets()
kwargs['training_data'] = (train_inp, train_tar)

print('Training set size:   {}'.format(train_inp.shape[1]))
print('Validation set size: {}'.format(val_inp.shape[1]))
print('Test set size: {}'.format(test_inp.shape[1]))

# Instantiate backprop with init values
sp = LevenbergMarquardBackprop(** kwargs)

iteration_count = 50
logspace = np.logspace(1., np.log(iteration_count), 100)
plot_points = [int(i) for i in list(logspace)]

# Interactive plotting of the mean squared error
plt.ion()

ax1 = plt.subplot(2, 1, 1)
ax1.set_title(r'RMS: training (blue) validation (green)')
plt.axis([1, 10. * iteration_count, 1e-1, 1.])
plt.yscale('log')
plt.xscale('log')

ax2 = plt.subplot(2, 1, 2)
ax2.set_title(r'Histogram $y-\hat{y_{test}}$ at min rms$_{val}$')

rms_val_min = np.finfo('float64').max
weights_val_min = sp.weights_to_x()
updated_val_min = True

for i in range(1, iteration_count):

    converged = sp.train_step()

    rms_train = nn_utils.get_rms_error(train_inp, train_tar, sp)
    rms_val = nn_utils.get_rms_error(val_inp, val_tar, sp)

    if rms_val < rms_val_min:
        print('Updated rms_val_min to {} in iteration {}'.format(
            rms_val_min,
            i))
        rms_val_min = rms_val
        weights_val_min = sp.weights_to_x()
        updated_val_min = True

    if i < 5 or i == iteration_count-1 or i in plot_points or converged or updated_val_min:

        print(
            'Iteration: {:5} rms_train: {:.8f} rms_val={:.8f} g_norm: {:.6f} converged: {}'.format(
                i, rms_train, rms_val, sp.g_norm, converged))
        sys.stdout.flush()

        # Training vs. validation rms (error)a
        plt.subplot(2, 1, 1)
        plt.scatter(i, rms_train, c='b')
        plt.scatter(i, rms_val, c='g')

        if updated_val_min:
            plt.subplot(2, 1, 2)
            plt.cla()
            ax2.set_title(r'Histogram $y-\hat{y}_{test}$ at min rms$_{val}=%.2f$' % rms_val_min)

            yhat = sp.get_response(test_inp)[0]
            diff = test_tar[0] - yhat

            plt.hist(diff, 50)
            updated_val_min = False

        plt.show()
        plt.pause(.0000001)

    if converged:
        break

plt.savefig('hagan_sinus_early_stopping.png')
