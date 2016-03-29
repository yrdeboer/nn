import numpy as np
import utils as nn_utils
import plot_utils

# Don't forget to add the parent directory to PYTHONPATH shell variable
from nets.levenberg_marquard_backprop import LevenbergMarquardBackprop

DATA_INPUT_DIR = 'data_output'
f_name_weights = 'weights_val_min.npy'
f_name_data_6_tup = 'data_6_tup.npy'
f_name_s1 = 's1.npy'

weights_val_min = np.load('{}/{}'.format(DATA_INPUT_DIR, f_name_weights))
data_6_tup = np.load('{}/{}'.format(DATA_INPUT_DIR, f_name_data_6_tup))
S1 = np.load('{}/{}'.format(DATA_INPUT_DIR, f_name_s1))

print('S1={}'.format(S1))
print('weights_val_min:\n{}'.format(weights_val_min))
# print('data_6_tup:\n{}'.format(data_6_tup))


DATA_DIR_CR = 'boats/computer_readable_data'
feature_names = np.load('{}/feature_names.npy'.format(DATA_DIR_CR))
builder_names = np.load('{}/builder_names.npy'.format(DATA_DIR_CR))
file_paths = np.load('{}/file_paths.npy'.format(DATA_DIR_CR))

print('feature_names (count={}): {}'.format(len(feature_names), feature_names))
print('builder_names (count={}): {}'.format(len(builder_names), builder_names))

BINCOUNT = 30  # Histograms

R = len(feature_names) + len(builder_names)
S2 = 1

kwargs = dict()
kwargs['input_dim'] = R
kwargs['layer1_neuron_count'] = S1
kwargs['layer2_neuron_count'] = S2

kwargs['layer1_transfer_function'] = nn_utils.logsig
kwargs['layer2_transfer_function'] = nn_utils.purelin

kwargs['layer1_transfer_function_derivative'] = nn_utils.dlogsig
kwargs['layer2_transfer_function_derivative'] = nn_utils.dpurelin

train_inp, train_tar, val_inp, val_tar, test_inp, test_tar = data_6_tup
kwargs['training_data'] = (train_inp, train_tar)

print('Training set size:   {}'.format(train_inp.shape[1]))
print('Validation set size: {}'.format(val_inp.shape[1]))
print('Test set size: {}'.format(test_inp.shape[1]))

sp = LevenbergMarquardBackprop(** kwargs)
W1, b1vec, W2, b2vec = sp.x_to_weights(weights_val_min)
sp.W1 = W1
sp.b1vec = b1vec
sp.W2 = W2
sp.b2vec = b2vec

plot_utils.plot_ols(train_inp,
                    train_tar,
                    test_inp,
                    test_tar,
                    BINCOUNT,
                    1,
                    'errors_ols_reproduced.png')

plot_utils.plot_net_error(train_inp,
                          train_tar,
                          test_inp,
                          test_tar,
                          sp,
                          BINCOUNT,
                          2,
                          'errors_net_reproduced.png')
