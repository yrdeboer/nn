import numpy as np
from nets.levenberg_marquard_backprop import LevenbergMarquardBackprop
import utils as nn_utils

"""
Here we test the Levenberg-Marquard implementation by the numeric example
given by Hagen in P.12.5 on p. 12-42
"""

def get_training_set():

    inp = np.array([[1., 2.]])
    tar = np.array([[1., 2.]])

    return (inp, tar)


kwargs = dict()
kwargs['input_dim'] = 1
kwargs['layer1_neuron_count'] = 1
kwargs['layer2_neuron_count'] = 1

kwargs['layer1_transfer_function'] = lambda n: n*n
kwargs['layer2_transfer_function'] = lambda n: n

kwargs['layer1_transfer_function_derivative'] = lambda n: 2. * n
kwargs['layer2_transfer_function_derivative'] = lambda n: np.ones(n.shape)

# W1.shape = (R, S1)
W1 = np.array([[1.]])

# b1vec shape = (S1, 1)
b1vec = np.array([[0.]])

# W2.shape = (S1, S2)
W2 = np.array([[2.]])

# b2vec.shape = (S2, 1)
b2vec = np.array([[1.]])

kwargs['layer1_initial_weights'] = [W1, b1vec]
kwargs['layer2_initial_weights'] = [W2, b2vec]

(V, y) = get_training_set()
kwargs['training_data'] = (V, y)

# Instantiate backprop with init values
sp = LevenbergMarquardBackprop(** kwargs)
sp.train_step()

print('\nJacobian:\n{}\n'.format(sp.Jac))

# Check values
error_count = 0

GoodJac = np.array([[-4., -4., -1., -1.],[-16., -8., -4., -1.]])

R = V.shape[0]
Q = V.shape[1]

for h in range(sp.S2 * Q):
    for l in range(sp.S1 * R + sp.S1 + sp.S2 * sp.S1 + sp.S2):

        if not GoodJac[h,l] == sp.Jac[h,l]:
            error_count += 1

if error_count == 0:
    print('SUCCESS')
else:
    print('ERROR (error_count = {})'.format(error_count))
