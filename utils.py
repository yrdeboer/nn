import numpy as np


def logsig(x):

    return 1. / (1. + np.exp(-x))


def dlogsig(x):
    
    e = np.exp(-x)
    d = (1. + e)
    return e / (d * d)


def tansig(x):
    return np.tanh(x)


def dtansig(x):
    
    tanh = np.tanh(x)
    return 1. - tanh*tanh


def purelin(x):
    return x


def dpurelin(x):
    return 1.


def get_rms_error(dat_inp, dat_tar, sp):

    """
    This function obtains the error, given
    the neural net and train/target data pairs.

    The mean squared error is used.

    Args:
      dat_inp:  Input data, columns are input vectors
      dat_tar: Target data, columns are target values
      sp:        Instance of a net, should have
                 a function "get_response"

    Returns:
      A float, the rms error.
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

        if i == 5:
            print('pvec = {} pnet = {} diff = {}'.format(pvec, pnet, diff))


    return mse / float(Ncol)
