import numpy as np


DEBUG = False


def print_dbg(* args):

    if not DEBUG:
        return 

    s = ''
    for arg in args:
        s += arg

    print(s)


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
    return np.ones(x.shape)


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

    return mse / float(Ncol)


def get_sse_from_error(ERR):

    """
    This calculates the Sum Squared Error.

    NOTE: The input must be of shape (1, N)
    """

    if not len(ERR.shape) == 2:
        raise ValueError('Invalid shape for ERR (axis count)')
    
    if not ERR.shape[0] == 1:
        raise ValueError('Invalid shape for ERR (dim)')

    return np.sum(np.power(ERR, 2.))


def get_sensitivity_diag(df, n):

    """
    This function calculates the diagonal matrix of the layer sensitivities
    as per Hagan eq. (11.34).

    Args:
      df: The derivatie function of the layer transfer function.
          Must be able to work with arrays.
      n:  The net input

    Returns:
      A diagonal matrix with dimensions the same as
      the row count of n and the diagonal entries D_ii
      equal to df(n_i).
    """

    return np.diag(np.transpose(df(n))[0])

