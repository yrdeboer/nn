import numpy as np
import sys


DEBUG = False


def print_dbg(* args):

    if not DEBUG:
        return

    s = ''
    for arg in args:
        s += arg

    print(s)
    sys.stdout.flush()


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
        mse += np.sum(diff * diff)

    return np.sqrt(mse / float(Ncol + sp.S1))


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


def all_data_to_data_sets(all_data_inp,
                          all_data_tar,
                          frac_train,
                          frac_val):

    """
    This function splits all training and target data into randomised
    data sets used for training, validation and testing.

    Args:
      all_data_inp: The training input data of shape (Nrow, Ncol), where Nrow is
                    the input vector column size and Ncol the number of
                    input data training vectors.
      all_data_tar: The training data target vectors of shape (S, Ncol),
                    where S usually is the output layer neuron count
                    of the (assumed) network.
      frac_train:   The fraction of the data that should become training data
      frac_val:     The fraction of the data that should become target data

      Note: The fraction of the data that becomes testing data should be > 0.
            equal to 1. - frac_train - frac_val.

    Returns:
      A 6-tuple with 3 data sets, training, validation and test data input and targets.
    """

    (Nrow, Ncol) = all_data_inp.shape
    tr_inp = np.zeros((Nrow, Ncol))
    tr_tar = np.zeros((1, Ncol))
    val_inp = np.zeros((Nrow, Ncol))
    val_tar = np.zeros((1, Ncol))
    tst_inp = np.zeros((Nrow, Ncol))
    tst_tar = np.zeros((1, Ncol))

    ran_sample = np.random.random_sample(Ncol)

    # Distribute data randomly
    i_tr = i_val = i_tst = 0
    for i in range(Ncol):

        inp = all_data_inp[:, [i]]
        tar = all_data_tar[:, [i]]
        ran = ran_sample[i]

        if ran < frac_train:
            tr_inp[:, [i_tr]] = inp
            tr_tar[:, [i_tr]] = tar
            i_tr += 1

        elif ran < frac_train + frac_val:

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

    tot = np.sum(all_data_inp[0])
    tr = np.sum(tr_inp[0])
    val = np.sum(val_inp[0])
    tst = np.sum(tst_inp[0])
    if tot - tr - val - tst > 1e-13:
        raise ValueError('Sum of lengths not well')

    tot = np.sum(all_data_tar[0])
    tr = np.sum(tr_tar[0])
    val = np.sum(val_tar[0])
    tst = np.sum(tst_tar[0])
    diff = tot - tr - val - tst
    if diff > 1e-11:
        raise ValueError('Diff of prices not well: {}'.format(diff))

    return tr_inp, tr_tar, val_inp, val_tar, tst_inp, tst_tar


def get_beta_in_ols(Y, X):

    """
    This function returns the vector b, which minismises
    the error function ||Y-b^T*X||^2.

    Y has shape (1, N_data)
    X has shape (N_features, N_data)
    beta has shape (N_features, 1)
    """
    XYT = np.dot(X, np.transpose(Y))
    XXT = np.dot(X, np.transpose(X))
    XXT_inv = np.linalg.inv(XXT)
    return np.dot(XXT_inv, XYT)
