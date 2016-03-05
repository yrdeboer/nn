import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_P = '/Users/ytsen/data/hagan_nnd/CaseStudyData/ball_p.txt'
DATA_FILE_T = '/Users/ytsen/data/hagan_nnd/CaseStudyData/ball_t.txt'


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


def get_normalised_data():

    """
    Returns the normalised training data in a 2-tuple as follows:

    (V, y) with V.shape = and y.shape = 
    
    """

    tvec = np.loadtxt('/Users/ytsen/data/hagan_nnd/CaseStudyData/ball_t.txt')
    P = np.loadtxt('/Users/ytsen/data/hagan_nnd/CaseStudyData/ball_p.txt')
    
    v1vec = normalise_vector(P[0])
    v2vec =normalise_vector(P[1])
    y = normalise_vector(tvec)

    V = np.zeros((2, len(v1vec)))
    V[0] = v1vec
    V[1] = v2vec

    return (V, y)


def plot_data():

    # Figure 18.4 from book, plots v1 and v2 (67 columns of V) in the y-range: [-1, 1]

    (V, y) = get_normalised_data()

    plt.plot(y, V[0], 'ro', y, V[1], 'bo')
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


def logsig(x):

    return 1. / (1. + np.exp(-x))

def get_sse(W1, b1vec, W2, b2vec, V, y):

    """
    The sum squared error is calculated
    from the current network weights and
    a data set of inputs p and targets t.
    """

    yhat = get_network_response(W1, b1vec, W2, b2vec, V, y)

    return np.sum(np.power(y - yhat, 2))


def backprop():

    # Constants
    R = 2   # Input dimension
    S1 = 10  # Neuron count layer 1
    S2 = 1  # Neuron count layer 2
    alpha = 0.1

    iteration_count = 500000
    sse_spacing = 1001

    # Initial weights
    mult_factor = 0.5
    W1 = (np.random.random_sample((S1, R)) - 0.5).reshape(S1, R) * mult_factor
    b1vec = (np.random.random_sample((S1)) - 0.5).reshape((S1, 1)) * mult_factor
    W2 = (np.random.random_sample(S1) - 0.5).reshape(1, S1) * mult_factor
    b2vec = (np.random.random_sample(S2).reshape((S2, 1)) - 0.5) * mult_factor

    V, y = get_normalised_data()

    ssevecx = np.arange((1. + int(iteration_count / sse_spacing)))
    ssevecy = np.zeros((1. + int(iteration_count / sse_spacing)))
    k = 0

    print('len(trange) = {}'.format(len(y)))

    # Iterate
    for i in range(iteration_count):
        for j in range(len(y)):
            pvec = V[:, [j]]
            tvec = np.array([[y[j]]])

            a1vec = logsig(np.dot(W1, pvec) + b1vec)
        
            # print('a1vec = {} shape = {}'.format(a1vec, a1vec.shape))
        
            # Propagate second layer
            a2vec = np.dot(W2, a1vec) + b2vec
        
            # print('vec = {} shape = {}'.format(a2vec, a2vec.shape))
        
            # Backprop starting at last layer
            Fdot2 = np.array([[1.0]])

            s2 = -2 * np.dot(Fdot2, tvec - a2vec)
        
            # print('s2 = {} shape = {}'.format(s2, s2.shape))

            dig = np.zeros(a1vec.shape[0])
            for i2 in range(a1vec.shape[0]):
                dig[i2] = (1. - a1vec[i2,0]) * a1vec[i2,0]
            Fdot1 = np.diag(dig)

            s1 = np.dot(np.dot(Fdot1, np.transpose(W2)) ,s2)
        
            # print('s1 = {} shape = {}'.format(s1, s1.shape))

            # Done, update weights
            W2 = W2 - alpha  * s2 * np.transpose(a1vec)
            b2vec = b2vec - alpha * s2
        
            W1 = W1 - alpha * s1 * np.transpose(pvec)
            b1vec = b1vec - alpha * s1

        if i > 0 and i % sse_spacing == 0:
            sse = get_sse(W1, b1vec, W2, b2vec, V, y)
            ssevecy[k] = sse
            print('sse[{}] = {}'.format(k, sse))
            k += 1


        if i > 0 and i % 1000 == 0:
                print('Iteration {}'.format(i))


    print('\nUpdated weightes and biases:')
    print('W1 = {}'.format(W1))
    print('b1vec = {}'.format(b1vec))
    print('W2 = {}'.format(W2))
    print('b2vec = {}'.format(b2vec))

    # Also take lowest sse
    ssevecy[k] = get_sse(W1, b1vec, W2, b2vec, V, y)
    k += 1

    ssevecx = np.resize(ssevecx, k)
    ssevecy = np.resize(ssevecy, k)
    print('Final SSEs = {}'.format(ssevecy[-1]))

    # Plot stats
    plot(W1, b1vec, W2, b2vec, V, y, ssevecx, ssevecy)

backprop()

# plot_data()
