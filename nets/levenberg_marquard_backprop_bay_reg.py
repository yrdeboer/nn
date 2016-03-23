import numpy as np
import utils as nn_utils
from utils import print_dbg


np.set_printoptions(threshold=np.nan)


class LevenbergMarquardBackprop():

    """
    Note: we are implementing the Bayes regularisation
          scheme in this class.
          For now, still testing ...
    """

    def __init__(self, * args, ** kwargs):

        self.R = kwargs.get('input_dim', None)
        self.S1 = kwargs.get('layer1_neuron_count', None)
        self.S2 = kwargs.get('layer2_neuron_count', None)
        self.f1 = kwargs.get('layer1_transfer_function', None)
        self.f2 = kwargs.get('layer2_transfer_function', None)
        self.df1 = kwargs.get('layer1_transfer_function_derivative', None)
        self.df2 = kwargs.get('layer2_transfer_function_derivative', None)
        self.mu = kwargs.get('mu', 0.01)
        self.theta = kwargs.get('theta', 10.0)

        training_data = kwargs.get('training_data', None)
        if training_data:
            self.V = training_data[0]
            self.y = training_data[1]
        else:
            print_dbg('Warning: no training data')

        W1inits = kwargs.get('layer1_initial_weights', None)
        if W1inits:
            self.W1 = W1inits[0]
            self.b1vec = W1inits[1]
        else:
            mult_factor = .5
            self.W1 = (np.random.random_sample(
                (self.S1, self.R)) - 0.5) * mult_factor
            self.b1vec = (np.random.random_sample(
                (self.S1, 1)) - 0.5) * mult_factor

        W2inits = kwargs.get('layer2_initial_weights', None)
        if W2inits:
            self.W2 = W2inits[0]
            self.b2vec = W2inits[1]
        else:
            self.W2 = (np.random.random_sample(
                (self.S2, self.S1)) - 0.5) * mult_factor
            self.b2vec = (np.random.random_sample(
                (self.S2, 1)) - 0.5) * mult_factor

        # Initialise Bayesian regularisation parameters
        # Note, the objective function as per
        # Hagan eq. (13.4) p. 13-8:
        #
        #    F(x) = beta * Ed + alpha * Ew

        self.N = self.V.shape[1] * self.S1
        self.n = self.S1 * self.R + self.S1 + self.S2 * self.S1 + self.S2

        Ed = self.get_sse_error()
        Ew = np.sum(np.power(self.weights_to_x(), 2))
        self.gamma = self.n
        self.alpha = 0.5 * self.gamma / Ew
        self.beta = 0.5 * (self.N - self.gamma) / Ed
        self.Fx = self.beta * Ed + self.alpha * Ew

        print('Init gammma={:.4f} alpha={:.4f} beta={:.4f} Fx={:.4f}'.format(
            self.gamma,
            self.alpha,
            self.beta,
            self.Fx))

    def get_response(self,
                     P,
                     W1=np.array([]),
                     b1vec=np.array([]),
                     W2=np.array([]),
                     b2vec=np.array([])):

        """
        This function returns the response of the net.

        If any of the weights or biases are given, then
        those are used.
        """

        if not W1.any():
            W1 = self.W1
        if not b1vec.any():
            b1vec = self.b1vec
        if not W2.any():
            W2 = self.W2
        if not b2vec.any():
            b2vec = self.b2vec

        N1 = np.dot(W1, P) + b1vec
        A1 = self.f1(N1)
        N2 = np.dot(W2, A1) + b2vec
        return self.f2(N2)

    def get_layer(self, l):

        """
        Returns the current layer, given l.
        """

        Nmax_1 = self.S1 * self.R + self.S1
        Nmax_2 = Nmax_1 + self.S2 * self.S1 + self.S1

        if l < Nmax_1:
            return 1
        elif l < Nmax_2:
            return 2

        raise IndexError('Value l out of bounds: {}'.format(l))

    def get_layer_dim(self, m):

        if m == 0:
            return self.R
        if m == 1:
            return self.S1
        elif m == 2:
            return self.S2

        raise IndexError('Value m out of bounds: {} (neuron count)'.format(m))

    def get_jq(self, h):

        """
        j and q are needed to calculate a^(m-1)_(j,q)

        h runs over Q batches of size S2

        q is the batch index corresponding to h and
        j is the offset within that batch
        """

        return (h % self.S2, int(h / self.S2))

    def get_v_from_error(self, ERR):

        """
        This function takes the error ERR, which is a 2-dimensional
        numpy array of shape (S2, Q).

        v is constructed as per Hagan eq. (12.35), which is a numpy
        array of shape (S2 * Q, 1)
        """

        Q = ERR.shape[1]
        v = np.zeros((self.S2 * Q, 1))
        for q in range(Q):
            v[q * self.S2: (q+1) * self.S2] = ERR[:, [q]]

        return v

    def get_layer_output(self, m, A1, A2):

        """
        This function returns the layer output, for layer m.
        For the "0'th layer" (m = 0) the input is returned.
        """

        if m == 0:
            return self.V
        elif m == 1:
            return A1
        elif m == 2:
            return A2

        raise IndexError('Value m out of bounds: {} (layer output)'.format(m))

    def get_mia(self, h, l, q, A1, A2):

        """
        This function returns a 3-tuple, with a,i and m.

        Args:
          h:   Row index of the Jacobian in [0, Q*S2]
          l:   Column index of the Jacobian
               in [0, R*S1 (W1) + R (b1vec) + S1*S2 (W2) + S2 (b2vec)]
          q:   Referring to the q'th training point
          A1:  Net output of layer 1
          A2:  Net output of layer 2

        Returns:
           m:  Layer index in [1, 2]
           i:  Referring to the i't relevant neuron in layer m,
               depending on l
           a:  The element a^(m-1)_(j,q) as per Hagan eq. (12.43)
        """

        S1R = self.S1 * self.R
        if l < S1R:  # W1

            m = 1
            i = int(l / self.R)

            j = l % self.R
            a = self.get_layer_output(m-1, A1, A2)[j, q]

            # print_dbg(
            #     '    l={} i={} j={} q={} a={} (W1)'.format(l, i, j, q, a))

            return (m, i, a)

        S1RS1 = S1R + self.S1
        if l < S1RS1:  # b1

            m = 1
            i = (l - S1R)
            a = 1.0

            # print_dbg('    l={} i={} a={} (b1vec)'.format(l, i, a))

            return (m, i, a)

        S1RS1S2S1 = S1RS1 + self.S2 * self.S1
        if l < S1RS1S2S1:  # W2

            m = 2
            # i = (l - S1RS1) % self.S2  # Wrong (?) as per 2016 mar 18
            i = int((l - S1RS1) / self.S1)

            j = (l - S1RS1) % self.S1
            a = self.get_layer_output(m-1, A1, A2)[j, q]

            # print_dbg(
            #     '    l={} i={} j={} q={} a={} (W2)'.format(l, i, j, q, a))

            return (m, i, a)

        S1RS1S2S1S2 = S1RS1S2S1 + self.S2
        if l < S1RS1S2S1S2:  # b2

            m = 2
            i = (l - S1RS1S2S1) % self.S2
            a = 1.0

            # print_dbg('    l={} i={} a={} (b2vec)'.format(l, i, a))

            return (m, i, a)

        raise IndexError('Value l out of bounds: {}'.format(l))

    def weights_to_x(self,
                     W1=np.array([]),
                     b1vec=np.array([]),
                     W2=np.array([]),
                     b2vec=np.array([])):

        """
        This function returns the vector x as in Hagan eq. (12.36).
        However, here we return it as a numpy column vector.

        Any argument weight that is None, is taken from self by
        the same name.

        Returns:
          Vector x with shape (N, 1)
        """

        R = self.V.shape[0]              # Input vector size

        w1_cnt = self.S1 * R
        b1_cnt = self.S1
        w2_cnt = self.S2 * self.S1
        b2_cnt = self.S2

        w1b1_cnt = w1_cnt + b1_cnt
        w1b1w2_cnt = w1b1_cnt + w2_cnt
        w1b1w2b2_cnt = w1b1w2_cnt + b2_cnt

        x = np.zeros(w1b1w2b2_cnt)

        if not W1.any():
            W1 = self.W1
        if not b1vec.any():
            b1vec = self.b1vec
        if not W2.any():
            W2 = self.W2
        if not b2vec.any():
            b2vec = self.b2vec

        x[0:w1_cnt] = W1.reshape((w1_cnt))
        x[w1_cnt:w1b1_cnt] = b1vec.reshape((b1_cnt))
        x[w1b1_cnt:w1b1w2_cnt] = W2.reshape((w2_cnt))
        x[w1b1w2_cnt:w1b1w2b2_cnt] = b2vec.reshape((b2_cnt))

        return x.reshape((w1b1w2b2_cnt, 1))

    def x_to_weights(self, x):

        """
        This function takes an x (as in Hagan eq. (12.36))
        and returns the corresponding weight matrices and
        vectors.
        """

        R = self.V.shape[0]

        w1_cnt = self.S1 * R
        b1_cnt = self.S1
        w2_cnt = self.S2 * self.S1
        b2_cnt = self.S2

        w1b1_cnt = w1_cnt + b1_cnt
        w1b1w2_cnt = w1b1_cnt + w2_cnt
        w1b1w2b2_cnt = w1b1w2_cnt + b2_cnt

        W1 = x[0:w1_cnt].reshape((self.S1, R))
        b1vec = x[w1_cnt:w1b1_cnt].reshape((self.S1, 1))
        W2 = x[w1b1_cnt:w1b1w2_cnt].reshape((self.S2, self.S1))
        b2vec = x[w1b1w2_cnt:w1b1w2b2_cnt].reshape((self.S2, 1))

        return (W1, b1vec, W2, b2vec)

    def get_sse_error(self,
                      W1=np.array([]),
                      b1vec=np.array([]),
                      W2=np.array([]),
                      b2vec=np.array([])):

        """
        This function returns the sum
        squared error.
        """

        Q = self.V.shape[1]

        sse = 0.
        for i in range(Q):

            pvec = self.V[:, [i]]
            yhat = self.get_response(pvec, W1, b1vec, W2, b2vec)

            diff = self.y[:, [i]] - yhat
            sse += np.sum(diff * diff)

        return sse

    def get_rms_error(self,
                      W1=np.array([]),
                      b1vec=np.array([]),
                      W2=np.array([]),
                      b2vec=np.array([])):

        """
        This function return the root mean squared
        error.
        """

        Q = self.V.shape[1]
        sse = self.get_sse_error(W1, b1vec, W2, b2vec)
        return np.sqrt(sse / float(Q + self.S1))

    def train_step(self):

        """
        This function performs 1 iteration using Levenberg-Marquard.
        It returns True if the algorithm has converged,
        otherwise False.
        """

        V = self.V                  # Input data shape = (R, Q)
        y = self.y                  # Input targets shape = (S2, Q)
        Q = V.shape[1]              # Input training vector count
        R = V.shape[0]              # Input vector size
        W1 = self.W1
        b1vec = self.b1vec
        W2 = self.W2
        b2vec = self.b2vec
        S1 = self.S1
        S2 = self.S2
        f1 = self.f1
        df1 = self.df1
        df2 = self.df2

        print_dbg('  R={} Q={} S1={} S2={}'.format(R, Q, S1, S2))

        # Algorithm from Hagan p. 12-25
        #
        # 1a. Compute network in- and out-puts N2 and A2
        N1 = np.dot(W1, V) + b1vec
        A1 = f1(N1)
        N2 = np.dot(W2, A1) + b2vec
        A2 = self.get_response(V)

        # print_dbg('N1:\n{}'.format(N1))
        # print_dbg('A1:\n{}'.format(A1))
        # print_dbg('N2:\n{}'.format(N2))
        # print_dbg('A2:\n{}'.format(A2))
        # print_dbg('y:\n{}'.format(self.y))

        # 1b. Calculate the errors
        ERR = y - A2
        # self.sse = self.get_sse_error_br()  # Updates self.alpha and self.beta

        print_dbg('ERR:\n{}'.format(ERR))
        # print_dbg('  self.sse init to: {}'.format(self.sse))

        v_cur = self.get_v_from_error(ERR)
        x_cur = self.weights_to_x()

        # 2a. Initialise and compute the sensitivies
        #     using Eq. (12.46) and Eq. (12.47) and
        #     also the augmented Marquard sensitiviy
        S_aug_2 = np.zeros((S2, S2 * Q))
        S_aug_1 = np.zeros((S1, S2 * Q))

        for q in range(Q):

            F2q = nn_utils.get_sensitivity_diag(df2, N2[:, [q]])
            S2q = -F2q
            S_aug_2[:, range(q * S2, (q+1) * S2)] = S2q

            F1q = nn_utils.get_sensitivity_diag(df1, N1[:, [q]])
            S1q = np.dot(
                np.dot(F1q, np.transpose(W2)),
                S2q)
            S_aug_1[:, range(q * S2, (q+1) * S2)] = S1q

        S_augs = (S_aug_1, S_aug_2)
        print_dbg('  S_augs[0].shape = {}\n  S_augs[0] =\n{}'.format(
            S_augs[0].shape,
            S_augs[0]))
        print_dbg('  S_augs[1].shape = {}\n  S_augs[1] =\n{}'.format(
            S_augs[1].shape,
            S_augs[1]))

        # 2b. Compute the elements of the Jacobian using
        #     eqs. (12.43) and (12.44)

        Nrow_j = S2 * Q
        Ncol_j = S1 * R + S1 + S2 * S1 + S2

        print_dbg('  Nrow_j = {} (h) Ncol_j = {} (l)'.format(Nrow_j, Ncol_j))

        self.Jac = np.zeros((Nrow_j, Ncol_j))

        J = self.Jac
        JT = np.transpose(J)

        for h in range(Nrow_j):

            # print_dbg('  h = {}'.format(h))

            for l in range(Ncol_j):

                q = int(h / self.S2)
                (m, i, a) = self.get_mia(h, l, q, A1, A2)

                s = S_augs[m-1][i][h]

                # print_dbg(
                #     '      S_augs[{}][{}][{}] = {}\n'.format(m-1, i, h, s))

                self.Jac[h, l] = s*a

        J = self.Jac
        JT = np.transpose(J)
        JTJ = np.dot(JT, J)

        # print_dbg('JT = {}'.format(JT))
        # print_dbg('v_cur = {}'.format(v_cur))

        g = 2. * np.dot(JT, v_cur)
        self.g_norm = np.linalg.norm(g)

        k = 0

        # print_dbg('  J={} '.format(J))
        # print_dbg('  x={} '.format(np.transpose(x_cur)))

        while True:

            k += 1

            print_dbg(
                '  k={}: Before multiply: k={} mu={}'.format(
                    k, k, self.mu))

            if self.mu < np.finfo('float64').eps:
                print(
                    '  Raising mu from {} to {}'.format(
                        self.mu,
                        self.theta*self.mu))
                self.mu = self.mu * self.theta

            # 3. Solve eq. (12.32) to obtain dx_k
            det = JTJ + self.mu * np.identity(Ncol_j)
            det_inv = np.linalg.inv(det)

            # print_dbg('det_inv = {}'.format(det_inv))

            jtv = np.dot(np.transpose(J), v_cur)

            # Convert the error to a columns vector as per Hagan eq. 12.35
            dx = -np.dot(det_inv, jtv)

            # 4a. Peek Fx
            x_peek = x_cur + dx
            W1, b1vec, W2, b2vec = self.x_to_weights(x_peek)

            Ed_peek = self.get_sse_error(W1, b1vec, W2, b2vec)
            Ew_peek = np.sum(np.power(x_peek, 2))

            Fx_peek = self.beta * Ed_peek + self.alpha * Ew_peek

            print_dbg('mu={} self.Fx={:.3f} Fx_peek={:.3f}'.format(
                self.mu,
                self.Fx,
                Fx_peek))

            # 4b. Update
            if Fx_peek < self.Fx:

                self.Fx = Fx_peek
                self.mu /= self.theta
                self.W1 = W1
                self.b1vec = b1vec
                self.W2 = W2
                self.b2vec = b2vec

                self.update_bay_reg_params(JTJ, Ed_peek, Ew_peek, Fx_peek)

                break

            else:
                self.mu *= self.theta

            if self.mu > 1e10:
                print_dbg(
                    'Converged, breaking out, k = {} mu = {}'.format(
                        k,
                        self.mu))

                return True

        return False

    def update_bay_reg_params(self, JTJ, Ed, Ew, Fx_peek):

        # Compute eff. nr. of wgts using old alpha
        H = 2. * (self.beta * JTJ + self.alpha * np.identity(
            JTJ.shape[0]))
        H_inv = np.linalg.inv(H)
        self.gamma = self.n - 2. * self.alpha * np.trace(H_inv)

        # Only now we recompute alpha and beta
        self.alpha = 0.5 * self.gamma / Ew
        self.beta = 0.5 * (self.N - self.gamma) / Ed
        self.Fx = self.beta * Ed + self.alpha * Ew

        self.dFx = self.Fx - Fx_peek
        

        # print('Updated gammma={:.4f} alpha={:.4f} beta={:.4f} Fx={:.4f} (dFx={:.4f})'.format(
        #     self.gamma,
        #     self.alpha,
        #     self.beta,
        #     self.Fx,
        #     np.abs(self.Fx - Fx_peek)))

    def print_weights(self):

        print('\nW1    = {}'.format(self.W1))
        print('b1vec = {}'.format(self.b1vec))
        print('W2    = {}'.format(self.W2))
        print('b2vec = {}'.format(self.b2vec))
