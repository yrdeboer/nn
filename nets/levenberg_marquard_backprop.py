import numpy as np
import utils as nn_utils


class LevenbergMarquardBackprop():

    def __init__(self, * args, ** kwargs):

        self.R = kwargs.get('input_dim', None)
        self.S1 = kwargs.get('layer1_neuron_count', None)
        self.S2 = kwargs.get('layer2_neuron_count', None)
        self.f1 = kwargs.get('layer1_transfer_function', None)
        self.f2 = kwargs.get('layer2_transfer_function', None)
        self.df1 = kwargs.get('layer1_transfer_function_derivative', None)
        self.df2 = kwargs.get('layer2_transfer_function_derivative', None)

        training_data = kwargs.get('training_data', None)
        if training_data:
            self.V = training_data[0]
            self.y = training_data[1]
        else:
            print('Warning: no training data')

        W1inits = kwargs.get('layer1_initial_weights', None)
        if W1inits:
            self.W1 = W1inits[0]
            self.b1vec = W1inits[1]
        else:
            mult_factor = .5
            self.W1 = (np.random.random_sample(
                (self.S1, self.R)) - 0.5).reshape(self.S1, self.R) * mult_factor
            self.b1vec = (np.random.random_sample(
                (self.S1)) - 0.5).reshape((self.S1, 1)) * mult_factor

        W2inits = kwargs.get('layer2_initial_weights', None)
        if W2inits:
            self.W2 = W2inits[0]
            self.b2vec = W2inits[1]
        else:
            self.W2 = (np.random.random_sample(
                self.S1) - 0.5).reshape(1, self.S1) * mult_factor
            self.b2vec = (np.random.random_sample(
                self.S2).reshape((self.S2, 1)) - 0.5) * mult_factor

        self.mu = 0.01
        self.theta = 10.


    def get_response(self, P, W1=None, b1vec=None, W2=None, b2vec=None):

        """
        This function returns the response of the net.

        If any of the weights or biases are given, then
        those are used.
        """

        if not W1:
            W1 = self.W1
        if not b1vec:
            b1vec = self.b1vec
        if not W2:
            W2 = self.W2
        if not b2vec:
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


    def get_jq(self, h):

        """
        j and q are needed to calculate a^(m-1)_(j,q)

        h runs over Q batches of size S2

        q is the batch index corresponding to h and
        j is the offset within that batch
        """

        return (h % self.S2, h / self.S2)
        

    def get_a(self, l, h, m_minus_1, A1, A2):

        """
        This function returns a^(m-1)_(j,q) as in Hagan eq. (12.43)
        """

        S1R = self.S1 * self.R
        if l < S1R:

            A_mm1 = self.get_layer_output(m_minus_1, A1, A2)
            j, q = self.get_jq(h)
            return A_mm1[j,q]
        
        S1RS1 = S1R + self.S1
        if l < S1RS1:
            return 1.0

        S1RS1S2S1 = S1RS1 + self.S2 * self.S1
        if l < S1RS1S2S1:

            A_mm1 = self.get_layer_output(m_minus_1, A1, A2)
            j, q = self.get_jq(h)
            return A_mm1[j,q]

        S1RS1S2S1S2 = S1RS1S2S1 + self.S2
        if l < S1RS1S2S1S2:
            return 1.0

        raise IndexError('Value l out of bounds: {}'.format(l))


    def weights_to_x(self, W1=None, b1vec=None, W2=None, b2vec=None):

        """
        This function returns the vector x as in Hagan eq. (12.36)

        Any argument weight that is None, is taken from self by
        the same name.
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

        if not W1:
            W1 = self.W1
        if not b1vec:
            b1vec = self.b1vec
        if not W2:
            W2 = self.W2
        if not b2vec:
            b2vec = self.b2vec

        x[0:w1_cnt] = W1.reshape((w1_cnt))
        x[w1_cnt:w1b1_cnt] = b1vec.reshape((b1_cnt))
        x[w1b1_cnt:w1b1w2_cnt] = W2.reshape((w2_cnt))
        x[w1b1w2_cnt:w1b1w2b2_cnt] = b2vec.reshape((b2_cnt))

        return x


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
        b1vec = x[w1_cnt:w1b1_cnt].reshape((self.S1))
        W2 = x[w1b1_cnt:w1b1w2_cnt].reshape((self.S2, self.S1))
        b2vec = x[w1b1w2_cnt:w1b1w2b2_cnt].reshape((self.S2))

        return (W1, b1vec, W2, b2vec)



    def get_rms_error(self, W1=None, b1vec=None, W2=None, b2vec=None):

        Q = self.V.shape[1]

        mse = 0.
        for i in range(Q):
    
            pvec = self.V[:, [i]]
            yhat = self.get_response(pvec, W1, b1vec, W2, b2vec)
    
            diff = self.y[:, [i]] - yhat
            mse += diff * diff
    
        return mse / float(Q)
    

    def train_step(self):

        V = self.V                  # Input data shape = (R, Q)
        y = self.y                  # Input targets shape = (S2, Q)
        Q = V.shape[1]              # Input training vector count
        R = V.shape[0]              # Input vector size
        W1 = self.W1              
        b1vec = self.b1vec
        W2 = self.W2
        b2vec = self.b2vec
        S1 = self.S1
        S2 = self.S1
        f1 = self.f1
        f2 = self.f2
        df1 = self.df1
        df2 = self.df2

        # Algorithm from Hagan p. 12-25
        # 
        # 1a. Compute network in- and out-puts N2 and A2
        N1 = np.dot(W1, V) + b1vec
        A1 = f1(N1)
        N2 = np.dot(W2, A1) + b2vec
        A2 = self.get_response(V)

        # 1b. Calculate the errors
        ERR = y - A2
        rms = self.get_rms_error()

        # 2a. Initialise and compute the sensitivies
        #     using Eq. (12.46) and Eq. (12.47) and
        #     also the augmented Marquard sensitiviy
        S_aug_2 = np.zeros((S2, S2 * Q))
        S_aug_1 = np.zeros((S1, S1 * Q))
        for q in range(Q):

            F2q = nn_utils.get_sensitivity_diag(df2, N2[:, [q]])
            S2q = -F2q
            S_aug_2[:, range(q * S2, (q+1) * S2)] = S2q

            F1q = nn_utils.get_sensitivity_diag(df1, N1[:, [q]])
            S1q = np.dot(
                np.dot(F1q, np.transpose(W2)),
                S2q)
            S_aug_1[:, range(q * S1, (q+1) * S1)] = S1q

        S_augs = (S_aug_1, S_aug_2)

        # 2b. Compute the elements of the Jacobian using
        #     eqs. (12.43) and (12.44)

        Nrow_j = S2 * Q
        Ncol_j =  S1 * R + S1 + S2 * S1 + S2

        self.Jac = np.zeros((Nrow_j, Ncol_j))

        for h in range(Nrow_j):
            for l in range(Ncol_j):

                # Calculate s^m_(i,h)
                m = self.get_layer(l)
                n = self.get_layer_dim(m)
                i = h % S2
                q = h / S2
                s = S_augs[m-1][i][q+i]

                # Calculate a^(m-1)_(j,q) as per Hagan eqs. (12.43) and (12.44)
                a = self.get_a(l, h, m-1, A1, A2)
                
                self.Jac[h,l] = s * a

                J = self.Jac
                JT = np.transpose(J)

        while True:

            # 3. Solve eq. (12.32) to obtain dx_k
            det = np.dot(JT, J) + self.mu * np.identity(Ncol_j)
            det_inv = np.linalg.inv(det)
            det_inv_jt = np.dot(det_inv, np.transpose(J))

            import ipdb
            ipdb.set_trace()

            dx_k = -np.dot(det_inv_jt, ERR)
    
            # 4a. Recompute rms
            xk = self.get_x()
            x_peek = xk + dx_k.reshape(xk.shape)
    
            W1, b1vec, W2, b2vec = self.x_to_weights(x_peek)
            rms_peek = self.get_rms_error(W1, b1vec, W2, b2vec)
    
            # 4b. Update
            if rms_peek < rms:

                rms = rms_peek
                self.mu /= self.theta
                self.W1 = W1
                self.b1vec = b1vec
                self.W2 = W2
                self.b2vec = b2vec

                break

            else:
                self.mu *= self.theta
    

    def print_weights(self):

        print('\nW1    = {}'.format(self.W1))
        print('b1vec = {}'.format(self.b1vec))
        print('W2    = {}'.format(self.W2))
        print('b2vec = {}'.format(self.b2vec))
