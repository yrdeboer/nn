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


    def get_response(self, P):

        N1 = np.dot(self.W1, P) + self.b1vec
        A1 = self.f1(N1)
        N2 = np.dot(self.W2, A1) + self.b2vec
        return self.f2(N2)
          

    def get_layer(self, l):

        """
        Returns the current layer, given l.
        """

        if l < 2 * self.S1 + self.R:
            return 1
        elif l < 2 * self.S1 + self.R + 2 * self.S2 + self.S1:
            return 2

        raise ValueError('Value l out of bounds: {}'.format(l))

    def get_layer_dim(self, m):

        if m == 0:
            return self.R
        if m == 1:
            return self.S1
        elif m == 2:
            return self.S2

        raise ValueError('Value m out of bounds: {} (neuron count)'.format(m))
            
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

        raise ValueError('Value m out of bounds: {} (layer output)'.format(m))


    def get_jq(self, l):

        Q = self.V.shape[1]
        if Q == 0:
            raise ValueError('Q is 0 (get_jq)')

        return (l / Q, l % Q)
        

    def get_a(self, h, m_minus_1):

        """
        This function returns a^(m-1)_(j,q) as in Hagan eq. (12.43)
        """

        S1R = self.S1 * self.R
        if h < N:

            A_mm1 = get_layer_output(m_minus_1, A1, A2)
            j, q = get_jq(h)
            return A_mm1[j,q]
        
        S1RS1 = S1R + self.S1
        if h < S1RS1:
            return 1.0

        S1RS1S2S1 = S1RS1 + self.S2 * self.S1
        if h < S1RS1S2S1:

            A_mm1 = get_layer_output(m_minus_1, A1, A2)
            j, q = get_jq(h)
            return A_mm1[j,q]

        S1RS1S2S1S2 = S1RS1S2S1 + self.S2
        if h < S1RS1S2S1S2:
            return 1.0

        raise ValueError('Value h out of bounds: {}'.format(h))


    def train_step(self):

        V = self.V                  # Input data shape = (R, Q)
        y = self.y                  # Input targets shape = (S2, Q)
        Q = V.shape[1]             # Input training vector count
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

        Jac = np.zeros((Nrow_j, Ncol_j))

        import ipdb
        ipdb.set_trace()

        for h in range(Nrow_j):
            for l in range(Ncol_j):

                # Calculate s^m_(i,h)
                m = self.get_layer(l)
                n = self.get_layer_dim(m)
                i = h % S2
                q = h / S2
                s = S_aug[m-1][i][q+i]

                # Calculate a^(m-1)_(j,q) as per Hagan eqs. (12.43) and (12.44)
                a = self.get_a(h, m-1, A1, A2)
                
                Jac[h,l] = s * a
                
                

            
        
        
            









            Fdot2 = get_sensitivity_diag(self.S2, self.df2, n2vec)
            s2 = -2 * np.dot(Fdot2, tvec - a2vec)

            Fdot1 = get_sensitivity_diag(self.S1, self.df1, n1vec)
            s1 = np.dot(np.dot(Fdot1, np.transpose(self.W2)), s2)

            self.W2 = self.W2 - self.alpha  * np.dot(s2, np.transpose(a1vec))
            self.b2vec = self.b2vec - self.alpha * s2
            self.W1 = self.W1 - self.alpha * np.dot(s1, np.transpose(pvec))
            self.b1vec = self.b1vec - self.alpha * s1


    def print_weights(self):

        print('\nW1    = {}'.format(self.W1))
        print('b1vec = {}'.format(self.b1vec))
        print('W2    = {}'.format(self.W2))
        print('b2vec = {}'.format(self.b2vec))
