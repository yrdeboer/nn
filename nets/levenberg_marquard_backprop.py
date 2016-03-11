import numpy as np


def get_sensitivity_diag(df, n):
    return np.diag(np.transpose(df(n))[0])


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


    def get_sse(ERR):

        """
        This calculates the Sum Squared Error.

        NOTE: The input must be of shape (1, N)
        """

        if not len(ERR.shape) == 2:
            raise ValueError('Invalid shape for ERR (axis count)')
        
        if not ERR.shape[0] == 1:
            raise ValueError('Invalid shape for ERR (dim)')

        return np.sum(np.power(ERR, 2.))
            

    def train_step(self):

        # Input_count Q (Hagan notation)
        Q == self.V.shape[1]

        # Algorithm from Hagan p. 12-25
        # 
        # 1a. Compute network in- and out-puts N2 and A2
        N1 = np.dot(self.W1, self.V) + self.b1vec
        A1 = self.f1(N1)
        N2 = np.dot(self.W2, A1) + self.b2vec
        A2 = self.get_response(self.V)

        # 1b. Calculate the errors
        ERR = self.y - A2

        # 1c. Compute sum squared errors using Eq. (12.34)
        sse = self.get_sse(ERR)

        # 2a. Initialise and compute the sensitivies
        #     using Eq. (12.46) and Eq. (12.47) and
        #     also the augmented Marquard sensitiviy
        S_aug_2 = np.zeros(self.S2, self.S2 * Q)
        S_aug_1 = np.zeros(self.S1, self.S1 * Q)
        for q in range(Q):

            F2q = get_sensitivity_diag(self.df2, N2[q])
            S2q = -F2q
            S_aug_2[:, range(q * self.S2, (q+1) * self.S2)]

            F1q = get_sensitivity_diag(self.df1, N1[q])
            S1q = np.dot(
                np.dot(F1q, np.transpose(self.W2)),
                S2q)
            S_aug_1[:, range(q * self.S1, (q+1) * self.S1)]

        F_aug = (F1q, F2q)

        # 2b. Compute the elements of the Jacobian using
        #     eqs. (12.43) and (12.44)

        def get_layer(l):

            """
            Returns the current layer, given l.
            """

            if l < 2 * self.S1 + self.R:
                return 1
            elif l < 2 * self.S1 + self.R + 2 * self.S2 + self.S1:
                return 2

            raise ValueError('Value l out of bounds: {}'.format(l))

        def get_layer_dim(m):

            if m == 0:
                return self.R
            if m == 1:
                return self.S1
            elif m == 2:
                return self.S2

            raise ValueError('Value m out of bounds: {} (neuron count)'.format(m))
                
        def get_layer_output(m, A1, A2):

            if m == 0:
                return self.V
            elif m == 1:
                return A1
            elif m == 2:
                return A2

            raise ValueError('Value m out of bounds: {} (layer output)'.format(m))

        def get_a(h):

            S1R = self.S1 * self.R
            if h < N:
                row = h / self.S1
                col = h % self.S1
                return self.W1[row, col]
            
            S1RS1 = S1R + self.S1
            if h < S1RS1:
                i = h - S1R
                row = i % self.S1
                return self.b1vec[row, 0]

            S1RS1S2S1 = S1RS1 + self.S2 * self.S1
            if h < S1RS1S2S1:
                i = h - S1RS1
                

            S1RS1S2S1S2 = S1RS1S2S1 + self.S2
            if h < S1RS1S2S1S2:
                return h - S1RS1S2S1

            raise ValueError('Value h out of bounds: {}'.format(h))
        
        Nrow_j = self.S2 * Q
        Ncol_j =  self.S1 * self.R + self.S1 + self.S2 * self.S1 + self.S2

        for h in range(Nrow_j):
            for l in range(Ncol_j):

                # Calculate s^m_(i,h)
                m = get_layer(l)
                n = get_layer_dim(m)
                i = h % self.S2
                q = h / self.S2
                s = F_aug[m-1][i][q+i]

                # Calculate a^(m-1)_(j,q)
                A_m = get_layer_output(m-1, A1, A2)
                a = get_a(h)
                

            
        
        
            









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
