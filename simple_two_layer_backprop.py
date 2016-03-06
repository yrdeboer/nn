import numpy as np


def get_sensitivity_diag(S, df, a):

    dig = np.zeros(S)
    for i in range(S):
        dig[i] = df(a[i])
    return np.diag(dig)


class SimpleTwoLayerBackprop():

    def __init__(self, * args, ** kwargs):

        self.R = kwargs.get('input_dim', None)
        self.S1 = kwargs.get('layer1_neuron_count', None)
        self.S2 = kwargs.get('layer2_neuron_count', None)
        self.alpha = kwargs.get('learning_rate', None)
        self.f1 = kwargs.get('layer1_transfer_function', None)
        self.f2 = kwargs.get('layer2_transfer_function', None)
        self.df1 = kwargs.get('layer1_transfer_function_derivative', None)
        self.df2 = kwargs.get('layer2_transfer_function_derivative', None)

        W1inits = kwargs.get('layer1_initial_weights', (None, None))
        self.W1 = W1inits[0]
        self.b1vec = W1inits[1]

        W2inits = kwargs.get('layer2_initial_weights', (None, None))
        self.W2 = W2inits[0]
        self.b2vec = W2inits[1]

        training_data = kwargs.get('training_data', None)

        self.V = training_data[0]
        self.y = training_data[1]


    def get_response(self, pvec):

        n1vec = np.dot(self.W1, pvec) + self.b1vec
        a1vec = self.f1(n1vec)
        n2vec = np.dot(self.W2, a1vec) + self.b2vec
        return self.f2(n2vec)


    def train_step(self):
    
        for j in range(len(self.y)):

            pvec = self.V[:, [j]]
            tvec = self.y[:, [j]]

            n1vec = np.dot(self.W1, pvec) + self.b1vec
            a1vec = self.f1(n1vec)

            n2vec = np.dot(self.W2, a1vec) + self.b2vec
            a2vec = self.f2(n2vec)
            
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
