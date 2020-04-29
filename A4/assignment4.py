import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.matlib import repmat
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sns.set_style('darkgrid')


class RNN():
    """A recurrent neural network. """

    def __init__(self, dataset, hidden_state=100, eta=1e-1, sigma=1e-2):
        super().__init__()
        self.text = dataset.text
        self.K = dataset.K
        self.char_to_ind = dataset.char_to_ind
        self.ind_to_char = dataset.ind_to_char
        self.m = hidden_state
        self.eta = eta

        self.U = np.random.normal(0, sigma, size=(hidden_state, self.K))
        self.V = np.random.normal(0, sigma, size=(self.K, hidden_state))
        self.W = np.random.normal(0, sigma, size=(hidden_state, hidden_state))
        self.b = np.zeros((hidden_state, 1))
        self.c = np.zeros((self.K, 1))

    def synthesize(self, hidden_state, input_state, sequence_length):
        """Method used to synthesize text

        Arguments:
            hidden_state {integer} -- The hidden state of the network
            input_state {array} -- Input vector
            sequence_length {integer} -- The length of the sequence of text to synthesize

        Returns:
            [array] -- An array of probabilities for each character at the current time step
        """

        xnext = np.zeros((self.K, 1))
        xnext[input_state] = 1

        h = hidden_state
        text = ''
        for t in range(sequence_length):
            p, _, h, _ = self.evaluate(xnext, h)
            # Draw random sample using the probabilities
            ix = np.random.choice(range(self.K), p=p.flat)
            xnext = np.zeros((self.K, 1))
            xnext[ix] = 1  # 1-hot-encoding
            text += self.ind_to_char[ix]

        return text

    def evaluate(self, input_state, hidden_state):
        """
        Calculates and returns the probability, the intermediary activation values as well as the cost
        and loss of the recurrent neural network.
        """
        a = self.W.dot(hidden_state) + \
            self.U.dot(input_state) + self.b     # input
        h = np.tanh(a)                              # hidden
        o = self.V.dot(h) + self.c                  # output
        # p = np.exp(o) / np.sum(np.exp(o), axis=0)   # softmax

        # TODO: Temp. solution, can probably be removed once grads have been clipped
        p = np.exp(o - np.max(o, axis=0)) / \
                np.exp(o - np.max(o, axis=0)).sum(axis=0)   

        return p, o, h, a

    def compute_gradients(self, inputs, y, hidden_state):

        # forward pass
        n = len(inputs)
        loss = 0
        p, o, h, a, x = {}, {}, {}, {}, {}
        h[0] = hidden_state
        for t in range(1, n):
            xnext = np.zeros((self.K, 1))
            xnext[inputs[t]] = 1
            x[t] = xnext

            p[t], o[t], h[t], a[t] = self.evaluate(xnext, h[t-1])
            loss -= np.log(p[t][y[t]][0])

        # backward pass
        hnext = np.zeros(h[0].shape)

        dU, dV, dW = np.zeros(self.U.shape), np.zeros(
            self.V.shape), np.zeros(self.W.shape)
        db, dc = np.zeros(self.b.shape), np.zeros(self.c.shape)
        for t in reversed(range(1, n)):

            do = -(y[t] - p[t])

            dh = self.V.T.dot(do) + hnext
            da = np.multiply(dh, (1 - h[t]**2))

            dV += np.dot(do, h[t].T)
            dU += np.dot(da, x[t].T)
            dW += np.dot(da, h[t-1].T)

            db += da
            dc += do

            hnext = np.dot(self.W.T, da)

        grads = [dU, dV, dW, db, dc]

        # TODO: Clip gradients after they've been fixed / verified to be correct

        return grads, loss, h[n-1]

    def train(self, epochs, sequence_length):
        """
        Train the network by calculating the weights and bias that minimizes the loss function
        using the Adaptive Gradient Descent approach.
        """

        h = np.zeros((self.m, 1))

        memory_params = [np.zeros(self.U.shape), np.zeros(self.W.shape),
                         np.zeros(self.V.shape), np.zeros(self.b.shape),
                         np.zeros(self.c.shape)]

        e = 0
        for i in tqdm(range(epochs), desc='Training model'):
            if i > len(self.text)-sequence_length-1:
                e = 0
            # Grab a sequence of input characters from the text
            X_chars = self.text[e:e+sequence_length]
            Y_chars = self.text[e+1:e+sequence_length+1]

            # Convert to one-hot encoding
            X = np.asarray([self.char_to_ind[char] for char in X_chars])
            Y = np.asarray([self.char_to_ind[char] for char in Y_chars])

            grads, loss, h = self.compute_gradients(X, Y, h)

            if i == 0:
                smooth_loss = loss

            smooth_loss *= 0.999 + 0.001 * loss

            # print some synthesized text here
            if i % 500 == 0:
                text = self.synthesize(h, X[0], 200)
                print(text)
                print('Smooth loss:', smooth_loss)

            # do adagrad update of parameters
            # TODO: implement a neater/cleaner solution
            layers = [self.U, self.V, self.W, self.b, self.c]
            for idx, (layer, grad) in enumerate(zip(layers, grads)):
                memory_params[idx] = np.power(grad, 2)
                layer -= self.eta / \
                    np.sqrt(memory_params[idx] + np.finfo(float).eps) * grad

            e += 1


class TextSource:
    """
    Class used to load and preprocess the text into a usable data set.
    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.get_text()
        self.text
        self.K
        self.char_to_ind
        self.ind_to_char

    def get_text(self):

        with open(self.filename) as f:
            text = f.read()

        self.text = text

        # Find unique number of characters
        characters = "".join(sorted(set("".join(text))))

        self.chars = list(characters)
        self.K = len(characters)

        # Create a map from between a character and its one-hot encoding
        self.char_to_ind = dict((char, idx)
                                for idx, char in enumerate(characters))
        self.ind_to_char = dict((idx, char)
                                for idx, char in enumerate(characters))


""" Functions for model comparison and presentation of the results. """

# TODO: Rework this method to work with RNN
@DeprecationWarning
def compute_grads_num(network, X, Y, h):
    """
    Compute the gradient using the Central Difference approximation.
    """
    grad_weights, grad_bias = [], []
    grad_gamma, grad_beta = [], []
    for idx, layer in enumerate(network.layers):
        W = layer.W
        b = layer.b

        gradW = np.matlib.zeros(W.shape)
        gradb = np.matlib.zeros(b.shape)

        for i in range(len(b)):
            network.layers[idx].b[i] -= h
            _, _, c1, _, _ = network.forward_pass(X, Y)
            network.layers[idx].b[i] += 2*h
            _, _, c2, _, _ = network.forward_pass(X, Y)
            gradb[i] = (c2-c1) / (2*h)
            network.layers[idx].b[i] -= h

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                network.layers[idx].W[i, j] -= h
                _, _, c1, _, _ = network.forward_pass(X, Y)
                network.layers[idx].W[i, j] += 2*h
                _, _, c2, _, _ = network.forward_pass(X, Y)
                gradW[i, j] = (c2-c1) / (2*h)
                network.layers[idx].W[i, j] -= h

        grad_weights.append(gradW)
        grad_bias.append(gradb)

    return grad_weights, grad_bias


def check_gradient(tol=1e-5):
    #TODO: Implement
    pass


text = TextSource('data/goblet_book.txt')

network = RNN(text)
network.train(1000, 25)
