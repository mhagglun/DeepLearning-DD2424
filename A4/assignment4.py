import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.matlib import repmat
from tabulate import tabulate
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sns.set_style('darkgrid')


class RNN():
    """A recurrent neural network. """

    def __init__(self, dataset=None, hidden_nodes=100, sigma=1e-2, config=None):
        """Constructor

        Keyword Arguments:
            dataset {Source} -- The (Source) object used to represent the text file (default: {None})
            hidden_nodes {int} -- The number of nodes in the hidden layer (default: {100})
            sigma {[type]} -- [description] (default: {1e-2})
        """
        super().__init__()
        if config is not None:
            self.load_config(config)
        else:
            self.dataset = dataset
            self.encode = dataset.encode
            self.text = dataset.text
            self.K = dataset.K
            self.char_to_ind = dataset.char_to_ind
            self.ind_to_char = dataset.ind_to_char
            self.m = hidden_nodes
            self.smooth_loss = 0
            self.parameters = {'U': np.random.normal(0, sigma, size=(hidden_nodes, self.K)),
                               'V': np.random.normal(0, sigma, size=(self.K, hidden_nodes)),
                               'W': np.random.normal(0, sigma, size=(hidden_nodes, hidden_nodes)),
                               'b': np.zeros((hidden_nodes)),
                               'c': np.zeros((self.K))}

    def load_config(self, filename):
        """Load a saved config into the model

        Arguments:
            filename {str} -- The path to the saved config file
        """
        config = np.load(filename, allow_pickle=True)
        self.m = config.f.hidden_nodes
        self.smooth_loss = config.f.smooth_loss
        self.dataset = Source(str(config.f.source))
        self.encode = self.dataset.encode
        self.text = self.dataset.text
        self.K = self.dataset.K
        self.char_to_ind = self.dataset.char_to_ind
        self.ind_to_char = self.dataset.ind_to_char
        self.parameters = {'U': config['U'], 'V': config['V'],
                           'W': config['W'], 'b': config['b'], 'c': config['c']}

    def save(self, filename):
        """Method for saving the current model configuration

        Arguments:
            filename {str} -- The path to save the file to
        """
        np.savez_compressed(
            filename,
            **{name: self.parameters[name] for name in self.parameters},
            source=self.dataset.filename,
            hidden_nodes=self.m,
            smooth_loss=self.smooth_loss
        )

    def synthesize(self, hidden_state, input_state, sequence_length):
        """Method used to synthesize text

        Arguments:
            hidden_state {integer} -- The hidden state of the network
            input_state {array} -- Input vector
            sequence_length {integer} -- The length of the sequence of text to synthesize

        Returns:
            [array] -- An array of probabilities for each character at the current time step
        """
        xnext = input_state
        h = hidden_state
        text = ''
        for _ in range(sequence_length):
            p, h = self.predict(xnext, h)
            # Draw random sample using the probabilities
            ix = np.random.choice(range(self.K), p=p.flat)
            xnext = np.zeros((self.K))
            xnext[ix] = 1  # 1-hot-encoding
            text += self.ind_to_char[ix]

        return text

    def predict(self, input_state, hidden_state):
        """Calculates and returns the probability and the intermediary activation values of the recurrent neural network.

        Arguments:
            input_state {ndarray} -- The one-hot encoded input character
            hidden_state {ndarray} -- The hidden layer of the RNN

        Returns:
            p {ndarray} -- The probabilities for the next character
            h {ndarray} -- The intermediary activation values
        """
        a = self.parameters['W'].dot(hidden_state) + \
            self.parameters['U'].dot(input_state) + \
            self.parameters['b']
        h = np.tanh(a)
        o = self.parameters['V'].dot(
            h) + self.parameters['c']
        p = np.exp(o) / np.sum(np.exp(o), axis=0)

        return p, h

    def compute_gradients(self, inputs, targets, hidden_state):
        """Computes the gradients of the parameters of the network by performing a forward and backward pass.


        Arguments:
            inputs {ndarray} -- The one-hot encoded input sequence
            targets {ndarray} -- The target one-hot encoded sequence
            hidden_state {ndarray} -- The current hidden state of the network

        Returns:
            grads [dict] -- The gradients of the network parameters stored as a dictionary
            loss [float] -- The current loss
            h [ndarray] -- The updated state of the hidden layer
        """
        # forward pass
        n = inputs.shape[1]
        p, h = np.zeros((self.K, n)), np.zeros((self.m, n+1))

        h[:, 0] = hidden_state
        for t in range(n):
            p[:, t], h[:, t+1] = self.predict(inputs[:, t], h[:, t])

        logarg = np.multiply(targets, p)
        loss = -sum(np.log(np.multiply(targets, p).sum(axis=0)))

        # backward pass
        grads = {'U': np.zeros(self.parameters['U'].shape), 'V': np.zeros(self.parameters['V'].shape),
                 'W': np.zeros(self.parameters['W'].shape), 'b': np.zeros(self.parameters['b'].shape),
                 'c': np.zeros(self.parameters['c'].shape)}

        hprevious = np.zeros((self.m))
        do = -(targets - p).T

        grads['V'] = np.dot(do.T, h[:, 1:].T)
        grads['c'] = do.sum(axis=0)

        dh, da = np.zeros((self.m, n)), np.zeros((self.m, n))

        dh[:, -1] = np.dot(do.T[:, -1], self.parameters['V'])
        da[:, -1] = np.multiply(dh[:, -1], (1 - h[:, -1]**2))

        for t in reversed(range(n-1)):
            dh[:, t] = np.dot(do[t, :], self.parameters['V']) + \
                np.dot(da[:, t+1], self.parameters['W'])
            da[:, t] = np.multiply(dh[:, t], (1 - h[:, t+1]**2))

        grads['U'] = np.dot(da, inputs.T)
        grads['W'] = np.dot(da, h[:, :-1].T)
        grads['b'] = da.sum(axis=1)

        return grads, loss, h[:, -1]

    def train(self, epochs, eta, sequence_length, verbose=True):
        """Train the network by calculating the weights and bias that minimizes the loss function
        using the Adaptive Gradient Descent approach.

        Arguments:
            epochs {integer} -- The number epochs (passes over the training data) to train for
            sequence_length {integer} -- The length of each sequence of text passed

        Keyword Arguments:
            verbose {bool} -- If true, display network metrics, synthesized text and training progress (default: {True})
        """
        memory_params = {'U': np.zeros(self.parameters['U'].shape), 'V': np.zeros(self.parameters['V'].shape),
                         'W': np.zeros(self.parameters['W'].shape), 'b': np.zeros(self.parameters['b'].shape),
                         'c': np.zeros(self.parameters['c'].shape)}

        iterations = int(epochs * len(self.dataset.text) / sequence_length)
        smooth_loss = np.zeros((iterations))

        e = 0
        h = np.zeros((self.m))
        print(len(self.dataset.text))
        for i in tqdm(range(iterations), desc='Training model', disable=(verbose)):
            if e > len(self.dataset.text)-sequence_length-1:
                e = 0
                h = np.zeros((self.m))

            # Reset hidden state after new tweet
            # if e % 10*sequence_length == 0:
            #     h = np.zeros((self.m))

            # Grab a sequence of input characters from the text
            input_chars = self.dataset.text[e:e+sequence_length]
            target_chars = self.dataset.text[e+1:e+sequence_length+1]

            # Convert to one-hot encoding
            input_labels = self.encode(input_chars)
            target_labels = self.encode(target_chars)

            grads, loss, h = self.compute_gradients(
                input_labels, target_labels, h)

            if i == 0:
                self.smooth_loss = loss

            self.smooth_loss = self.smooth_loss * 0.999 + 0.001 * loss
            smooth_loss[i] = self.smooth_loss

            # print some synthesized text
            if i % 10000 == 0 and verbose:
                text = self.synthesize(h, input_labels[:, 0], 200)
                print('\nIteration: ', i, '\tSmooth loss:', self.smooth_loss)
                print('Synthesized text: \n', text)

            # clip grads if they become too large
            for key in grads:
                grads[key] = np.clip(grads[key], -5, 5)

            # Adagrad update of the parameters
            for key in self.parameters:
                memory_params[key] += np.power(grads[key], 2)
                self.parameters[key] -= eta / \
                    np.sqrt(memory_params[key] +
                            np.finfo(float).eps) * grads[key]

            e += sequence_length

        for xc in range(int(len(self.dataset.text)/sequence_length), int(epochs * len(self.dataset.text) / sequence_length), int(len(self.dataset.text)/sequence_length)):
            plt.axvline(x=xc, color='orange')
        plt.plot(smooth_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()


class Source:
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
        """Reads the input text file, determine the number of unique characters in the source text and
        create mappings between characters and indices for the one-hot representation
        """
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

    def encode(self, input_text):
        """Generates one-hot encodings of input and target text

        Arguments:
            input_text {string} -- The input text to convert to one-hot encoding

        Returns:
            one_hot_encoding {ndarray} -- The one-hot matrix representation of the input text
        """
        indices = [self.char_to_ind[char] for char in input_text]
        one_hot_encoding = (np.eye(self.K)[indices]).T

        return one_hot_encoding


""" Functions for model comparison and presentation of the results. """


def compute_grads_num(network, X, Y, h):
    """Compute the gradient using the Central Difference approximation.

    Arguments:
        network {RNN} -- The RNN object
        X {ndarray} -- The input sequence
        Y {ndarray} -- The target sequence
        h {ndarray} -- The hidden state of the network

    Returns:
        {dict} -- The gradients of the network parameters stored in a dictionary
    """
    n = len(network.parameters)
    hprev = np.zeros((network.m))

    grads = {'U': np.zeros(network.parameters['U'].shape), 'V': np.zeros(network.parameters['V'].shape),
             'W': np.zeros(network.parameters['W'].shape), 'b': np.zeros(network.parameters['b'].shape),
             'c': np.zeros(network.parameters['c'].shape)}

    for key in network.parameters:
        for i in range(network.parameters[key].shape[0]):
            if network.parameters[key].ndim == 1:
                network.parameters[key][i] -= h
                _, l1, _ = network.compute_gradients(X, Y, hprev)
                network.parameters[key][i] += 2*h
                _, l2, _ = network.compute_gradients(X, Y, hprev)
                grads[key][i] = (l2-l1)/(2*h)
                network.parameters[key][i] -= h
            else:
                for j in range(network.parameters[key].shape[1]):
                    network.parameters[key][i, j] -= h
                    _, l1, _ = network.compute_gradients(X, Y, hprev)
                    network.parameters[key][i, j] += 2*h
                    _, l2, _ = network.compute_gradients(X, Y, hprev)
                    grads[key][i, j] = (l2-l1)/(2*h)
                    network.parameters[key][i, j] -= h

    return grads


def check_gradient(m=5, sigma=1e-2, sequence_length=25, tol=1e-4):
    """Verify the implemented analytical gradients using numerically obtained gradients

    Keyword Arguments:
        m {integer} -- The number of nodes in the hidden layer (default: {5})
        sigma {float} -- The standard deviation used to initiate the normally distributed weights (default: {1e-2})
        sequence_length {integer} -- [description] (default: {25})
        tol {float} -- Tolerance used for calculating the relative error (default: {1e-4})
    """
    text = Source('data/goblet_book.txt')

    network = RNN(text, hidden_nodes=m, sigma=sigma)

    input_chars = network.text[0:sequence_length]
    target_chars = network.text[1:sequence_length+1]

    input_labels = text.encode(input_chars)
    target_labels = text.encode(target_chars)

    hprev = np.zeros((network.m))

    grads, _, _ = network.compute_gradients(input_labels, target_labels, hprev)
    num_grads = compute_grads_num(network, input_labels, target_labels, tol)

    table = []
    for key in grads:
        rel_error = np.sum(abs(grads[key] - num_grads[key])) / np.maximum(tol,
                                                                          np.sum(abs(grads[key])) + np.sum(abs(num_grads[key])))
        table.append(['d'+key, rel_error])

    table = tabulate(
        table, headers=['Gradient', 'Relative error'], tablefmt='github')
    print(table)


def main(source='data/goblet_book.txt', config=None, epochs=10, eta=1e-1, sequence_length=25, filename=None, verbose=True):

    if config is not None:
        network = RNN(config=config)

    else:
        text = Source(source)
        network = RNN(text)
        network.train(epochs, eta, sequence_length, verbose=verbose)

        if filename is not None:
            network.save(filename)

    return network


# Example use

# check_gradient()

# network = main(source='data/raw_tweets.txt', epochs=20, sequence_length=140, filename='weights/trump_weights_20.npz')
# network = main(source='data/goblet_book.txt', epochs=50, eta=1e-1, sequence_length=25, filename='weights/goblet_weights_nepochs_50.npz')

# network = main(config='weights/goblet_weights.npz')
# network = main(config='weights/trump_weights.npz')


# print('Smooth loss of the network: ', network.smooth_loss)
# text = network.synthesize(np.zeros((network.m)), x, 1000)
# print(text)
