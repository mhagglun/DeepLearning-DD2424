import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage import util
from numpy.matlib import repmat
from tqdm import tqdm
from tabulate import tabulate

sns.set_style('darkgrid')


class Network():

    def __init__(self, input_dim, output_dim, hidden_layers=[50], l=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = self.initiate_layers(hidden_layers)
        self.l = l

    class Layer:
        """
        The inner class object used to construct the layers of the network.
        Contains the weights and biases for a layer of the network.
        """

        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.W = np.random.normal(
                0, 1 / np.sqrt(input_dim), (output_dim, input_dim))
            self.b = np.zeros((output_dim, 1))


    def initiate_layers(self, layer_size):
        """
        Sets the layers of the network.
        Takes a list that contains the number of nodes in each layer and
        initiates the layers of the network.
        """
        num_nodes = np.r_[self.input_dim, layer_size, self.output_dim]
        layers = []
        for i in range(len(num_nodes)-1):
            layers.append(self.Layer(num_nodes[i], num_nodes[i+1]))

        return layers

    def forward_pass(self, X, Y):
        """
        Calculates and returns the probability, the intermediary activations and the cost
        of the multi-layer neural network.
        """
        h = X
        activations = [h]
        loss = 0
        for layer in self.layers:
            s = np.dot(layer.W, h) + layer.b
            h = np.maximum(0, s)
            activations.append(h)
            loss += self.l * np.sum(np.power(layer.W, 2))

        P = np.exp(s) / np.sum(np.exp(s), axis=0)

        loss += np.sum(-np.log(np.multiply(Y, P).sum(axis=0))) / X.shape[1]

        return P, activations, loss

    def backward_pass(self, X, Y, P, activations, eta):

        G = P - Y
        for i, layer in reversed(list(enumerate(self.layers))):
            x = activations[i]
            N = X.shape[1]
            gradb = G * np.ones((N, 1)) / N
            gradW = np.dot(G, x.T) / N + 2 * self.l * layer.W

            # Update weights and bias of current layer
            self.layers[i].W -= eta * gradW
            self.layers[i].b -= eta * gradb

            # Propagate the gradient backwards
            G = layer.W.T * G
            G[x == 0] = 0

    def compute_accuracy(self, X, Y):
        """
        Calculates the prediction accuracy of the network on the data X 
        with weights W and bias b.
        """
        y = np.argmax(Y, axis=0)
        P, _, _ = self.forward_pass(X, Y)
        P = np.argmax(P, axis=0)

        return 100*np.sum(P == y)/y.shape[0]

    def train(self, training_data, validation_data, eta, n_epochs, n_batches, noise=None):
        """
        Train the network by calculating the weights and bias that minimizes the cost function
        using the Mini-Batch Gradient Descent approach.
        """

        X, Y, y = training_data['X'], training_data['Y'], training_data['y']
        Xval, Yval, yval = validation_data['X'], validation_data['Y'], validation_data['y']

        cost_train, accuracy_train, cost_val, accuracy_val = np.zeros((n_epochs+1, 1)), np.zeros(
            (n_epochs+1, 1)), np.zeros((n_epochs+1, 1)), np.zeros((n_epochs+1, 1))

        _, _, cost_train[0, :] = self.forward_pass(X, Y)
        _, _, cost_val[0, :] = self.forward_pass(Xval, Yval)
        accuracy_train[0, :] = self.compute_accuracy(X, Y)
        accuracy_val[0, :] = self.compute_accuracy(Xval, Yval)

        for i in tqdm(range(1, n_epochs+1), desc="Model training progress: "):
            for j in range(1, int(X.shape[1]/n_batches)+1):
                jstart = (j-1) * n_batches + 1
                jend = j * n_batches

                Xbatch = X[:, jstart:jend]
                Ybatch = Y[:, jstart:jend]

                if noise is not None:
                    Xbatch = util.random_noise(
                        Xbatch, mode=noise, seed=None, clip=True)

                P, activations, loss = self.forward_pass(Xbatch, Ybatch)

                self.backward_pass(Xbatch, Ybatch, P, activations, eta)
            _, _, cost_train[i, :] = self.forward_pass(X, Y)
            _, _, cost_val[i, :] = self.forward_pass(Xval, Yval)
            accuracy_train[i, :] = self.compute_accuracy(X, Y)
            accuracy_val[i, :] = self.compute_accuracy(Xval, Yval)

        return cost_train.flatten(), cost_val.flatten(), accuracy_train.flatten(), accuracy_val.flatten()


class CIFAR:
    """
    Class used to load and preprocess the CIFAR10 data set.
    """

    def __init__(self):
        self.input_dim = 32*32*3
        self.num_labels = 10

    def get_labels(self):
        """
        Returns the (named) labels of the data.
        """
        with open('cifar-10-batches-py/batches.meta', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            labels = [x.decode('ascii') for x in data[b'label_names']]
        return labels

    def get_batch(self, name):
        """
        Loads a batch of data into a dictionary.
            - X         d-by-n matrix containing n images. Each column corresponds to an image 
                        with dimensionality d (32*32*3 in this case).

            - Y         Matrix of size K-by-n. The one-hot representation of the true label for the
                        corresponding image in X (columnwise).

            - y         Vector of size n. Each element i corresponds to the label of the image in
                        column i of X.
        """
        with open('cifar-10-batches-py/'+name, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

            X = np.matrix(data[b'data'], dtype=float).T/255
            y = np.array(data[b'labels'])
            Y = (np.eye(self.num_labels)[y]).T

            batch = {
                'batch_name': data[b'batch_label'],
                'X': X,
                'Y': Y,
                'y': y,
            }
        return batch

    def normalize(self, data, mean, std):
        """
        Method used to normalize the data.
        """
        data -= repmat(mean, 1, np.size(data, axis=1))
        data /= repmat(std, 1, np.size(data, axis=1))
        return data


""" Functions for model comparison and presentation of the results. """


def compute_gradients(network, X, Y):

    P, activations, loss = network.forward_pass(X, Y)
    G = P - Y
    grad_weights = []
    grad_bias = []
    for i, layer in reversed(list(enumerate(network.layers))):
        x = activations[i]
        N = X.shape[1]
        gradb = G * np.ones((N, 1)) / N
        gradW = np.dot(G, x.T) / N + 2 * network.l * layer.W

        grad_weights.append(gradW)
        grad_bias.append(gradb)

        # Propagate the gradient backwards
        G = layer.W.T * G
        G[x == 0] = 0

    return grad_weights, grad_bias


def compute_grads_num(network, X, Y, h):
    """
    A numerical approach based on Finite Difference method to calculate the gradients.
    """
    grad_weights = []
    grad_bias = []
    for idx, layer in enumerate(network.layers):
        print('num on layer', idx)
        W = layer.W
        b = layer.b

        gradW = np.matlib.zeros(W.shape)
        gradb = np.matlib.zeros(b.shape)

        _, _, cost = network.forward_pass(X, Y)

        for i in tqdm(range(b.shape[0])):
            network.layers[idx].b[i] += h
            _, _, c2 = network.forward_pass(X, Y)
            gradb[i] = (c2 - cost) / h
            network.layers[idx].b[i] -= h

        for i in tqdm(range(W.shape[0])):
            for j in (range(W.shape[1])):
                network.layers[idx].W[i, j] += h
                _, _, c2 = network.forward_pass(X, Y)
                gradW[i, j] = (c2 - cost) / h
                network.layers[idx].W[i, j] -= h

        grad_weights.append(gradW)
        grad_bias.append(gradb)

    return grad_weights, grad_bias


def compute_grads_num_slow(network, X, Y, h):
    """
    Compute the gradient using the Central Difference approximation.
    Slightly slower but more accurate than the finite difference approach.
    """
    grad_weights = []
    grad_bias = []
    for idx, layer in enumerate(network.layers):
        print('slow num on layer', idx)
        W = layer.W
        b = layer.b

        gradW = np.matlib.zeros(W.shape)
        gradb = np.matlib.zeros(b.shape)

        for i in range(len(b)):
            network.layers[idx].b[i] -= h
            _, _, c1 = network.forward_pass(X, Y)
            network.layers[idx].b[i] += 2*h
            _, _, c2 = network.forward_pass(X, Y)
            gradb[i] = (c2-c1) / (2*h)
            network.layers[idx].b[i] -= h

        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                network.layers[idx].W[i, j] -= h
                _, _, c1 = network.forward_pass(X, Y)
                network.layers[idx].W[i, j] -= 2*h
                _, _, c2 = network.forward_pass(X, Y)
                gradW[i, j] = (c2-c1) / (2*h)
                network.layers[idx].W[i, j] += h

        grad_weights.append(gradW)
        grad_bias.append(gradb)

    return grad_weights, grad_bias


def check_gradient(batchSize=20, tol=1e-5):
    """
    Method used to check that the simple gradient is accurate by comparing with
    a more computer intensive but accurate method.
    """
    np.random.seed(400)
    dataset = CIFAR()
    training_data = dataset.get_batch('data_batch_1')

    X = training_data['X']
    Y = training_data['Y']

    # Calculate mean and standard deviation of training data
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)

    # Normalize the data w.r.t training data
    X = dataset.normalize(X, mean, std)

    network = Network(dataset.input_dim, dataset.num_labels)

    gradW, gradb = compute_gradients(network,
                                     X[:, :batchSize], Y[:, :batchSize])

    num_gradW, num_gradb = compute_grads_num(
        network, X[:, :batchSize], Y[:, :batchSize], tol)

    slow_num_gradW, slow_num_gradb = compute_grads_num_slow(
        network, X[:, :batchSize], Y[:, :batchSize], tol)

    table = []
    for i, (gradW, ngradW, slow_gradW) in enumerate(zip(gradW, num_gradW, slow_num_gradW)):
        rel_error = np.sum(abs(ngradW - gradW)) / np.maximum(tol,
                                                             np.sum(abs(ngradW)) + np.sum(abs(gradW)))
        rel_error_slow = np.sum(abs(slow_gradW - gradW)) / np.maximum(tol,
                                                                      np.sum(abs(slow_gradW)) + np.sum(abs(gradW)))

        table.append([i+1, rel_error, rel_error_slow])

    table = tabulate(table, headers=['Layer #', 'Relative error Analytical vs Numerical',
                                     'Relative error Analytical vs Slow Numerical'], tablefmt='github')
    print(table)


def plot_performance(cost_train, cost_val, accuracy_train, accuracy_val, filename=None):
    """
    Plots the cost and accuracy against the epochs for the two data sets.
    """
    plt.figure(figsize=(12.0, 5.0))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(accuracy_train)), accuracy_train,
             label='Training set')
    plt.plot(np.arange(len(accuracy_val)), accuracy_val,
             label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(cost_train)), cost_train, label='Training set')
    plt.plot(np.arange(len(cost_val)), cost_val, label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if filename is not None:
        plt.savefig('performance_{}.png'.format(filename))
        plt.clf()
        plt.close()


def model_summary(dataset, training_data, validation_data, parameters):
    """
    Generates a summary of the network performance based on the given data and parameters.
    """

    # Initialize the network
    network = Network(dataset.input_dim, dataset.num_labels, layer_size=[50])

    # Check initial accuracy
    init_acc_train = network.compute_accuracy(
        training_data['X'], training_data['Y'])
    init_acc_val = network.compute_accuracy(
        validation_data['X'], validation_data['Y'])

    # Format string to ouput model stats
    model_parameters = 'Model parameters: \n' + \
        '   lambda:  \t{}\n'.format(parameters['l']) + \
        '   eta:  \t{}\n'.format(parameters['eta']) + \
        '   n_epochs: \t{}\n'.format(parameters['n_epochs']) + \
        '   n_batches: \t{}\n'.format(parameters['n_batches'])

    print(model_parameters)

    # Train network
    cost_train, cost_val, accuracy_train, accuracy_val = network.train(
        training_data, validation_data, parameters['eta'], parameters['n_epochs'], parameters['n_batches'])

    model_performance = 'Training data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(init_acc_train) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(accuracy_train[-1]) + \
                        '   cost (final): \t\t{:.2f}\n'.format(cost_train[-1]) + \
                        'Validation data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(init_acc_val) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(accuracy_val[-1]) + \
                        '   cost (final): \t\t{:.2f}\n'.format(cost_val[-1])

    print(model_performance)

    plot_performance(cost_train, cost_val, accuracy_train, accuracy_val)
    plt.show()


def report(l=0.0, eta=0.01, n_epochs=40, n_batches=100):
    """
    Method that loads and preprocesses the data and then trains the model for the given parameters in order to generate
    a summary of the model performance.
    """
    np.random.seed(400)
    dataset = CIFAR()
    training_data = dataset.get_batch('data_batch_1')
    validation_data = dataset.get_batch('data_batch_2')

    mean = np.mean(training_data['X'], axis=1)
    std = np.std(training_data['X'], axis=1)

    training_data['X'] = dataset.normalize(training_data['X'], mean, std)
    validation_data['X'] = dataset.normalize(validation_data['X'], mean, std)

    parameters = {'l': l,  'eta': eta,
                  'n_epochs': n_epochs, 'n_batches': n_batches}

    model_summary(dataset, training_data, validation_data, parameters)

# TODO: Rework the numerical gradient computations
# check_gradient()


report()
 