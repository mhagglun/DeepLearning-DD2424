import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import preprocessing
from numpy.matlib import repmat

sns.set_style('darkgrid')


class CIFAR:
    def __init__(self):
        self.input_dim = 32 * 32 * 3
        self.num_labels = 10
        self.getLabels()
        self.batches = {}

    def getLabels(self):
        """
        Returns the (named) labels of the data
        """
        with open('cifar-10-batches-py/batches.meta', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            labels = [x.decode('ascii') for x in data[b'label_names']]
        return labels

    def getBatch(self, name):
        """
        Loads a batch of data into a dictionary
        """
        with open('cifar-10-batches-py/'+name, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            encoder = preprocessing.LabelBinarizer()
            encoder.fit([x for x in range(self.num_labels)])

            # Add batch to batches
            self.batches[name] = {
                'batch_name': data[b'batch_label'],
                'X': np.matrix(data[b'data'], dtype=float).T/255,
                'Y': np.array(encoder.transform(data[b'labels'])).T,
                'y': np.array(data[b'labels']),
            }
        return self.batches[name]

    def getBatches(self, *args):
        """
        Loads and returns a list of batches of data
        """
        self.batches = [self.getBatch(name) for name in args]
        return self.batches


def normalize(data, mean, std):
    """
    Normalizes the data
    """
    data -= repmat(mean, 1, np.size(data, axis=1))
    data /= repmat(std, 1, np.size(data, axis=1))
    return data

# --------------------------------------------------------------------------------#


class Network():
    """
    A simple one layer network.
    """

    def __init__(self, inputDim, outputDim, l=0):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.W = np.random.normal(0, 0.01, (outputDim, inputDim))
        self.b = np.random.normal(0, 0.01, (outputDim, 1))
        self.l = l

    def evaluateClassifier(self, X, W=None, b=None):
        """
        Calculates and returns the softmax
        """
        if W is None:
            W = self.W

        if b is None:
            b = self.b

        s = np.dot(W, X) + b
        e = np.exp(s)
        return e / np.sum(e, axis=0)

    def computeCost(self, X, Y, W=None, b=None):
        """
        Calculates the cost
        """
        P = self.evaluateClassifier(X, W, b)

        J1 = np.sum(-np.log(np.multiply(Y, P).sum(axis=0))) / X.shape[1]
        J2 = self.l * np.sum(self.W**2)

        return J1+J2

    def computeAccuracy(self, X, y):
        """
        Calculates the prediction accuracy of a network with weights W and bias b
        """
        P = self.evaluateClassifier(X)
        P = np.argmax(P, axis=0)

        return 100*np.sum(P == y)/y.shape[0]

    def computeGradients(self, X, Y):
        """
        Fast method for determining the gradients
        """
        P = self.evaluateClassifier(X)
        N = X.shape[1]

        gradient = P - Y

        gradW = gradient.dot(X.T) / N + 2 * self.l * self.W
        gradb = gradient.dot(np.ones((N, 1))) / N

        return gradW, gradb

    def train(self, training_data, validation_data, eta, n_epochs, n_batches, shuffle=False):
        """
        Trains the network by calculating weights and bias using Mini-Batch Gradient Descent
        """

        X, Y, y = training_data['X'], training_data['Y'], training_data['y']
        Xval, Yval, yval = validation_data['X'], validation_data['Y'], validation_data['y']

        cost_train, accuracy_train, cost_val, accuracy_val = np.zeros((n_epochs, 1)), np.zeros(
            (n_epochs, 1)), np.zeros((n_epochs, 1)), np.zeros((n_epochs, 1))

        cost_train[0, :] = self.computeCost(X, Y)
        cost_val[0, :] = self.computeCost(Xval, Yval)
        accuracy_train[0, :] = self.computeAccuracy(X, y)
        accuracy_val[0, :] = self.computeAccuracy(Xval, yval)

        for i in range(1, n_epochs):

            if(shuffle):
                rand = np.random.permutation(X.shape[1])
                X = X[:, rand]
                Y = Y[:, rand]
                y = np.asarray([y[idx] for idx in rand])

            for j in range(1, int(X.shape[1]/n_batches)+1):
                jstart = (j-1) * n_batches + 1
                jend = j * n_batches

                Xbatch = X[:, jstart:jend]
                Ybatch = Y[:, jstart:jend]

                # Compute gradients
                gradW, gradb = self.computeGradients(Xbatch, Ybatch)

                # Update weights and bias
                self.W -= eta * gradW
                self.b -= eta * gradb

            cost_train[i, :] = self.computeCost(X, Y)
            cost_val[i, :] = self.computeCost(Xval, Yval)
            accuracy_train[i, :] = self.computeAccuracy(X, y)
            accuracy_val[i, :] = self.computeAccuracy(Xval, yval)

        return cost_train.flatten(), cost_val.flatten(), accuracy_train.flatten(), accuracy_val.flatten()

# -------------------------------------------------------------------------#


"""
Functions to present the results
"""


def computeGradsNum(network, X, Y, W, b, h):
    """
    Slow but accurate method to determine the gradients
    """

    grad_w = np.zeros(W.shape)
    grad_b = np.zeros(b.shape)

    cost = network.computeCost(X, Y)

    for i in range(b.shape[0]):
        b[i] += h
        c2 = network.computeCost(X, Y)
        grad_b[i] = (c2 - cost) / h
        b[i] -= h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] += h
            c2 = network.computeCost(X, Y)
            grad_w[i, j] = (c2 - cost) / h
            W[i, j] -= h

    return grad_w, grad_b


def checkGradient(network, X, Y, batchSize=20, tol=1e-6):
    """
    Method used to check that the simple gradient is accurate by comparing with
    a more computer intensive but accurate method.
    """
    gradW, gradb = network.computeGradients(X[:, 0:batchSize], Y[:, 0:20])
    ngradW, ngradb = computeGradsNum(
        network, X[:, 0:batchSize], Y[:, 0:20], network.W, network.b, tol)

    rel_error = np.sum(abs(ngradW - gradW)) / np.maximum(tol,
                                                         np.sum(abs(ngradW)) + np.sum(abs(gradW)))
    return rel_error


def plot_weights(W, labels, show=True, save=False, filename=None):
    """
    Plots the learnt weights of the network.
    This is a representation of the 'class template image'
    """
    fig, axes = plt.subplots(2, 5, figsize=(12.0, 4.0))
    for w, label, ax in zip(W, labels, axes.ravel()):
        im = w.reshape(3, 32, 32)
        im = (im - im.min()) / (im.max() - im.min())
        im = im.T
        im = np.rot90(im, k=3)
        ax.imshow(im)
        ax.set_title(label)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.axis('tight')

    if filename is not None:
        plt.savefig('results/weights_plot_{}.eps'.format(filename))
        plt.clf()
        plt.close()

    if show:
        plt.show()


def plot_performance(cost_train, cost_val, accuracy_train, accuracy_val, show=True, filename=None):
    """
    Plots the cost and accuracy against the epochs for the two data sets
    """
    plt.figure()
    plt.plot(np.arange(len(cost_train)), cost_train, label='Training set')
    plt.plot(np.arange(len(cost_val)), cost_val, label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if filename is not None:
        plt.savefig('results/cost_plot_{}.eps'.format(filename))
        plt.clf()
        plt.close()

    plt.figure()
    plt.plot(np.arange(len(accuracy_train)), accuracy_train,
             label='Accuracy on training set')
    plt.plot(np.arange(len(accuracy_val)), accuracy_val,
             label='Accuracy on validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if filename is not None:
        plt.savefig('results/accuracy_plot_{}.eps'.format(filename))
        plt.clf()
        plt.close()

    if show:
        plt.show()


def summarize(dataset, training_data, validation, parameters):
    """
    Provides a summary of the network performance based on the given parameters
    """

    # Initialize the network
    network = Network(dataset.input_dim, dataset.num_labels, l=parameters['l'])
    modelnr = parameters['model_nr']

    # Check initial accuracy
    init_acc_train = network.computeAccuracy(
        training_data['X'], training_data['y'])
    init_acc_val = network.computeAccuracy(
        validation_data['X'], validation_data['y'])

    # Format string to ouput model stats
    model_parameters = ' Model parameters: \n' + \
        '  lambda:  \t{}\n'.format(parameters['l']) + \
        '  eta:  \t{}\n'.format(parameters['eta']) + \
        '  n_epochs: \t{}\n'.format(parameters['n_epochs']) + \
        '  n_batches: \t{}\n'.format(parameters['n_batches']) + \
        '  shuffle: \t{}\n'.format(parameters['shuffle'])

    print(model_parameters)
    # Train network
    cost_train, cost_val, accuracy_train, accuracy_val = network.train(
        training_data, validation_data, parameters['eta'], parameters['n_epochs'], parameters['n_batches'], parameters['shuffle'])

    # Print accuracy and cost for training and validation data
    model_performance = 'Training data:\n' + \
                        '  accuracy (untrained): \t{:.2f}%\n'.format(init_acc_train) + \
                        '  accuracy (trained): \t\t{:.2f}%\n'.format(accuracy_train[-1]) + \
                        '  cost (final): \t\t{:.2f}%\n'.format(cost_train[-1]) + \
                        'Validation data:\n' + \
                        '  accuracy (untrained): \t{:.2f}%\n'.format(init_acc_val) + \
                        '  accuracy (trained): \t\t{:.2f}%\n'.format(accuracy_val[-1]) + \
                        '  cost (final): \t\t{:.2f}%\n'.format(cost_val[-1])

    print(model_performance)

    # Write model stats to file
    with open('results/model_summary_{}.txt'.format(modelnr), 'w') as f:
        f.write(model_parameters + model_performance)

    # Generate and save plots
    plot_performance(cost_train, cost_val, accuracy_train,
                     accuracy_val, show=False, filename=modelnr)
    plot_weights(network.W, dataset.getLabels(), show=False, filename=modelnr)


# -------------------------------------------------------------------------#

# Load data
np.random.seed(400)
dataset = CIFAR()
loaded_data = dataset.getBatches('data_batch_1', 'data_batch_2', 'test_batch')
training_data = loaded_data[0]
validation_data = loaded_data[1]
test_data = loaded_data[2]

# Calculate mean and standard deviation of training data
mean = np.mean(training_data['X'], axis=1)
std = np.std(training_data['X'], axis=1)

# Normalize the data w.r.t training data
training_data['X'] = normalize(training_data['X'], mean, std)
validation_data['X'] = normalize(validation_data['X'], mean, std)
test_data['X'] = normalize(test_data['X'], mean, std)


GDparams = [
    {'model_nr': 1,     'l': 0.0,     'eta': 0.001,
        'n_epochs': 40,     'n_batches': 100,   'shuffle': True, },
    {'model_nr': 2,     'l': 0.0,     'eta': 0.1,
        'n_epochs': 40,     'n_batches': 100,   'shuffle': True, },
    {'model_nr': 3,     'l': 0.1,     'eta': 0.001,
        'n_epochs': 40,     'n_batches': 100,   'shuffle': True, },
    {'model_nr': 4,     'l': 1.0,     'eta': 0.001,
        'n_epochs': 40,     'n_batches': 100,   'shuffle': True, }
]

for parameters in GDparams:
    summarize(dataset, training_data, validation_data, parameters)


# Initiate the network
# network = Network(data.input_dim, data.num_labels)
# # Verify that the gradients calculated are good enough
# print('Relative error between analytical and numerical gradient:',
#       checkGradient(network, training_data['X'], training_data['Y']))
