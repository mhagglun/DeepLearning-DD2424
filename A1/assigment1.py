import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import preprocessing
from numpy.matlib import repmat
sns.set_style('darkgrid')


class CIFAR10:
    def __init__(self):
        self.input_dim = 32 * 32 * 3
        self.num_labels = 10
        self.getLabels()
        self.batches = {}

    def getLabels(self):
        """
        Returnes the (named) labels of the data
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


def normalize(X, mean, std):
    """
    Normalizes the data
    """
    X -= repmat(mean, 1, np.size(X, axis=1))
    X /= repmat(std, 1, np.size(X, axis=1))
    return X

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
        J2 = self.l * np.power(self.W, 2).sum()

        return J1+J2

    def computeAccuracy(self, X, y):
        """
        Calculates the prediction accuracy of a network with weights W and bias b
        """
        P = self.evaluateClassifier(X)
        P = np.argmax(P, axis=0)

        return np.sum(P == y)/y.shape[0]

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

    def computeGradsNum(self, X, Y, h):
        """
        Slow but accurate method to determine the gradients
        """

        grad_w = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)

        cost = self.computeCost(X, Y)

        for i in range(self.b.shape[0]):
            self.b[i] += h
            c2 = self.computeCost(X, Y)
            grad_b[i] = (c2 - cost) / h
            self.b[i] -= h

        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i, j] += h
                c2 = self.computeCost(X, Y)
                grad_w[i, j] = (c2 - cost) / h
                self.W[i, j] -= h

        return grad_w, grad_b

    def checkGradient(self, X, Y, batchSize=20, tol=1e-6):
        """
        Verify that the simple gradient is accurate by comparing with
        a more computer intensive but accurate method.
        """
        gradW, gradb = self.computeGradients(X[:, 0:batchSize], Y[:, 0:20])
        ngradW, ngradb = self.computeGradsNum(
            X[:, 0:batchSize], Y[:, 0:20], tol)

        rel_error = np.sum(abs(ngradW - gradW)) / np.maximum(tol,
                                                             np.sum(abs(ngradW)) + np.sum(abs(gradW)))

        return rel_error

    def train(self, training_data, validation_data, eta, n_epochs, n_batches, shuffle=False):
        """
        Trains the network by calculating weights and bias using Mini-Batch Gradient Descent
        """

        X = training_data['X']
        Y = training_data['Y']

        Xval = validation_data['X']
        Yval = validation_data['Y']

        cost_train = np.zeros((n_epochs, 1))
        cost_val = np.zeros((n_epochs, 1))
        cost_train[0, :] = self.computeCost(X, Y)
        cost_val[0, :] = self.computeCost(Xval, Yval)
        for i in range(1, n_epochs):

            if(shuffle):
                rand = np.random.permutation(X.shape[1])
                X = X[:, rand]
                Y = Y[:, rand]

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

        return cost_train, cost_val

# -------------------------------------------------------------------------#


"""
Functions to present the results
"""


def plot_weights(W, labels):
    """
    Plots the learnt weights of the network.
    This is a representation of the 'class template image'
    """
    fig, axes = plt.subplots(2, 5)
    for coef, label, ax in zip(W, labels, axes.ravel()):
        im = coef.reshape(3, 32, 32)
        im = (im - im.min()) / (im.max() - im.min())
        im = im.T
        im = np.rot90(im, k=3)
        ax.imshow(im)
        ax.set_title(label)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.axis('tight')
    plt.show()


# TODO: Add method to easily plot accuracy and loss graphs

# TODO: Add method to show information about the data sets

# -------------------------------------------------------------------------#

# Load data
np.random.seed(400)
data = CIFAR10()
loaded_data = data.getBatches('data_batch_1', 'data_batch_2', 'test_batch')
training_data = loaded_data[0]
validation_data = loaded_data[1]
test_data = loaded_data[2]

# Normalize the data w.r.t training data
training_data['X'] = normalize(training_data['X'], np.mean(training_data['X'], axis=1),
                               np.std(training_data['X'], axis=1))

validation_data['X'] = normalize(validation_data['X'], np.mean(training_data['X'], axis=1),
                                 np.std(test_data['X'], axis=1))

test_data['X'] = normalize(loaded_data[2]['X'], np.mean(training_data['X'], axis=1),
                           np.std(training_data['X'], axis=1))


# Initiate the network
network = Network(data.input_dim, data.num_labels)


# Check performance of untrained network
print('Accuracy on training data is', network.computeAccuracy(
    training_data['X'], training_data['y']), 'with an untrained network')

# Verify that the gradients calculated are good enough
print('Relative error between analytical and numerical gradient:',
      network.checkGradient(training_data['X'], training_data['Y']))

# Train network
print('Training the network...')
cost_train, cost_val = network.train(
    training_data, validation_data, 0.001, 40, 100)

# Accuracy on training data
print('Accuracy on training data: ', network.computeAccuracy(
    training_data['X'], training_data['y']))

# Accuracy on test data
print('Accuracy on test data: ',  network.computeAccuracy(
    test_data['X'], test_data['y']))

# Determine performance

plt.figure()
plt.plot(np.arange(len(cost_train)), cost_train, label='Training set')
plt.plot(np.arange(len(cost_val)), cost_val, label='Validation set')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plot_weights(network.W, data.getLabels())
