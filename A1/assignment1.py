import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.matlib import repmat
from tabulate import tabulate

sns.set_style('darkgrid')

class Network():
    """
    A simple one layer network.
    """

    def __init__(self, inputDim, outputDim, loss='cross', l=0):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.W = np.random.normal(0, 0.01, (outputDim, inputDim))
        self.b = np.random.normal(0, 0.01, (outputDim, 1))
        self.l = l
        if loss == 'svm':
            self.computeCost = self.computeSVMCost
            self.computeGradients = self.computeSVMGradients
        else:
            self.computeCost = self.computeCrossCost
            self.computeGradients = self.computeCrossGradients

    def evaluateClassifier(self, X, W=None, b=None):
        """
        Calculates and returns the softmax of the input weights and bias.
        """
        if W is None:
            W = self.W

        if b is None:
            b = self.b

        s = np.dot(W, X) + b
        e = np.exp(s)
        return e / np.sum(e, axis=0)

    def computeCrossCost(self, X, Y, W=None, b=None):
        """
        Calculates the cost of the loss function in the cross-entropy sense 
        for the given data, weights and bias.
        """
        if W is None:
            W = self.W

        if b is None:
            b = self.b

        P = self.evaluateClassifier(X, W, b)

        loss = np.sum(-np.log(np.multiply(Y, P).sum(axis=0))) / X.shape[1]
        regularization = self.l * np.sum(self.W**2)

        return loss+regularization

    def computeSVMCost(self, X, Y, W=None, b=None):
        """
        Calculates the cost of the loss function in the SVM multi-class sense 
        for the given data, weights and bias.
        """

        if W is None:
            W = self.W

        if b is None:
            b = self.b

        s = W.dot(X) + b
        sy = repmat(s.T[Y.T == 1], Y.shape[0], 1)

        N = X.shape[1]

        # Take the scores minus the "true" score
        margins = np.maximum(0, s - sy + 1)
        margins[Y == 1] = 0
        loss = margins.sum() / N

        regularization = 0.5 * self.l * np.sum(W**2)

        return loss + regularization

    def computeAccuracy(self, X, y):
        """
        Calculates the prediction accuracy of the network on the data X 
        with weights W and bias b.
        """
        P = self.evaluateClassifier(X)
        P = np.argmax(P, axis=0)

        return 100*np.sum(P == y)/y.shape[0]

    def computeCrossGradients(self, X, Y):
        """
        Computes the gradients of the weights and bias used to minimize
        the cross-entropy loss.
        """
        P = self.evaluateClassifier(X)
        N = X.shape[1]

        gradient = P - Y

        gradW = gradient.dot(X.T) / N + 2 * self.l * self.W
        gradb = gradient.dot(np.ones((N, 1))) / N

        return gradW, gradb

    def computeSVMGradients(self, X, Y):
        """
        Computes the gradients of the weights and bias which are used
        to minimize the SVM multi-class loss.
        """
        y = np.argmax(Y, axis=0)
        s = self.W.dot(X) + self.b                        # Calculate scores
        # Scores of the correct classifications, repeated columnwise
        sy = repmat(s.T[Y.T == 1], Y.shape[0], 1)
        N = X.shape[1]

        margins = np.maximum(0, s - sy + 1)
        margins[Y == 1] = 0
        # Convert to indicator matrix, for each entry of the matrix
        # if wj*x - wy*x + 1 > 1 then the entry is 1, else 0
        margins[margins > 0] = 1
        

        count = np.sum(margins, axis=0)

        # Each element should be placed in corresponding Y==1 position
        for i in range(margins.shape[1]):
            margins[y[i], i] = -count[:, i]

        grad_W = np.dot(margins, X.T) / N + self.l * self.W
        grad_b = np.reshape(np.sum(margins, axis=1) /
                            margins.shape[1], self.b.shape)

        return grad_W, grad_b

    def train(self, training_data, validation_data, eta, n_epochs, n_batches, shuffle=False):
        """
        Train the network by calculating the weights and bias that minimizes the cost function
        using the Mini-Batch Gradient Descent approach.
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

            # Add decay to learning rate
            # eta *= 0.9

            cost_train[i, :] = self.computeCost(X, Y)
            cost_val[i, :] = self.computeCost(Xval, Yval)
            accuracy_train[i, :] = self.computeAccuracy(X, y)
            accuracy_val[i, :] = self.computeAccuracy(Xval, yval)

        return cost_train.flatten(), cost_val.flatten(), accuracy_train.flatten(), accuracy_val.flatten()


class CIFAR:
    """
    Class used to load and preprocess the CIFAR10 data set.
    """

    def __init__(self):
        self.input_dim = 32*32*3
        self.num_labels = 10

    def getLabels(self):
        """
        Returns the (named) labels of the data.
        """
        with open('cifar-10-batches-py/batches.meta', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            labels = [x.decode('ascii') for x in data[b'label_names']]
        return labels

    def getBatch(self, name):
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

def computeGradsNum(network, X, Y, h):
    """
    A numerical approach based on Finite Difference method to calculate the gradients.
    """
    W = network.W
    b = network.b

    gradW = np.matlib.zeros(W.shape)
    gradb = np.matlib.zeros(b.shape)

    cost = network.computeCost(X, Y)

    for i in range(b.shape[0]):
        b_try = np.array(b)
        b_try[i] += h
        c2 = network.computeCost(X, Y, W, b_try)
        gradb[i] = (c2 - cost) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = network.computeCost(X, Y, W_try, b)
            gradW[i, j] = (c2 - cost) / h

    return gradW, gradb


def computeGradsNumSlow(network, X, Y, h):
    """
    Compute the gradient using the Central Difference approximation.
    Slightly slower but more accurate than the finite difference approach.
    """
    W = network.W
    b = network.b

    gradW = np.matlib.zeros(W.shape)
    gradb = np.matlib.zeros(b.shape)

    for i in range(len(b)):
        b -= h
        c1 = network.computeCost(X, Y)
        b_try = np.array(b)
        b_try[i] += h
        c2 = network.computeCost(X, Y, W, b_try)
        gradb[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = network.computeCost(X, Y, W_try, b)
            W_try = np.array(W)
            W_try[i, j] += h
            c2 = network.computeCost(X, Y, W_try, b)
            gradW[i, j] = (c2-c1) / (2*h)

    return gradW, gradb


def checkGradient(loss='cross', batchSize=20, tol=1e-6):
    """
    Method used to check that the simple gradient is accurate by comparing with
    a more computer intensive but accurate method.
    """
    np.random.seed(400)
    dataset = CIFAR()
    training_data = dataset.getBatch('data_batch_1')

    X = training_data['X']
    Y = training_data['Y']

    # Calculate mean and standard deviation of training data
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)

    # Normalize the data w.r.t training data
    X = dataset.normalize(X, mean, std)
    

    # Initiate the network
    network = Network(dataset.input_dim, dataset.num_labels, loss=loss)


    gradW, gradb = network.computeGradients(X[:, :batchSize], Y[:, :batchSize])
    
    ngradW, ngradb = computeGradsNum(network, X[:, :batchSize], Y[:, :batchSize], tol)

    slow_gradW, slow_gradb = computeGradsNumSlow(network, X[:, :batchSize], Y[:, :batchSize], tol)

    rel_error = np.sum(abs(ngradW - gradW)) / np.maximum(tol,
                                                         np.sum(abs(ngradW)) + np.sum(abs(gradW)))
    
    rel_error_slow = np.sum(abs(slow_gradW - gradW)) / np.maximum(tol,
                                                         np.sum(abs(slow_gradW)) + np.sum(abs(gradW)))

    
    table = [['Analytical', np.mean(gradW), np.min(gradW), np.max(gradW)],
             ['Finite difference (Numerical)', np.mean(ngradW), np.min(ngradW), np.max(ngradW)],
             ['Central difference (Numerical)', np.mean(slow_gradW), np.min(slow_gradW), np.max(slow_gradW)]]

    table = tabulate(table, headers=['Gradient', 'Mean Weight', 'Min Weight', 'Max Weight'], tablefmt='github')
    print('Relative error between the Analytical and Finite Difference approach is', rel_error,
                  '\nRelative error between the Analytical and Central Difference approach is', rel_error_slow,'\n\n')
    print(table)


def plot_weights(W, labels, show=True, save=False, filename=None):
    """
    Plots the learnt weights of the network, representing the 'template image' for each class.
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
        plt.savefig('results/classTemplates_{}.png'.format(filename))
        plt.clf()
        plt.close()
    

def plot_performance(cost_train, cost_val, accuracy_train, accuracy_val, filename=None):
    """
    Plots the cost and accuracy against the epochs for the two data sets.
    """
    plt.figure(figsize=(12.0, 5.0))
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(accuracy_train)), accuracy_train,
             label='Accuracy on training set')
    plt.plot(np.arange(len(accuracy_val)), accuracy_val,
             label='Accuracy on validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(len(cost_train)), cost_train, label='Training set')
    plt.plot(np.arange(len(cost_val)), cost_val, label='Validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.tight_layout()

    if filename is not None:
        plt.savefig('results/performance_{}.png'.format(filename))
        plt.clf()
        plt.close()
    

def model_summary(dataset, training_data, validation_data, test_data, parameters, loss='cross', save=False):
    """
    Generates a summary of the network performance based on the given parameters.
    """

    # Initialize the network
    network = Network(dataset.input_dim, dataset.num_labels,
                      l=parameters['l'], loss=loss)

    # Check initial accuracy
    init_acc_train = network.computeAccuracy(
        training_data['X'], training_data['y'])
    init_acc_val = network.computeAccuracy(
        validation_data['X'], validation_data['y'])
    init_acc_test = network.computeAccuracy(
        test_data['X'], test_data['y'])

    # Format string to ouput model stats
    model_parameters = 'Model parameters: \n' + \
        '   loss:   \t{}\n'.format(loss) + \
        '   lambda:  \t{}\n'.format(parameters['l']) + \
        '   eta:  \t{}\n'.format(parameters['eta']) + \
        '   n_epochs: \t{}\n'.format(parameters['n_epochs']) + \
        '   n_batches: \t{}\n'.format(parameters['n_batches']) + \
        '   shuffle: \t{}\n'.format(parameters['shuffle'])

    print(model_parameters)

    # Train network
    cost_train, cost_val, accuracy_train, accuracy_val = network.train(
        training_data, validation_data, parameters['eta'], parameters['n_epochs'], parameters['n_batches'], parameters['shuffle'])

    # Compute accuracy on test data
    accuracy_test = network.computeAccuracy(test_data['X'], test_data['y'])
    cost_test = network.computeCost(test_data['X'], test_data['Y'])

    model_performance = 'Training data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(init_acc_train) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(accuracy_train[-1]) + \
                        '   cost (final): \t\t{:.2f}\n'.format(cost_train[-1]) + \
                        'Validation data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(init_acc_val) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(accuracy_val[-1]) + \
                        '   cost (final): \t\t{:.2f}\n'.format(cost_val[-1]) + \
                        'Test data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(init_acc_test) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(accuracy_test) + \
                        '   cost (final): \t\t{:.2f}\n'.format(cost_test)

    print(model_performance)

    if save:
        filename = loss+"_lambda"+str(parameters['l'])+"_eta"+str(parameters['eta'])
        with open('results/summary_{}.txt'.format(filename), 'w') as f:
            f.write(model_parameters + model_performance)
    else:
        filename = None

    # Generate and save plots
    plot_performance(cost_train, cost_val, accuracy_train,
                     accuracy_val, filename=filename)
    plot_weights(network.W, dataset.getLabels(), filename=filename)
    plt.show()

# -------------------------------------------------------------------------#

def report(loss='cross', l=0.0, eta=0.001, n_epochs=40, n_batches=100, shuffle=False, save=False):

    np.random.seed(400)
    dataset = CIFAR()
    training_data = dataset.getBatch('data_batch_1')
    validation_data = dataset.getBatch('data_batch_2')
    test_data = dataset.getBatch('test_batch')

    # Calculate mean and standard deviation of training data
    mean = np.mean(training_data['X'], axis=1)
    std = np.std(training_data['X'], axis=1)

    # Normalize the data w.r.t training data
    training_data['X'] = dataset.normalize(training_data['X'], mean, std)
    validation_data['X'] = dataset.normalize(validation_data['X'], mean, std)
    test_data['X'] = dataset.normalize(test_data['X'], mean, std)

    param = {'l': l,  'eta': eta,   'n_epochs': n_epochs, 'n_batches': n_batches,   'shuffle': shuffle, }

    model_summary(dataset, training_data, validation_data,
                  test_data, param, loss=loss, save=save)


# # Example use:
# checkGradient()
# report(loss='cross', l=0, eta=0.001, n_epochs=40, n_batches=100, shuffle=False)