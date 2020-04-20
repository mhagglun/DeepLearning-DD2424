import pickle

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage import util
from numpy.matlib import repmat
from tabulate import tabulate
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sns.set_style('darkgrid')


class Network():
    """A multi-layer neural network.

    Args:
        input_dim            The dimension of the input data
        output_dim           The number of output classes
        hidden_layers        A list of the number of nodes to use for each layer
        l                    The parameter to control the degree of L2 regularization of the network

    Attributes:
        input_dim            The dimension of the input data
        output_dim           The number of output classes
        layers               A list of the Layer-objects that define the network
        l                    The parameter to control the degree of L2 regularization of the network

    """

    def __init__(self, input_dim, output_dim, hidden_layers=None, useBN=True, alpha=0.9, initialization=None, dropout=None, l=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = self.initiate_layers(hidden_layers, initialization)
        self.BN = useBN;
        self.alpha = alpha
        self.dropout = dropout
        self.l = l

    class Layer:
        """
        The inner class object used to construct the layers of the network.
        Contains the weights and biases for a layer of the network.
        """

        def __init__(self, input_dim, output_dim, initialization):
            super().__init__()
            # Parameters for batch norm
            self.gamma = np.ones((output_dim,1))
            self.beta = np.zeros((output_dim,1))
            self.mu_av = np.zeros((output_dim,1))
            self.var_av = np.zeros((output_dim,1))
            if initialization=='xavier':
                bound = np.sqrt(6) / (input_dim + output_dim)
                self.W = np.random.uniform(-bound, bound, size=(output_dim, input_dim) )
                self.b = np.zeros((output_dim, 1))
            elif initialization=='he':
                self.W = np.random.normal(
                0, np.sqrt(2 / (input_dim+output_dim)), size=(output_dim, input_dim) )
                self.b = np.zeros((output_dim, 1))
            else:
                self.W = np.random.normal(
                    0, 1 / np.sqrt(input_dim), size=(output_dim, input_dim))
                self.b = np.zeros((output_dim, 1))

    def initiate_layers(self, hidden_layers, initialization):
        """
        Sets the layers of the network.
        Takes a list that contains the number of nodes in each layer and
        initiates the layers of the network.
        """
        if hidden_layers is not None:
            num_nodes = np.r_[self.input_dim, hidden_layers, self.output_dim]
        else:
            num_nodes = [self.input_dim, self.output_dim]

        layers = []
        for i in range(len(num_nodes)-1):
            layers.append(self.Layer(num_nodes[i], num_nodes[i+1], initialization))

        return layers

    def forward_pass(self, X, Y, training=True):
        """
        Calculates and returns the probability, the intermediary activation values as well as the cost
        and loss of the multi-layer neural network.
        """
        input_values = X
        activations = [input_values]
        cost = 0

        BatchNorm = {
            'mean': [],
            'variance': [],
            'scores': [],
            'scores_normalized': []
        }

        for i, layer in enumerate(self.layers):
            s = np.dot(layer.W, input_values) + layer.b
            
            if self.BN and i < len(self.layers)-1:
                if training:
                    mean = np.mean(s, axis=1, keepdims=True)
                    variance = np.std(s, axis=1, keepdims=True) + np.finfo(np.float64).eps
                    if (self.layers[i].mu_av == np.ones(mean.shape)).all():                 # Check if it has been initiated
                        self.layers[i].mu_av = mean
                        self.layers[i].var_av = variance
                    else:
                        self.layers[i].mu_av = self.alpha * self.layers[i].mu_av + (1-self.alpha) * mean
                        self.layers[i].var_av = self.alpha * self.layers[i].var_av + (1-self.alpha) * variance
                else:
                    mean = layer.mu_av
                    variance = layer.var_av

                var = np.diagflat(np.power(variance, -0.5))                
                s_normalized = var.dot((s - mean))

                BatchNorm['mean'].append(mean)
                BatchNorm['variance'].append(variance)
                BatchNorm['scores'].append(s)
                BatchNorm['scores_normalized'].append(s_normalized) 

                s = np.multiply(layer.gamma, s_normalized) + layer.beta 
                
            input_values = np.maximum(0, s)

            if self.dropout is not None:
                p = np.random.binomial(1, (1-self.dropout), size=input_values.shape[1])
                input_values *= p

            activations.append(input_values)
            cost += self.l * np.sum(np.power(layer.W, 2))

        P = np.exp(s) / np.sum(np.exp(s), axis=0)

        loss = np.sum(-np.log(np.multiply(Y, P).sum(axis=0))) / X.shape[1]

        cost += loss

        return P, activations, cost, loss, BatchNorm

    # def backward_pass(self, X, Y, P, activations):
    #     """
    #     Backward propagation to calculate and return the gradients of each layer
    #     in the neural network.
    #     """
    #     N = X.shape[1]
    #     G = -(Y - P)

    #     grads_W, grads_b = [], []
    #     for i, layer in reversed(list(enumerate(self.layers))):
    #         gradb = G.sum(axis=1, keepdims=True) / N
    #         gradW = np.dot(G, activations[i].T) / N + 2 * self.l * layer.W

    #         grads_W.append(gradW)
    #         grads_b.append(gradb)

    #         # Propagate the gradient backwards
    #         G = np.dot(layer.W.T, G)
    #         G[activations[i] == 0] = 0

    #     return grads_W[::-1], grads_b[::-1]


    def backward_pass(self, X, Y, P, activations, BatchNorm=None):
        """
        Backward propagation using batch normalization to calculate and return the gradients of each layer
        in the neural network.
        """
        N = X.shape[1]
        G = -(Y - P)

        grads_W, grads_b = [], []
        grads_gamma, grads_beta = [], []
        for i, layer in reversed(list(enumerate(self.layers))):

            if i < len(self.layers)-1 and self.BN:
                mean = BatchNorm['mean'][i]
                variance = BatchNorm['variance'][i]
                score = BatchNorm['scores'][i]
                scores_normalized = BatchNorm['scores_normalized'][i]

                gradGamma = np.multiply(G, scores_normalized).sum(axis=1, keepdims=True) / N
                gradBeta = G.sum(axis=1, keepdims=True) / N
                
                grads_gamma.append(gradGamma)
                grads_beta.append(gradBeta)

                # Propagate gradient through scale and shift
                G = np.multiply(G, layer.gamma)
                G = self.batchnorm_backpass(G, score, mean, variance)


            gradb = G.sum(axis=1, keepdims=True) / N
            gradW = np.dot(G, activations[i].T) / N + 2 * self.l * layer.W

            grads_W.append(gradW)
            grads_b.append(gradb)

            # Propagate the gradient backwards
            G = np.dot(layer.W.T, G)
            G[activations[i] == 0] = 0

        return grads_W[::-1], grads_b[::-1], grads_gamma[::-1], grads_beta[::-1]

    def batchnorm_backpass(self, G, s, mean, variance):
        
        N = G.shape[1]

        sigma1 = np.power(variance, -0.5)
        sigma2 = np.power(variance, -1.5)
    
        G1 = np.multiply(G, sigma1)
        G2 = np.multiply(G, sigma2)

        D = s - mean
        c = np.multiply(G2, D).sum(axis=1, keepdims=True)

        G = G1 - np.dot(G1.sum(axis=1, keepdims=True), np.ones((1,N)))/N - np.multiply(D, c)/N
        return G


    def compute_performance(self, X, Y, training=False):
        """
        Calculates the prediction accuracy of the network on the data X 
        with weights W and bias b.
        """
        y = np.argmax(Y, axis=0)
        P, _, cost, loss, _ = self.forward_pass(X, Y, training=training)
        P = np.argmax(P, axis=0)

        return 100*np.sum(P == y)/y.shape[0], cost, loss

    def train(self, training_data, validation_data, n_batches, cycles=2, n_s=500, eta_min=1e-5, eta_max=1e-1,
                 sampling_rate=100, shuffle=False, noise=None):
        """
        Train the network by calculating the weights and bias that minimizes the cost function
        using the Mini-Batch Gradient Descent approach.
        """

        num_results = int(cycles * 2 * n_s/sampling_rate) + 1

        X, Y, y = training_data['X'], training_data['Y'], training_data['y']
        Xval, Yval, yval = validation_data['X'], validation_data['Y'], validation_data['y']

        train_results, val_results = np.zeros(
            (num_results, 5)), np.zeros((num_results, 5))

        # Store step number, accuracy, cost and loss columnwise, each row corresponds to an epoch
        train_results[0, 0] = 0
        train_results[0, 1], train_results[0,
                                           2], train_results[0, 3] = self.compute_performance(X, Y)

        val_results[0, 0] = 0
        val_results[0, 1], val_results[0, 2], val_results[0,
                                                          3] = self.compute_performance(Xval, Yval)

        for t in tqdm(range(1, cycles*2*n_s+1), desc='Training model', disable=False):
            j = (t % int(X.shape[1]/n_batches)) + 1
            jstart = (j-1) * n_batches + 1
            jend = j * n_batches


            Xbatch = X[:, jstart:jend]
            Ybatch = Y[:, jstart:jend]

            if noise is not None:
                Xbatch = util.random_noise(
                    Xbatch, mode=noise, seed=None, clip=True)

            l = int(t / (2*n_s))

            if t <= (2*l+1)*n_s:
                eta = eta_min + (t - 2*l*n_s)/n_s * (eta_max - eta_min)
            else:
                eta = eta_max - (t - (2*l+1)*n_s)/n_s * (eta_max - eta_min)

            P, activations, _, _, BatchNorm = self.forward_pass(Xbatch, Ybatch)

    
            if self.BN:
                gradsW, gradsb, gradsGamma, gradsBeta = self.backward_pass(Xbatch, Ybatch, P, activations, BatchNorm)  

                # Update the trainable parameters for each layer
                for i, (gradW, gradb, gradGamma, gradBeta) in enumerate(zip(gradsW, gradsb, gradsGamma, gradsBeta)):
                    self.layers[i].W -= eta * gradW
                    self.layers[i].b -= eta * gradb
                    self.layers[i].gamma += gradGamma
                    self.layers[i].beta += gradBeta
                    
            else:
                gradsW, gradsb, _, _ = self.backward_pass(Xbatch, Ybatch, P, activations)

                # Update the weights and biases of each layer
                for i, (gradW, gradb) in enumerate(zip(gradsW, gradsb)):
                    self.layers[i].W -= eta * gradW
                    self.layers[i].b -= eta * gradb

            if t % sampling_rate == 0:
                i = int(t/sampling_rate)
                train_results[i, 0] = t
                train_results[i, 1], train_results[i,
                                                   2], train_results[i, 3] = self.compute_performance(X, Y)
                train_results[i, 4] = eta

                val_results[i, 0] = t
                val_results[i, 1], val_results[i, 2], val_results[i,
                                                                  3] = self.compute_performance(Xval, Yval)
                val_results[i, 4] = eta

                if shuffle:
                    rand = np.random.permutation(X.shape[1])
                    X = X[:, rand]
                    Y = Y[:, rand]

        return train_results, val_results


class CIFAR:
    """
    Class used to load and preprocess the CIFAR10 data set.
    """

    def __init__(self):
        self.input_dim = 32*32*3
        self.num_labels = 10
        self.sample_size = 10000

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

            X = np.array(data[b'data'], dtype=float).T/255
            y = np.array(data[b'labels'])
            Y = (np.eye(self.num_labels)[y]).T

            batch = {
                'batch_name': data[b'batch_label'],
                'X': X,
                'Y': Y,
                'y': y,
            }
        return batch

    def get_aggregated_batches(self, num_batches=5):
        """
        Loads the number of batches and aggregates the batches into one single large batch
        """
        batches = [self.get_batch('data_batch_{}'.format(num))
                   for num in range(1, num_batches+1)]

        X = np.zeros((self.input_dim, num_batches*self.sample_size))
        Y = np.zeros((self.num_labels, num_batches*self.sample_size))
        y = np.zeros((num_batches*self.sample_size))

        for i, batch in enumerate(batches):
            X[:, i*self.sample_size:(i+1)*self.sample_size] = batch['X']
            Y[:, i*self.sample_size:(i+1)*self.sample_size] = batch['Y']
            y[i*self.sample_size:(i+1)*self.sample_size] = batch['y']

        new_batch = {
            'X': X,
            'Y': Y,
            'y': y,
        }

        return new_batch

    def load_datasets(self, num_training_batches=1, num_validation_samples=1000):
        """
        Method used to load and return the training, validation and test data sets.
        The user may specify the number of batches to use for the training set.
        """
        if 1 < num_training_batches and num_training_batches <= 5:
            data = self.get_aggregated_batches(num_training_batches)

            training_data, validation_data = {}, {}

            training_data['X'] = data['X'][:, :-num_validation_samples]
            training_data['Y'] = data['Y'][:, :-num_validation_samples]
            training_data['y'] = data['y'][:-num_validation_samples]

            validation_data['X'] = data['X'][:, -num_validation_samples:]
            validation_data['Y'] = data['Y'][:, -num_validation_samples:]
            validation_data['y'] = data['y'][-num_validation_samples:]

        else:
            training_data = self.get_batch('data_batch_1')
            validation_data = self.get_batch('data_batch_2')

        test_data = self.get_batch('test_batch')

        return training_data, validation_data, test_data

    def normalize(self, data, mean, std):
        """
        Method used to normalize the data using the specified mean and standard deviation.
        """
        data -= repmat(mean, 1, np.size(data, axis=1))
        data /= repmat(std, 1, np.size(data, axis=1))
        return data


""" Functions for model comparison and presentation of the results. """

def compute_gradients(network, X, Y):

    P, activations, cost, loss, BatchNorm = network.forward_pass(X, Y, training=True)

    N = X.shape[1]
    G = -(Y - P)

    grads_W, grads_b = [], []
    grads_gamma, grads_beta = [], []
    for i, layer in reversed(list(enumerate(network.layers))):

        if i < len(network.layers)-1 and network.BN:
            mean = BatchNorm['mean'][i]
            variance = BatchNorm['variance'][i]
            score = BatchNorm['scores'][i]
            scores_normalized = BatchNorm['scores_normalized'][i]

            gradGamma = np.multiply(G, scores_normalized).sum(axis=1, keepdims=True) / N
            gradBeta = G.sum(axis=1, keepdims=True) / N
            
            grads_gamma.append(gradGamma)
            grads_beta.append(gradBeta)

            # Propagate gradient through scale and shift
            G = np.multiply(G, layer.gamma)
            G = network.batchnorm_backpass(G, score, mean, variance)


        gradb = G.sum(axis=1, keepdims=True) / N
        gradW = np.dot(G, activations[i].T) / N + 2 * network.l * layer.W

        grads_W.append(gradW)
        grads_b.append(gradb)

        # Propagate the gradient backwards
        G = np.dot(layer.W.T, G)
        G[activations[i] == 0] = 0

    return grads_W[::-1], grads_b[::-1], grads_gamma[::-1], grads_beta[::-1]


def compute_grads_num(network, X, Y, h):
    """
    Compute the gradient using the Central Difference approximation.
    Slightly slower but more accurate than the finite difference approach.
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

        if network.BN:

            gamma = layer.gamma
            beta = layer.beta
            gradGamma = np.matlib.zeros(gamma.shape)
            gradBeta = np.matlib.zeros(beta.shape)

            for i in range(gamma.shape[0]):
                network.layers[idx].gamma[i,:] -= h
                _, _, c1, _, _ = network.forward_pass(X, Y)
                network.layers[idx].gamma[i, :] += 2*h
                _, _, c2, _, _ = network.forward_pass(X, Y)
                gradGamma[i, :] = (c2-c1) / (2*h)
                network.layers[idx].gamma[i, :] -= h
            
            for i in range(beta.shape[0]):
                network.layers[idx].beta[i,:] -= h
                _, _, c1, _, _ = network.forward_pass(X, Y)
                network.layers[idx].beta[i, :] += 2*h
                _, _, c2, _, _ = network.forward_pass(X, Y)
                gradBeta[i, :] = (c2-c1) / (2*h)
                network.layers[idx].beta[i, :] -= h

            grad_gamma.append(gradGamma)
            grad_beta.append(gradBeta)
        

    return grad_weights, grad_bias, grad_gamma, grad_beta


def check_gradient(hidden_layers=(50,50), useBN=True, dimensions=10, batchSize=20, tol=1e-5):
    """
    Method used to check that the simple gradient is accurate by comparing with
    a more computer intensive but accurate method.
    """
    np.random.seed(400)
    dataset = CIFAR()
    training_data, _, _ = dataset.load_datasets()

    X = training_data['X']
    Y = training_data['Y']

    # Calculate mean and standard deviation of training data
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)

    # Normalize the data w.r.t training data
    X = dataset.normalize(X, mean, std)

    network = Network(dimensions, dataset.num_labels, hidden_layers=hidden_layers, useBN=useBN)

    gradients_W, gradients_b, gradients_gamma, gradients_beta = compute_gradients(network,
                                                 X[:dimensions, :batchSize], Y[:, :batchSize])

    num_gradW, num_gradb, num_gradGamma, num_gradBeta = compute_grads_num(
        network, X[:dimensions, :batchSize], Y[:, :batchSize], tol)

    table = []
    for i, (gradW, ngradW, layer) in enumerate(zip(gradients_W, num_gradW, network.layers)):
        rel_error_W = np.sum(abs(ngradW - gradW)) / np.maximum(tol,
                                                             np.sum(abs(ngradW)) + np.sum(abs(gradW)))
        
        if useBN and i < len(network.layers)-1:
            rel_error_gamma = np.sum(abs(num_gradGamma[i] - gradients_gamma[i])) / np.maximum(tol,
                                                                np.sum(abs(num_gradGamma[i])) + np.sum(abs(gradients_gamma[i])))

            table.append([i+1, layer.W.shape[0], rel_error_W, rel_error_gamma])
        else:
            table.append([i+1, layer.W.shape[0], rel_error_W])


    table = tabulate(table, headers=['Layer #', 'Number of nodes', 'Relative error Weights', 'Relative error Gamma'], tablefmt='github')
    print(table)


def parameter_search(lmin=-1, lmax=-5, num_parameters=8, filename=None):
    """
    Performs a coarse search over a broad range of values of the regularization parameter
    to find a good value.
    """
    # Generate regularization parameters uniformly, from 10^lmin to 10^lmax
    np.random.seed(400)
    lambdas = lmin + (lmax - lmin)*np.random.uniform(size=num_parameters)
    lambdas = np.power(10, lambdas)

    table = []
    for l in lambdas:
        output, _, _ = report(l=l, num_training_batches=5, parameter_search=True)
        table.append(output)

    table = tabulate(table, headers=['lambda', 'cycles', 'n_s', 'n_batches', 'eta_min', 'eta_max', 'accuracy (train)',
                                     'accuracy (val)', 'accuracy (test)'], tablefmt='github')

    print(table)
    if filename is not None:
        with open('{}.md'.format(filename), 'w') as f:
            f.write(table)


def search_lr(lower_limit, upper_limit, verbose=False):
    """
    Used to search for a good learning rate by plotting the accuracy vs the learning rate for the specified range.
    """
    _, train_results, val_results = report(cycles=1, eta_min=lower_limit, eta_max=upper_limit, num_training_batches=5, verbose=verbose, sampling_rate=50, parameter_search=True)
    n = int(len(val_results)/2)

    plt.plot(val_results[1:n,4], val_results[1:n, 1])
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy %')
    plt.show()



def plot_performance(train_results, val_results, filename=None):
    """
    Plots the cost, loss and accuracy against the update steps for the two data sets.
    """
    plt.figure(figsize=(12.0, 5.0))
    plt.subplot(1, 3, 1)
    plt.plot(train_results[:, 0], train_results[:, 1],
             label='Training set')
    plt.plot(val_results[:, 0], val_results[:, 1],
             label='Validation set')
    plt.xlabel('Update steps')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_results[:, 0], train_results[:, 2], label='Training set')
    plt.plot(val_results[:, 0], val_results[:, 2], label='Validation set')
    plt.xlabel('Update steps')
    plt.ylabel('Cost')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_results[:, 0], train_results[:, 3], label='Training set')
    plt.plot(val_results[:, 0], val_results[:, 3], label='Validation set')
    plt.xlabel('Update steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig('performance_{}.png'.format(filename))
        plt.clf()
        plt.close()


def model_summary(dataset, training_data, validation_data, test_data, parameters, num_nodes=50, verbose=True, sampling_rate=100, parameter_search=False):
    """
    Generates a summary of the network performance based on the given data and parameters.
    """

    # Initialize the network
    network = Network(dataset.input_dim, dataset.num_labels,
                      hidden_layers=num_nodes, useBN=parameters['useBN'], alpha=parameters['alpha'], initialization=parameters['initialization'], dropout=parameters['dropout'], l=parameters['l'])

    # Format string to ouput model stats
    model_parameters = 'Model parameters: \n' + \
        '   hidden layers:  \t{}\n'.format(num_nodes) + \
        '   BN:  \t\t{}\n'.format(parameters['useBN']) + \
        '   lambda:  \t\t{}\n'.format(parameters['l']) + \
        '   cycles: \t\t{}\n'.format(parameters['cycles']) + \
        '   n_s: \t\t{}\n'.format(parameters['n_s']) + \
        '   n_batches: \t\t{}\n'.format(parameters['n_batches']) + \
        '   eta_min: \t\t{}\n'.format(parameters['eta_min']) + \
        '   eta_max: \t\t{}\n'.format(parameters['eta_max'])

    if parameters['initialization'] is not None:
        model_parameters = model_parameters + \
        '   initialization: \t{}\n'.format(parameters['initialization'])

    if parameters['dropout'] is not None:
        model_parameters = model_parameters + \
        '   dropout: \t\t{}\n'.format(parameters['dropout'])

    if parameters['shuffle'] is not False:
        model_parameters = model_parameters + \
            '   shuffle: \t\t{}\n'.format(parameters['shuffle'])

    if parameters['noise'] is not None:
        model_parameters = model_parameters + \
            '   noise: \t\t{}\n'.format(parameters['noise'])

    if verbose and (not parameter_search):
        print(model_parameters)

    test_accuracy_initial, _, _ = network.compute_performance(
        test_data['X'], test_data['Y'], True)

    # Train network
    train_results, val_results = network.train(training_data, validation_data, n_batches=parameters['n_batches'],
                                 cycles=parameters['cycles'], n_s=parameters['n_s'], eta_min=parameters['eta_min'],
                                 eta_max=parameters['eta_max'], sampling_rate=sampling_rate, noise=parameters['noise'])

    test_accuracy_final, test_cost, _ = network.compute_performance(
        test_data['X'], test_data['Y'])

    model_performance = 'Training data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(train_results[0, 1]) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(train_results[-1, 1]) + \
                        '   cost (final): \t\t{:.2f}\n'.format(train_results[-1, 2]) + \
                        'Validation data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(val_results[0, 1]) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(val_results[-1, 1]) + \
                        '   cost (final): \t\t{:.2f}\n'.format(val_results[-1, 2]) + \
                        'Test data:\n' + \
                        '   accuracy (untrained): \t{:.2f}%\n'.format(test_accuracy_initial) + \
                        '   accuracy (trained): \t\t{:.2f}%\n'.format(test_accuracy_final) + \
                        '   cost (final): \t\t{:.2f}\n'.format(test_cost)

    if verbose and not parameter_search:
        print(model_performance)

    if parameter_search:
        result = ['{:.6f}'.format(parameters['l']), parameters['cycles'], parameters['n_s'], parameters['n_batches'], parameters['eta_min'],
                  parameters['eta_max'], '{:.2f}%'.format(train_results[-1, 1]), '{:.2f}%'.format(val_results[-1, 1]), '{:.2f}%'.format(test_accuracy_final)]
        return result, train_results, val_results

    else:
        plot_performance(train_results, val_results)
        plt.show()

def report(num_nodes=50, l=0.01, cycles=2, n_s=500, eta_min=1e-5, eta_max=1e-1, useBN=False, alpha=0.9, n_batches=100, num_training_batches=1, sampling_rate=100, parameter_search=False, dropout=None, shuffle=False, initialization=None,  noise=None, verbose=True):
    """
    Method that loads and preprocesses the data and then trains the model for the given parameters in order to generate
    a summary of the model performance which will be used for the report.
    """
    np.random.seed(400)
    dataset = CIFAR()

    training_data, validation_data, test_data = dataset.load_datasets(
        num_training_batches)

    mean = np.mean(training_data['X'], axis=1, keepdims=True)
    std = np.std(training_data['X'], axis=1, keepdims=True)

    training_data['X'] = dataset.normalize(training_data['X'], mean, std)
    validation_data['X'] = dataset.normalize(validation_data['X'], mean, std)
    test_data['X'] = dataset.normalize(test_data['X'], mean, std)

    n_s = int( 5 * 45000 / n_batches )

    parameters = {'l': l, 'n_batches': n_batches, 'cycles': cycles, 'n_s': n_s, 'eta_min': eta_min,
                  'eta_max': eta_max, 'useBN': useBN, 'alpha': alpha, 'dropout': dropout, 'shuffle': shuffle, 'initialization': initialization, 'noise': noise}

    summary = model_summary(dataset, training_data, validation_data,
                            test_data, parameters, num_nodes=num_nodes, verbose=verbose, sampling_rate=sampling_rate, parameter_search=parameter_search)
    return summary


# Check error of gradients for each layer
# Layer #1 is the input layer
# check_gradient((50,50,50), useBN=True)


# Train 3 layer network with (50,50) hidden nodes with
# n_batch=100, eta_min=1e-5, eta_max=1e-1, lambda=0.005 for two cycles and
# n_s = 5 * 45000 / n_batch
# use shuffle, Xavier or He initialization

report(num_nodes=(50,50), l=0.005, cycles=2, eta_min=1e-5, eta_max=1e-1, useBN=False, alpha=0.9, n_batches=100,
         num_training_batches=5, shuffle=True, initialization='he')