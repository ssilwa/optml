import time
import numpy as np
import matplotlib.pyplot as plt

import dataaccess

from sklearn import random_projection
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from sklearn.manifold import TSNE          as scipyTSNE
from MulticoreTSNE    import MulticoreTSNE as multiTSNE
from openTSNE         import TSNE          as openTSNE


class TSNE:
    ''' abstracts away from the multiple TSNE implementations
        makes fit and fit_transform do the same thing
        kind   -- open, multi, scipy
        kwargs -- whatever you want to pass to your favoriate implementation of tsne
    '''
    def __init__(self, kind='open', **kwargs):
        if   kind == 'open' : self.tsne = openTSNE(**kwargs)
        elif kind == 'multi': self.tsne = multiTSNE(**kwargs)
        elif kind == 'scipy': self.tsne = scipyTSNE(**kwargs)
        else: raise Exception('TSNE type {} not recognized'.format(kind))
        self.open = kind == 'open'

    def fit(self, data):
        if self.open: return self.tsne.fit(data)
        else:         return self.tsne.fit_transform(data)

    def fit_transform(self, data): return self.fit(data)

def nearest_neighbor_check(X, y):
    ''' Given points X and labels y, returns the accuracy from checking label to nearest neighbor '''
    
    # nearest neighbor using ball trees
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # get the second nearest neighbor since the first is the point itself
    indices = indices[:,1]

    # predicted values
    pred_y = y[indices]

    # number of correct predictions
    correct = (pred_y == y).sum()

    # accuracy
    acc = correct / len(y)

    return acc

def do_random_projection(X, d):
    ''' returns the data X after undergoing a random projection to dimension d'''

    transformer = random_projection.GaussianRandomProjection(n_components = d)
    X_new = transformer.fit_transform(X)
    return X_new


def run(kind, n, shape, X, y, tsne_dim = 2, verbose = True, showplot = False):
    ''' run tsne using scipy implementation'''

    tsne = TSNE(kind, n_components = tsne_dim, method = 'barnes_hut' if tsne_dim<3 else 'exact')

    start = time.perf_counter()
    data = tsne.fit_transform(X)

    if verbose : print('T-SNE ({}) -- dim: {}, time: {}s'.format(kind, tsne_dim , time.perf_counter() - start))

    # get accuracy using label of nearest neighbor
    acc = nearest_neighbor_check(data, y)

    if verbose : print('Accuracy:',acc)
    if showplot : plot(data, y, 10)

    return acc

def plot(X_2d, y, num_classes):    
    # add more colors if num_classes > 10
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'r', 'w', 'orange', 'purple']

    for i, c in zip(range(num_classes), colors):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c)

    plt.show()

class Trial:
    '''
        datasets   -- mnist, kmnist, fashion-mnist, cifar10, norb
        tsne_kind  -- scipy, multi, open
        n          -- specify if you only want to use a subset of n images
        tsne_dim   -- dim of tsne, default is 2
        dim_reduce -- specify randomly reduce initial data to this many dimensions
        verbose    -- print stuff along the way
        buf        -- write results to buffer
    '''
    def __init__(self, dataset, tsne_kind='open', n=None, tsne_dim=2, dim_reduce=None, verbose=True, buf=None):
        # get correct dataset
        if verbose: print('Loading dataset', dataset)
        if 'mnist' in dataset:
            get_dataset = lambda : dataaccess.get_mnist(dataset)
        elif dataset == 'cifar10':
            get_dataset = dataaccess.get_cifar10
        elif dataset == 'norb':
            get_dataset = dataaccess.get_norb
        else : raise Error('Dataset {} not recognized'.format(dataset))

        default_n, shape, X, y = get_dataset()

        # possibly trim dataset
        n = n or default_n
        X = X[:n]
        y = y[:n]

        # possibly reduce dataset
        if dim_reduce:
            X_new = do_random_projection(X, d=dim_reduce)
            X = X_new

        if verbose: print('(# examples, dim) =', X.shape)

        self.X = X
        self.y = y

        self.tsne_dim = tsne_dim
        self.tsne_kind = tsne_kind
        self.tsne = TSNE(self.tsne_kind, n_components = self.tsne_dim)

        self.verbose = verbose
        self.buf     = buf

    def run(self):
        start = time.perf_counter()
        if self.verbose : print('> Running TSNE ({}).....'.format(self.tsne_kind),end='\r')        
        data = self.tsne.fit_transform(self.X)

        if self.verbose : print('T-SNE ({}) -- dim: {}, time: {}s'.format(self.tsne_kind, self.tsne_dim , time.perf_counter() - start))

        # get accuracy using label of nearest neighbor
        acc = nearest_neighbor_check(data, self.y)

        if self.verbose : print('Accuracy:',acc)
        return acc

if __name__ == '__main__':

    Trial(

        'cifar10',

        tsne_kind='scipy',
        dim_reduce = 30,
        n=1000,

        ).run()

