import gzip, pickle, time, os
import numpy as np
from sklearn.manifold import TSNE as scipytsne
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import random_projection
#from matplotlib import pyplot as plt
from openTSNE import TSNE as opentsne
from sklearn.neighbors import NearestNeighbors


def get_mnist(source="mnist"):
    ''' returns number_of_images, shape_of_images, images, labels for different types of mnist dataset '''
    ''' source can be mnist, fashion-mnist, kmnist '''
    im_shape = im_width, im_height, channels = 28, 28, 1
    num_images = 60000


    with gzip.open('datasets/' + source +  '/train-images-idx3-ubyte.gz','r') as f:
        f.read(16)
        buf = f.read(im_width * im_height * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X = data.reshape(num_images, im_width * im_height)

    with gzip.open('datasets/' + source +  '/train-labels-idx1-ubyte.gz','r') as f:
        f.read(8)
        buf = f.read(num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        y = data.flatten()

    return num_images, im_shape, X, y


def get_cifar10():
    ''' returns number_of_images, shape_of_images, images, labels for cifar10 dataset '''

    im_shape = im_width, im_height, channels = 32, 32, 3
    num_images = 10000 * 5

    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d

    data_dir = 'datasets/cifar10/cifar-10-batches-py'

    X = np.empty((0, im_width * im_height * 3), dtype=np.int)
    y = np.empty(0, dtype=np.int)
    for batch_num in range(1,6):
        d = unpickle(os.path.join(data_dir,'data_batch_{}'.format(batch_num)))
        X = np.vstack((X, d[b'data'  ]))
        y = np.hstack((y, d[b'labels']))

    return num_images, im_shape, X, y.flatten()

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
    acc = correct / n

    return acc

def do_random_projection(X, d):
    ''' returns the data X after undergoing a random projection to dimension d'''

    transformer = random_projection.GaussianRandomProjection(n_components = d)
    X_new = transformer.fit_transform(X)
    return X_new


def run_scipy(n, shape, X, y, tsne_dim = 2, verbose = True, showplot = False):
    ''' run tsne using scipy implementation'''

    tsne = scipytsne(n_components=tsne_dim, random_state=0, method = 'barnes_hut' if tsne_dim < 3 else 'exact')

    start = time.process_time()
    data = tsne.fit_transform(X)
    if verbose : print('T-SNE -- dim: {}, time: {}s'.format(tsne_dim , time.process_time() - start))

    # get accuracy using label of nearest neighbor
    acc = nearest_neighbor_check(data, y)

    if verbose : print('Accuracy:',acc)
    if showplot : plot(data, y, 10)

    return acc

def run_open(n, shape, X, y, tsne_dim = 2, verbose = True, showplot = False):
    ''' run tsne using open tsne implementation'''

    start = time.process_time()
    data = opentsne().fit(X)
    if verbose : print('T-SNE -- dim: {}, time: {}s'.format(tsne_dim , time.process_time() - start))

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

if __name__ == '__main__':
    n, shape, X, y = get_mnist("fashion-mnist")

    dim_reduce = True
    # reduce dimension of the data
    if dim_reduce:
        X_new = do_random_projection(X, d=50)
        print('Reducing dim to:', X_new.shape)
        X = X_new
    
    dim = 2
    # all 60k takes a while
    # gets to 88% with n = 1k, 94% with n = 5k, and 95% with n = 10k (approximately)
    n = 60000
    print('n:', n)
    X = X[:n]
    y = y[:n]

    accuracy = run_open(n, shape, X, y, dim, showplot = False)