import gzip, pickle, os
import numpy as np

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

    X = np.empty((0, im_width * im_height * 3), dtype=np.uint8)
    y = np.empty(0, dtype=np.uint8)
    for batch_num in range(1,6):
        d = unpickle(os.path.join(data_dir,'data_batch_{}'.format(batch_num)))
        X = np.vstack((X, d[b'data'  ]))
        y = np.hstack((y, d[b'labels']))

    return num_images, im_shape, X, y.flatten()

def get_norb():
    im_shape = im_width, im_height, channels = 96, 96, 1
    num_images = 24300 * 2

    with gzip.open('datasets/norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz','r') as f:
        f.read(24)
        buf = f.read(im_width * im_height * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        X = data.reshape(num_images, im_width * im_height)

    with gzip.open('datasets/norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz','r') as f:
        f.read(8)
        buf = f.read(num_images * 2)
        data = np.frombuffer(buf, dtype=np.uint16)
        y = data.flatten()

    return num_images, im_shape, X, y    
