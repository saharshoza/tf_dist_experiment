from __future__ import print_function
import gzip, binascii, struct, numpy
import time
import os
from six.moves.urllib.request import urlretrieve
import tensorflow as tf

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = "/tmp/mnist-data"
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000
BATCH_SIZE = 60
NUM_CHANNELS = 1
SEED = 42

def maybe_download(filename):
    """A helper to download the data files if not present."""
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
  
    For MNIST data, the number of channels is always 1.

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

def get_data():
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    train_data = extract_data(train_data_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    return train_data, test_data

def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)

def get_labels():
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    train_labels = extract_labels(train_labels_filename, 60000)
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
    test_labels = extract_labels(test_labels_filename, 10000)
    return train_labels, test_labels