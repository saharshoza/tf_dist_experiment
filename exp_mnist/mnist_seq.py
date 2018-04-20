
# coding: utf-8

# # MNIST from scratch
# 
# This notebook walks through an example of training a TensorFlow model to do digit classification using the [MNIST data set](http://yann.lecun.com/exdb/mnist/). MNIST is a labeled set of images of handwritten digits.
# 
# An example follows.

# In[1]:

from __future__ import print_function

import time
# We're going to be building a model that recognizes these digits as 5, 0, and 4.
# 
# # Imports and input data
# 
# We'll proceed in steps, beginning with importing and inspecting the MNIST data. This doesn't have anything to do with TensorFlow in particular -- we're just downloading the data archive.

# In[2]:


import os
from six.moves.urllib.request import urlretrieve

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
#SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
# for those who have no access to google storage, use lecun's repo please
WORK_DIRECTORY = "/tmp/mnist-data"

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

train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')


import gzip, binascii, struct, numpy


# In[7]:


IMAGE_SIZE = 28
PIXEL_DEPTH = 255

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

train_data = extract_data(train_data_filename, 60000)
test_data = extract_data(test_data_filename, 10000)


# A crucial difference here is how we `reshape` the array of pixel values. Instead of one image that's 28x28, we now have a set of 60,000 images, each one being 28x28. We also include a number of channels, which for grayscale images as we have here is 1.
# 
# Let's make sure we've got the reshaping parameters right by inspecting the dimensions and the first two images. (Again, mangled input is a very common source of errors.)



# Looks good. Now we know how to index our full set of training and test images.

# ### Label data
# 
# Let's move on to loading the full set of labels. As is typical in classification problems, we'll convert our input labels into a [1-hot](https://en.wikipedia.org/wiki/One-hot) encoding over a length 10 vector corresponding to 10 digits. The vector [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], for example, would correspond to the digit 1.

# In[9]:


NUM_LABELS = 10

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

train_labels = extract_labels(train_labels_filename, 60000)
test_labels = extract_labels(test_labels_filename, 10000)


# As with our image data, we'll double-check that our 1-hot encoding of the first few values matches our expectations.

# In[10]:


print('Training labels shape', train_labels.shape)
print('First label vector', train_labels[0])
print('Second label vector', train_labels[1])


# The 1-hot encoding looks reasonable.
# 
# ### Segmenting data into training, test, and validation
# 
# The final step in preparing our data is to split it into three sets: training, test, and validation. This isn't the format of the original data set, so we'll take a small slice of the training data and treat that as our validation set.

# In[11]:


VALIDATION_SIZE = 5000

validation_data = train_data[:VALIDATION_SIZE, :, :, :]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, :, :, :]
train_labels = train_labels[VALIDATION_SIZE:]

train_size = train_labels.shape[0]

print('Validation shape', validation_data.shape)
print('Train size', train_size)


# # Defining the model
# 
# Now that we've prepared our data, we're ready to define our model.
# 
# The comments describe the architecture, which fairly typical of models that process image data. The raw input passes through several [convolution](https://en.wikipedia.org/wiki/Convolutional_neural_network#Convolutional_layer) and [max pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer) layers with [rectified linear](https://en.wikipedia.org/wiki/Convolutional_neural_network#ReLU_layer) activations before several fully connected layers and a [softmax](https://en.wikipedia.org/wiki/Convolutional_neural_network#Loss_layer) loss for predicting the output class. During training, we use [dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout_method).
# 
# We'll separate our model definition into three steps:
# 
# 1. Defining the variables that will hold the trainable weights.
# 1. Defining the basic model graph structure described above. And,
# 1. Stamping out several copies of the model graph for training, testing, and validation.
# 
# We'll start with the variables.

# In[12]:


import tensorflow as tf

# We'll bundle groups of examples during training for efficiency.
# This defines the size of the batch.
BATCH_SIZE = 60
# We have only one channel in our grayscale images.
NUM_CHANNELS = 1
# The random seed that defines initialization.
SEED = 42

# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step, which we'll write once we define the graph structure.
train_data_node = tf.placeholder(
  tf.float32,
  shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE, NUM_LABELS))

# For the validation and test data, we'll just hold the entire dataset in
# one constant node.
validation_data_node = tf.constant(validation_data)
test_data_node = tf.constant(test_data)

# The variables below hold all the trainable weights. For each, the
# parameter defines how the variables will be initialized.
conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                      stddev=0.1,
                      seed=SEED))
conv1_biases = tf.Variable(tf.zeros([32]))
conv2_weights = tf.Variable(
  tf.truncated_normal([5, 5, 32, 64],
                      stddev=0.1,
                      seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
fc1_weights = tf.Variable(  # fully connected, depth 512.
  tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                      stddev=0.1,
                      seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
fc2_weights = tf.Variable(
  tf.truncated_normal([512, NUM_LABELS],
                      stddev=0.1,
                      seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

print('Done')


# Now that we've defined the variables to be trained, we're ready to wire them together into a TensorFlow graph.
# 
# We'll define a helper to do this, `model`, which will return copies of the graph suitable for training and testing. Note the `train` argument, which controls whether or not dropout is used in the hidden layer. (We want to use dropout only during training.)

# In[13]:


def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # Max pooling. The kernel size spec ksize also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

print('Done')


# Having defined the basic structure of the graph, we're ready to stamp out multiple copies for training, testing, and validation.
# 
# Here, we'll do some customizations depending on which graph we're constructing. `train_prediction` holds the training graph, for which we use cross-entropy loss and weight regularization. We'll adjust the learning rate during training -- that's handled by the `exponential_decay` operation, which is itself an argument to the `MomentumOptimizer` that performs the actual training.
# 
# The validation and prediction graphs are much simpler to generate -- we need only create copies of the model with the validation and test inputs and a softmax classifier as the output.

# In[14]:


# Training computation: logits + cross-entropy loss.
logits = model(train_data_node, True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  labels=train_labels_node, logits=logits))

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0)
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate,
                                       0.9).minimize(loss,
                                                     global_step=batch)

# Predictions for the minibatch, validation set and test set.
train_prediction = tf.nn.softmax(logits)
# We'll compute them only once in a while by calling their {eval()} method.
validation_prediction = tf.nn.softmax(model(validation_data_node))
test_prediction = tf.nn.softmax(model(test_data_node))

print('Done')


# # Training and visualizing results
# 
# Now that we have the training, test, and validation graphs, we're ready to actually go through the training loop and periodically evaluate loss and error.
# 
# All of these operations take place in the context of a session. In Python, we'd write something like:
# 
#     with tf.Session() as s:
#       ...training / test / evaluation loop...
#   
# But, here, we'll want to keep the session open so we can poke at values as we work out the details of training. The TensorFlow API includes a function for this, `InteractiveSession`.
# 
# We'll start by creating a session and initializing the variables we defined above.

# In[15]:


# Create a new interactive session that we'll use in
# subsequent code cells.
s = tf.InteractiveSession()

# Use our newly created session as the default for 
# subsequent operations.
s.as_default()

# Initialize all the variables we defined above.
tf.global_variables_initializer().run()


# Now we're ready to perform operations on the graph. Let's start with one round of training. We're going to organize our training steps into batches for efficiency; i.e., training using a small set of examples at each step rather than a single example.

# In[16]:


BATCH_SIZE = 60

# Grab the first BATCH_SIZE examples and labels.
batch_data = train_data[:BATCH_SIZE, :, :, :]
batch_labels = train_labels[:BATCH_SIZE]

# This dictionary maps the batch data (as a numpy array) to the
# node in the graph it should be fed to.
feed_dict = {train_data_node: batch_data,
             train_labels_node: batch_labels}

# Run the graph and fetch some of the nodes.
_, l, lr, predictions = s.run(
  [optimizer, loss, learning_rate, train_prediction],
  feed_dict=feed_dict)

print('Done')


# Let's take a look at the predictions. How did we do? Recall that the output will be probabilities over the possible classes, so let's look at those probabilities.

# In[17]:


print(predictions[0])


# As expected without training, the predictions are all noise. Let's write a scoring function that picks the class with the maximum probability and compares with the example's label. We'll start by converting the probability vectors returned by the softmax into predictions we can match against the labels.

# In[18]:


# The highest probability in the first entry.
print('First prediction', numpy.argmax(predictions[0]))

# But, predictions is actually a list of BATCH_SIZE probability vectors.
print(predictions.shape)

# So, we'll take the highest probability for each vector.
print('All predictions', numpy.argmax(predictions, 1))


# Next, we can do the same thing for our labels -- using `argmax` to convert our 1-hot encoding into a digit class.

# In[19]:


print('Batch labels', numpy.argmax(batch_labels, 1))


# Now we can compare the predicted and label classes to compute the error rate and confusion matrix for this batch.

# In[20]:


correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(batch_labels, 1))
total = predictions.shape[0]

print(float(correct) / float(total))

confusions = numpy.zeros([10, 10], numpy.float32)
bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(batch_labels, 1))
for predicted, actual in bundled:
  confusions[predicted, actual] += 1

# plt.grid(False)
# plt.xticks(numpy.arange(NUM_LABELS))
# plt.yticks(numpy.arange(NUM_LABELS))
# plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');


# Now let's wrap this up into our scoring function.

# In[21]:


def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = numpy.zeros([10, 10], numpy.float32)
    bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    
    return error, confusions

print('Done')


# We'll need to train for some time to actually see useful predicted values. Let's define a loop that will go through our data. We'll print the loss and error periodically.
# 
# Here, we want to iterate over the entire data set rather than just the first batch, so we'll need to slice the data to that end.
# 
# (One pass through our training set will take some time on a CPU, so be patient if you are executing this notebook.)

# In[22]:


# Train over the first 1/4th of our training set.
total_time = 0
steps = train_size // BATCH_SIZE
for step in range(steps):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    # This dictionary maps the batch data (as a numpy array) to the
    # node in the graph it should be fed to.
    feed_dict = {train_data_node: batch_data,
                 train_labels_node: batch_labels}
    # Run the graph and fetch some of the nodes.
    start = time.time()
    _, l, lr, predictions = s.run(
      [optimizer, loss, learning_rate, train_prediction],
      feed_dict=feed_dict)
    
    total_time += (time.time() - start)

    # Print out the loss periodically.
    if step % 100 == 0:
        error, _ = error_rate(predictions, batch_labels)
        print('Step %d of %d' % (step, steps))
        print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
        print('Validation error: %.1f%%' % error_rate(
              validation_prediction.eval(), validation_labels)[0])

print('Total time for seq is ' + str(total_time))
print('Time for seq per step is ' + str(total_time/steps))

# The error seems to have gone down. Let's evaluate the results using the test set.
# 
# To help identify rare mispredictions, we'll include the raw count of each (prediction, label) pair in the confusion matrix.

# In[23]:


test_error, confusions = error_rate(test_prediction.eval(), test_labels)
print('Test error: %.1f%%' % test_error)

# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.grid(False)
# plt.xticks(numpy.arange(NUM_LABELS))
# plt.yticks(numpy.arange(NUM_LABELS))
# plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');

# for i, cas in enumerate(confusions):
#     for j, count in enumerate(cas):
#         if count > 0:
#             xoff = .07 * len(str(count))
#             plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')


# We can see here that we're mostly accurate, with some errors you might expect, e.g., '9' is often confused as '4'.
# 
# Let's do another sanity check to make sure this matches roughly the distribution of our test set, e.g., it seems like we have fewer '5' values.

# In[25]:


# plt.xticks(numpy.arange(NUM_LABELS))
# plt.hist(numpy.argmax(test_labels, 1));


# Indeed, we appear to have fewer 5 labels in the test set. So, on the whole, it seems like our model is learning and our early results are sensible.
# 
# But, we've only done one round of training. We can greatly improve accuracy by training for longer. To try this out, just re-execute the training cell above.
