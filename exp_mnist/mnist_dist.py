from __future__ import print_function
import gzip, binascii, struct, numpy
import time
import os
import utils
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
import argparse
import sys
from tensorflow.python.client import device_lib

SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
WORK_DIRECTORY = "/tmp/mnist-data"
IMAGE_SIZE = 28
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000
BATCH_SIZE = 60
NUM_CHANNELS = 1
SEED = 42

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


def main(_):

  train_data, test_data = utils.get_data()
  train_labels, test_labels = utils.get_labels()

  validation_data = train_data[:VALIDATION_SIZE, :, :, :]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:, :, :, :]
  train_labels = train_labels[VALIDATION_SIZE:]

  train_size = train_labels.shape[0]

  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  print("ps: " + str(ps_hosts))
  print("worker: " + str(worker_hosts))
  print("job_name: " + str(FLAGS.job_name))
  print("task_index: " + str(FLAGS.task_index))
  print("cluster: " + str(cluster))

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  print("Device: " + str(device_lib.list_local_devices()))

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    train_size = train_size / 2

    if FLAGS.task_index == 0:
      validation_data = train_data[:train_size, :, :, :]
      validation_labels = train_labels[:train_size]
      train_data = train_data[:train_size, :, :, :]
      train_labels = train_labels[:train_size]


    else:
      validation_data = train_data[train_size:, :, :, :]
      validation_labels = train_labels[train_size:]
      train_data = train_data[train_size:, :, :, :]
      train_labels = train_labels[train_size:]


    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

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
      global_step = tf.train.get_or_create_global_step()
      # Use simple momentum for the optimization.
      optimizer = tf.train.MomentumOptimizer(learning_rate,
                                             0.9).minimize(loss,
                                                           global_step=global_step)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    validation_prediction = tf.nn.softmax(model(validation_data_node))
    test_prediction = tf.nn.softmax(model(test_data_node))

    #hooks=[tf.train.StopAtStepHook(last_step=1000000)]
    config = tf.ConfigProto(log_device_placement=True)

    # Train over the first 1/4th of our training set.
    with tf.train.MonitoredTrainingSession(config=config,
                                            master=server.target,
                                           is_chief=(FLAGS.task_index == 0)) as sess:
                                           #hooks=hooks) as sess:
      # tf.global_variables_initializer().run()
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
          _, l, lr, predictions = sess.run(
            [optimizer, loss, learning_rate, train_prediction],
            feed_dict=feed_dict)
          
          total_time += (time.time() - start)

          # Print out the loss periodically.
          if step % 100 == 0:
              error, _ = error_rate(predictions, batch_labels)
              print('Step %d of %d' % (step, steps))
              print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
              print('Validation error: %.1f%%' % error_rate(
                    validation_prediction.eval(session=sess), validation_labels)[0])

      print('Total time for seq is ' + str(total_time))
      print('Time for seq per step is ' + str(total_time/steps))

      test_error, confusions = error_rate(test_prediction.eval(session=sess), test_labels)
      print('Test error: %.1f%%' % test_error)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  # tf.app.run()
