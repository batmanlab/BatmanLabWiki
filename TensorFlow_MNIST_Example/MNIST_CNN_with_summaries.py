# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

 This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pdb
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
  """Create a weight variable with appropriate initialization."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# ReLU( CONV2D( X, W) + B )
def conv2d(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME' ):
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = weight_variable([filter_height, filter_width,
                              input_channels, num_filters])
    biases = bias_variable([num_filters])
    
    conv = tf.nn.conv2d(x, weights, strides=[1, stride_y, stride_x, 1], padding=padding)
    
    # Add biases
    bias = tf.nn.bias_add(conv, biases)
    #bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
    return relu
  
def max_pool(x, filter_height, filter_width, stride_y, stride_x,
             name, padding='SAME'):
  
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides=[1, stride_y, stride_x, 1], padding=padding, name=name)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)

def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases
    weights = weight_variable([num_in, num_out])
    biases = weight_variable([num_out])

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)
      return relu
    else:
      return act
  
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var) 

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.

  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

def feed_dict1(train,mnist):
  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  if train:
    xs, ys = mnist.train.next_batch(100)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

    
def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 10)
  
  # Creating the graph
  
  #Convolution Layer 1
  hidden_conv1 = conv2d(x_image, filter_height=5, filter_width=5, num_filters=32, stride_y=1, stride_x=1, name='conv1',
         padding='SAME' )
  #Max Pool Layer 1
  hidden_maxPool1 = max_pool(hidden_conv1, filter_height=2, filter_width=2, stride_y=2, stride_x=2, name='pool1', padding='SAME') #Output Dimension: 14 x 14x 32
  #Convolution Layer 2
  hidden_conv2 = conv2d(hidden_maxPool1, 5, 5, 64, 1, 1, name='conv2')
  #Max Pool Layer 2
  hidden_maxPool2 = max_pool(hidden_conv2, 2, 2, 2, 2, name='pool2') #Output dimension : 7 x 7 x 64
  #Flatten the Max pool layer
  hidden_maxPool2_flattened = tf.reshape(hidden_maxPool2, [-1, 7*7*64])
  #Fully Connected Layer
  hidden_fc1 = fc(hidden_maxPool2_flattened, num_in=7*7*64, num_out=1024, name='fc1')
  #Dropout Layer
  keep_prob = tf.placeholder(tf.float32)
  hidden_fc1_dropout = dropout(hidden_fc1, keep_prob)
  #Fully connected layer with un-scaled activations
  y_predicted = fc(hidden_fc1_dropout, num_in=1024, num_out=10, name='fc2',relu = False )  
    
  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_predicted)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries
  
  #Training Phase
  for i in range(FLAGS.max_steps):
    xs, ys = mnist.train.next_batch(100)
    if i % 100 == 0: # Record execution stats
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, acc = sess.run([merged, train_step],
                            feed_dict={
        x:xs, y_: ys, keep_prob: FLAGS.dropout},
                            options=run_options,
                            run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
    else: # Record a summary
      summary, acc = sess.run([merged, train_step], feed_dict={
        x:xs, y_: ys, keep_prob: FLAGS.dropout})
      train_writer.add_summary(summary, i)
      
  
  # Record summaries and test-set accuracy
  summary, acc = sess.run([merged, accuracy], feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
  test_writer.add_summary(summary,1)
  print('Testing Accuracy: %s' % (acc))

  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/pylon2/ms4s88p/singla/MNIST_Data/',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/pylon2/ms4s88p/singla/MNIST_Data/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
