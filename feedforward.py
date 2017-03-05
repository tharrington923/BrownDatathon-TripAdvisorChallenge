"""
feedforward.py

Skeleton file for the 2-Layer Feed-Forward Neural Network for the MNIST Task.

Creates and trains the model, then evaluates performance on the test set.

Run via: `python feedforward.py`
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

# Fetch MNIST Dataset using the supplied Tensorflow Utility Function
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Setup the Model Parameters (Layer sizes, Batch Size, etc.)
INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 784, 100, 10  
BATCH_SIZE, NUM_TRAINING_STEPS = 100, 1000

### Start Building the Computation Graph ###

# Initializer - initialize our variables from standard normal with stddev 0.1
initializer = tf.random_normal_initializer(stddev=0.1)

# Setup Placeholders => None argument in shape lets us pass in arbitrary sized batches
X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])  
Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

######################## FILL ME IN ###########################
# Hidden Layer Variables
W_1 = tf.get_variable("Hidden_W", shape=[INPUT_SIZE, HIDDEN_SIZE], initializer=initializer)
b_1 = tf.get_variable("Hidden_b", shape=[HIDDEN_SIZE], initializer=initializer)

# Hidden Layer Transformation
hidden = tf.nn.relu(tf.matmul(X, W_1) + b_1)

# Output Layer Variables
W_2 = tf.get_variable("Output_W", shape=[HIDDEN_SIZE, OUTPUT_SIZE], initializer=initializer)
b_2 = tf.get_variable("Output_b", shape=[OUTPUT_SIZE], initializer=initializer)

# Output Layer Transformation
output = tf.matmul(hidden, W_2) + b_2
###############################################################

# Compute Loss
loss = tf.losses.softmax_cross_entropy(Y, output)

# Compute Accuracy
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Setup Optimizer
train_op = tf.train.AdamOptimizer().minimize(loss)

### Launch the Session, to Communicate with Computation Graph ###
with tf.Session() as sess:
    # Initialize all variables in the graph
    sess.run(tf.global_variables_initializer())

    # Training Loop
    for i in range(NUM_TRAINING_STEPS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        curr_acc, _ = sess.run([accuracy, train_op], feed_dict={X: batch_x, Y: batch_y})
        if i % 100 == 0:
            print("Step %d Current Training Accuracy: %.3f" % (i, curr_acc))
    
    # Evaluate on Test Data
    print('Test Accuracy: %.3f' % sess.run(accuracy, feed_dict={X: mnist.test.images, 
                                                                Y: mnist.test.labels}))
