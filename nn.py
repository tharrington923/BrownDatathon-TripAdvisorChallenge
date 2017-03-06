# Note: This neural network example is based off of the examples found in:
# https://hackernoon.com/tensorflow-in-a-nutshell-part-three-all-the-models-be1465993930#.clv8zj7xg

# Import the important libraries that will be used

import csv
import tensorflow as tf
from random import shuffle

# Define model params

nFeatures, nOutputLayer = 13, 2
nHiddenLayer = 200

# Define training variables

nBatch = 100
nTrainSteps = 2000
bIndex = 0

nTrainingSamples = 200000 # Note: This must be equal to nBatch*nTrainSteps
nClass1Samples = 100000 # We set the number of class 1 to train the network
nClass0Samples = nTrainingSamples - nClass1Samples

# Loading the training data that was divided into the two classes

with open('class1.csv', 'rb') as csvfile:
    nextline = csv.reader(csvfile, delimiter=' ')
    class1 = []
    for x in nextline:
        yList = []
        for y in x[0].split(','):
            yList.append(float(y))
        class1.append(yList)

with open('class0.csv', 'rb') as csvfile:
    nextline = csv.reader(csvfile, delimiter=' ')
    class0 = []
    for x in nextline:
        yList = []
        for y in x[0].split(','):
            yList.append(float(y))
        class0.append(yList)


print("The number of class 1 samples: %d" % (len(class1)))
print("The number of class 0 samples: %d" % (len(class0)))

# Shuffle the data sets so the training data is randomly selected each time

shuffle(class1)
shuffle(class0)

# Create lists to store the data for training and the data for testing separately
# Storing the input and output vectors for each training sample as a tuple
# training_data is a list of tuples containing (input, output) vectors

training_data = []
test_x = []
test_y = []

for x in range(nClass1Samples):
    training_data.append((class1[x],[0,1]))

for x in range(nClass0Samples):
    training_data.append((class0[x],[1,0]))

for x in range(nClass1Samples,len(class1)):
    test_x.append(class1[x])
    test_y.append([0,1])

for x in range(nClass0Samples,len(class0)):
    test_x.append(class0[x])
    test_y.append([1,0])

# Shuffle the data so that it does not alternate between the two classes

shuffle(training_data)

# Create a list to store the percentage of properly indentified test samples for
# each neural network

test_accuracies = []

# TF initializer for normal distribution (will be used to initialize network weights)

initializer = tf.random_normal_initializer(stddev=0.1)

# Create TF placeholders for the input and output vectors to our network
# None is used as the first argument of shape to allow a value to be passed into shape (used for variable batch size)
# Shape takes two arguments for a 2D tensor (matrix)

X = tf.placeholder(tf.float32, shape=[None, nFeatures])
Y = tf.placeholder(tf.float32, shape=[None, nOutputLayer])

# Create the hidden layer weight matrix and bias

wHidden = tf.get_variable("wHidden", shape=[nFeatures, nHiddenLayer], initializer=initializer)
bHidden = tf.get_variable("bHidden", shape=[nHiddenLayer], initializer=initializer)

# Propagate input through hidden layer

hOutput = tf.nn.relu(tf.matmul(X, wHidden) + bHidden)

# Output layer weight matrix and bias

wOutput = tf.get_variable("wOutput", shape=[nHiddenLayer, nOutputLayer], initializer=initializer)
bOutput = tf.get_variable("bOutput", shape=[nOutputLayer], initializer=initializer)

# Output layer vector

output = tf.matmul(hOutput, wOutput) + bOutput

# Compute the loss- difference between output of network and the expected output

loss = tf.losses.softmax_cross_entropy(Y, output)

# Variables to store the accuracy and the correct prediction Compute Accuracy

cPrediction = tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
accuracy = 100 * tf.reduce_mean(tf.cast(cPrediction, tf.float32))

# Setup the TF optimizer. We chose the AdamOptimizer

train_op = tf.train.AdamOptimizer().minimize(loss)

# Now we launch the TF session to train the network
with tf.Session() as sess:

    # The number of neural networks we train
    for k in range(500):

        # Initialize the TF variables
        sess.run(tf.global_variables_initializer())

        def next_batch(batchIndex):
            return (batchIndex,batchIndex+nBatch)

        # Start the loop to train the neural network in batches
        # The number of times we loop and train the network with all the training data
        # We can adjust this to train the network multiple times with the training data
        for j in range(1):

            # Zero the batch index
            bIndex = 0

            # Loop over all the batches
            for i in range(nTrainSteps):

                # Set the start and end indices for the training batch
                batch_start, batch_end = next_batch(bIndex)
                bIndex += nBatch

                # Variables to hold the batch of training data
                batch_x, batch_y = zip(*training_data[batch_start:batch_end])

                # Variable to hold the current accuracy
                cAccuracy, _ = sess.run([accuracy, train_op], feed_dict={X: batch_x, Y: batch_y})

                # Print the training accuracy every 100 Steps
                if i % 100 == 0:
                    print("Batch %d: Current Training Accuracy: %.3f" % (i/100, cAccuracy))

        # Evaluate on Test Data and print the accuracy
        test_accuracies.append(sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
        print('Test Accuracy: %.3f' % test_accuracies[-1])

    # Print all
    print(test_accuracies)
