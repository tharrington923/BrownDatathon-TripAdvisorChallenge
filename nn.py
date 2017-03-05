import csv
import tensorflow as tf
from random import shuffle

# Setup the Model Parameters (Layer sizes, Batch Size, etc.)
#INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 13, 500, 2 # 58.7%
#INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 13, 1000, 2 # 50.9%
#INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 13, 200, 2 # 61%
INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 13, 200, 2
#BATCH_SIZE, NUM_TRAINING_STEPS = 200, 1000
BATCH_SIZE, NUM_TRAINING_STEPS = 100, 2000
BATCH_INDEX = 0

NUMBER_TRAINING_SAMPLES = 200000 #BATCH_SIZE*NUM_TRAINING_STEPS
NUMBER_CLASS_1_SAMPLES = 100000
# NUMBER_TRAINING_SAMPLES = 750000 #BATCH_SIZE*NUM_TRAINING_STEPS
# NUMBER_CLASS_1_SAMPLES = 150000
NUMBER_CLASS_0_SAMPLES = NUMBER_TRAINING_SAMPLES - NUMBER_CLASS_1_SAMPLES

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

print("The ratio is ",len(class1)," to ",len(class0))

shuffle(class1)
shuffle(class0)

training_data = []
test_x = []
test_y = []

for x in range(NUMBER_CLASS_1_SAMPLES):
    training_data.append((class1[x],[0,1]))

for x in range(NUMBER_CLASS_0_SAMPLES):
    training_data.append((class0[x],[1,0]))

for x in range(NUMBER_CLASS_1_SAMPLES,len(class1)):
    test_x.append(class1[x])
    test_y.append([0,1])

for x in range(NUMBER_CLASS_0_SAMPLES,len(class0)):
    test_x.append(class0[x])
    test_y.append([1,0])

# for x in range(BATCH_SIZE*NUM_TRAINING_STEPS/2):
#     training_data.append((class1[x],[0,1]))
#     training_data.append((class0[x],[1,0]))
#
# for x in range(BATCH_SIZE*NUM_TRAINING_STEPS/2,len(class1)):
#     test_x.append(class1[x])
#     test_y.append([0,1])
#
# for x in range(BATCH_SIZE*NUM_TRAINING_STEPS/2,len(class0)):
#     test_x.append(class0[x])
#     test_y.append([1,0])


# for x in range(BATCH_SIZE*NUM_TRAINING_STEPS/2,BATCH_SIZE*NUM_TRAINING_STEPS):
#     test_x.append(class1[x])
#     test_x.append(class0[x])
#     test_y.append([0,1])
#     test_y.append([1,0])

shuffle(training_data)
print(len(training_data))
print(len(test_y))
print(len(test_x))

test_accuracies = []

initializer = tf.random_normal_initializer(stddev=0.1)


# def init_weights(shape):
#     return tf.Variable(tf.random_normal(shape, stddev=0.01))
#
# def model(X, w_h, w_o):
#     h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
#     return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

# Setup Placeholders => None argument in shape lets us pass in arbitrary sized batches
X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

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

#tf.constant(value,dtype=, shape=None,name="BATCH_SIZE")

### Launch the Session, to Communicate with Computation Graph ###
with tf.Session() as sess:
    # Initialize all variables in the graph
    for k in range(500):
        sess.run(tf.global_variables_initializer())

        def next_batch(batchIndex):
            return (batchIndex,batchIndex+BATCH_SIZE)

        # Training Loop
        for j in range(1):
            BATCH_INDEX = 0
            for i in range(NUM_TRAINING_STEPS):
                batch_start, batch_end = next_batch(BATCH_INDEX)
                BATCH_INDEX += BATCH_SIZE
                batch_x, batch_y = zip(*training_data[batch_start:batch_end])
                curr_acc, _ = sess.run([accuracy, train_op], feed_dict={X: batch_x, Y: batch_y})
                if i % 100 == 0:
                    print("Step %d Current Training Accuracy: %.3f" % (i, curr_acc))

        # Evaluate on Test Data
        test_accuracies.append(sess.run(accuracy, feed_dict={X: test_x,
                                                                    Y: test_y}))
        print('Test Accuracy: %.3f' % test_accuracies[-1])

    print(test_accuracies)
