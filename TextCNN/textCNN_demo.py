import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# define some Hyperparameters and constants
embedding_dims = 2
sequence_length = 3
num_class = 2
filter_sizes = [2, 2, 2]
num_filters = 3
epoch_num = 1500
learning_rate = 0.001

# training data
sentences = ["i love you","he loves me", "she likes baseball", "i hate you","sorry for that", "this is awful"]
labels = [1,1,1,0,0,0]

# step 1: generate datasets
def generate_datasets():
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    vocab_size = len(word_list)
    word_dict = {}
    for key, val in enumerate(word_list):
        word_dict[val] = key
    reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

    return word_list, vocab_size, word_dict, reverse_dict

word_list, vocab_size, word_dict, reverse_dict = generate_datasets()


# step 2: define inputs, outputs and weights bias etc.
x_train = []
for sen in sentences:
    x_train.append([word_dict[i] for i in sen.split()])
y_train = []
for label in labels:
    y_train.append(np.eye(num_class)[label])

x_train = np.array(x_train)
y_train = np.array(y_train)

X = tf.placeholder(tf.int32, shape=[None, sequence_length], name='x-inputs')
y = tf.placeholder(tf.int32, shape=[None, num_class], name='y-inputs')
embedding_vectors = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_dims], minval=-1.0, maxval=1.0))
embeds = tf.nn.embedding_lookup(embedding_vectors, X)
embeds = tf.expand_dims(embeds, -1)        # [None, 3, 2, 1]

# step 3: define the model'structure
pool_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_dims, 1, num_filters]
    weights_filter = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1))
    bias_filter = tf.Variable(tf.constant(0.1, shape=[num_filters]))

    conv = tf.nn.conv2d(input=embeds,
                        filter=weights_filter,
                        strides=[1,1,1,1],
                        padding='VALID')
    relu_conv = tf.nn.relu(tf.nn.bias_add(conv, bias_filter))
    pool = tf.nn.max_pool(value=relu_conv,
                          ksize = [1, sequence_length-filter_size+1, 1, 1],
                          strides=[1, 1, 1, 1],
                          padding='VALID')
    pool_outputs.append(pool)

vector_concat = tf.concat(pool_outputs, axis=3)
flat_vector = tf.reshape(vector_concat, [-1, len(filter_sizes)*num_filters])
weights_full = tf.Variable(tf.truncated_normal(shape=[num_filters*len(filter_sizes), num_class], stddev=0.1))
bias_full = tf.Variable(tf.constant(0.1, shape=[num_class]))
pred = tf.nn.xw_plus_b(flat_vector, weights_full, bias_full)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# step 4: train

loss_sav = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_num):
        _, loss = sess.run([optimizer, cost], feed_dict={X:x_train, y:y_train})
        loss_sav.append([epoch, loss])
        if epoch % 100 == 0:
            print("Epoch: ", epoch, " Loss: ", loss)


def test(test):
    x_train = np.array([[word_dict[word] for word in test.split()]])
    predict = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predict = sess.run(pred, feed_dict={X:x_train})
    if(np.argmax(predict[0]) == 1):
        print("Good meaning.")
    else:
        print("Bad meaning.")


def plot_loss(data):
    data = np.array(data)
    plt.figure()
    plt.plot(data[:,0], data[:,1])
    plt.title("Loss")
    plt.grid(True)
    plt.show()


test_sen = "you love me"
test(test_sen)
plot_loss(loss_sav)
