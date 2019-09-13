import numpy as np
import tensorflow as tf
import zipfile
import collections
import random
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

filename = 'text8.zip'
n_iter = 100001
learning_rate = 1.0
vocabulary_size = 5000
batch_size = 128
embedding_size = 128
n_skips = 2
window_skips = 1
valid_size = 16
valid_window = 100
num_sample = 64


# step 1: get data list
def read_files(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

data = read_files(filename)

# step 2: generate dictionary, reverse_dictionary and get length etc.
def generate_dataset(data):
    num_words = len(data)
    count = [['UNK', -1]]
    count.extend(collections.Counter(data).most_common(vocabulary_size-1))

    dic = {}
    for val, count in count:
        dic[val] = len(dic)

    reverse_dic = dict(zip(dic.values(), dic.keys()))

    num_data = []
    for item in data:
        if item in dic:
            num_data.append(dic[item])
        else:
            num_data.append(0)

    return dic, reverse_dic, num_data, num_words

dic, reverse_dic, num_data, num_words = generate_dataset(data)
del(data)

# step 3: generate the batch data
index_data = 0
def generate_batch(batch_size, n_skips, window_skip):
    global index_data
    assert batch_size % n_skips == 0
    assert n_skips <= 2*window_skip

    batch_x = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch_y = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * window_skip + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(num_data[index_data])
        index_data = (index_data + 1)%num_words

    for i in range(batch_size // n_skips):
        target = window_skip
        avoid_target = [window_skip]

        for j in range(n_skips):
            while target in avoid_target:
                target = random.randint(0, span-1)
            avoid_target.append(target)
            batch_x[i*n_skips+j] = buffer[window_skip]
            batch_y[i*n_skips+j] = buffer[target]
        buffer.append(num_data[index_data])
        index_data = (index_data+1) % num_words

    return batch_x, batch_y

#### step 5: construct model

valid_exampled = np.random.choice(valid_window, valid_size, replace=False)

input_x = tf.placeholder(tf.int32, shape=[batch_size])
input_y = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_data = tf.constant(valid_exampled)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size]))
embed = tf.nn.embedding_lookup(embeddings, input_x)

nce_weight = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                              stddev=1.0 / np.sqrt(embedding_size)))
nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                      biases=nce_bias,
                      labels=input_y,
                      inputs=embed,
                      num_classes=vocabulary_size,
                      num_sampled=num_sample))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
norm_embeddings = embeddings / norm

valid_embeddings = tf.nn.embedding_lookup(norm_embeddings, valid_data)
similarity = tf.matmul(valid_embeddings, norm_embeddings, transpose_b=True)

# step 6: train model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Variables initialized.')

    average_loss = 0
    for step in range(n_iter):
        batch_x, batch_y = generate_batch(batch_size, n_skips, window_skips)
        feed_dict = {input_x: batch_x, input_y: batch_y}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            average_loss /= 2000
            print('Step: ', step, ' Avg loss: ', average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dic[valid_exampled[i]]
                top_k = 8
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word

                for k in range(top_k):
                    close_word = reverse_dic[nearest[k]]
                    log_str = "%s %s" % (log_str, close_word)
                print(log_str)
    final_embeddings = norm_embeddings.eval()






