import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sentences = [ "i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]
word_sequences = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
dic = {val: key for key, val in enumerate(word_list)}

batch_size = 20
embedding_size = 2
num_sampled = 10
vocabulary_size = len(word_list)


skip_gram = []

for i in range(1, len(word_sequences)-1):
    target = dic[word_sequences[i]]
    context_pre = dic[word_sequences[i-1]]
    context_post = dic[word_sequences[i+1]]
    skip_gram.append([target, context_pre])
    skip_gram.append([target, context_post])

def random_batch(data, size):
    batch_x = []
    batch_y = []

    random_indice = np.random.choice(len(data), size, replace=False)
    for i in random_indice:
        batch_x.append(data[i][0])
        batch_y.append([data[i][1]])
    return batch_x, batch_y

# Model
x = tf.placeholder(dtype=tf.int32, shape=[batch_size])
y = tf.placeholder(dtype=tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1.0, maxval=1.0))
embeds = tf.nn.embedding_lookup(embeddings, x)

nce_weights = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1.0, maxval=1.0))
nce_bias = tf.Variable(tf.zeros([vocabulary_size]))

cost = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                      biases=nce_bias,
                      num_sampled=num_sampled,
                      num_classes=vocabulary_size,
                      inputs=embeds,
                      labels=y))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(5000):
        batch_x, batch_y = random_batch(skip_gram, batch_size)
        feed = {x: batch_x, y:batch_y}
        _, loss = sess.run([optimizer, cost], feed_dict=feed)

        if epoch % 500 == 0:
            print('Epcoh : ', epoch, ' Loss : ', loss)

    final_embeddings = embeddings.eval()

plt.figure()
for i, label in enumerate(word_list):
    x,y = final_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()




