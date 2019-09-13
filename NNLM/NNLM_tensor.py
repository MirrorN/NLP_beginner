import numpy as np
import collections
import tensorflow as tf

sentences = ['i like dog', 'i love coffee', 'i hate milk']

wordlist = list(set(" ".join(sentences).split()))

dic = {}
temp = collections.Counter(" ".join(sentences).split()).most_common()
for key, val in temp:
    dic[key] = len(dic)
reverse_dic = dict(zip(dic.values(), dic.keys()))
n_class = len(wordlist)

n_step = 2
n_hidden = 2

# produce batch data
def generate_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [dic[i] for i in word[:-1]]
        target = dic[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])
    return input_batch, target_batch


inp, tar = generate_batch(sentences)
# tar = np.reshape(tar, [-1, 7])


# struct model
X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])

input = tf.reshape(X, shape=[-1, n_step*n_class])
W = tf.Variable(tf.random_normal([n_step*n_class, n_hidden]))
b1 = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b2 = tf.Variable(tf.random_normal([n_class]))

tanh = tf.nn.tanh(b1+ tf.matmul(input, W))
output = b2+tf.matmul(tanh, U)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
predict = tf.argmax(output, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(5000):
        _, lo = sess.run([optimizer, loss], feed_dict={X:inp, Y:tar})
        if epoch % 500 == 0:
            print('step:', epoch, ' loss: ', lo)

# predict
    res = sess.run([predict], feed_dict={X:inp})
    print(res)

print(reverse_dic)



