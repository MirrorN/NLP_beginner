import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

sentences = ['I like dog', 'I love coffee', 'I hate milk']
hidden = 2
iter_n = 100

# step 1: produce wordlist, dictionary, reverse dictionary, length of dictionary
def build_dataset():
    word_list = list(" ".join(sentences).split(" "))
    word_set = set(word_list)
    counter_temp = Counter(word_list).most_common()
    dic = {}
    for val, cou in counter_temp:
        dic[val] = len(dic)
    reverse_dic = dict(zip(dic.values(), dic.keys()))
    vocabulary_size = len(word_set)

    return dic, reverse_dic, vocabulary_size

dic, reverse_dic, vocabulary_size = build_dataset()
print(vocabulary_size)
print(reverse_dic)
# step 2: generate batch data
def generate_batch():
    X_batch = []
    Y_batch = []

    for sentence in sentences:
        words = sentence.split(" ")
        x_index = [dic[term] for term in words[:-1]]
        y_index = dic[words[-1]]

        X_batch.append(np.eye(vocabulary_size)[x_index])
        Y_batch.append(np.eye(vocabulary_size)[y_index])

    return X_batch, Y_batch

X_batch, Y_batch = generate_batch()   # X_batch.shape = [3, 2, 7]

print(X_batch)
print('-'*30)
print(Y_batch)

# step 3: construct the model
X = tf.placeholder(tf.float32, [None, 2, vocabulary_size])
Y = tf.placeholder(tf.float32, [None, vocabulary_size])
inputs = tf.reshape(X, shape=[-1, 2*vocabulary_size])

W1 = tf.Variable(tf.random_normal([2*vocabulary_size, hidden]))
b1 = tf.Variable(tf.random_normal([hidden]))
W2 = tf.Variable(tf.random_normal([hidden, vocabulary_size]))
b2 = tf.Variable(tf.random_normal([vocabulary_size]))

out_layer1 = tf.add(tf.matmul(inputs, W1), b1)
# out_layer1 = tf.matmul(inputs, W1) + b1
tanh_out_layer1 = tf.nn.tanh(out_layer1)

pred = tf.add(tf.matmul(tanh_out_layer1, W2), b2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

# step 4: train
step_record = []
loss_record = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(iter_n):
        loss_val, _ = sess.run([loss, optimizer], feed_dict={X: X_batch, Y:Y_batch})
        if step % 10 == 0:
            print('Step: ', step, ' Loss: ', loss_val)
        step_record.append(step)
        loss_record.append(loss_val)

    predict = sess.run(pred, feed_dict={X:X_batch})
    cou = 0
    for sen in X_batch:
        word1 = reverse_dic[np.argsort(sen[0])[-1]]
        word2 = reverse_dic[np.argsort(sen[1])[-1]]
        print(word1, ' ', word2, '-->', reverse_dic[np.argsort(predict[cou])[-1]])
        cou += 1


plt.figure()
plt.plot(step_record, loss_record)
plt.title('train-loss')
plt.show()








