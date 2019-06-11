# encoding=utf8
import collections
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl

filename = 'text_words.zip'


def read_data(filename):
    '''
    :param filename:  文件名(.zip)
    :return: 词语列表
    '''
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)

def remove_fre_stop_word(words):
    '''
    去掉一些高频的停用词 比如 的之类的
    :param words:  词语列表
    :return: 词语列表
    '''
    t = 1e-5  # t 值
    threshold = 0.8  # 剔除概率阈值

    int_word_counts = collections.Counter(words)
    total_count = len(words)
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(t / f) for w, f in word_freqs.items()}   # 计算删除概率
    train_words = [w for w in words if prob_drop[w] < threshold]
    return train_words

words = remove_fre_stop_word(words)
words_size = len(words)                     # words中分词数量(含重复)
vocabulary_size = len(set(words))           # words中分词数量(不含重复值)

def build_dataset(words):
    '''
    构建
    :param words:  词语列表
    :return: data 词语列表中每个词对应编号的列表 长度words_size
             count 词频统计 按词频大小降序  长度vocabulary_size
             dictionary 按词频生成词典 词频越大 序号越小 形式如 ('的'， 1)
             reverse_dictionary 将上面的词典翻转 形式如(1， ‘的’)
    '''
    count = [['UNK', -1]]
# 按照词频降序排列
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
# 词语不在词典中 则标记为unk_count
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
# 利用zip（）函数 翻转字典键值对
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words


data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    '''
    :param batch_size:  batch的大小
    :param num_skips:   对于每个词语的训练样本数量
    :param skip_window:  词语的上下文窗口半径
    :return: batch  batch_size * 1 的数组
             labels batch_size * 1 的数组
    '''
# data_index是全局变量，并且需要改变 所以声明global
    global data_index
# 两个判断 不符合条件提示错误信息
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
# 一个span包括前后各skip_window个词语 并包括自身，所以是 2*skip_window+1
    span = 2 * skip_window + 1
# 定义一个长度固定的队列 长度为span
    buffer = collections.deque(maxlen=span)
# 第一次在队列中填充数据
# 从这里可以看到 词语在训练数据的使用形式是编号，通过这个编号定位每个词语
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
# 详细解释这一部分
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels




batch_size = 128
embedding_size = 128
skip_window = 2
num_skips = 4
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)


nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

# loss = tf.reduce_mean(
#     tf.nn.sampled_softmax_loss(nce_weights, nce_biases, train_labels,
#                                embed, num_sampled, vocabulary_size))
# 这里设置num_sampled=num_sampled就是在负采样的时候默认执行 P(k) = (log(k + 2) - log(k + 1)) / log(range_max + 1)

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


num_steps = 100001
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]   # 自己与自己点乘肯定是最大的所以从1开始取  argsort是从小到大，所以可以取负号之后排序
                log_str = "nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary.get(nearest[k], None)
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

    final_embeddings = normalized_embeddings.eval()
    print("*" * 10 + "final_embeddings:" + "*" * 10 + "\n", final_embeddings)
    fp = open('vector_skip_gram.txt', 'w', encoding='utf8')
    for k, v in reverse_dictionary.items():
        t = tuple(final_embeddings[k])

        s = ''
        for i in t:
            i = str(i)
            s += i + " "

        fp.write(v + " " + s + "\n")

    fp.close()

# 词向量降维（2） + 绘图
def plot_with_labels(low_dim_embs, plot_labels, filename='tsne_skip_gram.png'):
    assert low_dim_embs.shape[0] >= len(plot_labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(plot_labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(u'{}'.format(label),
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文字符
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示正负号

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    plot_labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, plot_labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
