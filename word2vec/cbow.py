import zipfile
import numpy as np
import collections
import random
import math
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt


def read_file(filename):
    with zipfile.ZipFile(filename) as f:
        data =  tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_file('text_words.zip')

def remove_fre_stop_word(words):
    t = 1e-5
    threshold = 0.8

    int_word_counts = collections.Counter(words)
    total_count = len(words)
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(t / f) for w, f in word_freqs.items()}
    train_words = [w for w in words if prob_drop[w] < threshold]

    return train_words

words = remove_fre_stop_word(words)
vocabulary_size = len(set(words))

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            data.append(dictionary[word])
        else:
            data.append(0)
            unk_count += 1
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

data_index = 0
def generate_batch(batch_size, bag_window):
    global data_index
    span = 2 * bag_window + 1
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size):
        buffer_list = list(buffer)
        labels[i, 0] = buffer_list.pop(bag_window)
        batch[i] = buffer_list
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

batch_size = 128
embedding_size = 128
bag_window = 2
valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64

train_dataset = tf.placeholder(tf.int32, shape=[batch_size, bag_window * 2])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


embeds = tf.nn.embedding_lookup(embeddings, train_dataset)
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels,tf.reduce_sum(embeds, 1), num_sampled, vocabulary_size))

optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001

with tf.Session() as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, bag_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

    print("*" * 10 + "final_embeddings:" + "*" * 10 + "\n", final_embeddings)
    fp = open('vector_cbow.txt', 'w', encoding='utf8')
    for k, v in reverse_dictionary.items():
        t = tuple(final_embeddings[k])

        s = ''
        for i in t:
            i = str(i)
            s += i + " "

        fp.write(v + " " + s + "\n")

    fp.close()

def plot_with_labels(low_dim_embs, plot_labels, filename='tsne_cbow.png'):
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
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    plot_labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, plot_labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")