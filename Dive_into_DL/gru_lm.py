import torch
import torch.nn as nn
import math
import torch.optim as optim
import collections
import zipfile
import numpy as np
import time

batch_size = 32
hidden_num = 256
epoch_num = 10
learning_rate = 1e-3
clipping_theta = 1e-2
step_num = 5
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_file(filename):
    with zipfile.ZipFile(filename) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            # read file as a string variable --> f.read() method ---> decode()
            corpus = f.read().decode('utf-8')
    corpus = corpus.replace('\n', ' ').replace('\r', ' ')[: 10000]
    return corpus


filename = 'lm_data/jaychou_lyrics.txt.zip'
corpus = read_file(filename)

def prepare_data1(corpus):
    idx2char = list(set(corpus))
    char2idx = dict([(char, idx) for idx, char in enumerate(idx2char)])
    vocab_size = len(char2idx)
    corpus_indices = [char2idx[i] for i in corpus]
    return char2idx, idx2char, vocab_size, corpus_indices


def prepare_data2(corpus):
    count_corpus = dict(collections.Counter(corpus).most_common(8000))
    chars = list(count_corpus.keys())
    char2idx = dict(zip(chars, list(range(len(chars)))))
    idx2char = dict(zip(char2idx.values(), char2idx.keys()))
    vocab_size = len(chars)
    corpus_indices = [char2idx[char] for char in corpus]

    return char2idx, idx2char, vocab_size, corpus_indices


char2idx, idx2char, vocab_size, corpus_indices = prepare_data2(corpus)


def data_iter_random(corpus, batch_size, step_num, device):
    ins_num = (len(corpus)-1) // step_num
    batch_num = ins_num // batch_size
    ins_indices = list(range(ins_num))
    np.random.shuffle(ins_indices)

    def get_data(pos):
        return corpus[pos: pos+step_num]

    for i in range(batch_num):
        i = i * batch_size
        X = [get_data(j*step_num) for j in ins_indices[i: i+batch_size]]
        Y = [get_data(j*step_num+1) for j in ins_indices[i: i+batch_size]]
        yield torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float, device=device)


def data_iter_consecutive(corpus_indices, batch_size, step_num, device=None):
    corpus_indices = torch.tensor(corpus_indices,dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_len * batch_size].view(batch_size, -1)
    epoch_size = (batch_len-1) // step_num
    for i in range(epoch_size):
        i = i * step_num
        X = indices[:, i: i+step_num]
        Y = indices[:, i+1: i+step_num+1]
        yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
    '''
    x: [batch]
    return: [batch, n_class]
    '''
    x = x.long()
    tmp = torch.eye(n_class)
    return tmp[x]


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=hidden_num)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(hidden_num, vocab_size)
        self.state = None

    def forward(self, X, state):
        x = one_hot(X, vocab_size)  # [bsz, step_num, vocab_size]
        x = x.permute(1, 0, 2).to(device)  # [step_num, bsz, vocab_size]
        Y, self.state = self.gru(x, state)  # Y: [step_num, bsz, hidden_num]
        output = self.dense(Y.view(-1, Y.shape[-1]))  # [step_num*bsz, hidden_num]
        return output, self.state


def predict(prefix, char_num, model, vocab_size, device, idx2char, char2idx):
    state = None
    output = [char2idx[prefix[0]]]
    for t in range(char_num + len(prefix) - 1):
        X = torch.tensor(output[-1], device=device).view(1, 1)  # [bsz, step_num]
        if state is not None:
            if isinstance(state, tuple): # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)

        if t < len(prefix) - 1:
            output.append(char2idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return "".join([idx2char[i] for i in output])


def train(model, hidden_num, vocab_size, device, corpus_indices, idx2char, char2idx, epoch_num, step_num, lr,
          batch_size, pred_period, pred_len, prefix):
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(epoch_num):
        l_sum = 0.
        cnt = 0
        start = time.time()
        data_iter = data_iter_random(corpus_indices, batch_size, step_num, device)
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            (output, state) = model(X, state)

            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            cnt += y.shape[0]

        try:
            perplexity = math.exp(l_sum / cnt)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefix:
                print(' -', predict(
                    prefix, pred_len, model, vocab_size, device, idx2char,
                    char2idx))


model = GRUModel()
pred_period, pred_len, prefixes = 1, 50, ['分开', '不分开']
train(model, hidden_num, vocab_size, device, corpus_indices, idx2char, char2idx, epoch_num,
      step_num, learning_rate, batch_size, pred_period, pred_len, prefixes)
