import torch
import torch.nn as nn
import time
import zipfile
import torch.optim as optim
import numpy as np
import math

batch_size = 32
hidden_num = 256
epoch_num = 10
learning_rate = 1e-3
clipping_theta = 1e-2
step_num = 5
# device = 'cpu'
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_corpus(filename):
    with zipfile.ZipFile(filename) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    return corpus[: 10000]


filename = './lm_data/jaychou_lyrics.txt.zip'
corpus = read_corpus(filename)


def prepare_corpus(corpus):
    idx2char = list(set(corpus))
    char2idx = dict([(char, idx) for idx, char in enumerate(idx2char)])
    vocab_size = len(idx2char)
    corpus_indices = [char2idx[char] for char in corpus]
    return idx2char, char2idx, vocab_size, corpus_indices


idx2char, char2idx, vocab_size, corpus_indices = prepare_corpus(corpus)


def data_iter_random(corpus_indices, batch_size, step_num, device=None):
    ins_num = (len(corpus_indices)-1) // step_num
    batch_num = ins_num // batch_size
    ins_indices = list(range(ins_num))
    np.random.shuffle(ins_indices)

    def get_data(pos):
        return corpus_indices[pos: pos+step_num]

    for i in range(batch_num):
        i = i * batch_size
        X = [get_data(j*step_num) for j in ins_indices[i: i+batch_size]]
        Y = [get_data(j*step_num+1) for j in ins_indices[i: i+batch_size]]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


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


def clipping_grad(params, theta, device):
    norm = torch.tensor([0.], dtype=torch.float32, device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_num)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(hidden_num, vocab_size)
        self.state = None

    def forward(self, inputs, state):
        X = one_hot(inputs, self.vocab_size)
        X = X.permute(1, 0, 2).to(device)
        Y, self.state = self.rnn(X, state)
        # Y: [seq_len, bsz, hidden_num] -> view -> [seq_len * bsz, hidden_num]
        output = self.dense(Y.view(-1, Y.shape[-1]))
        # otuput: [seq_len* bsz, vocab_size]
        return output, self.state


def predict(prefix, chars_num, model, vocab_size, device, idx2char, char2idx):
    state = None
    output = [char2idx[prefix[0]]]
    for t in range(chars_num + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
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

# test
# model = RNNModel()
# model = model.to(device)
# print(predict('分开', 10, model, vocab_size, device, idx2char, char2idx))


def train(model, hidden_num, vocab_size, device, corpus_indices, idx2char, char2idx, epoch_num, step_num, lr,
          clipping_theta, batch_size, pred_period, pred_len, prefix):
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
            clipping_grad(model.parameters(), clipping_theta, device)
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


model = RNNModel()
pred_period, pred_len, prefixes = 1, 50, ['分开', '不分开']
train(model, hidden_num, vocab_size, device, corpus_indices, idx2char, char2idx, epoch_num,
      step_num, learning_rate, clipping_theta, batch_size, pred_period, pred_len, prefixes)