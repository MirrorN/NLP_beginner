"""
Multi-layers LSTM
任务：

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

data_type = torch.FloatTensor
# 注意这里的 sentence 的写法 使用了一个 ()
# 实际上这是一个字符串（不是三句话）
sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

word_list = list(set(sentence.split()))
word_index_dict = {w: i for i, w in enumerate(word_list)}
index_word_dict = dict(zip(word_index_dict.values(), word_index_dict.keys()))
n_class = len(word_list)  # 27
max_len = len(sentence.split())  # 27
n_hidden = 5


def make_batch(sentence):
    """
    input_batch 的 shape  (batch_size=26, n_step=27, input_size=27)
    target_batch 的 shape (batch_size=26)
    注意这里的任务是下一个单词的预测，因为数据有限 所以采用的了这种方式，例如： 文本是 a b c d e，结果生成的数据分别是：
    a 0 0 0 0 -> b
    a b 0 0 0 -> c
    a b c 0 0 -> d
    a b c d 0 -> e
    然后每个单词又使用的是 one-hot 编码，所以每个单词的维度
    """
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_index_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        target = word_index_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return Variable(torch.Tensor(input_batch)), torch.LongTensor(target_batch)


class Bi_LSTM(nn.Module):
    def __init__(self):
        super(Bi_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Parameter(torch.randn([2 * n_hidden, n_class]).type(data_type))
        self.b = nn.Parameter(torch.randn(n_class).type(data_type))

    def forward(self, x):
        batch_size = len(x)
        x = x.transpose(0, 1)

        init_hidden_state = Variable(torch.randn([2, batch_size, n_hidden]))
        init_cell_state = Variable(torch.randn([2, batch_size, n_hidden]))

        output, (_, _) = self.lstm(x, (init_hidden_state, init_cell_state))
        output = output[-1]
        final_output = torch.mm(output, self.W) + self.b
        return final_output


input_batch, target_batch = make_batch(sentence)
model = Bi_LSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 100 == 0:
        print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss))

    loss.backward()
    optimizer.step()

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([index_word_dict[n.item()] for n in predict.squeeze()])
