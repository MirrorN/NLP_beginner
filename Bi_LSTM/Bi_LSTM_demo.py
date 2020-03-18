"""
任务:
make、 need、coal 等等每个单词都被看做是一个字符序列，使用 LSTM 来通过前三个字符预测最后一个字符
例如 使用 mak -> e  所以在预测的时候也就相当于一个分类问题 ，类别的数量就是所有的字符的数量 26

使用 双向LSTM 建立模型， 基本内容与之前的 TextLSTM.py 一致只是稍微改动成 Bi-LSTM

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

data_type= torch.FloatTensor
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
word_index_dict = {w: i for i, w in enumerate(char_arr)}
index_word_dict = dict(zip(word_index_dict.values(), word_index_dict.keys()))
# number of class (number of vocab)
n_class = len(word_index_dict)  # 26

seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']


# TextLSTM 的参数
n_step = 3
n_hidden = 128


def make_batch(seq_data):
    """
    输入数据处理为 one-hot 编码
    """
    input_batch = []
    target_batch = []
    for seq in seq_data:
        input = [word_index_dict[c] for c in seq[:-1]]
        target = word_index_dict[seq[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    return Variable(torch.Tensor(input_batch)), Variable(torch.LongTensor(target_batch))


class Text_Bi_LSTM(nn.Module):
    def __init__(self):
        super(Text_Bi_LSTM, self).__init__()

        # 指定 bidirectional = True
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.W = nn.Parameter(torch.randn([2*n_hidden, n_class]).type(data_type))
        # 注意这里 全连接层 的 W 是 num_directions * hidden_size
        self.b = nn.Parameter(torch.randn([n_class]).type(data_type))

    def forward(self, x):
        batch_size = len(x)
        x = x.transpose(0, 1)

        # hidden_state 和 cell_state 的维度是相同的  num_layers * num_directions = 2
        init_hidden_state = Variable(torch.zeros(1*2, batch_size, n_hidden))
        init_cell_state = Variable(torch.zeros(1*2, batch_size, n_hidden))

        outputs, (_, _) = self.lstm(x, (init_hidden_state, init_cell_state))
        outputs = outputs[-1]
        final_output = torch.mm(outputs, self.W) + self.b

        return final_output


input_batch, target_batch = make_batch(seq_data)
model = Text_Bi_LSTM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)

    if (epoch+1) % 100 == 0:
        print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss))

    loss.backward()
    optimizer.step()


#  test
inputs = [sen[:3] for sen in seq_data]
predict = model(input_batch).data.max(1, keepdim=True)[1]
print(inputs, '->', [index_word_dict[n.item()] for n in predict.squeeze()])

