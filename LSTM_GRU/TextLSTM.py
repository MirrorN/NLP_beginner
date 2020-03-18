"""
任务:
make、 need、coal 等等每个单词都被看做是一个字符序列，使用 LSTM 来通过前三个字符预测最后一个字符
例如 使用 mak -> e  所以在预测的时候也就相当于一个分类问题 ，类别的数量就是所有的字符的数量 26

Pytorch 建立 LSTM 使用 torch.nn.LSTM()
def __init__(self,
             input_size: int,
             hidden_size: int,
             num_layers: int = ...,
             bias: bool = ...,
             batch_first: bool = ...,
             dropout: float = ...,
             bidirectional: bool = ...,
             nonlinearity: str = ...)

可以看到 torch.nn.LSTM() 与 torch.nn.RNN() 参数是一致的
同样需要对数据进行处理，来符合 LSTM 的要求：这里是batch_first为默认值 False

输入数据： 对于数据x: (sequence_length, batch_size, input_size)
           初始hidden state： (num_layers * num_directions，batch_size, hidden_size)
           初始cell state：与 hidden_state 相同

输出数据：output: (sequence_length, batch_size, hidden_size * num_directions)
          hidden_state: (num_layers * num_directions, batch_size, hidden_size)
          cell_state: 与 hidden_state 相同

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

data_type= torch.FloatTensor
# 这里我们是对字符进行预测 所以对26个字母进行编号 方便之后处理为 one-hot 形式
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


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(data_type))
        # torch.randn() 函数指定维度的时候可以不用加 []，但是显然加上显得清晰
        self.b = nn.Parameter(torch.randn([n_class]).type(data_type))

    def forward(self, x):
        # 调整数据从 [batch_size, n_step, dims(n_class)] 变成 [n_step, batch_size, dims(n_class)]
        batch_size = len(x)
        x = x.transpose(0, 1)
        # hidden_state 和 cell_state 的维度是相同的  num_layers * num_directions = 1
        # 这里使用了自定义的 hidden_state 和 cell_state 也可以不使用

        init_hidden_state = Variable(torch.zeros(1, batch_size, n_hidden))
        init_cell_state = Variable(torch.zeros(1, batch_size, n_hidden))

        # 输出结果会默认输出三个：所有时刻的output，最后时刻的 hidden_state 和 cell_state
        # outputs, (_, _) = self.lstm(x) 不使用自定义初始状态
        outputs, (_, _) = self.lstm(x, (init_hidden_state, init_cell_state))
        # 这里依然使用最后一个时刻的结果作为最后预测(分类的依据)
        outputs = outputs[-1]
        final_output = torch.mm(outputs, self.W) + self.b

        return final_output


input_batch, target_batch = make_batch(seq_data)
model = TextLSTM()

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

