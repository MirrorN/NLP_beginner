"""
任务： 使用 RNN 进行文本预测，例如 "i like dog" 使用 "i like" 来预测 "dog"

关于 pytorch 中 RNN 的实现
Pytorch 中可以通过两种方式来调用 RNN
torch.nn.RNNCell()：这种方式只能接收单步的输入 而且必须传入隐藏状态
torch.nn.RNN()：这种方式更简单 可以直接接收一个序列的输入
并且默认传入全0的隐藏状态或者传入自己预定义的隐藏状态

torch.nn.RNN 的定义：（这里的input_size 很多时候也写作 feature_size）
def __init__(self,
             input_size: int,   输入数据的特征数
             hidden_size: int,    隐藏层的特征数
             num_layers: int = ...,  网络层数
             bias: bool = ...,    是否使用偏置 默认是 true
             batch_first: bool = ..., 如果是True 那么输入的tensor的shape是[batch_size, n_step, input_size]
                                      输出的时候也是 [batch_size, n_step, input_size]
                                      默认为False 也就是 [n_step, batch_size, input_size]
             dropout: float = ...,   如果非零 则除了最后一层意外 其他层都会在输出时加上一个 dropout层
             bidirectional: bool = ...,   是否使用双向RNN 默认是 false
             nonlinearity: str = ...)  非线性激活函数，默认是 tanh

输入数据： 对于数据x: (sequence_length, batch_size, input_size)
           初始隐藏层： (num_layers * num_directions，batch_size, hidden_size)

输出结果： 对于输出数据（每一步都会有一个输出）  (sequence_length, batch, hidden_size * num_directions)
           输出状态（这里会输出最后一步的隐藏状态） (num_layers * num_directions, batch_size, hidden_size)

RNN 单元的计算公式为 h_t = tanh(W_ih * x_t + b_in + W_hh * h_t-1 + b_nn)
其中
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

data_type = torch.FloatTensor

sentences = ["i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
# 数据中每个单词都进行编号 方便之后处理为 one-hot 形式
word_index_dict = {w: i for i, w in enumerate(word_list)}
index_word_dict = dict(zip(word_index_dict.values(), word_index_dict.keys()))
vocab_len = len(word_list)
n_class = len(word_list)

batch_size = len(sentences)
# 步长 也就是RNN cell 展开有两个（两个单词）
n_step = 2
# 隐藏层单元数
n_hidden = 5


def make_batch(sentences):
    """
    生成 batch 数据集
    input 数组保存的是 one-hot 形式的数据 最后返回的结果 shape是 [batch, seq_length, 单词的 one-hot]
    target 数组保存的是 label  shape [batch]
    """
    inputs = []
    targets = []
    for sen in sentences:
        words = sen.split()
        input = [word_index_dict[word] for word in words[:-1]]
        target = word_index_dict[words[-1]]

        inputs.append(np.eye(vocab_len)[input])
        targets.append(target)
    return inputs, targets


# inputs_batch : (batch_size, seq_length, input_size(feature_dim))
inputs_batch, targets_batch = make_batch(sentences)
inputs = Variable(torch.Tensor(inputs_batch))
targets = Variable(torch.LongTensor(targets_batch))


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        # 两个重要的参数 input_size 是输入数据的特征数,这里每个单词都是使用的 one-hot 编码，所以特征数也就是 vocab 的长度
        # hidden_size 是隐藏层的特征数 这作为超参数指定
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        # 最后要对输出的结果(最后一个output，这里没有使用最后时刻的hidden state)连一个全连接分类器
        self.W = nn.Parameter(torch.randn([n_hidden, n_class]).type(data_type))
        self.b = nn.Parameter(torch.randn([n_class]).type(data_type))

    def forward(self, init_hidden_state, x):
        # 将输入的数据维度调整为 : [n_step, batch_size, n_class]
        x = x.transpose(0, 1)
        # 最后产生两个输出：
        # outputs : [n_step, batch_size, num_directions=1 * n_hidden]
        # 最后一个时刻的 hidden state : [num_layers(1) * num_directions(1), batch_size, n_hidden]
        # 这里没有使用 hidden_state 所以使用 _ 接收
        outputs, _ = self.rnn(x, init_hidden_state)
        # 只需要最后一个时刻的输出结果来进行分类 所以取-1
        # 这里注意 output 的输出格式 正好第一维是 n_step 也就是步长 所以直接对第一维度切片就可以
        outputs = outputs[-1]
        # 这里 self.W 的定义是 (n_hidden, n_class) 注意如果使用的是多层或者双向的RNN 需要对应改变W的定义
        final_outputs = torch.mm(outputs, self.W) + self.b
        return final_outputs


model = TextRNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5000):
    optimizer.zero_grad()
    hidden = Variable(torch.zeros(1, batch_size, n_hidden))

    pred = model(hidden, inputs)
    loss = criterion(pred, targets)

    if (epoch+1) % 1000 == 0:
        print("Epoch: {}, Loss: {:.4f}.".format(epoch+1, loss))

    loss.backward()
    optimizer.step()

# 测试代码
hidden = Variable(torch.zeros(1, batch_size, n_hidden))
predict = model(hidden, inputs).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [index_word_dict[n.item()] for n in predict.squeeze()])


