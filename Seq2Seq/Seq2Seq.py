"""
Seq2Seq 模型的简单示例
任务：模拟一个翻译问题 只是这里模拟了一下翻译的文本 所以实际上模型用来得到一个单词的反义词
例如：  man -> women , black -> white 等等

这里模型中使用了标准RNN单元作为 encoder 和 decoder，其中：
输入数据都被填充为长度为5的序列 并且添加decode开始以及decode输出的标志
在decoder阶段，只使用encode的最后状态作为初始状态（仅使用一次）
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

data_type = torch.FloatTensor
# 规定三种字符：
# S：解码输入的开始标志
# E：解码输出的结束标志
# P：填充标志
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
word_index_dict = {n: i for i, n in enumerate(char_arr)}

seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

# Parameters:
n_step = 5
n_hidden = 128
n_class = len(word_index_dict)
batch_size = len(seq_data)


def make_batch(seq_data):
    """
    input_batch：用于encoder输入
    output_batch: 用于decoder输入
    target_batch：标准输出 groud-truth
    """
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 这里进行一个padding操作 padding 长度也就是 n_step=5
        # 例如：man -> manPP
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))
        input = [word_index_dict[i] for i in seq[0]]
        # 为decoder使用的数据添加开始和结束标志 每个序列都增加了一个字符所以长度是6
        output = [word_index_dict[i] for i in ('S' + seq[1])]
        target = [word_index_dict[i] for i in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)  # 不使用 one-hot

    # 这里提醒一下注意 torch.Tensor() 和 torch.tensor() 是不同的，简单来说：
    # torch.Tensor() 相当于 torch.FloatTensor()  创建的是 float 类型数据
    # torch.tensor() 的数据类型则取决于传入的初始值的数据类型，创建的结果可以是任何数据类型
    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), torch.LongTensor(target_batch)


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # 这里令 max_len = n_step + 1 这是因为一开始把数据padding成n_step长度
        # 之后又分别加入了两个标志 'S', 'E' ，所以实际上数据的长度是max_len = n_step+1
        # enc_input: [max_len, batch_size, input_size]
        # dec_input: [max_len, batch_size, input_size]
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)

        # 对于encoder我们要保留的仅仅是最后一个时刻的state，作为decoder的初始state
        # enc_states : [num_layers*num_directions=1, batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        # outputs : [max_len, batch, num_directions*n_hidden=128]
        outputs, _ = self.dec_cell(dec_input, enc_states)

        # final_outputs: [max_len, batch_size, n_class]
        final_outputs = self.fc(outputs)
        return final_outputs


input_batch, output_batch, target_batch = make_batch(seq_data)

model = Seq2Seq()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    optimizer.zero_grad()

    init_enc_hidden = Variable(torch.zeros(1, batch_size, n_hidden))
    # output: [max_len, batch_size, n_class]
    output = model(input_batch, init_enc_hidden, output_batch)
    output = output.transpose(0, 1)
    loss = 0
    for i in range(len(output)):
        loss += criterion(output[i], target_batch[i])

    if (epoch + 1) % 1000 == 0:
        print('Epoch: {}, Loss:{:.4f}'.format(epoch+1, loss))
    loss.backward()
    optimizer.step()



# 测试效果
def translate(word):
    # 这里在测试与训练时候数据的不同
    # 训练的时候 decoder 的输入数据我们是已知的（监督学习嘛）
    # 但是测试时候答案是未知的，所以这里使用了等长的全 "P" 字符序列代替
    input_batch, output_batch, _ = make_batch([[word, 'P'*len(word)]])
    init_enc_hidden = Variable(torch.zeros(1, 1, n_hidden))
    # shape： [6, 1, 29] - [max_len, batch_size, n_class]
    output = model(input_batch, init_enc_hidden, output_batch)

    # [6, 1, 1] 找到最大的索引，这样就可以按照词典找到对应的单词了
    predict = output.data.max(2, keepdim=True)[1]
    decoded = [char_arr[i] for i in predict]
    # 找到结尾符号
    end = decoded.index('E')
    translated = ''.join(decoded[:end])
    # 把填充符号"P"去掉
    return translated.replace('P', '')


print('Text:')
print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('black ->', translate('black'))
print('upp ->', translate('upp'))
