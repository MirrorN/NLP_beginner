"""
添加 Attention mechanism 的 Seq2Seq 模型
    1，RNN单元使用的就是最原始的RNN
    2. attention score使用全连接方式计算
    3. 计算attention使用的是encoder每个时刻的输出
    4. 将context vector与每个时刻decoder的输出做拼接之后得到翻译输出

"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

data_type = torch.FloatTensor
# 定义三种特殊符号
# S: decoding input 的开始标志
# E: decoding output 的结束标志
# P: 填充字符

# 这里sentence数组保存训练数据 ，实际上数据只有一句话
# 任务也就是把 ich mochte ein bier 翻译成 i want a beer
# 这里省略了之前的Seq2Seq中的 padding以及添加特殊符号的过程 在make_batch的部分可以看出来
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_index_dict = {w: i for i, w in enumerate(word_list)}
index_word_dict = dict(zip(word_index_dict.values(), word_index_dict.keys()))

# 这里说明一下 为了简单起见 这里使用的是one-hot编码 所以词向量的维度等于词表长度也等于最后要分类的分类数目
n_class = len(word_index_dict)
embed_dims = n_class
n_hidden = 128


def make_batch(sentences):
    # 将encoder和decoder的输入数据处理成 one-hot形式
    # 其中 input_batch, output_batch: [n_step, embed_dims]
    # target_batch: [n_step(是target的长度，与原语言长度不一定相同), 1]
    input_batch = [np.eye(n_class)[[word_index_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_index_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_index_dict[n] for n in sentences[2].split()]]

    # make tensor
    return Variable(torch.Tensor(input_batch)), Variable(torch.Tensor(output_batch)), torch.LongTensor(target_batch)


class Atten_SeqSeq(nn.Module):
    def __init__(self):
        super(Atten_SeqSeq, self).__init__()

        self.enc_cell = nn.RNN(input_size=embed_dims, hidden_size=n_hidden)
        self.dec_cell = nn.RNN(input_size=embed_dims, hidden_size=n_hidden)

        # 计算attention的分数的时候又多种方法 这里使用的是一个简单的全连接层
        self.att = nn.Linear(n_hidden, n_hidden)
        # 这里最后分类器的输入是一个拼接的向量 [context vector, dec_output]
        self.fc = nn.Linear(n_hidden*2, n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        # 将数据格式调整为 [n_step=5, batch_size=1, embed_dims]
        enc_inputs = enc_inputs.transpose(0, 1)
        dec_inputs = dec_inputs.transpose(0, 1)

        # enc_outputs : [n_step=5, batch_size=1, n_hidden]
        # enc_hidden: [num_layers*num_directions=1, batch_size, n_hidden]
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)
        # 保存计算的 attention 的分数
        trained_att = []
        # decoder 依然使用encoder的最后一个时刻的 hidden_state作为初始hidden state
        hidden = enc_hidden
        # 接下来需要一步一步计算每一步decoder输入的attention向量 先获取decoder的数据步长
        # 因为origin和target语言的长度往往不相同 所以这里重新计算一下步长
        n_step = len(dec_inputs)
        # 这里的1 应该是因为batch_size为1 所以直接写了 保存最后的分类结果
        model = Variable(torch.empty([n_step, 1, n_class]))

        for i in range(n_step):
            # 这里每次循环只会运行一步 hidden是重复使用的
            # 这里取出dec_inputs的第i步，然后unqueeze()函数扩充1维
            # 这样将dec_inputs处理成符合RNN单元的数据 [1, 1, embed_dims]
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            # att_weights : [1, 1, n_hidden]
            # 传入get_att_weight()函数的 dec_output: [1, 1, n_hidden] enc_outputs: [n_step, 1, n_hidden]
            att_weights = self.get_att_weight(dec_output, enc_outputs)
            # att_weights: [1, 1, n_step(这里的n_step是origin的长度)]  squeeze() -> [n_step]
            trained_att.append(att_weights.squeeze().data.numpy())

            # 这里看起来很复杂 实际上只是对加权求context vector的过程的一个简单写法
            # 按照一般在博客中的理解，我们将每个分数乘以对应的encoder的输出或者是隐状态
            # 然后再相加得到最后的context 这个过程可以拆开写成多步，这里使用bmm函数只是
            # 将这个过程简化成一个矩阵乘法，写法上比较简单而已
            # [1, 1, n_step] * [1, n_step, n_hidden] = [1, 1, n_hidden]
            # torch.bmm()函数用于计算两个tensor的矩阵乘法： (b, h, w) * (b, w, h)
            context = att_weights.bmm(enc_outputs.transpose(0, 1))
            # dec_output原本是 [n_step=1, batch_size=1, n_hidden] 现在去掉n_step
            dec_output = dec_output.squeeze(0)
            # context去掉维度1
            context = context.squeeze(1)
            # model[i]: [1, n_hidden * 2]
            model[i] = self.fc(torch.cat((dec_output, context), 1))

        return model.transpose(0, 1).squeeze(0), trained_att

    def get_att_weight(self, dec_input, enc_outputs):
        """
        对于decoder每一步的输入，求解与encoder的输出的Attention的权重
        注意这里是对encoder的所有时刻的输出求权重 而非是每一步的隐藏状态输出
        此时 enc_outputs 的shape是 [n_step, batch_size=1, n_hidden]
        dec_input的shape是 [1, 1, embed_dims]
        """
        n_step = len(enc_outputs)
        # 每一步都会有一个分数 所以创建的tensor长度为 n_step
        att_scores = Variable(torch.zeros(n_step))

        for i in range(n_step):
            att_scores[i] = self.get_att_score(dec_input, enc_outputs[i])

        # 最终返回的结果: [1, 1, n_step]
        return F.softmax(att_scores, dim=0).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output):
        """
        enc_output的sahpe: [1, 1, n_hidden]
        dec_output的shape: [1, 1, n_hidden]
        """
        # [1, 1, n_hidden] * [n_hidden, n_hidden]
        score = self.att(enc_output)
        # [1, 1, n_hidden] * [1, 1, n_hidden]
        # 先view成一维向量，在进行点乘 最后返回的是一个标量
        return torch.dot(dec_output.view(-1), score.view(-1))


input_batch, output_batch, target_batch = make_batch(sentences)

# hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
hidden = Variable(torch.zeros(1, 1, n_hidden))

model = Atten_SeqSeq()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(2000):
    optimizer.zero_grad()
    output, _ = model(input_batch, hidden, output_batch)

    loss = criterion(output, target_batch.squeeze(0))
    if (epoch + 1) % 400 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimizer.step()


# Test
# 测试时候看不到target 所以传入一个假数据序列
test_batch = [np.eye(n_class)[[word_index_dict[n] for n in 'SPPPP']]]
test_batch = Variable(torch.Tensor(test_batch))
predict, trained_attn = model(input_batch, hidden, test_batch)
predict = predict.data.max(1, keepdim=True)[1]
print(sentences[0], '->', [index_word_dict[n.item()] for n in predict.squeeze()])

# Show Attention
# 可视化相关矩阵 主要是 plt.matshow()函数
# 可以参考 matplotlib 的官网样例
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.matshow(trained_attn, cmap='viridis')
ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
plt.show()