import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# torch.manual_seed(1)
input_num = 1000
batch_size = 10
learning_rate = 0.03
epoch_num = 30
feature_dim = 2
true_w = [2, -3.4]
true_b = 4.2


def produce_pseudo_data():
    inputs = torch.rand(input_num, feature_dim, dtype=torch.float32)
    labels = true_w[0] * inputs[:, 0] + true_w[1] * inputs[:, 1]
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    return inputs, labels


def data_plot(features, labels):
    plt.figure()
    plt.scatter(features[:, 1], labels)
    plt.show()


# data_plot(inputs, labels)


def prepare_data(bsz, inputs, labels):
    indices = list(range(input_num))
    random.shuffle(indices)
    # inputs = inputs[indices]
    # labels = labels[indices]
    batch_num = int(np.ceil(input_num / bsz))
    for i in range(0, input_num, bsz):
        j = torch.LongTensor(indices[i: min(i+bsz, input_num)])
        yield inputs.index_select(0, j), labels.index_select(0, j)
    # for i in range(batch_num):
    #     start_index = i * bsz
    #     end_index = min((i + 1) * bsz, input_num)
    #     yield inputs[start_index: end_index].clone().detach(), labels[start_index: end_index].clone().detach()


class Linear_Model(nn.Module):
    def __init__(self):
        super(Linear_Model, self).__init__()
        self.w = torch.tensor(np.random.normal(0, 0.01, (feature_dim, 1)), dtype=torch.float32, requires_grad=True)
        self.b = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, X):
        return torch.mm(X, self.w) + self.b


def loss_function(y_hat, y):
    '''
    loss function : square loss function
    '''
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, bsz):
    '''
    optimizer: SGD
    '''
    for param in params:
        param.data -= lr * param.grad / bsz


# Training
model = Linear_Model()
inputs, labels = produce_pseudo_data()
for epoch in range(epoch_num):
    for X, y in prepare_data(batch_size, inputs, labels):
        pred = model(X)
        l = loss_function(pred, y).sum()
        l.backward()
        sgd([model.w, model.b], learning_rate, batch_size)
        # print(model.w, model.b)

        model.w.grad.data.zero_()
        model.b.grad.data.zero_()
    each_epoch_loss = loss_function(model(inputs), labels)
    print("Epoch %d, Loss %f."%(epoch, each_epoch_loss.mean().item()))
print(model.w.data, model.b.data)




