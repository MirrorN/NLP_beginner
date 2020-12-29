import torch
import torch.nn as nn
import numpy as np

'''
For Convolution basis
'''


def corr2d(x, k):
    h, w = k.shape
    res = torch.zeros(x.shape[0]-h+1, x.shape[1]-w+1)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = (x[i: i+h, j:j+w] * k).sum()
    return res


# X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# K = torch.tensor([[0, 1], [2, 3]])
# print(corr2d(X, K))


class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


X = torch.ones(6, 8)
X[:, 2:6] = 0

K = torch.tensor([[1., -1]])
Y = corr2d(X, K)
# print(Y)

conv2d = Conv2d(kernel_size=(1, 2))

step = 20
lr = 0.01
for i in range(step):
    y_pred = conv2d(X)
    l = ((y_pred - Y) ** 2).sum()
    l.backward()

    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)

    print("Epoch: %d, Loss: %f." % (i, l))

print('Training done.')
print(conv2d.weight.data)
print(conv2d.bias.data)