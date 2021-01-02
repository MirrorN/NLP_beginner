import torch
import torch.nn as nn
import numpy as np

'''
For pooling (max and mean)
'''


def pooling(X, pooling_size, mode='max'):
    X = X.float()
    h, w = X.size()
    ph, pw = pooling_size
    res = torch.zeros(h-ph+1, w-pw+1)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if mode == 'max':
                res[i, j] = X[i: i+ph, j: j+pw].max()
            elif mode == 'mean':
                res[i, j] = X[i: i+ph, j: j+pw].mean()
    return res


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
res1 = pooling(X, (2, 2))
print(res1)

res2 = pooling(X, (2, 2), mode='mean')
print(res2)

X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2d(3, stride=1)
print(pool2d(X))