import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np

# 获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=True,
                                                download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=False,
                                               download=True, transform=transforms.ToTensor())

batch_size = 64
epoch_num = 10
feature_num = 28 * 28
labels_num = 10
learning_rate = 0.3


# 使用 torch.utils.data 创建 dataloader
train_dataloader = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_dataloader = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


class Linear_Model(nn.Module):
    def __init__(self):
        super(Linear_Model, self).__init__()
        self.w = torch.tensor(np.random.normal(0, 0.01, (feature_num, labels_num)), dtype=torch.float32, requires_grad=True)
        # self.w = torch.rand(feature_num, labels_num)
        # self.w.requires_grad_()
        self.b = torch.zeros(labels_num, dtype=torch.float32, requires_grad=True)
        # self.b.requires_grad_()

    def forward(self, X):
        return softmax(torch.matmul(X, self.w) + self.b)


def softmax(vec):
    vec_exp = torch.exp(vec)
    vec_exp_sum = torch.sum(vec_exp, dim=1, keepdim=True)
    return vec_exp / vec_exp_sum


def cross_entropy_loss(pred, y):
    return -torch.log(pred.gather(1, y.view(-1, 1)))


def sgd(params, bsz, learning_rate):
    for param in params:
        param.data -= learning_rate * param.grad / bsz


def evaluate(model, loader):
    acc_num = 0
    test_num = 0
    for X, y in loader:
        test_num += y.shape[0]
        pred = model(X.view(-1, feature_num))
        acc_num += (torch.argmax(pred, dim=1) == y).sum().float()
    return acc_num / test_num


model = Linear_Model()

for epoch in range(epoch_num):
    epoch_loss = 0.0
    input_num = 0.0
    for X, y in train_dataloader:
        pred = model(X.view(-1, feature_num))
        loss = cross_entropy_loss(pred, y).sum()

        if model.w.grad is not None:
            model.w.grad.data.zero_()
            model.b.grad.data.zero_()

        loss.backward()
        sgd([model.w, model.b], batch_size, learning_rate)
        epoch_loss += loss
        input_num += y.shape[0]
    accuracy = evaluate(model, test_dataloader)
    print("Epoch: %d, Loss: %f Test Acc: %f." % (epoch, epoch_loss/input_num, accuracy))
