import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

# 获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=True,
                                                download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=False,
                                               download=True, transform=transforms.ToTensor())

batch_size = 256
learning_rate = 0.1
epoch_num = 10
drop_prob1 = 0.2
drop_prob2 = 0.1
feature_num = 28*28
label_num = 10
hidden_num = 256


train_dataloader = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_dataloader = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self, drop_prob1, drop_prob2):
        super(MLP, self).__init__()
        self.w1 = torch.tensor(np.random.normal(0, 0.1, (feature_num, hidden_num)), dtype=torch.float)
        self.w1.requires_grad_(True)
        self.b1 = torch.zeros(hidden_num, dtype=torch.float)
        self.b1.requires_grad_(True)
        self.relu = nn.ReLU()
        self.drop_prob1 = drop_prob1
        self.w2 = torch.tensor(np.random.normal(0, 0.1, (hidden_num, label_num)), dtype=torch.float)
        self.w2.requires_grad_(True)
        self.b2 = torch.zeros(label_num, dtype=torch.float)
        self.b2.requires_grad_(True)
        self.drop_prob2 = drop_prob2

    def dropout(self, data, drop_prob):
        data = data.float()
        assert 0 <= drop_prob <= 1
        keep_prob = 1 - drop_prob
        if keep_prob == 0:
            return torch.zeros_like(data)
        mask = (torch.rand(data.shape) < keep_prob).float()
        return mask * data / keep_prob

    def forward(self, X):
        mid_res = torch.matmul(X, self.w1) + self.b1
        mid_res_relu = self.relu(mid_res)
        mid_res_relu_drop = self.dropout(mid_res_relu, self.drop_prob1)
        return self.dropout(torch.matmul(mid_res_relu_drop, self.w2) + self.b2, self.drop_prob2)


def sgd(params, batch_size, learning_rate):
    for param in params:
        param.data -= learning_rate * param.grad / batch_size


model = MLP(drop_prob1, drop_prob2)
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.cuda()

# training :
for epoch in range(epoch_num):
    epoch_loss = 0.0
    cou = 0
    for X, y in train_dataloader:
        pred = model(X.view(-1, feature_num))
        loss = loss_function(pred, y)
        if model.w1.grad is not None:
            model.w1.grad.data.zero_()
            model.w2.grad.data.zero_()
            model.b1.grad.data.zero_()
            model.b2.grad.data.zero_()

        loss.backward()
        sgd([model.w1, model.w2, model.b1, model.b2], batch_size, learning_rate)
        epoch_loss += loss
        cou += 1
    print("Epoch: %d, Loss: %f." % (epoch, epoch_loss/cou))