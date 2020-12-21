import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

'''
GPU 版本
除了model 和 loss_function 以及训练数据转移到GPU之外
记的 evaluate 的时候也要把数据转移！
'''

# 获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=True,
                                                download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=False,
                                               download=True, transform=transforms.ToTensor())

learning_rate = 0.1
batch_size = 256
epoch_num = 10
feature_num = 28 * 28
hidden_num = 256
label_num = 10


train_dataloader = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_dataloader = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


def evaluate_test(test_data, model):
    acc_num = 0
    ins_num = 0
    for X, y in test_data:
        X = X.cuda()
        y = y.cuda()
        pred = model(X.view(-1, feature_num))
        ins_num += y.shape[0]
        acc_num += (torch.argmax(pred, dim=1) == y).sum().float()
    return acc_num / ins_num


class MLP_Model(nn.Module):
    def __init__(self):
        super(MLP_Model, self).__init__()
        self.linear1 = nn.Linear(feature_num, hidden_num)
        self.linear2 = nn.Linear(hidden_num, label_num)
        self.relu = nn.ReLU()

    def forward(self, X):
        mid_res = self.relu(self.linear1(X))
        return self.linear2(mid_res)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP_Model()
model = model.cuda()
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.cuda()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(epoch_num):
    epoch_loss = 0.0
    cou = 0
    for X, y in train_dataloader:
        X = X.cuda()
        y = y.cuda()
        cou += 1
        pred = model(X.view(-1, feature_num))
        loss = loss_function(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    test_acc = evaluate_test(test_dataloader, model)
    print("Epoch: %d, Loss: %f, Acc: %f" % (epoch, epoch_loss/cou, test_acc))


