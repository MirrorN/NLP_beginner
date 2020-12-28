import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np

batch_size = 256
feature_dim = 28 * 28
hidden_dim = 1024
learning_rate = 0.3
label_num = 10
drop_prob = 0.2
epoch_num = 10

# 获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=True,
                                                download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=False,
                                               download=True, transform=transforms.ToTensor())

train_dataloader = Data.DataLoader(mnist_train, batch_size, shuffle=True)
test_dataloader = Data.DataLoader(mnist_test, batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self, drop_prob):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden_dim, label_num)

    def forward(self, X):
        mid_out = self.linear1(X)
        mid_out_relu = self.relu(mid_out)
        mid_out_relu_drop = self.dropout(mid_out_relu)
        output = self.linear2(mid_out_relu_drop)
        return output


def evaluate_test(test_data, model):
    acc_num = 0
    ins_num = 0
    for X, y in test_data:
        X = X.cuda()
        y = y.cuda()
        pred = model(X.view(-1, feature_dim))
        ins_num += y.shape[0]
        acc_num += (torch.argmax(pred, dim=1) == y).sum().float()
    return acc_num / ins_num


model = MLP(drop_prob)
model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    epoch_loss = 0
    cnt = 0
    for X, y in train_dataloader:
        X = X.cuda()
        y = y.cuda()
        pred = model(X.view(-1, feature_dim))
        loss = loss_function(pred, y)
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
    acc = evaluate_test(test_dataloader, model)
    print("Epoch: %d, Loss: %f, Acc: %f" % (epoch, epoch_loss / cnt, acc))

