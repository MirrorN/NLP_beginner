import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

# obtain the datasets
mnist_train = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=True,
                                                download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=False,
                                               download=True, transform=transforms.ToTensor())

batch_size = 256
learning_rate = 0.1
epoch_num = 10
# the drop probability of the layers nearer to the input often sets to smaller.
drop_prob1 = 0.2
drop_prob2 = 0.0
feature_num = 28*28
label_num = 10
hidden_num = 256


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


class MLP(nn.Module):
    def __init__(self, drop_prob1, drop_prob2):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(feature_num, hidden_num)
        self.linear2 = nn.Linear(hidden_num, label_num)
        self.drop1 = nn.Dropout(drop_prob1)
        self.drop2 = nn.Dropout(drop_prob2)
        self.relu = nn.ReLU()

    def forward(self, X):
        mid_res = self.linear1(X)
        mid_res_relu = self.relu(mid_res)
        mid_res_relu_drop = self.drop1(mid_res_relu)
        out = self.linear2(mid_res_relu_drop)
        return self.drop2(out)


model = MLP(drop_prob1, drop_prob2)
model = model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    epoch_loss = 0.0
    cou = 0
    for X, y in train_dataloader:
        X = X.cuda()
        y = y.cuda()
        pred = model(X.view(-1, feature_num))
        loss = loss_function(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss
        cou += 1
    test_acc = evaluate_test(test_dataloader, model)
    print("Epoch: %d, Loss: %f, Acc: %f" % (epoch, epoch_loss / cou, test_acc))

