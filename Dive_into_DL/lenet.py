import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=True,
                                                download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./fashion_mnist/', train=False,
                                               download=True, transform=transforms.ToTensor())


learning_rate = 0.001
batch_size = 256
feature_num = 28 * 28
label_num = 10
epoch_num = 10

train_dataloader = Data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
test_dataloader = Data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(   # [1, 28, 28]
            nn.Conv2d(1, 6, 5),      # [6, 24, 24]
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),      # [6, 12, 12]
            nn.Conv2d(6, 16, 5),     # [16, 8, 8]
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)       # [16, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        features = self.conv(img)
        out = self.fc(features.view(img.size(0), -1))
        return out


def evaluate(data_iter, model, device=None):
    if device is None and isinstance(model, nn.Module):
        device = list(model.parameters())[0].device
    acc_sum, cnt = 0., 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            if isinstance(model, nn.Module):
                model.eval()
                acc_sum += (model(X).argmax(-1) == y).float().sum().cpu().item()
                model.train()
            cnt += y.size(0)
    return acc_sum / cnt


def train(model, data_iter, optimizer, device, num_epoch):
    print('training on ', device)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(num_epoch):
        loss_epoch = 0.
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_function(pred, y)
            loss_epoch += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(test_dataloader, model)
        print("Epoch: %d, Loss: %f, acc: %f." % (epoch, loss_epoch/batch_size, acc))


if __name__ == '__main__':
    model = LeNet()
    model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    train(model, train_dataloader, optimizer, device, epoch_num)



