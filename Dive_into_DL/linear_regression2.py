import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import torch.optim as optim

'''
Simple version
'''

epoch_num = 10
input_num = 1000
batch_size = 10
feature_num = 2
learning_rate = 0.3
true_w = [3.2, 4.3]
true_b = 2.3


def produce_pseudo_data():
    inputs = torch.rand(input_num, feature_num, dtype=torch.float32)
    labels = true_w[0]*inputs[:, 0] + true_w[1]*inputs[:,1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    return inputs, labels


inputs, labels = produce_pseudo_data()
# 使用预置的 data 来读取数据 并进行 shuffle
dataset = Data.TensorDataset(inputs, labels)
data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for X, y in data_iter:
#     print(X)
#     print(y)
#     break


class Linear_Model(nn.Module):
    def __init__(self, feature_num):
        super(Linear_Model, self).__init__()
        self.linear = nn.Linear(feature_num, 1)

    def forward(self, inputs):
        return self.linear(inputs)


model = Linear_Model(feature_num)
print(model)

loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    for X, y in data_iter:
        pred = model(X)
        l = loss(pred, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("Epoch %d, Loss %f. " % (epoch, l.item()))

print(model.linear.weight)
print(model.linear.bias)
