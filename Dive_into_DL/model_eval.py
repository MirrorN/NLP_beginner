import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
'''
This code show the .eval() function.
model = MyClass()  -> forward predict
During test, we should negate the Dropout and Normalization
model1 = MyClass().eval()  -> evaluate()
'''

true_w1 = 3.2
true_w2 = 1.5
true_b = 2.2
feature_num = 2
label_num = 1
input_num = 100
batch_size = 50
epoch_num = 10
learning_rate = 0.2
inputs = torch.rand(input_num, feature_num)
labels = true_w1*inputs[:, 0] + true_w2*inputs[:, 1] + true_b
labels += torch.tensor(np.random.normal(0.0, 0.1, size=labels.size()))


train_dataset = Data.TensorDataset(inputs, labels)
train_dataloader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(2, 3)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(3, 1)

    def forward(self, x):
        out1 = self.linear1(x)
        out1_drop = self.dropout(out1)
        return self.linear2(out1_drop)


model = MyModel()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(epoch_num):
    epoch_loss = 0.0
    cnt = 0
    for x, y in train_dataloader:
        pred = model(x)
        loss = loss_function(pred, y)
        optimizer.zero_grad()
        epoch_loss += loss
        loss.backward()
        optimizer.step()
        cnt += 1
    print('Epoch: %d, Epoch Loss: %f.' % (epoch, epoch_loss / cnt))


'''
the result of five runs are different
tensor([4.5005], grad_fn=<AddBackward0>)
tensor([5.0422], grad_fn=<AddBackward0>)
tensor([4.5005], grad_fn=<AddBackward0>)
tensor([5.0422], grad_fn=<AddBackward0>)
tensor([4.3102], grad_fn=<AddBackward0>)
'''
test1 = torch.tensor([1.0, 2.0])
# for _ in range(5):
#     print(model(test1))

'''
.eval() can negate the dropout:
tensor([4.5882], grad_fn=<AddBackward0>)
tensor([4.5882], grad_fn=<AddBackward0>)
tensor([4.5882], grad_fn=<AddBackward0>)
tensor([4.5882], grad_fn=<AddBackward0>)
tensor([4.5882], grad_fn=<AddBackward0>)
'''
test_model = model.eval()
for _ in range(5):
    print(test_model(test1))
