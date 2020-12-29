import torch
import torch.nn as nn
import torch.optim as optim


def save_load():
    a = torch.ones(3, 3)
    print(a)
    torch.save(a, 'saves/a.pt')
    print('Done.')
    b = torch.load('saves/a.pt')
    print(b)

# save_load()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(3, 4)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(4, 1)

    def forward(self, x):
        out1 = self.linear1(x)
        act1 = self.act(out1)
        return self.linear2(act1)


model = MLP()
print(model)
print(model.state_dict())

optimizer = optim.SGD(model.parameters(), lr=0.3)
print(optimizer.state_dict())


'''
save and load models:
    1. save the models's state_dict
    2. save the whole model
'''

# 1.
torch.save(model.state_dict(), 'saves/model1.pt')
# model1 = TheModelClass(*args, **kwargs)
# model1.load_state_dict(torch.load('saves/model1.pt'))
model1 = MLP()
state_dict = torch.load('saves/model1.pt')
model1.load_state_dict(state_dict)

# 2.
torch.save(model, 'saves/model2.pt')
model2 = torch.load('saves/model2.pt')

# test:
input = torch.rand(2, 3)

out1 = model1(input)

out2 = model2(input)

print(out1)

print(out2)