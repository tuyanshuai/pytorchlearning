import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.lin1 = nn.Linear(16 * 6 * 6, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view((-1, self.num_flat_features(x)))

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def num_flat_features(self, x):
        numfeat = 1
        size = x.size()[1:]
        for s in size:
            numfeat *= spo

        return numfeat



net = Net()
print(net)

input = torch.randn(1,1,32,32)
out = net(input)

print(out)



out.backward(torch.randn(1,10))


output = net(input)
target = torch.ones(10)
target = target.view(1,-1)

criterion = nn.MSELoss()


import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)

for i in range(5000):
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print((i,loss))
    if loss.item() < 1e-6:
        break