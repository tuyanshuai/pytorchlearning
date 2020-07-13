from __future__ import print_function
import torch

x = torch.empty(5,3)
print(x)

x = torch.rand((5, 3))
print(x)

x = torch.zeros((5, 3), dtype=torch.long)
print(x)


x = torch.tensor([5.5, 3])
x = x.new_ones(5,3)
print(x)

x = torch.rand_like(x, dtype = torch.float)

print(x.size())


y = torch.rand((5,3))
print(x+y)

y.add_(x)
print(y)


x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1, 8 )

print(x)
print(y)
print(z)



import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)

np.add(a, 1, out=a)
print(a)
print(b)






print(torch.cuda.is_available())


device = torch.device("cuda")
y = torch.ones_like(x, device=device)
z = x + y

print(z)

print(z.to("cpu", torch.double))
