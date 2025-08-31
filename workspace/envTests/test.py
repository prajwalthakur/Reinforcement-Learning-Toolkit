import torch

x = torch.tensor([[1., 2., 3.]])   # shape (1, 3)
y = x.squeeze_(0)                   # shape (3,) but still shares storage

y[0] = 99
print(x)   # tensor([[99.,  2.,  3.]])  <-- original got modified!
