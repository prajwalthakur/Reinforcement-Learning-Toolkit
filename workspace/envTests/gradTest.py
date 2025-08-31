import torch

# Create a tensor that requires gradients
x = torch.tensor(2.0, requires_grad=True)
print(f"Original tensor x: {x}, requires_grad: {x.requires_grad}")

# Perform an operation on x
y = x * 3
print(f"Tensor y: {y}, requires_grad: {y.requires_grad}")

# Detach y from the graph
y_detached = y.detach()
print(f"Detached tensor y_detached: {y_detached}, requires_grad: {y_detached.requires_grad}")

# Operations on y_detached will not be tracked for gradients
z = y + 5
print(f"Tensor z: {z}, requires_grad: {z.requires_grad}")

# Attempting to backpropagate from z would not affect x
z.backward() #would raise an error if z was part of the graph and its computation depended on non-leaf tensors that require gradients

x.grad