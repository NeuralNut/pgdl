# %%
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# if torch.cuda.is_available():
#     torch.set_default_device('cuda:1')
#     device = torch.device('cuda:1')
device = torch.device('cpu')

# state space approximation of the underdamped spring-mass-damper system
class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, m, c, k):
        super(PhysicsInformedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # State space matrices for the underdamped spring
        self.A = torch.tensor([[0, 1], [-k/m, -c/m]], dtype=torch.float32)
        self.B = torch.tensor([[0], [1]], dtype=torch.float32)  # Assuming a zero input (free response)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def physics_loss(self, x):
        # x is the state vector
        dx = torch.matmul(self.A, x.T).T  # Calculate the state derivative
        return dx

# Example usage
m = 1.0  # Mass
c = 0.2  # Damping coefficient
k = 1.0  # Spring constant

model = PhysicsInformedNN(input_dim=2, hidden_dim=50, output_dim=2, m=m, c=c, k=k).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data (initial state and zero input for simplicity)
x = torch.randn(10, 2).to(device)

# Forward pass
output = model(x)

# Compute physics-based loss
loss = model.physics_loss(x).mean()  # Mean of the physics loss over the batch

loss


# %% [markdown]
# # Kolmorgorov-Arnold Network (KAN)

# %%
import random
import numpy as np
import torch
import torch.optim as optim
import models as pim
import losses as pil

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)
random.seed(0)


# Initialize model, optimizer, and training parameters
model = pim.KAN(
            layers_hidden=[1, 50, 50, 1],  # Define the layer sizes
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.Tanh,
            grid_eps=0.02,
            grid_range=[-1, 1]
        ).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

d, w0 = 2, 20
mu, k = 2*d, w0**2

d, w0 = torch.Tensor([d]).to(device), torch.Tensor([w0]).to(device)
mu, k = torch.Tensor([mu]).to(device), torch.Tensor([k]).to(device)

plosses = []
dlosses = []

# list to track moving average of loss
losses = []  # moving average of loss

# Training loop
for epoch in range(10000):
    loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    ploss, dloss, icloss = pil.physics_loss(model, mu, k, d, w0)
    ploss *= 1e-4
    loss += ploss
    loss += dloss
    loss += icloss
    # loss = dloss
    loss.backward()
    optimizer.step()

    # calculate moving average of loss
    if len(losses) < 100:
        losses.append(loss.item())
    else:
        losses.pop(0)
        losses.append(loss.item())
    
    # check if loss has converged
    std = np.std(losses)
    mean = np.mean(losses)
    if len(losses) == 100 and std < 5e-6 and epoch > 1000 and loss.item() < 1e-5:
        print(f'Converged at epoch {epoch}')
        break
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch},\
        Loss: {mean:.3e},\
        Physics Loss: {ploss.item():.3e},\
        Data Loss: {dloss.item():.3e},\
        IC Loss: {icloss.item():.3e},\
        Loss Std: {std:.3e}')
        scheduler.step()
    
    plosses.append(ploss.item())
    dlosses.append(dloss.item())


# %%

import matplotlib.pyplot as plt

# plot log loss curve
fig, ax = plt.subplots(figsize=(10, 5),num=1)
ax.plot(plosses, label='Physics Loss')
ax.plot(dlosses, label='Data Loss')

plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('PIKAN Loss Curve')
plt.legend()
plt.grid(True)
# plt.show(block=False)

# %%
#plot predicted vs true displacement
model.eval()
td = torch.linspace(0, 2, 120).reshape(-1, 1).requires_grad_(True)
t0 = torch.FloatTensor(1).fill_(0.0).reshape(-1, 1)
tn = torch.FloatTensor(119, 1).uniform_(0, 2)
# tL = torch.FloatTensor(1).fill_(2.0).reshape(-1, 1)
tp = torch.concat([t0, tn])
x = model(td).squeeze().detach().numpy()
x_true = pil.oscillator(d, w0, td).squeeze().detach().numpy()

td = td.squeeze().detach().numpy()
data_domain_mask = td<=0.5
physics_domain_mask = tp<=1.0



fig, ax = plt.subplots(figsize=(10, 5),num=2)
ax.plot(td, x, color="red", label='Predicted')
ax.plot(td, x_true, color="black", linestyle='--', label='True')
plt.scatter(td[data_domain_mask], x_true[data_domain_mask], color="tab:blue", alpha=0.8, label="Training data")
plt.scatter(tp[physics_domain_mask], 0*tp[physics_domain_mask], color="tab:green", alpha=0.8, label="Physics domain")
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('PIKAN Predicted vs True Displacement')
plt.legend()
plt.grid(True)
plt.show()

# %%



