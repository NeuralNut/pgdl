import torch

# if torch.cuda.is_available():
#     torch.set_default_device('cuda:1')
#     device = torch.device('cuda:1')

device = torch.device('cpu')
def oscillator(d, w0, t):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
    assert d < w0

    w = torch.sqrt(w0**2-d**2)
    phi = torch.arctan(-d/w)
    A = 1/(2*torch.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    x  = exp*2*A*cos
    return x


# Define the physics-informed loss
def physics_loss(model, mu, k, d, w0):
    """Defines the physics-informed loss function for the 1D underdamped harmonic oscillator problem."""

    # t0 is the initial time, tL is the final time, tn are the n time points in between
    t0 = torch.FloatTensor(1,1).fill_(0.0).requires_grad_(True).to(device)
    tn = torch.FloatTensor(29,1).uniform_(0, 1).requires_grad_(True).to(device)
    t = torch.concat([t0, tn], dim=0)
    mask = t.squeeze()<=0.5

    # True displacement
    x_true = oscillator(d, w0, t).reshape(-1, 1)
    
    # Predict displacement
    x = model(t)
    
    # Data loss
    dloss = torch.mean((x-x_true)[mask]**2)

    # Automatically compute derivatives
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]  # dx/dt
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]  # d^2x/dt^2
    
    # Compute the residual of the differential equation
    phy = x_tt + mu*x_t + k*x # = 0
    ploss = torch.mean(phy**2)

    ic1 = torch.mean((x[0]-1.)**2)
    ic2 = torch.mean((x_t[0]-0.)**2)
    ic = ic1 + ic2

    # Return the mean squared error against the expected zero
    return (ploss, # physics loss
            dloss, # data loss
            ic) # initial condition loss
