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
def oscillator_loss(model, mu, k, d, w0, n=30, t0=0.0, tL=1.0, train_frac=0.5, return_data=False):
    """
    Defines the physics-informed loss function for the 1D underdamped harmonic oscillator problem.
    The loss function is composed of three terms:
    - The physics loss, which enforces the differential equation
    - The data loss, which enforces the model to fit the training data
    - The initial condition loss, which enforces the model to satisfy the initial conditions

    """

    # t0 is the initial time, tL is the final time, tn are the n-1 time points in between
    tn = torch.FloatTensor(n-1,1).uniform_(t0, tL).requires_grad_(True).to(device)
    t0 = torch.FloatTensor(1,1).fill_(t0).requires_grad_(True).to(device)
    t = torch.concat([t0, tn], dim=0)
    
    
    # Mask for the training data
    t_train = tL*train_frac
    mask = t.squeeze()<=t_train

    # True displacement
    x_true = oscillator(d, w0, t).reshape(-1, 1)
    
    # Predict displacement
    x = model(t)
    
    # Data loss
    dloss = torch.mean((x-x_true)[mask]**2) # only consider the training data for data loss calculation

    # Automatically compute derivatives
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]  # dx/dt
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]  # d^2x/dt^2
    
    # Compute the residual of the differential equation
    phy = x_tt + mu*x_t + k*x # = 0
    ploss = torch.mean(phy**2)

    ic1 = torch.mean((x[0]-1.)**2) # x(0) = 1  ie initial displacement is 1
    ic2 = torch.mean((x_t[0]-0.)**2) # dx/dt(0) = 0  ie initial velocity is zero
    icloss = ic1 + ic2

    if return_data:
        return (ploss, # physics loss
                dloss, # data loss
                icloss, # initial condition loss
                x_true, # true displacement
                x, # predicted displacement
                t, # time points
                mask) # mask for training data
    
    else:
        # Return the mean squared error against the expected zero
        return (ploss, # physics loss
                dloss, # data loss
                icloss) # initial condition loss
