import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from layers import KANLinear



# NN implementation
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# Define the neural network model
class PIKAN(nn.Module):
    def __init__(self):
        super(PIKAN, self).__init__()

        # not implemented error
        raise NotImplementedError("NNSSM not implemented yet")
    
        self.net = KAN(
            layers_hidden=[1, 50, 50, 1],  # Define the layer sizes
            grid_size=100,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1]
        )

    def forward(self, t):
        return self.net(t)
    

# Kolmogorov-Arnold State-Space Model Implementation
class KASSM(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, kan_layers):
        super(KASSM, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the KANs for state transition and observation
        self.kan_state_transition = KAN([state_dim + input_dim] + kan_layers + [state_dim])
        self.kan_observation = KAN([state_dim + input_dim] + kan_layers + [output_dim])

    def forward(self, state, input, update_grid=False):
        # State transition: h_{t} = f(h_{t-1}, x_t)
        state_input = torch.cat([state, input], dim=-1)
        next_state = self.kan_state_transition(state_input, update_grid=update_grid)

        # Observation: y_t = g(x_t, h_t)
        observation = self.kan_observation(state_input, update_grid=update_grid)

        return next_state, observation
    

# Neural Network State-Space Model Implementation
class NNSSM(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, kan_layers):
        super(NNSSM, self).__init__()

        # not implemented error
        raise NotImplementedError("NNSSM not implemented yet")

        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Define the KANs for state transition and observation - NN(input_dim, hidden_dims, output_dim)
        self.kan_state_transition = KAN([state_dim + input_dim] + kan_layers + [state_dim])
        self.kan_observation = KAN([state_dim + input_dim] + kan_layers + [output_dim])

    def forward(self, state, input, update_grid=False):
        # State transition: h_{t} = f(h_{t-1}, x_t)
        state_input = torch.cat([state, input], dim=-1)
        next_state = self.kan_state_transition(state_input, update_grid=update_grid)

        # Observation: y_t = g(x_t, u_t)
        observation = self.kan_observation(state_input, update_grid=update_grid)

        return next_state, observation    