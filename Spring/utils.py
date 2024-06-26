import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# Plotting functions
def plot_loss_curves(train_losses, test_losses, val_losses, train_seq_length):
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.yscale('log')
    plt.title(f"Loss Curves for Train Seq Length {train_seq_length}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.show()


# Damped harmonic oscillator function
def oscillator(d, w0, t):
    """Defines the analytical solution to the 1D underdamped harmonic oscillator problem."""
    assert d < w0
    d, w0 = torch.Tensor([d]), torch.Tensor([w0])
    w = torch.sqrt(w0**2 - d**2)
    phi = torch.arctan(-d/w)
    A = 1/(2*torch.cos(phi))
    cos = torch.cos(phi + w*t)
    exp = torch.exp(-d*t)
    x = exp * 2 * A * cos
    return x


# Function to generate oscillator data with given parameters
def generate_oscillator_data(start_time, end_time, seq_length, num_sequences, d, w0):
    t = torch.linspace(start_time, end_time, seq_length)  # Adjust time scaling as needed
    data = oscillator(d, w0, t).unsqueeze(-1)
    sequences = data.repeat(num_sequences, 1, 1).reshape(num_sequences, seq_length, 1)
    times = t.repeat(num_sequences, 1, 1).reshape(num_sequences, seq_length, 1)
    return sequences, times



# def split_sequences(batch, train_split_start, train_split_end, test_split_point):
#     """
#     Splits a batch of sequences into training, validation, and testing batches with a randomized train/validation split point.

#     Parameters:
#     - batch (torch.Tensor): A tensor of shape (batch_size, sequence_length, *features).
#     - train_split_start (int): The start of the range for randomizing the train split point.
#     - train_split_end (int): The end of the range for randomizing the train split point.
#     - test_split_point (int): The time index to split the sequences into validation and testing.

#     Returns:
#     - train_batch (torch.Tensor): A tensor containing the training sequences.
#     - val_batch (torch.Tensor): A tensor containing the validation sequences.
#     - test_batch (torch.Tensor): A tensor containing the testing sequences.
#     - train_split_point (int): The randomized train split point.
#     """
#     # Ensure the train split point is within the valid range
#     if train_split_start < 0 or train_split_end > test_split_point:
#         raise ValueError("Train split points must be within the valid range.")
    
#     # Randomize the train split point within the specified range
#     train_split_point = random.randint(train_split_start, train_split_end)
    
#     train_batch = batch[:, :train_split_point]
#     val_batch = batch[:, train_split_point:test_split_point]
#     test_batch = batch[:, test_split_point:]
    
#     return train_batch, val_batch, test_batch, train_split_point


def split_sequences(batch, train_split_point, test_split_point):
    """
    Splits a batch of sequences into training, validation, and testing batches with a randomized train/validation split point.

    Parameters:
    - batch (torch.Tensor): A tensor of shape (batch_size, sequence_length, *features).
    - train_split_start (int): The start of the range for randomizing the train split point.
    - train_split_end (int): The end of the range for randomizing the train split point.
    - test_split_point (int): The time index to split the sequences into validation and testing.

    Returns:
    - train_batch (torch.Tensor): A tensor containing the training sequences.
    - val_batch (torch.Tensor): A tensor containing the validation sequences.
    - test_batch (torch.Tensor): A tensor containing the testing sequences.
    - train_split_point (int): The randomized train split point.
    """
    # Ensure the train split point is within the valid range
    if train_split_point < 0 or train_split_point > test_split_point:
        raise ValueError("Train split points must be within the valid range.")
    
    train_batch = batch[:, :train_split_point]
    val_batch = batch[:, train_split_point:test_split_point]
    test_batch = batch[:, test_split_point:]
    
    return train_batch, val_batch, test_batch

def plot_sequences(batch, train_split_point, test_split_point):
    """
    Plots the batch of sequences with different colors for train, validation, and test parts.

    Parameters:
    - batch (torch.Tensor): A tensor of shape (batch_size, sequence_length, *features).
    - train_split_start (int): The start of the range for randomizing the train split point.
    - train_split_end (int): The end of the range for randomizing the train split point.
    - test_split_point (int): The time index to split the sequences into validation and testing.
    """
    train_batch, val_batch, test_batch = split_sequences(batch, train_split_point, test_split_point)
    batch_size = batch.size(0)

    fig, axes = plt.subplots(batch_size, 1, figsize=(10, 2 * batch_size))

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        axes[i].plot(range(train_split_point), train_batch[i].numpy(), color='blue', label='Train')
        axes[i].plot(range(train_split_point, test_split_point), val_batch[i].numpy(), color='orange', label='Validation')
        axes[i].plot(range(test_split_point, batch.size(1)), test_batch[i].numpy(), color='green', label='Test')
        axes[i].set_title(f'Sequence {i+1}')

        if i==0:
            axes[i].legend()

    plt.tight_layout()
    plt.show()
