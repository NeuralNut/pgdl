#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Download the latest Miniconda3 installer
echo "Downloading the latest Miniconda3 installer..."
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Step 2: Install Miniconda3
echo "Installing Miniconda3..."
sh Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Step 3: Initialize Conda
echo "Initializing Conda..."
export PATH="$HOME/miniconda3/bin:$PATH"
conda init bash

# Step 4: Create a new Conda environment named "phd"
echo "Creating Conda environment 'phd2'..."
conda create -y -n phd2 python=3.11

# Step 5: Activate the environment and install required packages
echo "Activating environment and installing required packages..."
source $HOME/miniconda3/bin/activate
conda activate phd2

pip install "jax[cuda12]" flax optax pandas scipy matplotlib seaborn scikit-learn

# Step 6: Final message
echo "Setup complete. To use the 'phd' environment, run:"
echo "  conda activate phd"
