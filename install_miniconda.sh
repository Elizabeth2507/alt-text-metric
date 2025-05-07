#!/bin/bash
set -e

echo -e "\n[1/5] Downloading Miniconda installer with Python 3.11.5..."
cd ~
wget -O Miniconda3-py311_23.5.2-0.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh

echo -e "\n[2/5] Running the installer silently into ~/miniconda3..."
bash Miniconda3-py311_23.5.2-0.sh -b -p $HOME/miniconda3

echo -e "\n[3/5] Initializing Conda for bash shell..."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash

echo -e "\n[4/5] Updating Conda to the latest version..."
conda update -n base -c defaults conda -y

echo -e "\n[5/5] Cleaning up installer..."
rm -f ~/Miniconda3-py311_23.5.2-0.sh

echo -e "\nMiniconda (Python 3.11.5) installed successfully!"
echo "👉 Please restart your shell OR run:"
echo "    source ~/miniconda3/etc/profile.d/conda.sh"
