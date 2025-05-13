#!/bin/bash
set -e

project_root=$(pwd)
env_name="metric"

echo -e "\n[1/6] Checking if Conda is available..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda first:"
    echo "https://doc.lab-ia.fr/faq/#python-installation"
    exit 1
fi

echo -e "\n[2/6] Creating or reusing conda environment: $env_name"
eval "$(conda shell.bash hook)"
if conda env list | grep -q "^$env_name"; then
    echo "Environment '$env_name' already exists. Skipping creation."
else
    conda create -y -n "$env_name" python=3.11
fi

echo -e "\n[3/6] Activating conda environment..."
conda activate "$env_name"

echo -e "\n[4/6] Installing PyTorch (CPU only)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo -e "\n[5/6] Cloning EfficientViT repo if not already present..."
if [ ! -d "$project_root/efficientvit/.git" ]; then
    rm -rf "$project_root/efficientvit"
    git clone https://github.com/mit-han-lab/efficientvit.git "$project_root/efficientvit"
fi

echo -e "\n[6/6] Installing project dependencies..."
cd "$project_root/efficientvit"
pip install -r requirements.txt
pip install onnx onnxsim
pip install triton

# for smolvlm
pip install num2words
pip install hf_xet

echo -e "\nInstallation complete!"
echo "To activate your environment again:"
echo "    conda activate $env_name"
