#!/bin/bash
set -e

project_root=$(pwd)

echo -e "\n[1/6] Checking for virtualenv"
if ! command -v virtualenv &> /dev/null; then
    echo "Installing virtualenv..."
    pip install --user virtualenv
    export PATH="$HOME/.local/bin:$PATH"
fi

echo -e "\n[2/6] Cleaning any broken virtualenv cache"
rm -rf ~/.local/share/virtualenv/wheel

echo -e "\n[3/6] Creating Python virtual environment at $project_root/.venv"
#virtualenv "$project_root/.venv"
python -m virtualenv "$project_root/.venv"


echo -e "\n[4/6] Activating virtual environment"
#source "$project_root/.venv/bin/activate"
source "$project_root/.venv/Scripts/activate"

echo -e "\n[5/6] Installing PyTorch (CPU only)"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo -e "\n[6/6] Cloning EfficientViT if not already cloned"
if [ ! -d "$project_root/efficientvit/.git" ]; then
    rm -rf "$project_root/efficientvit"
    git clone https://github.com/mit-han-lab/efficientvit.git "$project_root/efficientvit"
fi

cd "$project_root/efficientvit"
pip install -r requirements.txt
pip install debugpy

echo -e "\n======== DONE =========="
echo "To activate your environment again:"
echo "    source $project_root/.venv/bin/activate"
