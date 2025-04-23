#  Installation

To set up the project environment and install all required dependencies, simply run the provided installation script:
```aiignore
chmod +x install.sh
./install.sh
```

This will:

Create a virtual environment in .venv/

Install PyTorch (CPU-only by default)

Clone and set up EfficientViT-SAM

Install all required Python packages
```commandline
pip install --user virtualenv
```

After installation, activate the environment with:
```aiignore
source .venv/bin/activate
```

### On Linux/macOS:
```bash
chmod +x install.sh
./install.sh
```
### On Windows (PowerShell or Git Bash):
```aiignore
bash install.sh
```

Triton library, which is required by EfficientViT's TritonRMSNorm2dFunc used in normalization.
Triton is a deep learning compiler designed for GPUs. It will install even if you’re on CPU, but it may not be used.
```aiignore
pip install triton
```