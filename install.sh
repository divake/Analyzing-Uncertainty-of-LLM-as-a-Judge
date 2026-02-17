#!/bin/bash
set -e

echo "========================================"
echo "Installing dependencies for LLM-as-a-Judge Uncertainty Analysis"
echo "========================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_py311

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install standard requirements
echo ""
echo "Installing standard requirements..."
pip install -r requirements.txt

# Install R2CCP (requires special installation)
echo ""
echo "Installing R2CCP..."
if [ ! -f "R2CCP-0.0.8-py3-none-any.whl" ]; then
    echo "Downloading R2CCP wheel..."
    wget https://files.pythonhosted.org/packages/py3/R/R2CCP/R2CCP-0.0.8-py3-none-any.whl
fi
pip install R2CCP-0.0.8-py3-none-any.whl --no-deps

# Install BoostedCP (from GitHub)
echo ""
echo "Installing BoostedCP..."
if [ ! -d "boosted-conformal" ]; then
    echo "Cloning BoostedCP repository..."
    git clone https://github.com/EliasCohen/boosted-conformal.git
fi
cd boosted-conformal
pip install -e .
cd ..

# Install CHR (Conformalized Histogram Regression)
echo ""
echo "Installing CHR..."
if [ ! -d "chr" ]; then
    echo "Cloning CHR repository..."
    git clone https://github.com/msesia/chr.git
fi
cd chr
pip install -e .
cd ..

# Initialize LVD submodule
echo ""
echo "Initializing LVD submodule..."
git submodule update --init --recursive

# Install LVD dependencies
echo ""
echo "Installing LVD..."
if [ -d "LVD" ] && [ "$(ls -A LVD)" ]; then
    cd LVD
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    if [ -f "setup.py" ]; then
        pip install -e .
    fi
    cd ..
else
    echo "Warning: LVD submodule is empty. Run 'git submodule update --init --recursive' manually."
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p model_paths
mkdir -p results
mkdir -p "results/oversampling"
mkdir -p model_logits

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. For GenAI-Bench example: cd Example_GenAI-Bench"
echo "2. For conformal prediction: cd 'conformal predictors'"
echo "3. For analysis notebooks: jupyter notebook analysis/"
echo ""
echo "Note: You may need to:"
echo "  - Set up model data in model_logits/ directory"
echo "  - Configure GPU/CUDA if using GPU acceleration"
echo "  - Download datasets (SummEval, DialSumm, ROSCOE, GenAI-Bench)"
