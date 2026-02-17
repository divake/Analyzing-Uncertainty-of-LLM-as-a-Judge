# Installation Guide

Complete installation guide to replicate all results from this repository.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for running LLM evaluations)
- 16GB+ RAM (32GB+ recommended for large models)
- Git

## Quick Start (Automated Installation)

```bash
# Run the automated installation script
bash install.sh
```

This will install all dependencies including special packages like R2CCP, BoostedCP, and CHR.

## Manual Installation (Step-by-Step)

### 1. Basic Setup

```bash
# Upgrade pip
pip install --upgrade pip

# Install standard dependencies
pip install -r requirements.txt
```

### 2. Install Special Conformal Prediction Packages

#### R2CCP (Regression-to-Classification Conformal Prediction)
```bash
wget https://files.pythonhosted.org/packages/py3/R/R2CCP/R2CCP-0.0.8-py3-none-any.whl
pip install R2CCP-0.0.8-py3-none-any.whl --no-deps
```

#### BoostedCP (Boosted Conformal Prediction)
```bash
git clone https://github.com/EliasCohen/boosted-conformal.git
cd boosted-conformal
pip install -e .
cd ..
```

#### CHR (Conformalized Histogram Regression)
```bash
git clone https://github.com/msesia/chr.git
cd chr
pip install -e .
cd ..
```

### 3. Initialize Git Submodules

```bash
# Initialize LVD submodule
git submodule update --init --recursive

# Install LVD if it has setup.py
cd LVD
pip install -e .
cd ..
```

### 4. Create Required Directories

```bash
mkdir -p model_paths
mkdir -p results/oversampling
mkdir -p model_logits/dsr1
mkdir -p summeval/data
mkdir -p summeval/prompts/summeval
```

## Dependency Overview

### Core Libraries
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers for LLMs
- **Datasets**: Hugging Face datasets library

### Data & Scientific Computing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions
- **Scikit-learn**: Machine learning utilities

### Conformal Prediction
- **R2CCP**: Regression-to-Classification Conformal Prediction
- **BoostedCP**: Boosted Conformal Prediction
- **CHR**: Conformalized Histogram Regression
- **MAPIE**: Model Agnostic Prediction Interval Estimator
- **LVD**: Learning under Varying Distribution (submodule)

### Visualization & Utilities
- **Matplotlib/Seaborn**: Plotting
- **Jupyter**: Interactive notebooks
- **tqdm**: Progress bars
- **qwen-vl-utils**: Qwen Vision-Language utilities

## GPU/CUDA Setup

For GPU acceleration (highly recommended):

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verification

Test your installation:

```python
# Test basic imports
import torch
import transformers
import pandas as pd
import numpy as np
from R2CCP.main import R2CCP
from mapie.quantile_regression import MapieQuantileRegressor

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Common Issues & Solutions

### Issue: R2CCP installation fails
**Solution**: Make sure to use `--no-deps` flag:
```bash
pip install R2CCP-0.0.8-py3-none-any.whl --no-deps
```

### Issue: BoostedCP or CHR not found
**Solution**: Make sure you cloned the repositories and installed with `-e`:
```bash
cd boosted-conformal && pip install -e . && cd ..
cd chr && pip install -e . && cd ..
```

### Issue: LVD submodule is empty
**Solution**: Initialize submodules:
```bash
git submodule update --init --recursive
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use model quantization in evaluation scripts

### Issue: qwen-vl-utils not found
**Solution**: Install directly from PyPI:
```bash
pip install qwen-vl-utils
```

## Data Setup

After installation, you'll need to obtain the datasets:

1. **GenAI-Bench**: Automatically loaded via `datasets` library
   ```python
   from datasets import load_dataset
   dataset = load_dataset("BaiqiL/GenAI-Bench")
   ```

2. **SummEval, DialSumm, ROSCOE**: Place data files in appropriate directories:
   - `summeval/data/summeval.json`
   - `summeval/data/dialsumm.jsonl`
   - Corresponding prompt files in `summeval/prompts/summeval/`

3. **Model Logits**: Run evaluation scripts first to generate logits, or place pre-computed logits in `model_logits/dsr1/`

## Next Steps

After installation:

1. **Test with Example**:
   ```bash
   cd Example_GenAI-Bench
   python VQA_eval.py
   ```

2. **Run Analysis Notebooks**:
   ```bash
   jupyter notebook analysis/
   ```

3. **Run Conformal Predictors**:
   ```bash
   cd "conformal predictors"
   python R2CCP_rancom.py
   ```

## System Requirements

**Minimum**:
- CPU: 4+ cores
- RAM: 16GB
- GPU: 8GB VRAM (for small models)
- Storage: 50GB

**Recommended**:
- CPU: 8+ cores
- RAM: 32GB+
- GPU: 24GB+ VRAM (for Qwen-32B, GPT-4 scale models)
- Storage: 100GB+
