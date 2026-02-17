# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation for **"Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction"** (https://arxiv.org/abs/2509.18658). This framework applies conformal prediction to construct prediction intervals for LLM-based evaluation scores, providing uncertainty quantification for LLM-as-a-judge systems.

**Key Concepts:**
- **Conformal Prediction**: Provides statistically valid prediction intervals with coverage guarantees
- **Ordinal Boundary Adjustment**: Adapts continuous intervals for discrete rating scales (e.g., 1-5)
- **Interval Midpoint**: Alternative scoring method using the midpoint of prediction intervals, offering lower bias than raw LLM scores
- **Judge Reprompting**: Re-evaluating judgments to improve reliability

## Quick Start

**First-time setup** (see [INSTALLATION.md](INSTALLATION.md) for details):
```bash
conda activate env_py311
pip install -r requirements.txt
# Clone and setup submodules (boosted-conformal, chr, LvD)
```

**Test installation**:
```bash
conda activate env_py311
python -c "from R2CCP.main import R2CCP; from mapie.quantile_regression import MapieQuantileRegressor; print('✓ Ready')"
```

**Run example**:
```bash
conda activate env_py311
cd Example_GenAI-Bench
python VQA_eval.py
```

## Repository Structure

### Core Components

**[Example_GenAI-Bench/](Example_GenAI-Bench/)** - Complete example applying the framework to GenAI-Bench dataset
- [VQA_eval.py](Example_GenAI-Bench/VQA_eval.py) - Vision-Language Model evaluation with logits extraction
- [interval_processing.py](Example_GenAI-Bench/interval_processing.py) - Range clipping, boundary adjustment, coverage/width calculation
- [evaluation_metrics.py](Example_GenAI-Bench/evaluation_metrics.py) - Correlation (Pearson/Spearman/Kendall) and error metrics (MSE/MAE/RMSE)
- [performance.ipynb](Example_GenAI-Bench/performance.ipynb) - Results visualization with scores, intervals, and evaluations

**[conformal predictors/](conformal%20predictors/)** - 7 conformal prediction methods (9 variants total)
- `R2CCP_rancom.py` - Regression-to-Classification Conformal Prediction (R2CCP)
- `CQR_random.py` - Conformalized Quantile Regression (2 variants)
- `BoostedCP_random.py` - Boosted Conformal Prediction (2 variants)
- `OrdinalRC_random.py` - Ordinal Regression Conformal
- `OrdinalAPS_random.py` - Ordinal Adaptive Prediction Sets
- `CHR_random..py` - Conformalized Histogram Regression
- `LVD_random.py` - Learning under Varying Distribution

**[analysis/](analysis/)** - Experimental analysis notebooks supporting paper results
- `statistics_intervals.ipynb` - Interval performance metrics
- `score_performance.py` - Score and midpoint performance calculation
- `R2CCP_distribution_shift.ipynb` - Cross-calibration between SummEval and DialSumm
- `R2CCP_validity.ipynb` - Temperature sensitivity analysis
- `prompt_sensitivity.ipynb` - GPT-4o vs GPT-4o mini comparison with different prompts
- `calsize_comparison.ipynb` / `calsize_experiment.ipynb` - Calibration set size experiments
- `heteroskedasticity_ht.ipynb` - Heteroskedasticity hypothesis testing
- `oversampling_raws.ipynb` - Oversampling and majority vote evaluation
- `human_performance.ipynb` - Human baseline construction
- `temperature_comparison.ipynb` - Judge model and temperature comparison

**[evaluations and reprompt on server/](evaluations%20and%20reprompt%20on%20server/)** - Server-based evaluations with Qwen and DeepSeek models
- `qwen_eval.py` / `qwen_eval_dialsumm.py` - SummEval and DialSumm evaluation
- `reasoning_eval.py` / `reasoning_eval_oversampling.py` - ROSCOE reasoning evaluation
- `reprompt_regrade.py` / `reprompt_regrade_reasoning.py` - Reprompt and regrade for SummEval/ROSCOE
- `reprompt_analysis.py` - Analysis of judge resistance to score changes
- `reprompt_improvement.py` - Reprompting improvement evaluation
- Bash scripts: `run_summ.sh`, `run_reasoning_eval.sh`, `run_reasoning_eval_oversampling.sh`

**[LVD/](LVD/)** - Git submodule for LVD method (https://github.com/zlin7/LVD.git)

## Key Architecture Patterns

### Evaluation Pipeline
1. **Model Inference** → Extract token logits from LLM judge at specific rating positions
2. **Conformal Predictor** → Generate prediction intervals from calibration set
3. **Interval Processing** → Apply range clipping + ordinal boundary adjustment
4. **Metrics Calculation** → Compute coverage, width, correlations, and errors

### Conformal Prediction Methods
All methods follow a common pattern:
```python
# Split data: 50% calibration, 50% test
X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

# Fit conformal predictor
model.fit(X_cal, y_cal)
intervals = model.get_intervals(X_test)  # Returns [(low, high), ...]

# Process intervals
low, high = range_modification(low, high, range_low=1, range_up=5)
low_adj = boundary_adjustment(low, label_set=[1,2,3,4,5], threshold=0.1)
high_adj = boundary_adjustment(high, label_set=[1,2,3,4,5], threshold=0.1)
```

### Logits Extraction
Models output logits for each possible rating. The framework extracts these for tokens corresponding to discrete scores (e.g., "1", "2", "3", "4", "5") and uses them as features for conformal prediction.

### Using BoostedCP and CHR
These methods require adding their directories to sys.path before importing:

```python
import sys
import os

# For BoostedCP
sys.path.insert(0, os.path.abspath("./boosted-conformal/"))
sys.path.insert(0, os.path.abspath("./boosted-conformal/third_party"))
from boostedCP.utils import cqr_preboost

# For CHR
sys.path.insert(0, os.path.abspath("./chr/"))
from chr.black_boxes import QNet
```

This pattern is already implemented in the conformal predictor scripts.

## Common Commands

### Running Evaluations

**SummEval evaluation (all dimensions):**
```bash
cd "evaluations and reprompt on server/"
bash run_summ.sh
```

**ROSCOE reasoning evaluation:**
```bash
cd "evaluations and reprompt on server/"
bash run_reasoning_eval.sh
# Or with oversampling:
bash run_reasoning_eval_oversampling.sh
```

**Single dimension evaluation:**
```bash
python qwen_eval.py \
    --prompt_fp summeval/prompts/summeval/con_detailed.txt \
    --summeval_fp summeval/data/summeval.json \
    --save_fp results/qwen_con_detailed.json \
    --model "Qwen/Qwen2.5-VL-32B-Instruct"
```

### Running Conformal Prediction

**R2CCP example:**
```bash
python "conformal predictors/R2CCP_rancom.py"
# Requires: model_logits/dsr1/Summeval_{dimension}_logits.csv
# Outputs: R2CCP_{dataset}_{dimension}_{seed}.csv
```

**General pattern for conformal predictors:**
```python
# Load data with logits as features, human labels as target
X = df.iloc[:, :-1]  # Logits/features
y = df.iloc[:, -1]   # Ground truth labels

# Run experiment
width, coverage = run_experiment(X, y, seed=42, dimension='consistency', dataset='summeval')
```

### Jupyter Notebooks

**Run analysis notebooks:**
```bash
jupyter notebook analysis/statistics_intervals.ipynb
jupyter notebook Example_GenAI-Bench/performance.ipynb
```

## Environment Setup

**Conda Environment**: This project uses `env_py311` (Python 3.11)
```bash
conda activate env_py311
```

**Complete Installation**: See [INSTALLATION.md](INSTALLATION.md) for full setup instructions
```bash
pip install -r requirements.txt
```

## Dependencies

Key Python packages with **specific version requirements**:
- **Model Inference**: `transformers`, `torch>=2.0.0`, `qwen_vl_utils`
- **Conformal Prediction**:
  - `R2CCP==0.0.8` (installed from wheel with `--no-deps`)
  - `mapie==0.8.6` ⚠️ **CRITICAL**: Code uses `MapieQuantileRegressor` which was renamed to `ConformalizedQuantileRegressor` in MAPIE v1.x. Must use 0.8.x for compatibility.
  - `pytorch_lightning>=2.0.0`, `configargparse`
- **Data & Analysis**: `datasets`, `pandas`, `numpy`, `scipy`, `sklearn`
- **Visualization**: `matplotlib`, `seaborn`, `jupyter`
- **Utilities**: `tqdm`, `psutil`

**R2CCP Installation**:
```bash
wget https://files.pythonhosted.org/packages/py3/R/R2CCP/R2CCP-0.0.8-py3-none-any.whl
pip install R2CCP-0.0.8-py3-none-any.whl --no-deps
pip install configargparse pytorch_lightning torchvision
```

## Git Submodules

This repository has **three submodule-style directories** at specific commits:

**LvD** (commit `1be7dc5`):
```bash
git clone https://github.com/zlin7/LVD.git LvD
cd LvD && git checkout 1be7dc54dbaf8205c1e40765d10dd0f5d5a84318 && cd ..
```

**boosted-conformal** (commit `41aba95`):
```bash
git clone https://github.com/ran-xie/boosted-conformal.git
cd boosted-conformal && git checkout 41aba95e2672c80836ee148bf4ae488b2b8d75e1 && cd ..
```

**chr** (commit `4f02607`):
```bash
git clone https://github.com/msesia/chr.git
cd chr && git checkout 4f02607f2c7f35237b4e042132db9a6404294b04 && cd ..
```

**Note**: These are tracked as git submodules in the repository but require manual cloning at specific commits.

## Important Notes

### Configuration
- **Environment**: Always use `conda activate env_py311` before running scripts
- **Alpha level**: Default significance level is α=0.10 (90% coverage target)
- **Rating scales**: Most experiments use 1-5 discrete rating scales
- **Calibration/test split**: Standard 50/50 split for conformal prediction
- **Seeds**: Experiments use varying random seeds for statistical robustness (typically 100 runs)

### Directory Structure
- **Model paths**: Conformal predictors save/load from `model_paths/` directory
- **Data paths**: Evaluation scripts expect data in `summeval/data/`, `results/`, `model_logits/` directories
- **Submodules**: Directory names: `LvD/` (lowercase 'v'), `boosted-conformal/`, `chr/`

### Common Issues
- **MAPIE version**: Code imports `from mapie.quantile_regression import MapieQuantileRegressor`. This class was renamed to `ConformalizedQuantileRegressor` in MAPIE v1.x, so version 0.8.6 is required for backward compatibility.
- **Directory paths**: Scripts use relative paths, run from repository root
- **GPU memory**: Large models (Qwen-32B) require 24GB+ VRAM
- **Jupyter kernel**: Select `env_py311` kernel when using notebooks
