# Paper Reference: Analyzing Uncertainty of LLM-as-a-Judge

**Full Title:** Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction
**ArXiv:** https://arxiv.org/abs/2509.18658
**Authors:** Huanxin Sheng, Xinyi Liu, Hangfeng He, Jieyu Zhao, Jian Kang

> This file captures everything needed to replicate the paper. Do not re-read the paper — use this file.

---

## 1. MODELS REQUIRED

### LLM Judges (3 models used)

| Model | HuggingFace ID | Type | Temperature | Used For |
|---|---|---|---|---|
| GPT-4o mini | `gpt-4o-mini-2024-07-18` | **API (OpenAI)** | 1 | SummEval, DialSumm (via G-Eval) |
| DeepSeek-R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | **Local (HuggingFace)** | 0 | SummEval, DialSumm, ROSCOE (via G-Eval + SocREval) |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct` | **Local (HuggingFace)** | 0 | SummEval, DialSumm, ROSCOE (via G-Eval + SocREval) |

**⚠️ IMPORTANT:** GPT-4o mini requires an **OpenAI API key**. The two local models run on GPU via HuggingFace Transformers — no API key needed.

**GPU Requirements:**
- DeepSeek-R1-Distill-Qwen-32B: ~40GB VRAM (fits on 1x RTX 6000 Ada)
- Qwen2.5-72B-Instruct: ~80GB VRAM (needs 2x RTX 6000 Ada with `device_map="auto"`)

### Judge Evaluation Frameworks (2 frameworks)

| Framework | Used With | Purpose |
|---|---|---|
| **G-Eval** (Liu et al., 2023) | All 3 LLMs | Summarization evaluation (SummEval, DialSumm) |
| **SocREval** (He et al., 2024) | All 3 LLMs | Reasoning evaluation (ROSCOE) |

---

## 2. DATASETS REQUIRED

### 2.1 Text Summarization

| Dataset | Samples | Dimensions | Annotators | Scale | Source |
|---|---|---|---|---|---|
| **SummEval** (Fabbri et al., 2021) | 1,600 | Consistency, Coherence, Fluency, Relevance | 3 human raters | Likert (1–5) | Need to download |
| **DialSumm** (Gao & Wan, 2022) | 1,400 | Consistency, Coherence, Fluency, Relevance | 3 human raters | Likert (1–5) → GPA scale | Need to download |

- Ground truth = **average of 3 human ratings**
- For SummEval/DialSumm, GPA-scale scores are mapped to 1–13 Likert scale by linear transformation (1.33×score − 2)

### 2.2 Reasoning (ROSCOE benchmark)

| Sub-dataset | Samples | Task | Scale |
|---|---|---|---|
| **CosmosQA** (Li et al., 2023) | ~200 | Reading comprehension | Likert |
| **DROP** (Dua et al., 2019) | ~200 | Discrete reasoning | Likert |
| **e-SNLI** (Camburu et al., 2018) | ~200 | Natural language inference | Likert |
| **GSM8K** (Cobbe et al., 2021) | ~200 | Math word problems | Likert |

- Evaluated using **SocREval** framework (overall quality annotations)

### 2.3 Vision (GenAI-Bench — bonus example)

| Dataset | Source | Auto-download |
|---|---|---|
| **GenAI-Bench** (Li et al., 2024a) | HuggingFace `BaiqiL/GenAI-Bench` | ✅ Yes, via `load_dataset()` |

---

## 3. PROMPTS

### Prompt Style
- **G-Eval framework**: Chain-of-Thought (CoT) prompt — instructs LLM to rate and output the score in a specific format
- **SocREval framework**: Reference-free reasoning evaluation prompt
- Prompts listed in **Appendix A.13** of the paper (not reproduced here — need from repo or authors)

### Expected Prompt Files (in `evaluations and reprompt on server/summeval/prompts/summeval/`)
```
con_detailed.txt   → Consistency
coh_detailed.txt   → Coherence
flu_detailed.txt   → Fluency
rel_detailed.txt   → Relevance
```

### Token Targeting Strategy
- LLM outputs logits at the rating token position
- Extract log-probabilities for all rating tokens: "1", "2", "3", "4", "5"
- Aggregate equivalent tokens (e.g., "two" and "2")
- Result: K=5 dimensional logit feature vector z per sample
- Rule-based strategy to locate rating token at most frequent position

---

## 4. CONFORMAL PREDICTION METHODS (9 total)

### 4.1 Regression-Based (7 methods, use token logits directly)

| # | Method | Short Name | Key Hyperparameters | Runtime (mean) |
|---|---|---|---|---|
| 1 | Conformalized Quantile Regression | **CQR** | MAPIE + gradient boosting regressor | 0.83s |
| 2 | Asymmetric CQR | **Asym CQR** | Same as CQR | 0.82s |
| 3 | Conformalized Histogram Regression | **CHR** | QNet: batch=32, hidden=256, lr=5e-4, epoch=1000 | 9.54s |
| 4 | Locally Valid and Discriminative | **LVD** | DNN_model, KernelMLKR(d=10, n_iters=500) | 93.67s |
| 5 | Boosted CQR | **Boosted CQR** | n_rounds_cv=500, lr=0.02 | 91.43s |
| 6 | Boosted LCP | **Boosted LCP** | Same as Boosted CQR | 87.27s |
| 7 | R2CCP | **R2CCP** | max_epochs=100 | 9.25s |

### 4.2 Ordinal Classification-Based (2 methods, use token probabilities after softmax)

| # | Method | Short Name | Notes |
|---|---|---|---|
| 8 | Ordinal Adaptive Prediction Sets | **OrdinalAPS** | Default settings |
| 9 | Ordinal Risk Control | **OrdinalRC** | WeightedCRPredictor, weight-based risk |

**Note:** Ordinal methods only evaluated **after** boundary adjustment.

---

## 5. EXPERIMENTAL SETUP

| Parameter | Value |
|---|---|
| Significance level α | 0.10 (targeting 90% coverage) |
| Data split | 50% calibration / 50% test |
| Number of random seeds | 30 (seeds 1–30) |
| Reported metrics | Mean interval width & coverage rate over 30 runs |
| Rating scale | 1–5 Likert (K=5) |
| Boundary adjustment threshold | Shrink to nearest integer label |

---

## 6. METRICS

### Interval Metrics (Tables 1, 2, 12, 13)
- **Interval Width** (lower = better efficiency)
- **Coverage Rate %** (target: ≥ 90%)
- Coverage < 85%: gray text; 85–90%: underlined; ≥ 90% with smallest width: **bold**

### Score/Midpoint Metrics (Table 3)
- **MSE** — Mean Squared Error (lower = better)
- **MAE** — Mean Absolute Error (lower = better)
- **Spearman ρ** — rank correlation (higher = better)
- **Kendall τ** — rank correlation (higher = better)
- Calculated over 30 experiments

### Compared Score Methods (Table 3)
1. **Raw Score** — direct LLM output
2. **Weighted Average** — weighted average of token probabilities
3. **Con R2CCP** — midpoint of R2CCP interval **before** boundary adjustment
4. **Dis R2CCP** — midpoint of R2CCP interval **after** boundary adjustment

---

## 7. PAPER RESULTS

### Table 1 — Interval Width / Coverage (%) BEFORE Boundary Adjustment
*(Format: Width / Coverage%)*

#### SummEval Evaluated with G-Eval

| Method | Consistency | Coherence | Fluency | Relevance |
|---|---|---|---|---|
| **GPT-4o mini** | | | | |
| CQR | 1.15 / 94.16% | **2.87 / 93.15%** | 1.44 / 92.92% | 2.09 / 90.92% |
| Asym CQR | 1.25 / 94.97% | 2.91 / 93.76% | 1.60 / 93.75% | 2.13 / 91.42% |
| CHR | 0.67 / 88.99% | 2.43 / 82.96% | 0.91 / 88.86% | 1.74 / 82.62% |
| LVD | 1.01 / 92.35% | 2.73 / 89.76% | 1.11 / 90.59% | 2.02 / 89.55% |
| Boosted CQR | 1.01 / 87.75% | 2.67 / 88.68% | 1.00 / 88.68% | 1.91 / 87.19% |
| Boosted LCP | 0.76 / 89.22% | 2.67 / 87.34% | 0.92 / 89.18% | 1.91 / 87.19% |
| **R2CCP** | **0.69 / 90.88%** | 2.62 / 89.63% | **0.92 / 89.36%** | **1.97 / 89.70%** |
| **DeepSeek-R1-Distill-Qwen-32B** | | | | |
| CQR | 1.16 / 93.88% | 2.67 / 92.50% | 1.31 / 93.01% | 2.13 / 91.05% |
| Asym CQR | 1.30 / 95.13% | 2.72 / 92.86% | 1.49 / 94.52% | 2.21 / 92.06% |
| CHR | 0.82 / 91.17% | 2.23 / 87.07% | 0.90 / 89.24% | 1.87 / 86.38% |
| LVD | 0.97 / 92.93% | 2.43 / 91.10% | 1.00 / 91.10% | 2.04 / 90.14% |
| Boosted CQR | 1.08 / 90.30% | 1.76 / 89.30% | 1.08 / 89.46% | 1.91 / 86.89% |
| Boosted LCP | 0.77 / 89.20% | 2.32 / 86.70% | 0.93 / 89.10% | 1.91 / 86.89% |
| **R2CCP** | **0.69 / 90.44%** | **2.30 / 90.12%** | **0.89 / 90.09%** | **2.00 / 89.84%** |
| **Qwen2.5-72B-Instruct** | | | | |
| CQR | 0.98 / 93.10% | 2.73 / 92.25% | 1.44 / 93.73% | 2.11 / 91.30% |
| Asym CQR | 1.10 / 94.47% | 2.80 / 93.13% | 1.63 / 94.79% | 2.17 / 92.21% |
| CHR | 0.66 / 92.21% | 2.14 / 86.13% | 0.98 / 88.93% | 1.61 / 85.78% |
| LVD | 0.85 / 92.82% | 2.55 / 90.49% | 1.09 / 90.94% | 1.94 / 89.27% |
| Boosted CQR | 0.80 / 88.28% | 2.43 / 86.92% | 1.24 / 89.22% | 1.86 / 87.17% |
| Boosted LCP | 0.67 / 88.81% | 2.43 / 86.92% | 0.94 / 89.26% | 1.86 / 87.51% |
| **R2CCP** | **0.61 / 90.73%** | **2.44 / 89.54%** | **0.95 / 90.18%** | **1.98 / 90.45%** |

#### ROSCOE Evaluated with SocREval — BEFORE Boundary Adjustment

| Method | CosmosQA | DROP | e-SNLI | GSM8K |
|---|---|---|---|---|
| **GPT-4o mini** | | | | |
| CQR | **3.53 / 95.27%** | **3.82 / 96.70%** | 3.04 / 96.62% | **3.53 / 95.67%** |
| Asym CQR | 3.90 / 98.71% | 3.91 / 98.09% | 2.87 / 98.08% | 3.89 / 98.43% |
| CHR | 2.54 / 73.06% | 1.86 / 73.06% | 1.36 / 72.49% | 1.98 / 78.67% |
| LVD | 3.10 / 83.95% | 2.49 / 83.05% | 2.17 / 86.18% | 3.08 / 89.57% |
| Boosted CQR | 3.15 / 80.07% | 2.92 / 85.40% | 1.88 / 82.80% | 3.08 / 82.50% |
| Boosted LCP | 3.60 / 83.91% | 2.92 / 85.40% | 1.88 / 81.23% | 3.36 / 85.93% |
| R2CCP | 2.96 / 85.85% | 2.43 / 84.73% | 1.75 / 84.02% | 2.15 / 85.07% |
| **DeepSeek-R1-Distill-Qwen-32B** | | | | |
| CQR | 3.48 / 96.70% | 3.83 / 96.35% | 2.97 / 96.36% | 3.46 / 95.60% |
| Asym CQR | 3.84 / 99.08% | 3.95 / 99.27% | 2.86 / 96.45% | 3.84 / 98.43% |
| CHR | 2.66 / 76.50% | 1.95 / 78.00% | 1.38 / 71.97% | 2.01 / 81.60% |
| LVD | 3.25 / 88.10% | 2.62 / 88.06% | 2.24 / 90.96% | 3.02 / 90.63% |
| Boosted CQR | 3.17 / 82.72% | 2.79 / 85.46% | 1.79 / 80.96% | 2.91 / 79.83% |
| Boosted LCP | 3.48 / 81.60% | 2.79 / 85.46% | 1.81 / 80.61% | 3.43 / 85.23% |
| R2CCP | 2.94 / 86.97% | 2.29 / 86.35% | 1.85 / 87.87% | 1.88 / 85.33% |
| **Qwen2.5-72B-Instruct** | | | | |
| CQR | 3.37 / 94.80% | 3.79 / 97.02% | 3.01 / 97.37% | 3.35 / 95.33% |
| Asym CQR | 3.86 / 99.01% | 3.89 / 98.67% | 2.77 / 96.84% | 3.87 / 98.97% |
| CHR | 2.49 / 82.14% | 2.02 / 82.89% | 1.18 / 84.56% | 1.79 / 85.27% |
| LVD | 3.05 / 84.29% | 2.67 / 90.57% | 1.91 / 85.96% | 2.83 / 90.13% |
| Boosted CQR | 3.05 / 79.08% | 2.56 / 81.17% | 1.51 / 77.15% | 2.11 / 80.67% |
| Boosted LCP | 3.46 / 80.41% | 2.81 / 85.75% | 1.74 / 77.50% | 3.38 / 86.23% |
| R2CCP | 2.90 / 85.34% | 2.39 / 86.25% | 1.59 / 84.50% | 2.00 / 86.73% |

---

### Table 2 — Interval Width / Coverage (%) AFTER Boundary Adjustment

#### SummEval Evaluated with G-Eval

| Method | Consistency | Coherence | Fluency | Relevance |
|---|---|---|---|---|
| **GPT-4o mini** | | | | |
| CQR | 1.15 / 95.45% | 2.87 / 94.94% | 1.44 / 93.80% | 2.09 / 93.56% |
| Asym CQR | 1.25 / 96.02% | 2.90 / 95.41% | 1.60 / 94.57% | 2.14 / 94.14% |
| CHR | 0.70 / 91.79% | 2.41 / 87.78% | 0.94 / 90.60% | 1.74 / 88.10% |
| LVD | 1.01 / 94.11% | 2.73 / 93.72% | 1.12 / 92.70% | 2.03 / 93.82% |
| Boosted CQR | 0.99 / 92.81% | 2.68 / 93.53% | 1.54 / 94.80% | 2.00 / 93.40% |
| Boosted LCP | 0.74 / 91.90% | 2.68 / 93.53% | 0.90 / 90.88% | 1.91 / 92.70% |
| **R2CCP** | **0.68 / 92.15%** | **2.62 / 92.81%** | **0.92 / 90.99%** | **1.97 / 93.38%** |
| OrdinalAPS | 2.28 / 71.48% | 1.88 / 64.84% | 1.78 / 13.65% | 2.36 / 87.94% |
| OrdinalRC | 2.41 / 75.19% | 2.02 / 67.38% | 3.19 / 13.58% | 2.51 / 90.30% |
| **DeepSeek-R1-Distill-Qwen-32B** | | | | |
| CQR | 1.15 / 95.02% | 2.67 / 94.34% | 1.32 / 94.01% | 2.13 / 93.67% |
| Asym CQR | 1.31 / 95.99% | 2.72 / 94.83% | 1.49 / 95.77% | 2.21 / 94.53% |
| CHR | 0.87 / 93.96% | 2.23 / 91.42% | 1.01 / 91.98% | 1.87 / 90.84% |
| LVD | 0.97 / 95.01% | 2.44 / 94.58% | 1.00 / 93.21% | 2.04 / 94.12% |
| Boosted CQR | 1.08 / 93.55% | 2.32 / 92.37% | 1.15 / 93.48% | 2.01 / 92.81% |
| Boosted LCP | 0.76 / 92.03% | 2.32 / 92.37% | 0.93 / 91.34% | 1.92 / 92.81% |
| **R2CCP** | **0.68 / 91.57%** | **2.30 / 92.37%** | **0.83 / 91.80%** | **1.99 / 91.90%** |
| OrdinalAPS | 2.51 / 90.06% | 2.52 / 90.64% | 3.76 / 91.08% | 2.13 / 89.98% |
| OrdinalRC | 2.54 / 90.11% | 2.56 / 91.22% | 3.73 / 89.53% | 2.14 / 62.35% |
| **Qwen2.5-72B-Instruct** | | | | |
| CQR | 0.98 / 94.35% | 2.72 / 94.18% | 1.45 / 94.79% | 2.09 / 94.02% |
| Asym CQR | 1.10 / 95.47% | 2.79 / 94.70% | 1.64 / 95.63% | 2.17 / 94.44% |
| CHR | 0.66 / 92.21% | 2.41 / 88.17% | 0.91 / 91.16% | 1.61 / 85.78% |
| LVD | 0.85 / 95.11% | 2.56 / 94.05% | 1.09 / 93.45% | 1.95 / 93.86% |
| Boosted CQR | 0.81 / 92.36% | 2.44 / 92.26% | 1.20 / 93.66% | 1.86 / 94.90% |
| Boosted LCP | 0.65 / 91.26% | 2.44 / 92.26% | 0.93 / 91.20% | 1.86 / 92.57% |
| **R2CCP** | **0.59 / 91.83%** | **2.43 / 92.78%** | **0.95 / 92.12%** | **1.98 / 89.29%** |
| OrdinalAPS | 2.86 / 90.18% | 3.01 / 90.59% | 3.05 / 45.43% | 2.75 / 90.29% |
| OrdinalRC | 2.85 / 90.00% | 2.96 / 89.35% | 3.21 / 53.41% | 2.75 / 90.14% |

#### ROSCOE Evaluated with SocREval — AFTER Boundary Adjustment

| Method | CosmosQA | DROP | e-SNLI | GSM8K |
|---|---|---|---|---|
| **GPT-4o mini** | | | | |
| CQR | 3.53 / 95.34% | 3.82 / 97.05% | 3.04 / 96.89% | 3.53 / 95.67% |
| Asym CQR | 3.84 / 99.08% | 3.91 / 98.73% | 2.87 / 96.89% | 3.89 / 98.80% |
| CHR | 2.54 / 73.06% | 1.87 / 78.62% | 1.36 / 72.49% | 1.98 / 78.67% |
| LVD | 3.13 / 91.53% | 2.52 / 90.22% | 2.17 / 94.82% | 3.09 / 93.37% |
| Boosted CQR | 3.20 / 93.40% | 2.63 / 75.63% | 3.01 / 91.27% | 3.26 / 93.09% |
| Boosted LCP | 3.60 / 95.48% | 3.01 / 91.27% | 1.90 / 91.80% | 3.26 / 92.17% |
| **R2CCP** | **2.91 / 90.58%** | **2.41 / 77 / 90.11%** | **1.80 / 92.35%** | **2.09 / 86.93%** |
| OrdinalAPS | 0.73 / 47.52% | 0.83 / 55.08% | 0.72 / 52.76% | 0.58 / 73.90% |
| OrdinalRC | 0.82 / 49.46% | 0.91 / 57.11% | 0.80 / 54.61% | 0.60 / 74.43% |
| **DeepSeek-R1-Distill-Qwen-32B** | | | | |
| CQR | 3.48 / 96.80% | 3.82 / 96.54% | 2.96 / 96.90% | 3.45 / 95.63% |
| Asym CQR | 3.84 / 99.08% | 3.95 / 99.08% | 2.88 / 96.45% | 3.84 / 98.43% |
| CHR | 2.66 / 76.50% | 1.97 / 85.90% | 1.39 / 85.96% | 2.01 / 86.60% |
| LVD | 3.28 / 95.27% | 2.67 / 93.75% | 2.24 / 96.36% | 3.03 / 94.40% |
| Boosted CQR | 2.93 / 95.21% | 2.53 / 93.30% | 2.29 / 77.93% | 2.50 / 92.23% |
| Boosted LCP | 3.46 / 95.95% | 2.80 / 91.94% | 1.87 / 92.89% | 3.36 / 93.63% |
| **R2CCP** | **2.91 / 90.58%** | **1.80 / 92.35%** | **1.80 / 90.58%** | **1.80 / 86.93%** |
| OrdinalAPS | 1.32 / 60.00% | 1.26 / 78.22% | 1.46 / 87.85% | 1.50 / 85.67% |
| OrdinalRC | 1.44 / 62.35% | 1.29 / 78.22% | 1.33 / 86.07% | 1.55 / 86.07% |
| **Qwen2.5-72B-Instruct** | | | | |
| CQR | 3.36 / 95.07% | 3.79 / 97.08% | 3.01 / 97.68% | 3.34 / 95.33% |
| Asym CQR | 3.85 / 99.18% | 3.89 / 99.18% | 2.77 / 97.06% | 3.87 / 98.97% |
| CHR | 2.49 / 82.14% | 2.49 / 82.89% | 1.18 / 84.56% | 1.79 / 85.27% |
| LVD | 3.07 / 92.01% | 2.67 / 93.87% | 1.91 / 95.53% | 2.87 / 93.43% |
| Boosted CQR | 3.10 / 94.01% | 2.56 / 90.79% | 1.49 / 92.11% | 2.82 / 92.03% |
| Boosted LCP | 3.40 / 94.90% | 2.84 / 92.41% | 1.79 / 91.84% | 3.33 / 92.90% |
| **R2CCP** | **2.88 / 89.29%** | **2.41 / 90.00%** | **1.55 / 90.11%** | **1.96 / 88.57%** |
| OrdinalAPS | 0.71 / 55.99% | 0.25 / 56.83% | 0.67 / 77.68% | 0.16 / 70.87% |
| OrdinalRC | 0.75 / 57.28% | 0.29 / 56.83% | 0.80 / 79.74% | 0.19 / 71.37% |

---

### Table 3 — Midpoints vs Raw Score vs Weighted Average (SummEval + G-Eval)
*(Format: MSE / MAE / Spearman ρ / Kendall τ — mean over 30 seeds)*
**Bold** = better than both baselines | *Underline* = better than one baseline

#### GPT-4o mini

| Method | Coherence (MSE/MAE/ρ/τ) | Consistency (MSE/MAE/ρ/τ) | Fluency (MSE/MAE/ρ/τ) | Relevance (MSE/MAE/ρ/τ) |
|---|---|---|---|---|
| Raw Score | 1.729 / 1.055 / 0.446 / 0.373 | 1.674 / 1.073 / 0.480 / 0.437 | 3.907 / 1.977 / 0.219 / 0.197 | 1.009 / 0.786 / 0.512 / 0.427 |
| Weighted Avg | 1.643 / 1.037 / 0.514 / 0.379 | 1.548 / 1.066 / 0.478 / 0.383 | 3.412 / 1.733 / 0.319 / 0.250 | 0.865 / 0.737 / 0.567 / 0.419 |
| Con R2CCP | **0.791 / 0.716** / 0.512 / 0.373 | **0.510 / 0.432** / 0.455 / 0.371 | **0.442 / 0.491** / 0.330 / 0.261 | **0.418 / 0.509** / 0.546 / 0.403 |
| Dis R2CCP | **0.794 / 0.715** / 0.508 / 0.386 | **0.512 / 0.428** / 0.462 / 0.300 | **0.443 / 0.488** / 0.330 / 0.300 | **0.423 / 0.509** / 0.540 / 0.423 |

#### DeepSeek-R1-Distill-Qwen-32B

| Method | Coherence (MSE/MAE/ρ/τ) | Consistency (MSE/MAE/ρ/τ) | Fluency (MSE/MAE/ρ/τ) | Relevance (MSE/MAE/ρ/τ) |
|---|---|---|---|---|
| Raw Score | 1.010 / 0.775 / 0.549 / 0.457 | 1.229 / 0.770 / 0.467 / 0.425 | 1.549 / 0.387 / 0.355 / — | 0.763 / 0.682 / 0.520 / 0.437 |
| Weighted Avg | 0.869 / 0.734 / 0.599 / 0.447 | 1.439 / 1.065 / 0.468 / 0.375 | 2.783 / 1.564 / 0.420 / 0.332 | 0.646 / 0.632 / 0.565 / 0.419 |
| Con R2CCP | **0.599 / 0.619** / 0.663 / 0.484 | **0.564 / 0.441** / 0.445 / 0.361 | **0.373 / 0.455** / 0.391 / 0.311 | **0.431 / 0.513** / 0.555 / 0.412 |
| Dis R2CCP | **0.602 / 0.619** / 0.661 / 0.508 | **0.566 / 0.441** / 0.462 / 0.423 | **0.375 / 0.454** / 0.393 / 0.351 | **0.434 / 0.512** / 0.548 / 0.431 |

#### Qwen2.5-72B-Instruct

| Method | Coherence (MSE/MAE/ρ/τ) | Consistency (MSE/MAE/ρ/τ) | Fluency (MSE/MAE/ρ/τ) | Relevance (MSE/MAE/ρ/τ) |
|---|---|---|---|---|
| Raw Score | 1.432 / 0.981 / 0.426 / 0.358 | 2.068 / 1.237 / 0.458 / 0.416 | 4.476 / 1.958 / 0.310 / 0.281 | 1.188 / 0.903 / 0.498 / 0.420 |
| Weighted Avg | 1.282 / 0.932 / 0.539 / 0.395 | 1.847 / 1.213 / 0.483 / 0.387 | 4.236 / 1.928 / 0.363 / 0.285 | 1.091 / 0.885 / 0.555 / 0.412 |
| Con R2CCP | **0.675 / 0.659** / 0.603 / 0.444 | **0.469 / 0.396** / 0.465 / 0.378 | **0.414 / 0.486** / 0.340 / 0.269 | **0.407 / 0.502** / 0.571 / 0.425 |
| Dis R2CCP | **0.678 / 0.659** / 0.600 / 0.456 | **0.469 / 0.387** / 0.538 / 0.498 | **0.416 / 0.485** / 0.342 / 0.306 | **0.411 / 0.501** / 0.566 / 0.444 |

---

### Table 5 — Hyperparameter Settings

| Method | Hyperparameters |
|---|---|
| CQR | MAPIE + gradient boosting regressor with quantile loss |
| Asymmetric CQR | Same as CQR |
| CHR | QNet estimator, batch_size=32, hidden_dim=256, lr=5e-4, epoch=1000 |
| LVD | DNN_model, readout_layer=pretrain_general(seed=0, quiet=True, model_setting=0), KernelMLKR(d=10, seed=0, n_iters=500, norm=True, lr=1e-3) |
| Boosted LCP | len_local_boost: n_rounds_cv=500, learning_rate=0.02, store=True, verbose=False |
| Boosted CQR | len_cqr_boost: same as Boosted LCP |
| R2CCP | Default settings + max_epochs=100 |
| OrdinalAPS | Default settings |
| OrdinalRC | Default settings with WeightedCRPredictor |

### Table 6 — Runtime and Memory

| Method | Time Mean (s) | Time Std (s) | Memory Mean (MB) | Memory Std (MB) |
|---|---|---|---|---|
| CQR | 0.83 | 0.03 | 0.35 | 0.00 |
| Asymmetric CQR | 0.82 | 0.03 | 0.35 | 0.00 |
| CHR | 9.54 | 0.25 | 0.62 | 0.01 |
| LVD | 93.67 | 2.82 | 0.55 | 0.01 |
| Boosted CQR | 91.43 | 3.24 | 0.89 | 0.05 |
| Boosted LCP | 87.27 | 2.57 | 2.05 | 0.01 |
| R2CCP | 9.25 | 0.53 | 1.35 | 0.00 |
| OrdinalAPS | 0.01 | 0.00 | 0.20 | 0.00 |
| OrdinalRC | 0.03 | 0.00 | 0.19 | 0.00 |

---

## 8. KEY FINDINGS (for quick comparison against our results)

### Finding 1: Best Conformal Prediction Method
- **R2CCP** = best overall balance of coverage + efficiency (narrowest intervals at ≥90% coverage)
- **LVD** = slightly higher coverage but wider intervals
- **Boosted LCP** = comparable to R2CCP but less efficient
- **CQR/Asym CQR** = always achieve 90% coverage but with much wider intervals (>3 for ROSCOE)
- **OrdinalAPS/OrdinalRC** = poor on ROSCOE (often fail to reach 90% coverage)

### Finding 2: Best LLM Judge
- **DeepSeek-R1-Distill-Qwen-32B** = most consistent coverage (best for high-stakes applications)
- **Qwen2.5-72B-Instruct** = typically narrowest intervals (most efficient)
- **GPT-4o mini** = good performance but slightly worse than open-source alternatives

### Finding 3: Best Recommended Combination
- **High-stakes**: DeepSeek-R1-Distill-Qwen-32B + G-Eval + LVD
- **Most efficient**: Qwen2.5-72B-Instruct + R2CCP + SocREval

### Finding 4: Boundary Adjustment Always Improves Coverage
- All coverage rates improve after boundary adjustment
- Coverages in SummEval/DialSumm increase from 83–88% → consistently >90%
- LVD on e-SNLI (Qwen2.5-72B): 85.96% → 95.53% after adjustment

### Finding 5: Midpoints Reduce Bias vs Raw Score
- R2CCP midpoints reduce MSE by up to **88.7%** vs raw score
  - Example: Fluency (GPT-4o mini) — Raw: MSE=3.907 → Midpoint: MSE=0.443
- MAE consistently below 0.5 for midpoints (vs >1.0 for raw score)
- Correlation (ρ, τ) roughly comparable or slightly worse — accuracy tradeoff

### Finding 6: Calibration Size Matters
- Coverage increases and error bars shrink as calibration set size increases
- 50% calibration split is sufficient for stable ≥90% coverage

### Finding 7: Reprompting Has Limited Effect
- If initial score is **within** the prediction interval → reprompting strengthens confidence in that score
- If initial score is **outside** the interval → judge resists changing to a score within the interval
  - Reason: model constrained to integer outputs, finds moving to adjacent integer "unreasonable"

---

## 9. WHAT WE NEED TO REPRODUCE (Checklist)

### ✅ Already Done
- [x] All Python packages installed (env_py311)
- [x] R2CCP, MAPIE 0.8.6, PyTorch, Transformers installed
- [x] Submodules cloned (LvD, boosted-conformal, chr) at correct commits
- [x] GPU available (2x RTX 6000 Ada = 96GB VRAM)

### ❌ Still Needed

**Models:**
- [ ] **GPT-4o mini** → OpenAI API key required
- [ ] **DeepSeek-R1-Distill-Qwen-32B** → Download from HuggingFace (~64GB)
- [ ] **Qwen2.5-72B-Instruct** → Download from HuggingFace (~140GB)

**Datasets:**
- [ ] **SummEval** → `summeval/data/summeval.json` (1,600 samples)
- [ ] **DialSumm** → `summeval/data/dialsumm.jsonl` (1,400 samples)
- [ ] **ROSCOE** (CosmosQA, DROP, e-SNLI, GSM8K) → ~200 samples each
- [ ] **GenAI-Bench** → ✅ Auto-downloads

**Prompt Files:**
- [ ] G-Eval prompts: `con_detailed.txt`, `coh_detailed.txt`, `flu_detailed.txt`, `rel_detailed.txt`
- [ ] SocREval prompts for reasoning tasks
- [ ] Reprompt templates (3 files already in repo: `reprompt.txt`, `reprompt1.txt`, `reprompt2.txt`)

**Pre-computed Logits (needed for conformal predictors without re-running models):**
- [ ] `model_logits/dsr1/Summeval_{consistency,coherence,fluency,relevance}_logits.csv`
- [ ] `model_logits/qwen/Summeval_{dimension}_logits.csv`
- [ ] `model_logits/qwen/Dialsumm_{dimension}_logits.csv`
- [ ] `model_logits/qwen/SocREval_{dimension}_logits.csv`

---

## 10. REPRODUCTION ORDER

```
Step 1: Get datasets (SummEval, DialSumm, ROSCOE)
Step 2: Get/reconstruct prompt files
Step 3: Run LLM evaluation (qwen_eval.py, reasoning_eval.py) → generates JSON outputs
Step 4: Extract logits → generates *_logits.csv files in model_logits/
Step 5: Run conformal predictors (conformal predictors/*.py) → generates interval CSVs
Step 6: Run analysis notebooks (analysis/) → generates paper tables/figures
```

Or shortcut if logit CSVs are obtained from authors:
```
Step 5 + Step 6 only
```
