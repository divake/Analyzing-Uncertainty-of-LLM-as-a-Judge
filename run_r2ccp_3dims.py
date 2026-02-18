#!/usr/bin/env python3
"""Quick R2CCP run for 3 available dimensions (con, coh, flu) with seed=42"""

import sys, os
sys.path.insert(0, '.')
import numpy as np
import random
import torch
torch.set_float32_matmul_precision('medium')
import pandas as pd
import time
from R2CCP.main import R2CCP
from sklearn.model_selection import train_test_split

ALPHA = 0.10
os.makedirs('model_paths', exist_ok=True)

def merge_intervals(sample_intervals):
    if not sample_intervals:
        return (1, 5)
    return (min(l for l,h in sample_intervals), max(h for l,h in sample_intervals))

def run_experiment(X, y, seed, dimension):
    random.seed(seed); np.random.seed(seed)
    X = X.to_numpy().astype(np.float32)
    y = y.to_numpy().astype(np.float32)

    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    pth = 'model_paths/model_save_destination.pth'
    if os.path.exists(pth):
        os.remove(pth)

    model = R2CCP({'model_path': pth, 'max_epochs': 100, 'alpha': ALPHA})
    model.fit(X_cal, y_cal.flatten())
    intervals = model.get_intervals(X_test)
    intervals = [merge_intervals(si) for si in intervals]

    in_interval = [(l <= yt <= h) for (l,h), yt in zip(intervals, y_test)]
    coverage = np.mean(in_interval)
    width    = np.mean([h - l for l, h in intervals])

    del model; torch.cuda.empty_cache(); time.sleep(0.5)
    return width, coverage

FOLDER = 'model_logits/dsr1'
DIMS   = ["consistency", "coherence", "fluency"]

print("=" * 55)
print("  R2CCP | DeepSeek-R1-32B | SummEval | seed=42")
print("=" * 55)
print(f"{'Dimension':<14} {'Width':>8} {'Coverage':>10}")
print("-" * 55)

for dim in DIMS:
    fp = os.path.join(FOLDER, f"Summeval_{dim}_logits.csv")
    df = pd.read_csv(fp)
    X  = df.iloc[:, :-1]
    y  = df.iloc[:, -1]
    print(f"  Running {dim} ...")
    w, cov = run_experiment(X, y, 42, dim)
    print(f"  {dim:<12} {w:>8.4f} {cov:>10.4f}")

print("=" * 55)
