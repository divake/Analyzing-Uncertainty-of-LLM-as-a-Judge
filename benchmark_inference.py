"""
Inference speed benchmark for LLM judges.
Tests both DeepSeek-R1-Distill-Qwen-32B and Qwen2.5-72B-Instruct
with both GPUs using device_map="auto".

Measures:
  - Model load time
  - VRAM usage per GPU
  - Tokens/second (prefill + generation)
  - Time per sample (relevant for evaluation scale)
"""
import torch
import time
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Sample prompt mimicking the G-Eval summarization judge ──────────────────
SAMPLE_PROMPT = """You'll be provided with a task to evaluate. These are the introduction, criteria and evaluation steps: [...]
The task for you to evaluate is as follows:
Source: The study found that coffee consumption is associated with reduced risk of type 2 diabetes. Researchers analyzed data from over 1 million participants across 30 countries.
Summary: Coffee reduces diabetes risk according to researchers.
Consistency (1-5): Please rate the consistency of the summary with the source on a scale of 1 to 5, where 1 is the lowest and 5 is the highest.
Score Sheet (Only Score):
Consistency 1-5:"""

def benchmark_model(model_path: str, model_name: str, n_warmup: int = 2, n_runs: int = 10):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    # ── GPU state before load ────────────────────────────────────────────────
    print("\n[GPU] Before loading model:")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {used:.1f}GB used / {total:.1f}GB total")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── Load model across both GPUs ──────────────────────────────────────────
    print("[2/3] Loading model with device_map='auto' (both GPUs)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",          # spreads across all available GPUs
    )
    model.eval()
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.1f}s")

    # ── GPU state after load ─────────────────────────────────────────────────
    print("\n[GPU] After loading model:")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - used
        print(f"  GPU {i}: {used:.1f}GB used / {total:.1f}GB total ({free:.1f}GB free)")

    # ── Show device map summary ──────────────────────────────────────────────
    if hasattr(model, 'hf_device_map'):
        devices = set(str(v) for v in model.hf_device_map.values())
        print(f"\n  Model layers spread across devices: {sorted(devices)}")

    # ── Tokenize prompt ──────────────────────────────────────────────────────
    inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]
    n_input_tokens = input_ids.shape[1]
    # Move to first GPU
    input_ids = input_ids.to("cuda:0")
    print(f"\n[3/3] Running inference benchmark ({n_input_tokens} input tokens)")
    print(f"  Warmup runs: {n_warmup}  |  Timed runs: {n_runs}")

    # ── Warmup ───────────────────────────────────────────────────────────────
    print("  Warming up...", end="", flush=True)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
    torch.cuda.synchronize()
    print(" done")

    # ── Timed runs ───────────────────────────────────────────────────────────
    times = []
    output_tokens_list = []
    print("  Running timed inference...", flush=True)
    with torch.no_grad():
        for i in range(n_runs):
            torch.cuda.synchronize()
            t_start = time.time()
            out = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            torch.cuda.synchronize()
            elapsed = time.time() - t_start
            n_out = out.shape[1] - n_input_tokens
            times.append(elapsed)
            output_tokens_list.append(n_out)
            print(f"    Run {i+1:2d}/{n_runs}: {elapsed:.2f}s  |  {n_out} new tokens  |  {n_out/elapsed:.1f} tok/s")

    # ── Also extract logits for rating tokens (paper's actual use-case) ──────
    print("\n  Testing logits extraction (actual paper use-case)...")
    rating_tokens = {}
    for r in ["1", "2", "3", "4", "5"]:
        toks = tokenizer.encode(r, add_special_tokens=False)
        if toks:
            rating_tokens[r] = toks[0]
    print(f"  Rating token IDs: {rating_tokens}")

    t_logit = time.time()
    with torch.no_grad():
        out_logits = model(input_ids)
    torch.cuda.synchronize()
    logit_time = time.time() - t_logit
    last_logits = out_logits.logits[0, -1, :]
    extracted = {r: last_logits[tid].item() for r, tid in rating_tokens.items()}
    print(f"  Logit extraction time: {logit_time:.3f}s")
    print(f"  Sample logits for ratings 1-5: {extracted}")

    # ── Results ──────────────────────────────────────────────────────────────
    import statistics
    mean_time    = statistics.mean(times)
    median_time  = statistics.median(times)
    mean_toksec  = statistics.mean(o/t for o, t in zip(output_tokens_list, times))

    results = {
        "model": model_name,
        "model_path": model_path,
        "load_time_s": round(load_time, 1),
        "input_tokens": n_input_tokens,
        "mean_inference_time_s": round(mean_time, 3),
        "median_inference_time_s": round(median_time, 3),
        "mean_tokens_per_sec": round(mean_toksec, 1),
        "logit_extraction_time_s": round(logit_time, 3),
        "gpu_vram_after_load": {
            f"GPU_{i}": {
                "used_GB": round(torch.cuda.memory_reserved(i) / 1024**3, 1),
                "total_GB": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1),
            }
            for i in range(torch.cuda.device_count())
        },
        "all_times_s": [round(t, 3) for t in times],
    }

    print(f"\n{'─'*40}")
    print(f"  RESULTS SUMMARY: {model_name}")
    print(f"{'─'*40}")
    print(f"  Model load time:       {load_time:.1f}s")
    print(f"  Mean inference time:   {mean_time:.3f}s")
    print(f"  Median inference time: {median_time:.3f}s")
    print(f"  Mean throughput:       {mean_toksec:.1f} tokens/sec")
    print(f"  Logit extraction:      {logit_time:.3f}s (per sample, paper use-case)")

    # Estimate total eval time for paper's dataset
    n_summeval = 1600
    print(f"\n  Estimated time for SummEval ({n_summeval} samples):")
    print(f"    @ {mean_time:.1f}s/sample = {n_summeval * mean_time / 3600:.1f} hours per dimension")
    print(f"    × 4 dimensions           = {4 * n_summeval * mean_time / 3600:.1f} hours total")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["deepseek", "qwen", "both"], default="both")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_warmup", type=int, default=2)
    args = parser.parse_args()

    print(f"\nSystem: {torch.cuda.device_count()} GPU(s) available")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({total:.1f}GB VRAM)")

    all_results = []

    models_to_test = {
        "deepseek": ("models/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek-R1-Distill-Qwen-32B"),
        "qwen":     ("models/Qwen2.5-72B-Instruct",          "Qwen2.5-72B-Instruct"),
    }

    if args.model == "both":
        to_run = list(models_to_test.items())
    else:
        to_run = [(args.model, models_to_test[args.model])]

    for key, (path, name) in to_run:
        import os
        if not os.path.exists(path):
            print(f"\n⚠ Skipping {name} — model not found at {path}")
            continue
        r = benchmark_model(path, name, n_warmup=args.n_warmup, n_runs=args.n_runs)
        all_results.append(r)
        # Save after each model in case second crashes
        with open("benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to benchmark_results.json")

    print(f"\n{'='*60}")
    print("ALL BENCHMARKS COMPLETE")
    print(f"{'='*60}")
    if len(all_results) == 2:
        print("\nComparison:")
        for r in all_results:
            print(f"  {r['model']:40s}  {r['mean_tokens_per_sec']:6.1f} tok/s  "
                  f"load={r['load_time_s']}s  logit={r['logit_extraction_time_s']}s/sample")


if __name__ == "__main__":
    main()
