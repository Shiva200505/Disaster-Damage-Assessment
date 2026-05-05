"""
Benchmark PyTorch vs ONNX inference speed.
"""

import time
import torch
import numpy as np
import argparse
from pathlib import Path
from tabulate import tabulate

from utils.config import cfg
from member2_siamese.siamese_net import SiameseUNet, SiameseUNetV2, SiameseUNetV3
from member5_evaluation.onnx_inference import ONNXSiamese


def benchmark_pytorch(model, pre, post, device, n=100):
    model.to(device)
    model.eval()
    
    pre_t = torch.from_numpy(pre).unsqueeze(0).float().to(device)
    post_t = torch.from_numpy(post).unsqueeze(0).float().to(device)
    
    print("Warming up PyTorch model...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(pre_t, post_t)
            
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    print(f"Running {n} PyTorch inferences...")
    latencies = []
    for _ in range(n):
        t0 = time.time()
        with torch.no_grad():
            _ = model(pre_t, post_t)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
    return latencies, peak_mem


def benchmark_onnx(onnx_model, pre, post, n=100):
    print("Warming up ONNX model...")
    # ONNX inputs expect (H, W, C) array natively as pre-wrapper wrapper implements
    pre_arr = pre.transpose(1, 2, 0)
    post_arr = post.transpose(1, 2, 0)
    
    for _ in range(10):
        _ = onnx_model.predict(pre_arr, post_arr)
        
    print(f"Running {n} ONNX inferences...")
    latencies = []
    for _ in range(n):
        t0 = time.time()
        _ = onnx_model.predict(pre_arr, post_arr)
        latencies.append((time.time() - t0) * 1000)
        
    return latencies, 0


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on device: {device}")
    
    # ── PyTorch Initialization ─────────────────────────────────────────────
    MODEL_MAP = {"v1": SiameseUNet, "v2": SiameseUNetV2, "v3": SiameseUNetV3}
    ModelClass = MODEL_MAP.get(cfg.siamese.model_version, SiameseUNet)
    
    if cfg.siamese.model_version in ["v2", "v3"]:
        pt_model = ModelClass(in_channels=cfg.siamese.in_channels, pretrained=False, deep_supervision=False)
    else:
        pt_model = ModelClass(in_channels=cfg.siamese.in_channels, pretrained=False)
        
    # Dummy tensors
    H, W = 512, 512
    C = cfg.siamese.in_channels
    pre_data = np.random.randn(C, H, W).astype(np.float32)
    post_data = np.random.randn(C, H, W).astype(np.float32)

    # PyTorch Evaluation
    pt_lats, pt_mem = benchmark_pytorch(pt_model, pre_data, post_data, device, n=args.n)
    
    # ── ONNX Initialization ────────────────────────────────────────────────
    try:
        onnx_model = ONNXSiamese(args.onnx_path, device=device)
        onnx_lats, _ = benchmark_onnx(onnx_model, pre_data, post_data, n=args.n)
    except Exception as e:
        print(f"Failed to load/run ONNX model: {e}")
        return

    # ── Report Statistics ──────────────────────────────────────────────────
    def calc_stats(latencies, mem):
        mean_v = np.mean(latencies)
        p95_v = np.percentile(latencies, 95)
        through = 1000.0 / mean_v 
        return [f"{mean_v:.2f} ms", f"{p95_v:.2f} ms", f"{through:.2f} fps", f"{mem:.1f} MB"]

    pt_stats = calc_stats(pt_lats, pt_mem)
    onnx_stats = calc_stats(onnx_lats, 0) # ONNX handles memory outside torch tracker
    
    speedup = np.mean(pt_lats) / np.mean(onnx_lats)

    table = [
        ["PyTorch", *pt_stats],
        ["ONNX Runtime", *onnx_stats]
    ]
    
    print("\n" + "="*60)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*60)
    print(tabulate(table, headers=["Framework", "Mean Latency", "P95 Latency", "Throughput", "Peak VRAM (GPU)"], tablefmt="heavy_outline"))
    print("="*60)
    print(f"Overall Speedup: {speedup:.2f}x faster using ONNX Runtime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", type=str, required=True, help="Path to exported ONNX model")
    parser.add_argument("--n", type=int, default=100, help="Number of inference iterations")
    args = parser.parse_args()
    
    # Requirement modules loaded
    try:
        import tabulate
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        from tabulate import tabulate
    
    main(args)
