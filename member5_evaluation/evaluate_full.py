"""
Full model evaluation on the xBD test set.
Usage: python -m member5_evaluation.evaluate_full \
    --siamese-ckpt checkpoints/siamese_best.pth \
    --split test \
    --output-dir outputs/eval_results
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rich.table import Table
from rich.console import Console

# In testing framework we use absolute imports explicitly 
from member1_preprocessing.dataset import get_dataloader
from member2_siamese.siamese_net import SiameseUNet, SiameseUNetV2, SiameseUNetV3
from member5_evaluation.sliding_window import sliding_window_inference
from member5_evaluation.metrics import (
    compute_iou, compute_f1, compute_precision_recall,
    compute_ap, expected_calibration_error, xbd_localization_score, compute_per_class_iou
)
from member4_visualization.static_plots import plot_pr_curve, plot_roc_curve, plot_disaster_f1
from utils.checkpoint import load_checkpoint
from utils.config import cfg

def evaluate_full(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    console = Console()
    
    console.print(f"[bold blue]Loading model architecture '{args.model_version}'...[/]")
    MODEL_MAP = {"v1": SiameseUNet, "v2": SiameseUNetV2, "v3": SiameseUNetV3}
    ModelClass = MODEL_MAP.get(args.model_version, SiameseUNet)
    
    if args.model_version in ["v2", "v3"]:
        model = ModelClass(in_channels=cfg.siamese.in_channels, pretrained=False, deep_supervision=False).to(device)
    else:
        model = ModelClass(in_channels=cfg.siamese.in_channels, pretrained=False).to(device)

        
    load_checkpoint(args.siamese_ckpt, model, device=device)
    model.eval()

    console.print(f"[bold blue]Loading test dataset...[/]")
    # Using batch size of 1 directly retrieves elements without collation complications for dynamic sizes 
    test_loader = get_dataloader(args.split, augment=False, batch_size=1, add_spectral=cfg.preprocess.add_spectral)
    
    tile_results = []
    disaster_stats = {}
    
    all_probs = []
    all_targets = []
    
    console.print("[bold blue]Running inference...[/]")
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            # Tensors: pre/post usually inside dataset dict
            pre_t = batch['pre'].to(device)
            post_t = batch['post'].to(device)
            target_mask = batch['change_mask'].squeeze().cpu().numpy()

            
            # Use dataloader identifiers explicitly
            disaster = batch.get('disaster_name', ["unknown"])[0]
            stem = batch.get('stem', ["unknown"])[0]
            
            # Since sliding window is designed for numpy H, W, C natively:
            pre_img = pre_t.squeeze(0).permute(1,2,0).cpu().numpy()
            post_img = post_t.squeeze(0).permute(1,2,0).cpu().numpy()
            
            prob_map = sliding_window_inference(
                model=model,
                pre_img=pre_img,
                post_img=post_img,
                patch_size=cfg.preprocess.patch_size,
                stride=cfg.siamese.sliding_window_stride,
                device=device,
                use_tta=cfg.siamese.use_tta,
                n_augments=cfg.siamese.tta_n_augments
            )
            
            pred_mask = (prob_map > cfg.siamese.map_threshold).astype(np.uint8)
            
            iou = compute_iou(pred_mask, target_mask)
            f1 = compute_f1(pred_mask, target_mask)
            precision, recall = compute_precision_recall(pred_mask, target_mask)
            ap = compute_ap(prob_map, target_mask)
            
            tile_dict = {
                "stem": stem,
                "disaster": disaster,
                "iou": float(iou),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "ap": float(ap)
            }
            tile_results.append(tile_dict)
            
            if disaster not in disaster_stats:
                disaster_stats[disaster] = []
            disaster_stats[disaster].append(f1)
            
            # Subsample for global metrics safely keeping VRAM clean
            stride_s = 4 
            all_probs.append(prob_map[::stride_s, ::stride_s].flatten())
            all_targets.append(target_mask[::stride_s, ::stride_s].flatten())
    
    console.print("[bold blue]Computing global metrics...[/]")
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    all_preds = (all_probs > cfg.siamese.map_threshold).astype(np.uint8)
    
    global_iou = float(compute_iou(all_preds, all_targets))
    global_f1 = float(compute_f1(all_preds, all_targets))
    global_p, global_r = compute_precision_recall(all_preds, all_targets)
    global_ap = float(compute_ap(all_probs, all_targets))
    global_ece = float(expected_calibration_error(all_probs, all_targets))
    
    agg_disaster = {k: float(np.mean(v)) for k,v in disaster_stats.items()}
    
    report = {
        "summary": {
            "mean_iou": global_iou,
            "mean_f1": global_f1,
            "mean_precision": float(global_p),
            "mean_recall": float(global_r),
            "ap": global_ap,
            "ece": global_ece
        },
        "per_disaster": agg_disaster,
        "per_tile": tile_results
    }
    
    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    console.print(f"[bold green]Report saved to {report_path}[/]")
    
    # Table Printing
    table = Table(title="Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=20)
    table.add_column("Value")
    
    for metric, val in report["summary"].items():
        table.add_row(metric, f"{val:.4f}")
        
    console.print(table)
    
    # Plotting implementations
    try:
        plot_pr_curve(all_targets, all_probs, out_dir / "pr_curve.png")
        plot_roc_curve(all_targets, all_probs, out_dir / "roc_curve.png")
        plot_disaster_f1(agg_disaster, out_dir / "disaster_f1.png")
        console.print(f"[bold green]Visualizations saved to {out_dir}[/]")
    except Exception as e:
        console.print(f"[bold red]Failed to generate plots: {e}[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--siamese-ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_results", help="Outputs map")
    parser.add_argument("--model-version", type=str, default="v2", help="Model version (v1, v2, v3)")
    args = parser.parse_args()
    
    try:
        import rich
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
        
    evaluate_full(args)
