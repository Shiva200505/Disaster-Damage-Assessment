"""
member5_evaluation/sliding_window.py
────────────────────────────────────────────────────────────────────────────
Sliding window inference for large satellite images.
xBD images are 1024×1024 — our model expects 256×256 patches.
This module handles tiling, inference, and seamless stitching.
"""

import numpy as np
import torch
import torch.nn as nn
from member5_evaluation.tta import TTAWrapper


def gaussian_kernel_2d(kernel_size: int, sigma: float) -> np.ndarray:
    """Generate a 2D Gaussian kernel."""
    x = np.arange(kernel_size) - kernel_size // 2
    x_grid, y_grid = np.meshgrid(x, x)
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return kernel.astype(np.float32)


def sliding_window_inference(
    model: nn.Module,
    pre_img: np.ndarray,    # H×W×C
    post_img: np.ndarray,   # H×W×C
    patch_size: int = 256,
    stride: int = 128,      # 50% overlap for smoother blending
    device: str = "cuda",
    use_tta: bool = True,
    n_augments: int = 4,
) -> np.ndarray:
    """
    Tile the image into overlapping patches, run inference on each,
    and stitch results using Gaussian blending weights (center pixels
    weighted more than edges, reducing stitching artifacts).
    
    Returns: (H, W) float32 probability map in [0, 1]
    """
    H, W, C = pre_img.shape
    
    output_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)
    
    # 2D Gaussian mask (sigma = patch_size/4) for smoother blending
    gaussian_kernel = gaussian_kernel_2d(patch_size, patch_size / 4.0)
    
    if use_tta:
        infer_model = TTAWrapper(model, n_augments=n_augments)
    else:
        infer_model = TTAWrapper(model, n_augments=1) # 1 = original only
        
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Calculate coordinates ensuring we stay within image bounds
            y1 = min(H, y + patch_size)
            x1 = min(W, x + patch_size)
            y0 = max(0, y1 - patch_size)
            x0 = max(0, x1 - patch_size)
            
            pre_patch = pre_img[y0:y1, x0:x1, :]
            post_patch = post_img[y0:y1, x0:x1, :]
            
            # Ensure shape is precisely (patch_size, patch_size, C)
            if pre_patch.shape[:2] != (patch_size, patch_size):
                continue
                
            # Convert to Tensor (C, H, W) -> (1, C, H, W)
            pre_t = torch.from_numpy(pre_patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
            post_t = torch.from_numpy(post_patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # Run inference (returns averaged sigmoid from TTA logic)
            pred_t = infer_model.predict(pre_t, post_t) # (1, 1, H, W)
            pred = pred_t.squeeze().cpu().numpy()
            
            # Accumulate predictions
            output_sum[y0:y1, x0:x1] += pred * gaussian_kernel
            weight_sum[y0:y1, x0:x1] += gaussian_kernel
            
    final_output = output_sum / (weight_sum + 1e-8)
    return np.clip(final_output, 0.0, 1.0)
