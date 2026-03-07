"""
utils/checkpoint.py
────────────────────────────────────────────────────────────────────────────
Model checkpoint utilities — save and load model + optimizer state dicts.

Usage
-----
    from utils.checkpoint import save_checkpoint, load_checkpoint

    save_checkpoint(model, optimizer, epoch=5, val_f1=0.82,
                    path="checkpoints/siamese_best.pth")

    epoch, val_f1 = load_checkpoint("checkpoints/siamese_best.pth",
                                    model, optimizer)
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.logger import get_logger

log = get_logger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    path: str | Path,
    extra: Optional[dict] = None,
) -> None:
    """
    Save model and optimizer state to a .pth checkpoint file.

    Parameters
    ----------
    model     : PyTorch model
    optimizer : optimizer whose state will be saved
    epoch     : current epoch number
    val_f1    : validation F1 score at this checkpoint
    path      : file path to write (parent dirs created automatically)
    extra     : optional dict of additional metadata to store
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":     epoch,
        "val_f1":    val_f1,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)

    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}  (epoch={epoch}, val_f1={val_f1:.4f})")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[int, float]:
    """
    Load a checkpoint into model (and optionally optimizer).

    Parameters
    ----------
    path      : path to .pth checkpoint file
    model     : model to load weights into
    optimizer : if provided, optimizer state is also restored
    device    : torch device string

    Returns
    -------
    (epoch, val_f1) tuple from the checkpoint metadata
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    epoch  = state.get("epoch", 0)
    val_f1 = state.get("val_f1", 0.0)
    log.info(f"Checkpoint loaded ← {path}  (epoch={epoch}, val_f1={val_f1:.4f})")
    return epoch, val_f1


def get_best_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Scan a directory for .pth files and return the one with the highest
    val_f1 stored in its metadata.

    Returns None if the directory is empty or contains no valid checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_path, best_f1 = None, -1.0

    for ckpt in checkpoint_dir.glob("*.pth"):
        try:
            state = torch.load(ckpt, map_location="cpu")
            f1 = state.get("val_f1", -1.0)
            if f1 > best_f1:
                best_f1  = f1
                best_path = ckpt
        except Exception as exc:
            log.warning(f"Could not read {ckpt}: {exc}")

    if best_path:
        log.info(f"Best checkpoint: {best_path}  (val_f1={best_f1:.4f})")
    return best_path
