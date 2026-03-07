"""
utils/logger.py
────────────────────────────────────────────────────────────────────────────
Centralised logging + optional Weights & Biases experiment tracking.

Usage
-----
    from utils.logger import get_logger, init_wandb, log_metrics

    log = get_logger(__name__)
    log.info("Training started")
    log_metrics({"loss": 0.25, "iou": 0.78}, step=10)
"""

import logging
import sys
from typing import Any, Dict, Optional

_wandb_run = None   # module-level W&B run handle


# ── Standard Logger ───────────────────────────────────────────────────────────

def get_logger(name: str = "disaster_da", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ── W&B Integration ───────────────────────────────────────────────────────────

def init_wandb(project: str, entity: str, config: Optional[Dict[str, Any]] = None,
               run_name: Optional[str] = None) -> None:
    """
    Initialise a Weights & Biases run. Call once at the start of training.

    Parameters
    ----------
    project  : W&B project name
    entity   : W&B username or team
    config   : dict of hyperparameters to log
    run_name : optional human-readable run name
    """
    global _wandb_run
    try:
        import wandb  # type: ignore
        _wandb_run = wandb.init(
            project=project,
            entity=entity if entity else None,
            config=config,
            name=run_name,
            reinit=True,
        )
        get_logger().info(f"W&B run initialised: {_wandb_run.url}")
    except ImportError:
        get_logger().warning("wandb not installed – skipping W&B logging.")
    except Exception as exc:
        get_logger().warning(f"W&B init failed: {exc} – continuing without W&B.")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log a dict of scalar metrics.
    Writes to W&B if a run is active; otherwise, logs to stdout.
    """
    if _wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.log(metrics, step=step)
            return
        except Exception:
            pass
    # Fallback: print to logger
    msg = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    get_logger().info(f"[step={step}] {msg}")


def finish_wandb() -> None:
    """Close the W&B run if active."""
    global _wandb_run
    if _wandb_run is not None:
        try:
            _wandb_run.finish()
        except Exception:
            pass
        _wandb_run = None
