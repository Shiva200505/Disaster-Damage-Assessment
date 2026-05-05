"""
api/startup.py
──────────────────────────────────────────────────────────────────────────────
Model singleton registry implementation.
"""

import threading
from typing import Optional
import torch

from member2_siamese.siamese_net import SiameseUNet, SiameseUNetV2, SiameseUNetV3
from utils.checkpoint import load_checkpoint
from utils.config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


class ModelRegistry:
    """Thread-safe model registry loaded at startup."""
    _siamese: Optional[torch.nn.Module] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_siamese(cls) -> torch.nn.Module:
        with cls._lock:
            if cls._siamese is None:
                raise RuntimeError("Siamese model not preloaded.")
            return cls._siamese
    
    @classmethod
    def preload(cls, siamese_ckpt: str, device: str):
        with cls._lock:
            if cls._siamese is not None:
                return
                
            log.info(f"Preloading Siamese model {cfg.siamese.model_version} on {device}...")
            
            MODEL_MAP = {
                "v1": SiameseUNet, 
                "v2": SiameseUNetV2, 
                "v3": SiameseUNetV3
            }
            ModelClass = MODEL_MAP.get(cfg.siamese.model_version, SiameseUNet)
            
            in_ch = cfg.siamese.in_channels
            
            if cfg.siamese.model_version in ["v2", "v3"]:
                model = ModelClass(in_channels=in_ch, pretrained=False, deep_supervision=False).to(device)
            else:
                model = ModelClass(in_channels=in_ch, pretrained=False).to(device)
                
            load_checkpoint(siamese_ckpt, model, device=device)
            model.eval()
            cls._siamese = model
            log.info("Siamese model preloaded successfully.")
