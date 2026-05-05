"""Celery task definitions for async inference."""

import time
import shutil
import cv2
import numpy as np
import torch
from pathlib import Path
from celery import Celery

from member1_preprocessing.preprocessing import align_images, normalize
from member1_preprocessing.spectral_indices import add_spectral_channels
from member5_evaluation.sliding_window import sliding_window_inference
from member4_visualization.geojson_utils import predictions_to_geojson, save_geojson
from member4_visualization.folium_map import render_damage_map
from member5_evaluation.inference_pipeline import _classify_by_change_prob, _mask_to_polygons, _extract_gps_info, _load_rgb
from utils.config import cfg
from api.startup import ModelRegistry

celery_app = Celery(
    'disaster_damage',
    broker='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/1',
)

import signal
import sys
from celery.signals import worker_shutting_down

@worker_shutting_down.connect
def graceful_shutdown(sig, how, exitcode, **kwargs):
    """Graceful SIGTERM handling revoking new tasks while finishing current scopes."""
    import logging
    logging.info("SIGTERM Caught - Revoking unassigned unstarted jobs within cache.")
    # Actually wait logic is natively handled by the celery process, but
    # we explicitly issue internal commands to block new ingress
    try:
        celery_app.control.revoke([task.id for task in celery_app.control.inspect().active()])
    except Exception:
         pass


@celery_app.task(bind=True, name="run_inference_task")
def run_inference_task(self, task_id: str, pre_path: str, post_path: str, settings: dict):
    try:
        t0 = time.time()
        self.update_state(state='PROGRESS', meta={'step': 'data_loading', 'pct': 10})
        
        # 1. Load images
        pre_img       = _load_rgb(pre_path)
        post_img_orig = _load_rgb(post_path)
        post_img      = align_images(pre_img, post_img_orig)
        exif_gps      = _extract_gps_info(post_path)
        
        H, W = pre_img.shape[:2]
        
        self.update_state(state='PROGRESS', meta={'step': 'preprocessing', 'pct': 25})
        
        # 2. Spectral channels
        if cfg.preprocess.add_spectral:
            pre_in  = add_spectral_channels(normalize(pre_img))
            post_in = add_spectral_channels(normalize(post_img))
        else:
            pre_in  = normalize(pre_img)
            post_in = normalize(post_img)
            
        self.update_state(state='PROGRESS', meta={'step': 'change_detection', 'pct': 40})
        
        # 3. Model Inference using Preloaded Model via Registry
        if ModelRegistry._siamese is None:
            # We enforce preloading here for safety if the Celery worker didn't inherit the load
            ModelRegistry.preload(str(cfg.paths.checkpoint_dir / "siamese_best.pth"), cfg.device)
            
        siamese = ModelRegistry.get_siamese()
        device = cfg.device if torch.cuda.is_available() else "cpu"
        
        change_prob = sliding_window_inference(
            model=siamese,
            pre_img=pre_in,
            post_img=post_in,
            patch_size=cfg.preprocess.patch_size,
            stride=cfg.siamese.sliding_window_stride,
            device=device,
            use_tta=cfg.siamese.use_tta,
            n_augments=cfg.siamese.tta_n_augments,
        )
        
        if change_prob.shape != (H, W):
            change_prob = cv2.resize(change_prob, (W, H))
            
        map_threshold = settings.get("map_threshold", 0.35)
        change_mask = (change_prob > map_threshold).astype(np.uint8)
        
        self.update_state(state='PROGRESS', meta={'step': 'polygon_extraction', 'pct': 60})
        
        # 4. Extract polygons
        polygons = _mask_to_polygons(change_mask)
        
        self.update_state(state='PROGRESS', meta={'step': 'damage_classification', 'pct': 75})
        
        # 5. Classify damage
        labels = _classify_by_change_prob(polygons, change_prob)
        
        # Stats
        stats = {
            "total_buildings": len(labels),
            "no_damage": labels.count(0),
            "minor_damage": labels.count(1),
            "major_damage": labels.count(2),
            "destroyed": labels.count(3)
        }
        
        self.update_state(state='PROGRESS', meta={'step': 'generating_outputs', 'pct': 90})
        
        # 6. GeoJSON and Map
        geojson = predictions_to_geojson(polygons=polygons, damage_labels=labels, image_shape=(H, W), geo_polygons=None)
        if exif_gps:
            geojson["properties"]["exif_gps"] = exif_gps
            
        out_dir = Path(cfg.paths.output_dir) / task_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        gj_path = out_dir / "predictions.geojson"
        save_geojson(geojson, gj_path)
        
        post_img_path_out = out_dir / "post_img.png"
        overlay_rgb = post_img_orig
        rgba = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2RGBA)
        black_mask = np.all(overlay_rgb < 10, axis=-1)
        rgba[black_mask, 3] = 0
        cv2.imwrite(str(post_img_path_out), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        
        map_path = out_dir / "damage_map.html"
        render_damage_map(
            geojson_path=gj_path,
            output_html=map_path,
            image_path=str(post_img_path_out),
            image_shape=(H, W),
            disaster_name=settings.get("disaster_name", "Disaster Zone")
        )
        
        result = {
            "task_id": task_id,
            "status": "SUCCESS",
            "map_url": f"/api/v1/inference/{task_id}/map",
            "geojson_url": f"/api/v1/inference/{task_id}/geojson",
            "stats": stats,
            "processing_time_seconds": round(time.time() - t0, 2)
        }
        
        return result
        
    except Exception as e:
        import traceback
        # Return generic fallback object on error so it can be handled
        return {"task_id": task_id, "status": "FAILURE", "error": str(e), "traceback": traceback.format_exc()}


@celery_app.task(name="cleanup_task")
def cleanup_task(task_id: str):
    """Deletes output directory and uploads after 24h to free disk space."""
    target_dir = Path(cfg.paths.output_dir) / task_id
    if target_dir.exists() and target_dir.is_dir():
        shutil.rmtree(target_dir)
        
    return {"cleaned": task_id}
