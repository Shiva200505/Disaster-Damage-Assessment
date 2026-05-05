"""
Export trained PyTorch models to ONNX format.
Usage: python scripts/export_onnx.py --model siamese --ckpt checkpoints/siamese_best.pth
       python scripts/export_onnx.py --model classifier --ckpt checkpoints/classifier_best.pth
"""

import argparse
import json
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from datetime import datetime
import timm

from utils.config import cfg
from member2_siamese.siamese_net import SiameseUNet, SiameseUNetV2, SiameseUNetV3
from utils.checkpoint import load_checkpoint


def export_model(args):
    out_dir = Path("outputs/onnx")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cpu"
    
    if args.model == "siamese":
        # Check cfg for version
        MODEL_MAP = {"v1": SiameseUNet, "v2": SiameseUNetV2, "v3": SiameseUNetV3}
        ModelClass = MODEL_MAP.get(cfg.siamese.model_version, SiameseUNet)
        
        in_ch = cfg.siamese.in_channels
        if cfg.siamese.model_version in ["v2", "v3"]:
            model = ModelClass(in_channels=in_ch, pretrained=False, deep_supervision=False)
        else:
            model = ModelClass(in_channels=in_ch, pretrained=False)
            
        load_checkpoint(args.ckpt, model, device=device)
        model.to(device)
        model.eval()

        dummy_input = (
            torch.randn(1, in_ch, 256, 256), 
            torch.randn(1, in_ch, 256, 256)
        )
        
        dynamic_axes = {
            'pre':  {0: 'batch', 2: 'height', 3: 'width'},
            'post': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        }
        input_names = ['pre', 'post']
        output_names = ['output']
        onnx_path = out_dir / f"siamese_{cfg.siamese.model_version}.onnx"
        
        metadata = {
            "model": "SiameseUNet",
            "version": cfg.siamese.model_version,
            "in_channels": in_ch,
            "patch_size": 256,
            "export_date": datetime.utcnow().isoformat(),
            "onnx_opset": 16
        }
        
    elif args.model == "classifier":
        model = timm.create_model(cfg.classifier.model_name, pretrained=False, num_classes=cfg.classifier.num_classes)
        # Attempt to load if exists
        if Path(args.ckpt).exists():
            checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint.get("model", checkpoint))
            
        model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, cfg.classifier.in_channels, cfg.classifier.crop_size, cfg.classifier.crop_size)
        dynamic_axes = {'input': {0: 'batch'}, 'output': {0: 'batch'}}
        input_names = ['input']
        output_names = ['output']
        onnx_path = out_dir / f"classifier_{cfg.classifier.model_name}.onnx"
        
        metadata = {
            "model": "DamageClassifier",
            "version": cfg.classifier.model_name,
            "in_channels": cfg.classifier.in_channels,
            "crop_size": cfg.classifier.crop_size,
            "export_date": datetime.utcnow().isoformat(),
            "onnx_opset": 16
        }
    else:
        raise ValueError("Invalid model type specified.")

    print(f"Exporting {args.model} to {onnx_path}...")

    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        str(onnx_path), 
        export_params=True,
        opset_version=16,          
        do_constant_folding=True,  
        input_names=input_names,   
        output_names=output_names,  
        dynamic_axes=dynamic_axes
    )

    print("Checking ONNX model integrity...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model passed internal checks.")

    # Numerical Comparison
    print("Validating inference output consistency...")
    with torch.no_grad():
        if args.model == "siamese":
            pt_out = model(*dummy_input)
        else:
            pt_out = model(dummy_input)
            
    if isinstance(pt_out, (list, tuple)):
        pt_out = pt_out[0]
        
    try:
        ort_sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        
        if args.model == "siamese":
            ort_inputs = {
                'pre': dummy_input[0].numpy(), 
                'post': dummy_input[1].numpy()
            }
        else:
            ort_inputs = {'input': dummy_input.numpy()}
            
        ort_out = ort_sess.run(None, ort_inputs)[0]
        
        max_diff = np.max(np.abs(pt_out.numpy() - ort_out))
        print(f"Max absolute difference: {max_diff:.8f}")
        
        if max_diff < 1e-4:
            print("ONNX model numerical validation passed.")
        else:
            print("WARNING: Difference exceeds 1e-4 threshold!")
    except Exception as e:
        print(f"Failed ONNX numerical comparison checks due to missing modules or execution errors: {e}")

    # Save Metadata JSON
    json_path = onnx_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model and metadata successfully exported to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["siamese", "classifier"], help="Model to export")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best PyTorch checkpoint")
    args = parser.parse_args()
    
    export_model(args)
