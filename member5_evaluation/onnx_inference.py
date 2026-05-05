"""
ONNX Runtime inference session — faster than PyTorch on CPU,
compatible with TensorRT on GPU.
"""

import numpy as np
import onnxruntime as ort

class ONNXSiamese:
    """ONNX Runtime wrapper for the Siamese change detection network."""
    def __init__(self, onnx_path: str, device: str = "cpu"):
        requested_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        
        try:
            self.sess = ort.InferenceSession(onnx_path, providers=requested_providers)
        except Exception:
            # Fallback for environments lacking CUDA acceleration
            self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
        self.pre_name = self.sess.get_inputs()[0].name
        self.post_name = self.sess.get_inputs()[1].name
    
    def predict(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """(H,W,C) float32 images → (H,W) probability map."""
        pre_t = pre.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        post_t = post.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        ort_inputs = {
            self.pre_name: pre_t,
            self.post_name: post_t
        }
        
        logits = self.sess.run(None, ort_inputs)[0]
        
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
            
        probs = 1.0 / (1.0 + np.exp(-logits.squeeze()))
        return probs

class ONNXClassifier:
    """ONNX Runtime wrapper for the EfficientNet damage classifier."""
    def __init__(self, onnx_path: str, device: str = "cpu"):
        requested_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
        
        try:
            self.sess = ort.InferenceSession(onnx_path, providers=requested_providers)
        except Exception:
            self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
        self.input_name = self.sess.get_inputs()[0].name

    def predict_batch(self, crops: np.ndarray) -> list[int]:
        """(B,H,W,C) float32 array -> list of int labels."""
        crops_t = crops.transpose(0, 3, 1, 2).astype(np.float32)
        
        ort_inputs = {
            self.input_name: crops_t
        }
        
        logits = self.sess.run(None, ort_inputs)[0]
        
        if isinstance(logits, (list, tuple)): # Redundancy catch
            logits = logits[0]
            
        labels = np.argmax(logits, axis=1)
        return labels.tolist()
