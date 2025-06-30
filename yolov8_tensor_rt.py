import cv2
import numpy as np
import time
from ultralytics import YOLO
from mss import mss
import ctypes
import torch
import os

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

# ===== Configuration =====
TRIGGER_KEY = 0x02  # Right mouse
FOV = 160        # Smaller FOV = faster processing

# ===== GPU Optimization =====
torch.backends.cudnn.benchmark = True  # Auto-optimizes CUDA

# ===== Model Setup =====
# Load original model
if not os.path.exists('yolov8_20250629.engine'):
    print("Couldn't find engine, re-export model.")
    model = YOLO('yolov8_20250629.pt')

    # Fuse Conv+BN layers (permanent in exported model)
    model.fuse()  # Reduces layers by ~15%

    # Export to TensorRT (saves optimized model)
    model.export(
        format='engine',  # TensorRT format
        half=True,        # FP16 quantization (2x speed)
        simplify=True,    # Simplify ONNX first
        workspace=4,      # GPU memory in GB
        device=0          # GPU index
    )  # Saves as 'yolov8n-pose.engine'

model = YOLO('yolov8_20250629.engine')

# ===== Screen Capture =====
sct = mss()
center_x, center_y = 1280, 720  # Adjust to your resolution
region = {
    "top": max(0, center_y - FOV//2),
    "left": max(0, center_x - FOV//2),
    "width": min(FOV, 2560),
    "height": min(FOV, 1440)
}

# Pre-allocate memory
frame = np.zeros((FOV, FOV, 3), dtype=np.uint8)

# ===== Input =====
user32 = ctypes.windll.user32

def move_mouse(dx, dy):
    user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)

# ===== Optimized Detection =====
def get_head_position(results):
    if not results or results[0].keypoints is None:
        return None
    
    # Pure GPU tensor operations
    kpts = results[0].keypoints.xy  # shape=(n,17,2)
    if kpts.shape[0] == 0:
        return None
    
    # Get nose tensor (still on GPU)
    nose_tensor = kpts[0,0]  # First person's nose
    
    # Early exit if invalid
    if not torch.any(nose_tensor > 0):
        return None
    
    # GPU-accelerated prediction
    # predicted_tensor = predictor.update_and_predict(nose_tensor)
    
    # Single GPU->CPU transfer at the end
    return nose_tensor.cpu().numpy()

# ===== Main Loop =====
print("CS2 Aim Assist | Hold Right Mouse | F2 to exit")
try:
    while not ctypes.windll.user32.GetAsyncKeyState(0x71):  # F2 exit
        if ctypes.windll.user32.GetAsyncKeyState(TRIGGER_KEY):
            start_time = time.perf_counter()
            # 1. Fast screen capture
            # frame[:,:,:] = cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)
            
            # Pre-allocate once outside the loop
            sct_img = np.array(sct.grab(region))

            # Use in-place operations where possible
            cv2.cvtColor(sct_img, cv2.COLOR_BGRA2BGR, dst=frame)

            # 2. GPU-accelerated inference
            results = model(frame, verbose=False, half=True, device='0')  # FP16 acceleration
            
            inference_time = time.perf_counter() - start_time
            fps = 1 / inference_time
            print(f"Inference: {inference_time*1000:.1f}ms | FPS: {fps:.1f}")

            # 3. Fast head position extraction
            head_pos = get_head_position(results)
            
            if head_pos is not None:
                # 4. Direct movement calculation
                dx = (region["left"] + head_pos[0] - center_x) * 0.8
                dy = (region["top"] + head_pos[1] - center_y) * 0.8
                
                # 5. Thresholded movement
                if abs(dx) > 2 or abs(dy) > 2:  # 2-pixel deadzone
                    move_mouse(dx, dy)
            
            time.sleep(0.002)  # ~200Hz polling

finally:
    print("Script stopped")