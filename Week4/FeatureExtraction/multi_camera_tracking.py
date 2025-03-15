import argparse
import numpy as np
import json
from pathlib import Path
from sort import Sort
from ultralytics import YOLO
import cv2
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
from torchreid import models
from torchreid import utils
from torchreid import data

parser = argparse.ArgumentParser(description="Object tracking script")
parser.add_argument("--sequence", required=True, help="Nombre de la secuencia (ej. S01)")
parser.add_argument("--case", required=True, help="Nombre del caso (ej. c002)")
# parser.add_argument("--camera_id", required=True, help="Camera ID")
parser.add_argument("--visualize", action="store_true", help="Visualizar el seguimiento")
parser.add_argument("--parked", type=bool, default=False, help="Si es True, elimina los coches estacionados, si es False, los considera")
args = parser.parse_args()

sequence = args.sequence.lower()
case = args.case.lower()

# Camera ID
camera_id = args.case.lower()[-1]  # Unique camera identifier
# camera_id = args.camera_id  # Unique camera identifier
visualize = args.visualize
parked = args.parked
seq_case_name = f"{sequence}_{case}"

# Camera relationships based on spatial layout
camera_transitions = {
    "0": ["1"],
    "1": ["0", "2"],
    "2": ["1", "3"],
    "3": ["2", "4", "5"],
    "4": ["3"],
    "5": ["3"]
}

# Output directories
output_dir = Path(f"output/{seq_case_name}")
output_dir.mkdir(parents=True, exist_ok=True)

detections_json = output_dir / "detections.json"
features_json = output_dir / "features.json"
crop_dir = output_dir / "crops"
crop_dir.mkdir(parents=True, exist_ok=True)

# Load model and tracker
model = YOLO("yolov8l.pt")
tracker = Sort()

# Load VERI-Wild ReID model
reid_model = models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
utils.load_pretrained_weights(reid_model, 'osnet_x1_0_imagenet.pth')
reid_model.eval()
_, reid_transform = data.transforms.build_transforms(height=256, width=128, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

def extract_features(image_path):
    try:
        # Check if the file exists
        if not Path(image_path).exists():
            print(f"Warning: Image file not found: {image_path}")
            return [0] * 512  # Return a zero vector of appropriate size
            
        image = Image.open(image_path).convert("RGB")
        image = reid_transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = reid_model(image)
        return feature.squeeze().cpu().numpy().tolist()
    except Exception as e:
        print(f"Error extracting features for {image_path}: {e}")
        return [0] * 512  # Return a zero vector of appropriate size

def match_across_cameras(features_data, time_constraint=50):
    matched_tracks = {}
    for cam_id, cam_data in features_data.items():
        for track in cam_data:
            track_id = track["track_id"]
            features = np.array(track["features"])
            
            for neighbor_cam in camera_transitions.get(cam_id, []):
                if neighbor_cam in features_data:
                    for neighbor_track in features_data[neighbor_cam]:
                        neighbor_features = np.array(neighbor_track["features"])
                        similarity = 1 - cosine(features, neighbor_features)
                        
                        if similarity > 0.8 and abs(track["frame"] - neighbor_track["frame"]) <= time_constraint:
                            matched_tracks[track_id] = neighbor_track["track_id"]
    return matched_tracks

# Open video file
video_path = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/vdo.avi")
cap = cv2.VideoCapture(str(video_path))

# Store tracking results
tracking_results = []
feature_results = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx += 1
    results = model(frame)
    detections = []

    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) in [2, 7]:  # Consider only cars & trucks
            detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)
    tracked_objects = tracker.update(detections) if detections.shape[0] > 0 else []

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        width, height = x2 - x1, y2 - y1
        track_data = {
            "frame": frame_idx,
            "track_id": obj_id,
            "bbox": [x1, y1, width, height],
            "camera_id": camera_id
        }
        tracking_results.append(track_data)
        
        # Save cropped image for ReID
        try:
            # Save cropped image for ReID
            crop_img = frame[y1:y2, x1:x2]
            
            # Check if the crop is valid (not empty)
            if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                print(f"Warning: Invalid crop at frame {frame_idx}, track {obj_id}")
                continue
                
            crop_path = crop_dir / f"{frame_idx}_{obj_id}.jpg"
            cv2.imwrite(str(crop_path), crop_img)
            
            # Verify the file was saved
            if not crop_path.exists():
                print(f"Warning: Failed to save crop at {crop_path}")
                continue
                
            # Extract features for ReID
            features = extract_features(crop_path)
            feature_results.append({
                "frame": frame_idx,
                "track_id": obj_id,
                "camera_id": camera_id,
                "features": features
            })
        except Exception as e:
            print(f"Error processing detection at frame {frame_idx}, track {obj_id}: {e}")

    if visualize:
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

# Save tracking results to JSON
with open(detections_json, "w") as f:
    json.dump(tracking_results, f, indent=4)

# Save extracted features to JSON
with open(features_json, "w") as f:
    json.dump(feature_results, f, indent=4)

# Perform cross-camera ReID matching
with open(features_json, "r") as f:
    features_data = json.load(f)
camera_features = {}
for feature in features_data:
    camera_features.setdefault(feature["camera_id"], []).append(feature)

# matched_tracks = match_across_cameras(camera_features)
# print("Cross-camera matching completed.")

print(f"Tracking results saved to {detections_json}")
print(f"Feature extraction results saved to {features_json}")
