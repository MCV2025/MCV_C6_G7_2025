import cv2
from ultralytics import YOLO
from yolo_interactions import detect_with_yolo
from utils import extract_features, track_features_dense
import numpy as np
"""
Task 1.2 Pipeline
"""
# Initialize video capture, YOLO model, and optical flow parameters
cap = cv2.VideoCapture(str(video_path))
model = YOLO(weights_path)
prev_frame = None
predicions = {}  

while cap.has_frames():
    frame = cap.read_frame()
    
    detections, _ = detect_with_yolo(model, frame)  # conf_thresh = 0.7 default
    
    # Step 2: If previous frame exists, compute optical flow
    if prev_frame is not None:
        # Compute optical flow between prev_frame and current frame
        flow_vectors = compute_optical_flow(prev_frame, frame)
        
        # Step 3: For each detection, refine tracking using optical flow
        for detection in detections:
            bbox, class_id, confidence = detection

            feature_points = extract_features(frame, region=bbox)
            if feature_points is None or len(feature_points) == 0:
                
            
            # Track these points using the computed optical flow
            tracked_points = track_features_dense(feature_points, flow_vectors)
            
            
            # Aggregate tracked points to compute a measurement (e.g., centroid)
            if tracked_points is not None and len(tracked_points) > 0:
                centroid = np.mean(tracked_points, axis=0)  # (x, y) average

                # Kalman filter update with the aggregated measurement
                kalman_state = kalman_filter.predict()         # Predict next state
                kalman_state = kalman_filter.update(centroid)    # Update with new measurement

    # Use the updated state for further tracking or visualization
    updated_position = kalman_state[:2]
            # Optionally, refine bbox based on the movement of the feature points
            detection = refine_bbox(detection, tracked_points)
    
    # Step 4: Visualization and logging
    for detection in detections:
        draw_bbox(frame, detection)  # Draw YOLO (or refined) bounding box on frame
    for car_id, track in tracking_data.items():
        draw_tracking_line(frame, track)  # Visualize car trajectory
    
    log_results(detections, tracking_data)
    
    # Prepare for next iteration
    prev_frame = frame

# After processing the video, compute evaluation metrics (accuracy, precision, recall, etc.)
results = compute_evaluation_metrics(tracking_data)
save_results(results)
