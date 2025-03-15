import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
import time
from tqdm import tqdm  # For progress bars

# Specify your sequence and cases
sequence = "s03"
cases = ["c010", "c011"]  # Your two cameras

# Camera transitions (simplified for just two cameras)
camera_transitions = {
    "0": ["1"],
    "1": ["0"]
}

def match_across_cameras(features_data, time_constraint=50, similarity_threshold=0.8):
    print("Starting cross-camera matching...")
    start_time = time.time()
    matched_tracks = {}
    
    # Extract camera IDs
    cam_ids = list(features_data.keys())
    if len(cam_ids) < 2:
        print(f"Not enough cameras to match. Found: {cam_ids}")
        return matched_tracks
    
    # For each pair of cameras
    for i, cam_id1 in enumerate(cam_ids):
        # Only process if this camera has transitions
        if cam_id1 not in camera_transitions:
            continue
            
        print(f"Processing camera {cam_id1}...")
        
        # Get all tracks for this camera
        tracks1 = features_data[cam_id1]
        
        # Group tracks by ID to avoid duplicate comparisons
        tracks1_by_id = {}
        for track in tracks1:
            track_id = track["track_id"]
            if track_id not in tracks1_by_id:
                tracks1_by_id[track_id] = []
            tracks1_by_id[track_id].append(track)
        
        # For each neighboring camera
        for cam_id2 in camera_transitions[cam_id1]:
            if cam_id2 not in features_data:
                continue
                
            print(f"  Comparing with camera {cam_id2}...")
            
            # Get all tracks for neighboring camera
            tracks2 = features_data[cam_id2]
            
            # Group tracks by ID
            tracks2_by_id = {}
            for track in tracks2:
                track_id = track["track_id"]
                if track_id not in tracks2_by_id:
                    tracks2_by_id[track_id] = []
                tracks2_by_id[track_id].append(track)
            
            # For each track ID in first camera
            for track_id1, tracks1_same_id in tqdm(tracks1_by_id.items(), desc=f"Camera {cam_id1} tracks"):
                global_track_id1 = f"{cam_id1}_{track_id1}"
                
                # Skip if already matched
                if global_track_id1 in matched_tracks:
                    continue
                
                best_match = None
                best_similarity = similarity_threshold
                
                # For each track ID in second camera
                for track_id2, tracks2_same_id in tracks2_by_id.items():
                    global_track_id2 = f"{cam_id2}_{track_id2}"
                    
                    # Skip if already matched
                    if global_track_id2 in matched_tracks.values():
                        continue
                    
                    # Compare a representative feature from each track
                    # Using the middle frame's features for better representation
                    track1 = tracks1_same_id[len(tracks1_same_id)//2]
                    track2 = tracks2_same_id[len(tracks2_same_id)//2]
                    
                    features1 = np.array(track1["features"])
                    features2 = np.array(track2["features"])
                    
                    # Skip if arrays have different shapes
                    if features1.shape != features2.shape:
                        continue
                    
                    # Check time constraint
                    if abs(track1["frame"] - track2["frame"]) > time_constraint:
                        continue
                        
                    try:
                        similarity = 1 - cosine(features1, features2)
                        
                        # Update best match if better similarity found
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = global_track_id2
                    except Exception as e:
                        print(f"Error calculating similarity: {e}")
                        continue
                
                # If a good match was found, record it
                if best_match:
                    matched_tracks[global_track_id1] = best_match
            
            print(f"  Found {len(matched_tracks)} matches so far")
    
    elapsed_time = time.time() - start_time
    print(f"Cross-camera matching completed in {elapsed_time:.2f} seconds")
    print(f"Total matches found: {len(matched_tracks)}")
    
    return matched_tracks

# Load feature data
all_features = {}
for case in cases:
    camera_id = case[-1]
    seq_case_name = f"{sequence}_{case}"
    feature_path = Path(f"output/{seq_case_name}/features.json")
    
    if feature_path.exists():
        print(f"Loading features from {feature_path}...")
        with open(feature_path, "r") as f:
            features = json.load(f)
            print(f"Loaded {len(features)} feature entries for camera {camera_id}")
            all_features[camera_id] = features
    else:
        print(f"Warning: Feature file not found: {feature_path}")

if len(all_features) < 2:
    print("Not enough feature data loaded to perform matching")
else:
    # Perform cross-camera matching with more efficient algorithm
    matched_tracks = match_across_cameras(all_features, time_constraint=100, similarity_threshold=0.7)

    # Save the results
    output_path = Path(f"output/{sequence}_multicamera_tracks.json")
    with open(output_path, "w") as f:
        json.dump(matched_tracks, f, indent=4)

    print(f"Multi-camera tracking completed. Results saved to {output_path}")