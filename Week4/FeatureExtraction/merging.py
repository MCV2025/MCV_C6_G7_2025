import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

# Specify sequence and cases
sequence = "s03"
cases = ["c010", "c011"]  # Your two cameras

# Camera transitions (simplified for just two cameras)
camera_transitions = {
    "0": ["1"],
    "1": ["0"]
}

def match_across_cameras_knn(features_data, time_constraint=50, similarity_threshold=0.7, k=3):
    print("Starting cross-camera matching using KNN...")
    
    matched_tracks = {}
    
    # Extract camera IDs
    cam_ids = list(features_data.keys())
    if len(cam_ids) < 2:
        print(f"Not enough cameras to match. Found: {cam_ids}")
        return matched_tracks
    
    # Process each camera
    for cam_id1 in cam_ids:
        if cam_id1 not in camera_transitions:
            continue
            
        print(f"Processing camera {cam_id1}...")
        tracks1 = features_data[cam_id1]
        
        # Group features by track ID
        track_features1 = {}
        for track in tracks1:
            track_id = track["track_id"]
            if track_id not in track_features1:
                track_features1[track_id] = []
            track_features1[track_id].append(track["features"])
        
        # Process neighboring cameras
        for cam_id2 in camera_transitions[cam_id1]:
            if cam_id2 not in features_data:
                continue
                
            print(f"  Comparing with camera {cam_id2}...")
            tracks2 = features_data[cam_id2]
            
            track_features2 = {}
            for track in tracks2:
                track_id = track["track_id"]
                if track_id not in track_features2:
                    track_features2[track_id] = []
                track_features2[track_id].append(track["features"])
            
            # Prepare data for KNN
            all_features2 = []
            track_labels2 = []
            
            for track_id2, feature_list in track_features2.items():
                for feature in feature_list:
                    all_features2.append(feature)
                    track_labels2.append(track_id2)  # Assign track ID as label
            
            if len(all_features2) == 0:
                continue
            
            all_features2 = np.array(all_features2)
            
            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn.fit(all_features2, track_labels2)
            
            # Match tracks
            for track_id1, feature_list1 in tqdm(track_features1.items(), desc=f"Matching camera {cam_id1} to {cam_id2}"):
                global_track_id1 = f"{cam_id1}_{track_id1}"
                
                if global_track_id1 in matched_tracks:
                    continue
                
                best_match = None
                best_votes = 0
                
                # Predict using KNN
                for feature in feature_list1:
                    neighbor_ids = knn.predict([feature])  # Get predicted track ID
                    neighbor_id = neighbor_ids[0]
                    
                    # Count votes
                    votes = np.sum(knn.predict_proba([feature]) > similarity_threshold)
                    
                    if votes > best_votes:
                        best_votes = votes
                        best_match = f"{cam_id2}_{neighbor_id}"
                
                # If match found, record it
                if best_match:
                    matched_tracks[global_track_id1] = best_match
            
            print(f"  Found {len(matched_tracks)} matches so far")

    print(f"Cross-camera matching completed. Total matches: {len(matched_tracks)}")
    
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
    # Perform multi-camera tracking with KNN
    matched_tracks = match_across_cameras_knn(all_features, time_constraint=100, similarity_threshold=0.7, k=3)

    # Save the results
    output_path = Path(f"output/{sequence}_multicamera_tracks_knn.json")
    with open(output_path, "w") as f:
        json.dump(matched_tracks, f, indent=4)

    print(f"Multi-camera tracking with KNN completed. Results saved to {output_path}")
