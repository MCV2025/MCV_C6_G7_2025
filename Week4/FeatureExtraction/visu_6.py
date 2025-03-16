import cv2
import numpy as np
import time

# Video details
videos = []
sequence = "s03"
cases = ["c010", "c011", "c012", "c013", "c014", "c015"]  # Cameras

# Custom start delays (in seconds, None for automatic calculation)
custom_delays = {
    "c010": 3.5,  # Delay for video c010
    "c011": 3,  # Delay for video c011
    "c012": 2,
    "c013": 0,
    "c014": 1,  # Automatic delay calculation
    "c015": 0,
}

# Target FPS (standardize frame rate for all videos)
target_fps = 30  # Desired frame rate (e.g., 30 FPS)
frame_interval = 1 / target_fps  # Time between frames in seconds

# Gather video paths
for case in cases:
    video_path = f"aic19-track1-mtmc-train/train/{sequence}/{case}/vdo.avi"
    videos.append(video_path)

# Open video captures
captures = [cv2.VideoCapture(video) for video in videos]

# Retrieve frame counts and frame rates
frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
fps_list = [int(cap.get(cv2.CAP_PROP_FPS)) for cap in captures]

# Calculate automatic delays (to sync all videos to finish at the same time)
max_duration = max(frame_counts[i] / fps_list[i] for i in range(len(captures)))
delays = []
for i in range(len(captures)):
    custom_delay = custom_delays[cases[i]]
    if custom_delay is not None:
        delays.append(custom_delay)
    else:
        video_duration = frame_counts[i] / fps_list[i]
        delays.append(max_duration - video_duration)

# Resize parameters
video_width, video_height = 320, 240

# Initialize frame counters and timers
frame_indices = [0] * len(captures)
last_frame_time = time.time()

# Start time for elapsed time calculation
start_time = time.time()

while True:
    # Limit the loop to maintain the target FPS
    current_time = time.time()
    if current_time - last_frame_time < frame_interval:
        continue
    last_frame_time = current_time

    frames = []
    for i, cap in enumerate(captures):
        # Check if the video is in its delay period
        if frame_indices[i] < delays[i] * target_fps:
            # Add black frame during delay
            frames.append(np.zeros((video_height, video_width, 3), dtype=np.uint8))
        else:
            # Play video normally
            ret, frame = cap.read()
            if not ret:
                # Add black frame if video ends
                frames.append(np.zeros((video_height, video_width, 3), dtype=np.uint8))
            else:
                # Resize frame for consistency
                frame = cv2.resize(frame, (video_width, video_height))
                frames.append(frame)
        # Increment frame counter
        frame_indices[i] += 1

    # Create mosaic (2 rows x 3 columns)
    row1 = np.hstack(frames[:3])  # First row
    row2 = np.hstack(frames[3:])  # Second row
    mosaic = np.vstack([row1, row2])  # Combine rows

    # Calculate elapsed time
    elapsed_time = current_time - start_time
    time_text = f"Time: {elapsed_time:.2f} s"

    # Overlay time on the mosaic
    cv2.putText(mosaic, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display synchronized mosaic
    cv2.imshow("Synchronized Mosaic with Time Overlay", mosaic)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
