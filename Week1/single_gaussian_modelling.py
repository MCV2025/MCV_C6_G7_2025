import cv2
import numpy as np
from pathlib import Path

# Function to get total number of frames in the video
def get_total_frames(video_path):
    """
    Returns the total number of frames in the video file.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# Function to read video frames into a numpy array
def get_frames(video_path, start=0, end=None):
    """
    Reads frames from the video from `start` to `end`.
    Returns the frames as a numpy array of grayscale images.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if end is None:
        end = get_total_frames(video_path)  # Get total frames if not specified
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    # Read the frames between start and end frame numbers
    for _ in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    return np.array(frames, dtype=np.float32)

# Function to estimate the background of the video
def estimate_background(video_path, percent=0.25, use_median=False):
    """
    Estimates the background of the video by using the first `percent` of frames.
    The median or mean can be used for background estimation.
    """
    total_frames = get_total_frames(video_path) # Total frames in the video
    num_bg_frames = int(total_frames * percent)  # Number of frames used for background estimation

    print(f"Total frames: {total_frames}, Using the first {num_bg_frames} for background estimation")

    frames = get_frames(video_path, start=0, end=num_bg_frames) # Read the first `num_bg_frames` frames

    if len(frames) == 0: # No frames read
        print("Error: No frames were read for background estimation!")
        return None, None, num_bg_frames

    # Estimate background: Mean or Median
    if use_median:
        mean_bg = np.median(frames, axis=0)
    else:
        mean_bg = np.mean(frames, axis=0)
    
    # Calculate standard deviation for background estimation
    std_bg = np.std(frames, axis=0)

    return mean_bg, std_bg, num_bg_frames

# Function to segment the foreground in the video frames
def segment_foreground(video_path, mean_bg, std_bg, num_bg_frames, alpha=2.5, output_path=None, show_video=False):
    """
    Segments the foreground in the video based on the background estimation.
    It computes the difference between the current frame and the background.
    Foreground pixels are those where the difference exceeds `alpha` * std deviation.
    Optionally saves the output video to `output_path`.
    Optionally shows the video if `show_video` is True.
    """
    total_frames = get_total_frames(video_path) # Total frames in the video
    frames = get_frames(video_path, start=num_bg_frames, end=total_frames) # Read frames after background frames

    # Create the output video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
        fps = 30  # Frame rate
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        # Create foreground mask by comparing frame with background
        foreground_mask = np.abs(frame - mean_bg) >= (alpha * (std_bg + 2))
        fg_binary = (foreground_mask * 255).astype(np.uint8)

        # Show the foreground binary mask if show_video is True
        if show_video:
            cv2.imshow("Foreground Mask", fg_binary)

        # Write the processed frame to the output video
        if output_path:
            out_video.write(cv2.cvtColor(fg_binary, cv2.COLOR_GRAY2BGR))

        # Break on pressing 'ESC'
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

    # Release video writer if it was created
    if output_path:
        out_video.release()


# Main function 
def gaussian_modelling(video_path, percent=0.25, alpha=2.5, use_median=False, output_name=None, show_video=False):
    """
    This is the main function to process the video.
    It estimates the background and segments the foreground based on the background model.
    The processed video is saved in a folder named `Output_Videos`.
    Custom output video name.
    The video can be shown by setting `show_video` to True.
    """
    # Create the output folder if it doesn't exist
    output_folder = "Output_Videos"
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # If no output name is provided, use the input video's name with '_output' appended
    if not output_name:
        output_name = Path(video_path).stem + "_output"

    # Define the output video path
    output_video_path = Path(output_folder) / f"{output_name}.avi"

    # Estimate background
    mean_bg, std_bg, num_bg_frames = estimate_background(video_path, percent, use_median)

    if mean_bg is None or std_bg is None:  # Error in background estimation
        print("Background estimation failed. Check the video path and format.")
    else:
        # Segment foreground and save to video
        segment_foreground(video_path, mean_bg, std_bg, num_bg_frames, alpha, output_video_path, show_video)


if __name__ == "__main__":

    # Video Path
    video_path = str(Path("AICity_data") / "train" / "S03" / "c010" / "vdo.avi")

    # Single Gaussian Modelling 
    gaussian_modelling(video_path, # Video path
                       percent=0.25, # Percentage of frames used for background estimation
                       alpha=2.5, # Threshold for foreground segmentation
                       use_median=False, # Use median for background estimation
                       output_name="single_gaussian_output",  # Output video name
                       show_video=False) # Set to True to display the video
