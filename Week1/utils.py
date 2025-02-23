import cv2
import numpy as np


def get_total_frames(video_path: str) -> int:
    """Get the total number of frames in a video file.

    This function opens a video file using OpenCV, retrieves the total frame count, 
    and then releases the video capture object.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: The total number of frames in the video.

    Example:
        >>> get_total_frames("video.mp4")
        2400  # Example output
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def get_frames(video_path: str, start: int = 0, end: int | None = None) -> np.ndarray:
    """Extracts video frames as grayscale images and returns them as a NumPy array.

    This function reads frames from the video file between 'start' and 'end' (exclusive),
    converts them to grayscale, and stores them in a NumPy array.

    Args:
        video_path (str): Path to the video file.
        start (int, optional): The starting frame index (default is 0).
        end (int | None, optional): The ending frame index (exclusive). If None, 
            it defaults to the total number of frames in the video.

    Returns:
        np.ndarray: A NumPy array of shape (num_frames, height, width) containing 
        grayscale frames as float32 values.

    Example:
        >>> frames = get_frames("video.mp4", start=10, end=20)
        >>> frames.shape
        (10, 720, 1280)  # Example output (height and width depend on the video)

    Notes:
        - The function reads frames sequentially, so larger ranges may take more time.
        - If 'start' is greater than the total frame count, an empty array is returned.
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


def estimate_background(video_path: str, 
                        percent: float = 0.25, 
                        use_median: bool = False
                        ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Estimate the background of a video using the first 'percent' of frames.

    This function computes a background model by analyzing a portion of the video frames. 
    The background can be estimated using either the mean or median of the selected frames. 
    Additionally, the function calculates the standard deviation of the frames.

    Args:
        video_path (str): Path to the video file.
        percent (float, optional): The fraction (0 to 1) of the total frames to use 
            for background estimation (default is 0.25).
        use_median (bool, optional): If True, uses the median instead of the mean for 
            background estimation (default is False).

    Returns:
        tuple:
            - np.ndarray | None: The estimated background (mean or median) in grayscale, 
              or None if no frames were read.
            - np.ndarray | None: The standard deviation of the frames, or None if no 
              frames were read.
            - int: The number of frames used for background estimation.

    Example:
        >>> bg, std, num_frames = estimate_background("video.mp4", percent=0.2, use_median=True)
        Total frames: 1000, Using the first 200 for background estimation
        >>> bg.shape, std.shape, num_frames
        ((720, 1280), (720, 1280), 200)  # Example output (depends on video resolution)

    Notes:
        - If 'percent' is too small, the background estimate may be noisy.
        - If no frames are read, the function returns '(None, None, num_frames)'.
        - Ensure the video file is accessible and readable.
    """
    total_frames = get_total_frames(video_path)  # Total frames in the video
    num_bg_frames = int(total_frames * percent)  # Number of frames used for background estimation

    print(f"Total frames: {total_frames}, Using the first {num_bg_frames} for background estimation")

    frames = get_frames(video_path, start=0, end=num_bg_frames)  # Read the first `num_bg_frames` frames

    if len(frames) == 0:  # No frames read
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

def segment_foreground(
    video_path: str,
    mean_bg: np.ndarray,
    std_bg: np.ndarray,
    num_bg_frames: int,
    alpha: float = 2.5,
    output_path: str | None = None,
    show_video: bool = False
) -> None:
    """Segments the foreground in a video based on background estimation.

    This function processes a video by comparing each frame to a precomputed background model.
    It identifies foreground pixels where the difference from the background exceeds 
    `alpha * std_bg`. The resulting binary mask highlights moving objects.

    The function optionally saves the output as a video and can display the segmentation in real time.

    Args:
        video_path (str): Path to the input video file.
        mean_bg (np.ndarray): The estimated background image (grayscale).
        std_bg (np.ndarray): The standard deviation of the background model.
        num_bg_frames (int): The number of frames used to compute the background.
        alpha (float, optional): Threshold multiplier for background subtraction (default is 2.5).
        output_path (str | None, optional): Path to save the output segmented video.
            If None, the video is not saved (default is None).
        show_video (bool, optional): If True, displays the segmented foreground in real time (default is False).

    Returns:
        None: The function does not return a value but can save an output video.

    Example:
        >>> segment_foreground("video.mp4", mean_bg, std_bg, num_bg_frames, alpha=3.0, output_path="output.avi", show_video=True)

    Notes:
        - The function processes frames starting from `num_bg_frames` to the end of the video.
        - Foreground detection is based on the absolute difference exceeding `alpha * std_bg + 2`.
        - Press 'ESC' to stop the video display early.
    """
    total_frames = get_total_frames(video_path)  # Total frames in the video
    frames = get_frames(video_path, start=num_bg_frames, end=total_frames)  # Read frames after background frames

    # Create the output video writer if saving
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