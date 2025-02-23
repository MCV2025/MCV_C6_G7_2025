import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, Dict, List  # Python 3.9-compatible typing

# ---------------------------------------------------------------------------------
#                               TASK 1 FUNCTIONS
#----------------------------------------------------------------------------------
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


def get_frames(video_path: str, start: int = 0, end: Optional[int] = None) -> np.ndarray:
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
                        ) -> Tuple[Optional[np.ndarray],Optional[np.ndarray], int]:
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
    output_path: Optional[str] = None,
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


# ---------------------------------------------------------------------------------
#                               TASK 2 FUNCTIONS
#----------------------------------------------------------------------------------

def convert_xml_to_coco(xml_file: str, categories: Dict[str, int]) -> Dict:
    """Converts an XML annotation file (CVAT format) to COCO format.

    This function parses an XML file containing object annotations, filters out 
    non-car objects and parked cars, and converts the remaining annotations into 
    COCO format.

    Args:
        xml_file (str): Path to the XML annotation file.
        categories (dict[str, int]): Mapping of category names to COCO category IDs.

    Returns:
        dict: A dictionary in COCO format containing:
            - "images" (list[dict]): Metadata for each frame with annotations.
            - "annotations" (list[dict]): Bounding box annotations in COCO format.
            - "categories" (list[dict]): List of COCO category definitions.

    Example:
        >>> categories = {"car": 1}
        >>> coco_data = convert_xml_to_coco("annotations.xml", categories)
        >>> print(coco_data.keys())
        dict_keys(['images', 'annotations', 'categories'])

    Notes:
        - Only annotations labeled as "car" are included.
        - Parked cars are ignored based on the "parked" attribute.
        - The function assumes fixed image dimensions (1920x1080).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    images = []
    annotations = []
    annotation_id = 1

    for track in root.findall("track"):
        label = track.get("label")
        # if label != "car":  # Ignore non-car objects
        #     continue

        for box in track.findall("box"):
            frame = int(box.get("frame"))
            xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])
            if label == "car":
                parked = box.find("attribute[@name='parked']").text
                if parked == "true":  # Ignore parked cars
                    continue
            
            # Convert all labels into car-bike
            label = "car-bike"
            
            category_id = categories[label]
          
            image_info = {"id": frame, "file_name": f"{frame}.jpg", "height": 1080, "width": 1920}
            if image_info not in images:
                images.append(image_info)

            bbox = [xtl, ytl, xbr - xtl, ybr - ytl]  # COCO format: (x_min, y_min, width, height)
            annotation = {
                "id": annotation_id,
                "image_id": frame,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car-bike"}],
    }

    return coco_format


def extract_bounding_boxes(video_path: str, 
                           output_json: str="detected_bounding_boxes.json", 
                           threshold: int=127,
                           min_area: int=10) -> None:
    """
    Extracts bounding boxessss from a binary-segmented video.

    This function processes an input video, applying grayscale conversion 
    and thresholding to detect objectsss. It then extracts bounding boxes 
    around the detected regions and stores them in a JSON file.

    Args:
        video_filename (str): The name of the input video file, located in the "Output_Videos" directory.
        output_json (str, optional): The filename for saving the detected bounding boxes as a JSON file. 
                                     Defaults to "detected_bounding_boxes.json".
        threshold (int, optional): The binarization threshold for segmenting the video frames. 
                                   Defaults to 127.

    Returns:
        dict: A dictionary mapping frame numbers to a list of bounding box tuples. 
              Each tuple is in the format (x_min, y_min, x_max, y_max).

    Example:
        >>> boxes = extract_bounding_boxes("example_video.mp4")
        Processing video: Output_Videos/example_video.mp4
        Extracted bounding boxes saved as detected_bounding_boxes.json
    """

    cap = cv2.VideoCapture(video_path)
    detected_boxes = {}
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Ignore very small objects (noise)
            if w > 1 and h > 1 and area > min_area:  
                frame_bboxes.append((x, y, x + w, y + h))  # Store (x_min, y_min, x_max, y_max)

        # Save bounding boxes for this frame
        if frame_bboxes:
            detected_boxes[frame_number] = frame_bboxes

        frame_number += 1

    cap.release()

    # Save detected bounding boxes as a JSON file
    with open(output_json, "w") as f:
        json.dump(detected_boxes, f, indent=4)

    return output_json


def convert_detections_to_coco(detections: Dict[int, List[List[float]]], 
                               categories: Dict[str, int], 
                               output_json: str) -> Dict:
    """Converts object detections into COCO format.

    This function processes a dictionary of detections, where each frame ID maps 
    to a list of bounding boxes. It converts the detections into the COCO annotation 
    format and optionally saves them as a JSON file.

    Args:
        detections (dict[int, list[list[float]]]): Dictionary mapping frame IDs to a list of bounding boxes.
            Each bounding box is represented as [x_min, y_min, x_max, y_max].
        categories (dict[str, int]): Mapping of category names to COCO category IDs.
        output_json (str): Path to save the output JSON file in COCO format.

    Returns:
        dict: A dictionary in COCO format containing:
            - "images" (list[dict]): Metadata for each annotated frame.
            - "annotations" (list[dict]): Bounding box annotations in COCO format.
            - "categories" (list[dict]): List of COCO category definitions.

    Example:
        >>> detections = {1: [[100, 200, 300, 400], [50, 60, 200, 250]]}
        >>> categories = {"car-bike": 1}
        >>> coco_data = convert_detections_to_coco(detections, categories, "output.json")
        >>> print(coco_data.keys())
        dict_keys(['images', 'annotations', 'categories'])

    Notes:
        - The function assumes the "car-bike" category is present in `categories`.
        - The bounding boxes are converted to COCO format: (x_min, y_min, width, height).
        - The output JSON file is written to `output_json`.
    """
    images = []
    annotations = []
    annotation_id = 1

    for frame, boxes in detections.items():
        image_info = {"id": frame, "file_name": f"{frame}.jpg", "height": 1080, "width": 1920}
        images.append(image_info)

        for box in boxes:
            bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]  # Convert to COCO format
            annotation = {
                "id": annotation_id,
                "image_id": frame,
                "category_id": categories["car-bike"],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car-bike"}],
    }

    # Save COCO format to JSON file
    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    return coco_format
