import torch
import cv2
from typing import Any, Optional
from transformers import DetrForObjectDetection

def prepare_frame_for_detr(
    frame,
    frame_idx: int,
    annotations: dict[int, list[dict]],
    transform,
    device: torch.device,
    car_category_id: int
) -> tuple[Optional[torch.Tensor], Optional[list[dict]]]:
    """
    Utility that:
      - transforms a BGR frame into the model input tensor
      - extracts bounding boxes for “car” from annotations[frame_idx]
      - converts them to the huggingface DETR format
    Returns (input_tensor, labels_list):
      - input_tensor: shape [1, C, H, W] or None if no boxes
      - labels_list: a list of length=1: [{"class_labels": Tensor, "boxes": Tensor}] or None if no boxes
    """
    h, w, _ = frame.shape

    input_tensor = transform(frame).unsqueeze(0).to(device)

    # Get CVAT annotations for this frame
    gt_list = annotations.get(frame_idx, [])
    boxes_list = []
    labels_list = []

    for ann in gt_list:
        if ann["label"] == "car":
            (x1, y1, x2, y2) = ann["bbox"]
            box_w = x2 - x1
            box_h = y2 - y1
            # Convert to normalized [x, y, w, h]
            boxes_list.append([
                x1 / w,
                y1 / h,
                box_w / w,
                box_h / h
            ])
            labels_list.append(car_category_id)

    # If no car boxes skip frame
    if not boxes_list:
        return None, None

    boxes_tensor = torch.tensor(boxes_list, dtype=torch.float, device=device)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

    # HuggingFace DETR expects a list of length = batch_size.
    labels_dict = [{
        "class_labels": labels_tensor,
        "boxes": boxes_tensor
    }]

    return input_tensor, labels_dict


class Trainer:
    def __init__(
        self,
        video_path: str,
        train_end: int,
        val_end: int,
        annotation_file: str,
        transform,  
        car_category_id: int = 1
    ) -> None:
        """
        :param video_path: path to a video file (e.g. "vdo.avi")
        :param annotation_file: path to CVAT XML (e.g. "ai_challenge_s03_c010-full_annotation.xml")
        :param transform: a transform that takes a NumPy BGR frame -> PyTorch tensor
        :param car_category_id: label ID used for 'car'
        """
        self.video_path = video_path
        self.annotation = annotation_file
        self.transform = transform
        self.car_category_id = car_category_id
        self.train_end = train_end
        self.val_end = val_end

    def train(
        self,
        model: DetrForObjectDetection,
        optimizer: torch.optim.Optimizer,
        params: dict[str, Any],
        device: torch.device
    ) -> tuple[float, float]:
        """
        Runs a training pass on every frame in `self.video_path`.
        For each frame:
          1) transform the frame
          2) build bounding-box tensors for 'car' from the CVAT annotations
          3) pass them through the model
          4) compute loss, backprop, step

        Returns (avg_loss, avg_accuracy). 
        Accuracy for detection tasks is often 0 or omitted, but we return it for API consistency.
        """
        
        model.train().to(device)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video:", self.video_path)
            return 0.0, 0.0

        frame_idx = 0
        total_loss = 0.0
        frames_used = 0


        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < self.train_end:
                # Prepare frame for Detr
                input_tensor, labels_list = prepare_frame_for_detr(
                    frame=frame,
                    frame_idx=frame_idx,
                    annotations=self.annotations,
                    transform=self.transform,
                    device=device,
                    car_category_id=self.car_category_id
                )

                # If no annotations for this frame, skip
                if input_tensor is None:
                    frame_idx += 1
                    continue

                # Forward pass & compute loss
                optimizer.zero_grad()
                outputs = model(pixel_values=input_tensor, labels=labels_list)
                loss = outputs.loss

                # 5. Backprop
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            frames_used += 1
            frame_idx += 1

        cap.release()

        avg_loss = total_loss / frames_used if frames_used > 0 else 0.0
        return avg_loss, 0.0


    def validation(
        self,
        model: DetrForObjectDetection,
        _unused_loader,           # We keep this in the signature just to match your run_model usage
        params: dict[str, Any],
        device: torch.device
    ) -> tuple[float, float]:
        """
        Evaluates the model on each frame in self.video_path (just like 'Trainer', but no backprop).
        Returns (avg_loss, avg_accuracy).
        
        In detection tasks, "accuracy" is usually replaced by mAP or IoU-based metrics. 
        Here we return a placeholder of 0.0 for accuracy to keep the interface consistent.
        """
        model.eval().to(device)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video:", self.video_path)
            return 0.0, 0.0

        frame_idx = self.test_end
        total_loss = 0.0
        frames_used = self.test_end

        # We do not track gradients during validation
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # No more frames or error
                
                if self.train_end < frame_idx < self.val_end:
                    input_tensor, labels_list = prepare_frame_for_detr(
                        frame=frame,
                        frame_idx=frame_idx,
                        annotations=self.annotations,
                        transform=self.transform,
                        device=device,
                        car_category_id=self.car_category_id
                    )
                    
                    # If no annotations for this frame, skip
                    if input_tensor is None:
                        frame_idx += 1
                        continue

                    # Forward pass
                    outputs = model(pixel_values=input_tensor, labels=labels_list)
                    loss = outputs.loss

                    total_loss += loss.item()
                frames_used += 1
                frame_idx += 1

        cap.release()

        avg_loss = total_loss / frames_used if frames_used > 0 else 0.0
        # For detection tasks, "accuracy" is typically not relevant, so we return 0.0
        return avg_loss, 0.0
    
    def test(
        self,
        model: torch.nn.Module,
        params: dict[str, Any],
        device: torch.device
    ) -> tuple[float, float]:
        """
        Evaluates the model on test frames.
        Returns (avg_loss, avg_accuracy).
        """
        model.eval()  # Set model to evaluation mode
        model.to(device)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening test video:", self.video_path)
            return 0.0, 0.0

        frame_idx = 0
        total_loss = 0.0
        frames_used = 0

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # No more frames
                
                if frame_idx > self.val_end:
                    # Prepare frame
                    input_tensor, labels_list = prepare_frame_for_detr(
                        frame=frame,
                        frame_idx=frame_idx,
                        annotations=self.annotations,
                        transform=self.transform,
                        device=device,
                        car_category_id=self.car_category_id
                    )

                    if input_tensor is None:
                        frame_idx += 1
                        continue

                    # Forward pass (compute loss only, no backprop)
                    outputs = model(pixel_values=input_tensor, labels=labels_list)
                    loss = outputs.loss

                    total_loss += loss.item()
                frames_used += 1
                frame_idx += 1

        cap.release()

        avg_test_loss = total_loss / frames_used if frames_used > 0 else 0.0
        return avg_test_loss, 0.0  # Accuracy is 0.0 since DETR is detection-based