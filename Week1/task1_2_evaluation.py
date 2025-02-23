
from utils import convert_xml_to_coco, extract_bounding_boxes, convert_detections_to_coco

import json
from pathlib import Path
import cv2


import torch
import detectron2
print(torch.__version__)  # Should be 2.1.2+cu121
print(torch.version.cuda)  # Should be 12.1


# # Convert XML annotations and save them as a COCO formated JSON
# xml_file_path = "ai_challenge_s03_c010-full_annotation.xml"
# coco_gt = convert_xml_to_coco(xml_file_path, {"car-bike": 1})

# with open("ground_truth.json", "w") as f:
#     json.dump(coco_gt, f, indent=4)

# print("Ground truth annotations saved as ground_truth.json")

# video_filename = "task_1_1_mean_alpha2.5.avi"
# video_path = str(Path("Output_Videos") / video_filename)


# # Extract bounding boxes from the modelled video
# output_json = extract_bounding_boxes(video_path)   
# print(f"Extracted bounding boxes saved as {output_json}")

# # Example detected bounding boxes (replace with actual detections)
# with open(output_json, "r") as f:
#     detected_boxes = json.load(f)

# # Convert JSON keys (which are strings) to integers for consistency
# detected_boxes = {int(frame): boxes for frame, boxes in detected_boxes.items()}

# coco_preds = convert_detections_to_coco(detected_boxes, {"car-bike": 1}, output_json=output_json)

# with open("detections.json", "w") as f:
#     json.dump(coco_preds, f, indent=4)

# print("Predicted detections saved as detections.json")


# from detectron2.evaluation import COCOEvaluator
# from detectron2.data.datasets import load_coco_json
# from detectron2.data import DatasetCatalog, MetadataCatalog

# # Register dataset
# DatasetCatalog.clear()
# DatasetCatalog.register("ground_truth", lambda: load_coco_json("ground_truth.json", ""))
# DatasetCatalog.register("detections", lambda: load_coco_json("detections.json", ""))

# # Load metadata
# MetadataCatalog.get("ground_truth").thing_classes = ["car"]

# # Initialize evaluator
# evaluator = COCOEvaluator("ground_truth", ("bbox",), False, output_dir="./")
# evaluator.reset()

# # Evaluate using Detectron2
# results = evaluator.evaluate()
# print(results)
