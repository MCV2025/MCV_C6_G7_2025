# pyright: reportMissingImports=false

import argparse
from ultralytics import YOLO
from pathlib import Path

def finetune_yolo(output_folder, strategy, batch_size, epochs):
    model = YOLO("yolov8n.pt")
    model.train(
        data="config.yaml",
        epochs=epochs,
        batch=batch_size,
        freeze=10,
        project=f'{output_folder}',
        name=f"{strategy}",
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 model.")

    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("yolo_ft_output"),
        help="Path to the output folder for training results. (default: yolo_ft_output)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="A",
        help="Strategy label for this run. (default: A)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training. (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training. (default: 100)",
    )

    args = parser.parse_args()

    finetune_yolo(
        output_folder=args.output_folder,
        strategy=args.strategy,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
