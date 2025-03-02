import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

class ObjectDetectionTrainer:
    def train(
        self,
        model: nn.Module,  # e.g., DetrForObjectDetection
        train_loader: DataLoader,
        criterion,  # Not needed externally, DETR uses built-in Hungarian loss
        optimizer: optim.Optimizer,
        params: dict,
        device: torch.device
    ):
        model.train()
        total_loss = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            # labels is a list of dicts, each with "labels" and "boxes"
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # For detection, you might not compute “accuracy”; you track loss or do an mAP pass
        avg_accuracy = 0.0
        return avg_loss, avg_accuracy
