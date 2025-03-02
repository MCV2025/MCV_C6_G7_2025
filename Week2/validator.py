import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

class ObjectDetectionValidator:
    def validation(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion,
        params: dict,
        device: torch.device
    ):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = 0.0  # or compute detection metrics separately
        return avg_loss, avg_accuracy
