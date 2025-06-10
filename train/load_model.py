import torch
from pathlib import Path
import os


def train_load_checkpoint(train_obj, model_path, device=None):
    """
    Loads model, optimizer, scheduler states and training metrics into the Train object.
    """
    if device is None:
        device = torch.device('cpu')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Restore model and optimizer state
    train_obj.model.load_state_dict(checkpoint['model_state_dict'])
    train_obj.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_obj.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore epoch and best accuracy
    train_obj.start_epoch = checkpoint.get('epoch', 0)
    train_obj.best_val_acc = checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0.0))  # fallback if older checkpoint

    # Restore metrics
    train_obj.train_loss = checkpoint.get('train_loss', 0.0)
    train_obj.train_top1_acc = checkpoint.get('train_top1_acc', 0.0)
    train_obj.train_top5_acc = checkpoint.get('train_top5_acc', 0.0)
    train_obj.val_loss = checkpoint.get('val_loss', 0.0)
    train_obj.val_top1_acc = checkpoint.get('val_top1_acc', 0.0)
    train_obj.val_top5_acc = checkpoint.get('val_top5_acc', 0.0)

    train_obj.logger.info(
        f"Loaded checkpoint from {model_path} at epoch {train_obj.start_epoch} "
        f"with val_acc {train_obj.val_top1_acc:.4f} and best_val_acc {train_obj.best_val_acc:.4f}"
    )
