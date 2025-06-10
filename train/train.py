import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from utils import LoadConfig
from log_tracker import Logger
import math
import time


class Train:
    def __init__(self, model, train_loader, val_loader):
        self.config = LoadConfig.load_config("configs/train_config.yaml")
        self.logger = Logger.get_logger()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.model_name = self.config['model_name']
        self.epoch = self.config['epoch']
        self.lr = self.config['lr']
        self.save_dir = self.config['save_dir']
        self.linear_warm_up_epoch = self.config['linear_warm_up_epoch']
        self.weight_decay = self.config['weight_decay']
        self.save_number = self.config['save_number']
        self.save_freq = self.config['save_freq']

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: self._lr_lambda(epoch))
        self.start_epoch = 0

        self.train_loss = 0.0
        self.train_top1_acc = 0.0
        self.train_top5_acc = 0.0
        self.val_loss = 0.0
        self.val_top1_acc = 0.0
        self.val_top5_acc = 0.0
        self.total_time = 0

        self.best_checkpoints = []
        self.interval_checkpoints = []
        self.best_val_acc = 0.0

    def _save_checkpoint(self, path, epoch, val_acc):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'train_loss': self.train_loss,
            'train_top1_acc': self.train_top1_acc,
            'train_top5_acc': self.train_top5_acc,
            'val_loss': self.val_loss,
            'val_top1_acc': self.val_top1_acc,
            'val_top5_acc': self.val_top5_acc,
            'best_val_acc': self.best_val_acc
        }, path)

    def _lr_lambda(self, current_epoch):
        if current_epoch < self.linear_warm_up_epoch:
            return float(current_epoch) / float(max(1, self.linear_warm_up_epoch))
        else:
            progress = (current_epoch - self.linear_warm_up_epoch) / (self.epoch - self.linear_warm_up_epoch)
            return 0.5 * (1 + math.cos(math.pi * progress))

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            # Top-1
            _, pred_top1 = outputs.topk(1, dim=1)
            top1_correct += (pred_top1.squeeze() == labels).sum().item()

            # Top-5
            _, pred_top5 = outputs.topk(5, dim=1)
            top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        self.train_loss = running_loss / total_samples
        self.train_top1_acc = top1_correct / total_samples
        self.train_top5_acc = top5_correct / total_samples

    def validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                total_samples += labels.size(0)

                _, pred_top1 = outputs.topk(1, dim=1)
                top1_correct += (pred_top1.squeeze() == labels).sum().item()

                _, pred_top5 = outputs.topk(5, dim=1)
                top5_correct += (pred_top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        self.val_loss = running_loss / total_samples
        self.val_top1_acc = top1_correct / total_samples
        self.val_top5_acc = top5_correct / total_samples

    def start_training(self):
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epoch):
            self.logger.info(f"Epoch {epoch + 1}/{self.epoch}")
            self.train_one_epoch()
            self.scheduler.step()
            self.validate_one_epoch()

            # Log metrics after this epoch
            metrics = self.get_latest_metrics()
            self.logger.info(
                f"Epoch {epoch + 1} Metrics -- "
                f"LR: {metrics['lr']:.6f}, "
                f"Train Loss: {metrics['train_loss']:.4f}, "
                f"Train Top-1 Acc: {metrics['train_top1_acc']:.4f}, "
                f"Train Top-5 Acc: {metrics['train_top5_acc']:.4f}, "
                f"Val Loss: {metrics['val_loss']:.4f}, "
                f"Val Top-1 Acc: {metrics['val_top1_acc']:.4f}, "
                f"Val Top-5 Acc: {metrics['val_top5_acc']:.4f}, "
                f"Best Val Acc: {metrics['best_val_acc']:.4f}"
            )

            self.save_model(epoch + 1, val_acc=self.val_top1_acc)

        self.total_time = time.time() - start_time
        self.logger.info(f"Training complete in {self.total_time:.2f} seconds.")

    def save_model(self, epoch, val_acc):
        os.makedirs(self.save_dir, exist_ok=True)

        # --- Check if this epoch qualifies as a best checkpoint ---
        # Condition: fewer checkpoints saved or val_acc better than worst saved best checkpoint
        if len(self.best_checkpoints) < self.save_number or val_acc > min(acc for acc, _ in self.best_checkpoints):
            filename = f"{self.model_name}_best_epoch_{epoch}.pth"
            save_path = os.path.join(self.save_dir, filename)

            # Update the tracked best validation accuracy if this one is better
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc

            # Save checkpoint with val_acc (consistent key)
            self._save_checkpoint(save_path, epoch, val_acc)

            self.logger.info(f"Best model saved: {save_path}")
            # Track this checkpoint
            self.best_checkpoints.append((val_acc, save_path))

            # If exceed number of best checkpoints to keep, remove worst one
            if len(self.best_checkpoints) > self.save_number:
                worst = min(self.best_checkpoints, key=lambda x: x[0])
                try:
                    os.remove(worst[1])
                    self.logger.info(f"Deleted worst best checkpoint: {worst[1]}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete checkpoint {worst[1]}: {e}")
                self.best_checkpoints.remove(worst)

        # --- Handle interval checkpoints ---
        if (epoch % self.save_freq == 0):
            filename = f"{self.model_name}_interval_epoch_{epoch}.pth"
            save_path = os.path.join(self.save_dir, filename)

            self._save_checkpoint(save_path, epoch, val_acc)

            self.logger.info(f"Interval model saved: {save_path}")

            self.interval_checkpoints.append((epoch, save_path))

            # Remove oldest interval checkpoint if exceed limit
            if len(self.interval_checkpoints) > self.save_number:
                oldest = min(self.interval_checkpoints, key=lambda x: x[0])
                try:
                    os.remove(oldest[1])
                    self.logger.info(f"Deleted oldest interval checkpoint: {oldest[1]}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete checkpoint {oldest[1]}: {e}")
                self.interval_checkpoints.remove(oldest)

        # --- Always save latest model checkpoint ---
        filename = f"{self.model_name}_latest.pth"
        latest_path = os.path.join(self.save_dir, filename)
        self._save_checkpoint(latest_path, epoch, val_acc)

        self.logger.info(f"Latest model saved: {latest_path}")

    def get_trained_model(self):
        return self.model

    def get_latest_metrics(self):
        return {
            'lr': self.scheduler.get_last_lr()[0],
            'train_loss': self.train_loss,
            'train_top1_acc': self.train_top1_acc,
            'train_top5_acc': self.train_top5_acc,
            'val_loss': self.val_loss,
            'val_top1_acc': self.val_top1_acc,
            'val_top5_acc': self.val_top5_acc,
            'best_val_acc': self.best_val_acc,
            'total_time': self.total_time
        }
