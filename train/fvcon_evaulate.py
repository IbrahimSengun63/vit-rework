from log_tracker import Logger
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count


class Evaluate:
    def __init__(self, model, loader):
        self.logger = Logger.get_logger()
        self.model = model
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.model.eval()

        self.val_loss = 0.0
        self.val_top1_acc = 0.0
        self.val_top5_acc = 0.0
        self.total_flops = 0.0
        self.total_param = 0.0

    def evaluate_model(self):
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in self.loader:
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

    def calculate_flops(self, input_shape=(1, 3, 224, 224)):
        dummy_input = torch.randn(*input_shape).to(self.device)
        self.model.eval()

        # FLOPs calculation
        flops = FlopCountAnalysis(self.model, dummy_input)
        self.total_flops = flops.total()  # total FLOPs for 1 image

        # Parameter count
        params = parameter_count(self.model)
        self.total_param = sum(p for p in params.values())

    def get_evaluation_metrics(self):
        return {
            'val_loss': self.val_loss,
            'val_top1_acc': self.val_top1_acc,
            'val_top5_acc': self.val_top5_acc,
            'total_flops': self.total_flops,
            'total_params': self.total_param
        }

    def get_result(self):

        self.evaluate_model()
        self.calculate_flops()
        result = self.get_evaluation_metrics()

        self.logger.info(
            f"Val Loss: {result['val_loss']:.6f}, "
            f"Val Top 1 Acc: {result['val_top1_acc']:.6f}, "
            f"Val Top 5 Acc: {result['val_top5_acc']:.6f}, "
            f"Total Flops: {result['total_flops']:}, "
            f"Total Params: {result['total_params']:} "
        )
