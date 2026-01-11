import torch.nn as nn
from transformers import ViTModel

class ViTMultiTask(nn.Module):
    def __init__(self, model_name_or_path, num_labels=4):
        super(ViTMultiTask, self).__init__()
        self.backbone = ViTModel.from_pretrained(model_name_or_path)
        hidden_size = self.backbone.config.hidden_size

        # Classification Head
        self.classifier = nn.Linear(hidden_size, num_labels)
        # Regression Heads
        self.regressor_ssim = nn.Linear(hidden_size, 1)
        self.regressor_lpips = nn.Linear(hidden_size, 1)
        self.regressor_strength = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values, labels=None, ssim=None, lpips=None, strength=None):
        outputs = self.backbone(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(sequence_output)
        pred_ssim = self.regressor_ssim(sequence_output)
        pred_lpips = self.regressor_lpips(sequence_output)
        pred_strength = self.regressor_strength(sequence_output)

        return {
            "logits": logits,
            "pred_ssim": pred_ssim,
            "pred_lpips": pred_lpips,
            "pred_strength": pred_strength
        }