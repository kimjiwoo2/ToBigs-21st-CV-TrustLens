import torch
import torch.nn as nn
import timm

class MultiTaskSwinV2(nn.Module):
    def __init__(self, model_name='swinv2_small_window8_256', pretrained=False, num_classes=4):
        super(MultiTaskSwinV2, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features

        self.head_label = nn.Sequential(
            nn.Linear(num_features, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.head_ssim = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.head_lpips = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        return {
            "logits": self.head_label(features),
            "pred_ssim": self.head_ssim(features),
            "pred_lpips": self.head_lpips(features),
        }