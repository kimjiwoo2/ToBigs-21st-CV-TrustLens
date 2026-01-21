import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model_def import MultiTaskSwinV2

class SwinInference:
    def __init__(self, model_path, model_name='swinv2_tiny_window16_256', img_size=256):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = img_size
        
        # 모델 로드
        self.model = MultiTaskSwinV2(model_name=model_name)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # state_dict 키 정리 (module. 제거 등)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device).eval()

        # 전처리 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def reshape_transform(self, tensor):
        """Swin Transformer의 출력을 CAM용 2D 형태로 변환"""
        # Swin의 출력 형태: [Batch, Height, Width, Channels] -> [Batch, Channels, Height, Width]
        result = tensor.permute(0, 3, 1, 2)
        return result

    def predict(self, pil_image): # img_path 대신 pil_image를 받음
        # 1. 전처리 (경로 로드 생략하고 바로 리사이즈)
        image_rgb = pil_image.convert('RGB')
        raw_image_resized = image_rgb.resize((self.img_size, self.img_size))
        rgb_img_float = np.array(raw_image_resized, dtype=np.float32) / 255.0
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # 2. Grad-CAM 설정 (중략)
        target_layers = [self.model.backbone.layers[-1].blocks[-1].norm1]
        
        class CAMWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)["logits"]

        cam = GradCAM(model=CAMWrapper(self.model), 
                      target_layers=target_layers, 
                      reshape_transform=self.reshape_transform)

        # 3. 추론
        outputs = self.model(input_tensor)
        logits = outputs["logits"]
        pred_idx = torch.argmax(logits, dim=1).item()
        
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

        return {
            "label": pred_idx,
            "confidence": F.softmax(logits, dim=1)[0][pred_idx].item(),
            "ssim": outputs["pred_ssim"].item(),
            "lpips": outputs["pred_lpips"].item(),
            "heatmap": visualization
        }