import torch
import os
from transformers import ViTImageProcessor
from model_def import ViTMultiTask
# safetensors를 읽기 위한 전용 함수 임포트
from safetensors.torch import load_file 

class ViTInference:
    def __init__(self, model_path, base_model="google/vit-base-patch16-224-in21k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(base_model)
        self.model = self._load_model(model_path, base_model)

    def _load_model(self, model_path, base_model):
        model = ViTMultiTask(base_model)
        st_path = os.path.join(model_path, "model.safetensors")
        
        if os.path.exists(st_path):
            state_dict = load_file(st_path, device=self.device)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {st_path}")

        model.to(self.device)
        model.eval()
        return model

    def predict(self, image):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs['pixel_values'])
        
        return {
            "label": torch.argmax(outputs['logits'], dim=1).item(),
            "ssim": outputs['pred_ssim'].item(),
            "lpips": outputs['pred_lpips'].item(),
            "strength": outputs['pred_strength'].item()
        }