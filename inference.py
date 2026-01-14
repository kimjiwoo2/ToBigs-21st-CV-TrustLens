import torch
import os
import cv2
import numpy as np
from transformers import ViTImageProcessor
from model_def import ViTMultiTask
from safetensors.torch import load_file 

class ViTInference:
    def __init__(self, model_path, base_model="google/vit-base-patch16-224-in21k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(base_model)
        self.model = self._load_model(model_path, base_model)

    def _load_model(self, model_path, base_model):
        model = ViTMultiTask(base_model)
        model.backbone.set_attn_implementation("eager")
        st_path = os.path.join(model_path, "model.safetensors")
        
        if os.path.exists(st_path):
            state_dict = load_file(st_path, device=self.device)
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {st_path}")

        model.to(self.device)
        model.eval()
        return model
    
    def _generate_heatmap(self, image, attentions):
        # 1. 마지막 레이어 어텐션 추출 (batch, heads, seq, seq)
        # 모든 헤드의 평균을 구하고 CLS 토큰(0번)의 가중치만 추출
        last_layer_attn = attentions[-1] 
        avg_attn = torch.mean(last_layer_attn, dim=1)[0, 0, 1:] # (196,)
        
        # 2. 14x14 그리드로 변환 및 정규화
        grid_size = int(np.sqrt(avg_attn.size(0)))
        heatmap = avg_attn.reshape(grid_size, grid_size).cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 3. 이미지 크기에 맞게 확대 (224x224)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        
        # 4. 컬러맵 적용 및 원본 이미지와 합성
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_np = np.array(image.resize((224, 224)))
        # OpenCV 컬러는 BGR이므로 RGB 변환 후 합성
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
        
        return superimposed_img

    def predict(self, image):
        image_rgb = image.convert("RGB")
        inputs = self.processor(image_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs['pixel_values'], output_attentions=True)

        # 히트맵 생성
        heatmap_res = self._generate_heatmap(image_rgb, outputs['attentions'])
        
        return {
            "label": torch.argmax(outputs['logits'], dim=1).item(),
            "ssim": outputs['pred_ssim'].item(),
            "lpips": outputs['pred_lpips'].item(),
            "strength": outputs['pred_strength'].item(),
            "heatmap": heatmap_res
        }