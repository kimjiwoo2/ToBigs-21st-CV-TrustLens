import torch
import torch.nn as nn
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
        
        # 1. 모델 초기화
        self.model = MultiTaskSwinV2(model_name=model_name, num_classes=4) # num_classes 확인 필요
        
        # 2. 체크포인트 로드 (weights_only=False는 보안 경고 무시용, 신뢰할 수 있는 파일일 때 사용)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 3. state_dict 처리
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device).eval()

        # 4. 통계 정보 로드 (역정규화용)
        # 학습 코드에서 저장을 안 했다면 기본값(mean=0, std=1)을 사용하도록 처리
        if 'stats' in checkpoint:
            self.stats = checkpoint['stats']
            print(f"✅ Loaded Stats for Denormalization: {self.stats}")
        else:
            self.stats = None
            print("⚠️ No stats found in checkpoint. Using raw model outputs.")

        # 5. 전처리 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def reshape_transform(self, tensor):
        """Swin Transformer의 출력을 CAM용 2D 형태로 변환"""
        # [Batch, H, W, C] -> [Batch, C, H, W]
        result = tensor.permute(0, 3, 1, 2)
        return result
    
    def _calc_rm_pvr(self, pil_image, k=3.0):
        """SRM 잔차 기반 RM/PVR 계산"""
        img_rgb = np.array(pil_image.convert("RGB"))
        if img_rgb.ndim == 2:  # Grayscale check
             img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
             
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        filter_kernel = np.array([
            [0, -1,  2, -1, 0],
            [0,  2, -4,  2, 0],
            [0, -1,  2, -1, 0]
        ], dtype=np.float32) / 4.0

        res_map = cv2.filter2D(img_gray, -1, filter_kernel)
        abs_res = np.abs(res_map)

        rm = float(np.mean(abs_res))
        std_res = float(np.std(abs_res))
        threshold = k * std_res

        # 분모가 0이 되는 것을 방지
        size = abs_res.size if abs_res.size > 0 else 1
        pvr = float((np.sum(abs_res > threshold) / size) * 100.0)
        return rm, pvr, threshold

    def predict(self, pil_image):
        # 1. 전처리
        image_rgb = pil_image.convert('RGB')
        vis_image = image_rgb.resize((self.img_size, self.img_size))
        rgb_img_float = np.array(vis_image, dtype=np.float32) / 255.0
        
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # 2. Grad-CAM용 Wrapper 클래스 정의 (수정됨)
        class CAMWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                output = self.model(x)
                # [수정] 딕셔너리인 경우 'logits' 키로 접근, 아니면 인덱싱
                if isinstance(output, dict):
                    return output['logits']
                elif hasattr(output, 'logits'): # HuggingFace ModelOutput
                    return output.logits
                return output[0]

        # 3. Grad-CAM 설정
        target_layers = [self.model.backbone.layers[-1].blocks[-1].norm1]
        
        cam = GradCAM(model=CAMWrapper(self.model), 
                      target_layers=target_layers, 
                      reshape_transform=self.reshape_transform)

        # 4. 추론 수행 (수정됨)
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # [수정] 결과값 추출 로직 변경
        if isinstance(outputs, dict):
            # 딕셔너리인 경우 키값으로 추출
            logits = outputs['logits']
            
            # 주의: 나머지 키 이름이 'ssim', 'lpips'가 맞는지 확인이 필요합니다.
            # 만약 에러가 난다면 outputs.keys()를 출력해서 정확한 키 이름을 확인해야 합니다.
            # 여기서는 가장 확률 높은 키 이름을 사용합니다.
            # 키 이름이 다를 경우 아래 문자열('ssim', 'lpips')을 실제 키 이름으로 바꿔주세요.
            p_ssim_z = outputs.get('ssim', outputs.get('ssim_z', None)) 
            p_lpips_z = outputs.get('lpips', outputs.get('lpips_z', None))
            
            # 만약 키를 못 찾았다면 순서대로 가져오기 시도 (비상책)
            if p_ssim_z is None: p_ssim_z = list(outputs.values())[1]
            if p_lpips_z is None: p_lpips_z = list(outputs.values())[2]

        elif hasattr(outputs, 'logits'):
            # HuggingFace 객체인 경우
            logits = outputs.logits
            # 커스텀 필드가 있다면 속성으로 접근, 없다면 None 처리 등 필요
            p_ssim_z = getattr(outputs, 'ssim', None)
            p_lpips_z = getattr(outputs, 'lpips', None)
        else:
            # 기존 튜플 방식
            logits, p_ssim_z, p_lpips_z = outputs

        # 값 확인용 (이제는 텐서가 나와야 함)
        # print("Type:", type(logits)) 

        pred_idx = torch.argmax(logits, dim=1).item()
        confidence = F.softmax(logits, dim=1)[0][pred_idx].item()

        # 5. 역정규화 (Denormalization)
        ssim_val = p_ssim_z.item()
        lpips_val = p_lpips_z.item()

        if self.stats:
            ssim_val = (ssim_val * self.stats['ssim_std']) + self.stats['ssim_mean']
            lpips_val = (lpips_val * self.stats['lpips_std']) + self.stats['lpips_mean']
            
            ssim_val = max(0.0, min(1.0, ssim_val))
            lpips_val = max(0.0, lpips_val)

        # 6. Grad-CAM 생성
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

        # 7. RM / PVR 계산
        rm, pvr, _ = self._calc_rm_pvr(pil_image, k=3.0)

        return {
            "label": pred_idx,
            "confidence": confidence,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "rm": rm,
            "pvr": pvr,
            "heatmap": visualization
        }