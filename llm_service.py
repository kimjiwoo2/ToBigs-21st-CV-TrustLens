import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SolarLLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SOLAR_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar"
        )
        self.risk_desc = ["안전(Original)", "주의(Low)", "경고(Medium)", "위험(High)"]

    def generate_report(self, vit_results):
        prompt = f"""
        당신은 이미지 위변조 분석 전문가입니다. 아래 데이터를 바탕으로 리포트를 작성하세요.
        - 등급: {self.risk_desc[vit_results['label']]}
        - SSIM: {vit_results['ssim']:.4f}
        - Strength: {vit_results['strength']:.2f}
        
        전문적이면서 일반 사용자도 이해하기 쉽게 설명해 주세요.
        """
        
        response = self.client.chat.completions.create(
            model="solar-pro2",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class GPTLLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.risk_desc = ["안전(Original)", "주의(Low)", "경고(Medium)", "위험(High)"]

    def generate_report(self, vit_results):
        prompt = f"""
        당신은 디지털 포렌식 및 이미지 위변조 분석 전문가입니다. 
        Vision Transformer(ViT) 기반 탐지 모델이 분석한 아래 데이터를 바탕으로 전문적인 판독 리포트를 작성하세요.

        [분석 데이터]
        - 종합 판정: {self.risk_desc[vit_results['label']]}
        - SSIM(구조적 유사도): {vit_results['ssim']:.4f} (1에 가까울수록 원본과 구조가 동일)
        - LPIPS(지각적 유사도): {vit_results['lpips']:.4f} (0에 가까울수록 인간이 보기에 자연스러움)
        - 변형 강도(Strength): {vit_results['strength']:.2f}
        
        [시각적 증거]
        - 현재 분석 화면에는 'Attention Heatmap'이 함께 표시되고 있습니다. 
        - 히트맵의 붉은색 영역은 모델이 픽셀 불연속성이나 인위적인 노이즈를 포착한 지점입니다.

        [리포트 작성 가이드]
        1. 등급에 따른 현재 상태를 명확히 정의하세요.
        2. SSIM과 LPIPS 수치를 결합하여 왜 이런 등급이 나왔는지 논리적으로 설명하세요.
        3. 사용자가 함께 보고 있는 '히트맵'의 붉은 영역을 어떻게 해석해야 하는지 가이드(예: 경계면 합성 의심 등)를 포함하세요.
        4. 마크다운(Markdown) 형식을 사용하여 가독성 있게 작성하세요.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "당신은 신뢰할 수 있는 디지털 포렌식 분석가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7 # 리포트의 전문성과 창의성 균형
        )
        return response.choices[0].message.content