import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RISK_TABLE = {
    0: {
        "name": "Real (원본)",
        "state": "Real",
        "zone": "Zone A",
        "traits": [
            "원본 이미지"
        ],
    },
    1: {
        "name": "Sleek Fake (눈속임형)",
        "state": "Low Risk",
        "zone": "Zone B",
        "traits": [
            "SSIM Low / LPIPS Low (자연스러움)",
            "RM 지표상 원본과 유사하나 PVR로 변별 가능"
        ],
    },
    2: {
        "name": "Noisy Fake (노이즈형)",
        "state": "Mid Risk",
        "zone": "Zone C",
        "traits": [
            "SSIM High / LPIPS High (어색함)",
            "RM 및 PVR 수치 급증"
        ],
    },
    3: {
        "name": "Failure (망가짐)",
        "state": "High Risk",
        "zone": "Zone D",
        "traits": [
            "SSIM Low / LPIPS High (붕괴)",
            "모든 물리적 지표 최악"
        ],
    },
}

class SolarLLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("SOLAR_API_KEY"),
            base_url="https://api.upstage.ai/v1/solar"
        )
        self.risk_desc = ["안전(Original)", "주의(Low Risk)", "경고(Medium Risk)", "위험(High Risk)"]
        self.risk_table = RISK_TABLE

    def generate_report(self, vit_results):

        info = self.risk_table[vit_results["label"]]

        prompt = f"""
        당신은 디지털 포렌식 분석 전문가입니다. 아래의 모델 분석 데이터를 바탕으로 판독 리포트를 한국어로 작성하세요.

        - Risk Label: {vit_results['label']}
        - 명칭(표 기준): {info['name']}
        - 상태(표 기준): {info['state']}
        - Zone(표 기준): {info['zone']}
        - 신뢰도: {vit_results['confidence']*100:.2f}%
        - SSIM: {vit_results['ssim']:.4f}
        - LPIPS: {vit_results['lpips']:.4f}
        - RM (잔차 평균): {vit_results['rm']:.6f}
        - PVR (강한 피크 비율): {vit_results['pvr']:.2f}% 

        ## [표 기준 물리적 특징]
        {chr(10).join([f"- {t}" for t in info["traits"]])}

        ## [리포트 작성 규칙]
        1. 형식: 반드시 Markdown 형식을 사용하고, 섹션별로 구분하십시오.
        2. 톤: 객관적이고 신뢰감 있는 전문가적 어조를 유지하십시오.
        3. 핵심 지표 해석:
        - SSIM이 낮고 LPIPS가 높은 경우: 픽셀 단위의 미세한 변조 가능성 강조.
        - SSIM은 높으나 확신도가 낮은 경우: 정밀한 합성 기술이 적용되었을 가능성 언급.
        - RM이 높을수록 고주파 잔차(합성/편집 흔적)가 전반적으로 강할 가능성.
        - PVR이 높을수록 매우 강한 잔차 피크가 넓게 분포(거친 노이즈/깨짐/강한 합성 흔적)했을 가능성.
        - RM/PVR은 통계적 신호이므로 단독으로 확정 판정하지 말고, SSIM/LPIPS/신뢰도/히트맵과 종합하여 표현하시오.
        4. 히트맵 가이드: 사용자가 보고 있는 히트맵의 '붉은색 영역'을 모델이 집중적으로 위변조 징후를 포착한 근거로 설명하십시오.
        
        ## [리포트 필수 포함 구성]
        ### 1. 종합 판정 결과
        (등급과 확신도를 결합하여 현재 이미지의 상태를 한 줄 요약)

        ### 2. 세부 지표 분석
        (SSIM과 LPIPS 수치를 근거로 한 기술적 해석)

        ### 3. 히트맵 해석
        (시각적 근거인 히트맵을 어떻게 보아야 하는지 설명)

        ## [출력 형식 강제]
        - 아래 4개 섹션 제목을 그대로 사용하고, 섹션은 추가하지 마시오:
        1) 종합 판정  2) 지표 근거  3) 히트맵 해석 4) 결론
        - 각 섹션은 정확히 3문장 이내로 작성

        ## [출력 길이 제한]
        - 전체 500자 이내
        - 섹션별 2~3문장 이내
        - 불릿은 섹션당 최대 3개
        """
        
        response = self.client.chat.completions.create(
            model="solar-pro",
            messages=[
                {"role": "system", "content": "이미지 포렌식 리포트 작성 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # 분석 리포트이므로 낮은 온도로 일관성 유지,
            max_tokens=500
        )
        return response.choices[0].message.content

# class GPTLLMService:
#     def __init__(self):
#         self.client = OpenAI(
#             api_key=os.getenv("OPENAI_API_KEY")
#         )
#         self.risk_desc = ["안전(Original)", "주의(Low)", "경고(Medium)", "위험(High)"]

#     def generate_report(self, vit_results):
#         prompt = f"""
#         당신은 디지털 포렌식 및 이미지 위변조 분석 전문가입니다. 
#         Vision Transformer(ViT) 기반 탐지 모델이 분석한 아래 데이터를 바탕으로 전문적인 판독 리포트를 작성하세요.

#         [분석 데이터]
#         - 종합 판정: {self.risk_desc[vit_results['label']]}
#         - SSIM(구조적 유사도): {vit_results['ssim']:.4f} (1에 가까울수록 원본과 구조가 동일)
#         - LPIPS(지각적 유사도): {vit_results['lpips']:.4f} (0에 가까울수록 인간이 보기에 자연스러움)
#         - 변형 강도(Strength): {vit_results['strength']:.2f}
        
#         [시각적 증거]
#         - 현재 분석 화면에는 'Attention Heatmap'이 함께 표시되고 있습니다. 
#         - 히트맵의 붉은색 영역은 모델이 픽셀 불연속성이나 인위적인 노이즈를 포착한 지점입니다.

#         [리포트 작성 가이드]
#         1. 등급에 따른 현재 상태를 명확히 정의하세요.
#         2. SSIM과 LPIPS 수치를 결합하여 왜 이런 등급이 나왔는지 논리적으로 설명하세요.
#         3. 사용자가 함께 보고 있는 '히트맵'의 붉은 영역을 어떻게 해석해야 하는지 가이드(예: 경계면 합성 의심 등)를 포함하세요.
#         4. 마크다운(Markdown) 형식을 사용하여 가독성 있게 작성하세요.
#         """
        
#         response = self.client.chat.completions.create(
#             model="gpt-4o", 
#             messages=[
#                 {"role": "system", "content": "당신은 신뢰할 수 있는 디지털 포렌식 분석가입니다."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7 # 리포트의 전문성과 창의성 균형
#         )
#         return response.choices[0].message.content