# TrustLens: 상품의 진짜를 보는 시선 

TrustLens는 이커머스 환경에서 생성형 AI로 만든 이미지를 탐지하고, 통계적 검증과 설명 가능한 AI(XAI)을 기반으로 하는 4단계 리스크 리포트를 제공하는 서비스입니다. 

기존의 생성 확률 제시를 넘어, 정량적 지표로 위험도를 진단하고 사용자에게 명확한 판단 근거를 제시하는 신뢰 프레임워크를 제안합니다. 



## WorkFlow 

1. **데이터셋 구축:**
    원본 이미지와 생성 모델 Stable Diffusion을 활용한 생성 이미지 데이터셋 구축.
2. **데이터 레이블링:**
    정량적 지표에 따른 K-Means Clustering으로 위험 단계 레이블링(Zone A~D) 및 유효성 검증.
3. **모델 학습:** 
    직접 생성한 데이터셋을 이용하여 Pre-trained ViT 계열의 Swin Transformer 모델 Fine-Tuning.
4. **결과 해석 및 시각화:** 
    위험 단계 분류에 대한 해석 및 시각적 근거 도출.
5. **데모 서비스 구현:**
    도출된 근거를 바탕으로 LLM을 통한 리스크 리포트 생성 및 Streamlit UI 구현.



## Key Contents

1. **Stable Diffusion을 활용한 AI 이미지 생성:**
    - Prompt 및 모델 파라미터 실험을 통해 이미지 생성.
    - 원본과 생성본을 통해 모든 이미지에 대해 정량적 지표(SSIM, LPIPS) 계산.
    - 총 원본(REAL) 3,797장과 생성(GEN) 12,920장으로 구성된 데이터셋 구축.
2. **Risk Zone 레이블링:**
    - 원본(REAL)은 Zone A로 고정.
    - 나머지 생성본(GEN)에 대해 K-Means Clustering을 적용하여 3단계의 리스크 존(Zone B~D) 정의.
    - 보조 지표(RM, PVR)을 활용하여 통계 검증(ANOVA 및 Tukey's HSD)을 통해 유의미함을 확인. 
3. **Swin Transformer 기반 탐지 모델:**
    - **Architecture**: 계층적 어텐션 메커니즘을 가진 Swin v2 모델을 채택하여 이미지 내 미세한 부자연스러움 포착.
    - **Multi-task Learning**: 성능 향상을 위해 SSIM, LPIPS 정량적 지표를 함께 예측하는 학습 전략 채택. 
    - **Optimization**: 데이터 불균형 처리, 데이터 증강, 모델 하이퍼파라미터 튜닝, 손실함수 재구성 등의 실험을 통해 최적 모델 학습. 
4. **설명 가능한 리포트 (XAI & LLM):**
    - 모델 추론 단계에서 Grad-CAM 및 AAR 지표를 도출하여, 통계적 검증 후 설명 제공. 
    - 분석된 지표를 바탕으로 프롬프트 엔지니어링 기반 LLM을 통해 리스크 리포트 생성. 



## Tech Stack
- **Language**: Python
- **Generative AI**: Stable Diffusion (Hugging Face)
- **Deep Learning**: PyTorch, Swin Transformer (Hugging Face), Grad-CAM
- **Data Analysis**: Pandas, Scikit-learn, Matplotlib, Statsmodels (ANOVA/Tukey HSD)
- **Service**: Streamlit, Solar Pro2 API
- **Tools**: Google Colab, Google Drive (Data Management), Notion, Git



## Directory
```
TrustLens
├── DATASET            # REAL & GEN 이미지 데이터 (드라이브 관리)
├── EVAL_METRIC        # RM, PVR, AAR, NED 추출 및 분석 스크립트
├── ZONE               # K-Means 및 ANOVA 통계 검증 결과
├── TRAIN              # Swin Transformer 모델 학습 및 가중치 저장 
├── STREAMLIT          # 데모 서비스 코드
│   ├── app.py         # 메인 UI
│   ├── inference.py   # 모델 추론 로직
│   └── llm_service.py # 리포트 생성 로직
└── requirements.txt   # 환경 설정
```
