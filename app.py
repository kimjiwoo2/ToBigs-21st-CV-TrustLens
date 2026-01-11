import streamlit as st
from PIL import Image
from inference import ViTInference
from llm_service import SolarLLMService
from llm_service import GPTLLMService

# 1. 초기 설정 및 서비스 로드
st.set_page_config(page_title="AI 생성 이미지 판별 시스템", layout="wide")

@st.cache_resource
def init_services():
    vit_engine = ViTInference(model_path="./models/best_model")
    # llm_engine = SolarLLMService()
    llm_engine = GPTLLMService()
    return vit_engine, llm_engine

vit_engine, llm_engine = init_services()

# 2. UI 레이아웃
st.title("AI 이미지 보안 진단 서비스")
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, width="stretch", caption="업로드된 이미지")
        
        btn = st.button("진단 시작")
    
    if btn:
        # 1. 상태 표시를 위한 placeholder
        status_text = st.empty()
        
        try:
            # 2. ViT 모델 추론 (로컬 연산)
            status_text.info("🔍 1단계: ViT 모델이 이미지를 정밀 분석 중입니다...")
            results = vit_engine.predict(image)
            
            # 터미널 로그 출력 (VS Code 터미널에서 확인 가능)
            print(f"ViT Inference Result: {results}")
            st.success("✅ 1단계: 이미지 특성 추출 완료")

            # 3. GPT 리포트 생성 (API 호출)
            status_text.info("✍️ 2단계: LLM 모델이 분석 리포트를 작성 중입니다...")
            
            try:
                # llm_engine이 이제 GPTLLMService 인스턴스여야 함
                report = llm_engine.generate_report(results)
                status_text.empty() # 로딩 메시지 제거
            except Exception as e:
                report = f"⚠️ GPT API 호출 실패: {str(e)}"
                st.error("리포트 생성 중 문제가 발생했습니다. API 키와 잔액을 확인하세요.")

            # 4. 화면 결과 출력 (오른쪽 컬럼 col2)
            with col2:
                st.subheader("📊 데이터 분석 요약")
                
                # 가독성을 위해 메트릭 형태로 표시
                m1, m2 = st.columns(2)
                risk_labels = ["안전", "주의", "경고", "위험"]
                m1.metric("위험 등급", risk_labels[results['label']])
                m2.metric("SSIM 유사도", f"{results['ssim']:.4f}")
                
                st.write(f"**LPIPS:** {results['lpips']:.4f} | **변형 강도:** {results['strength']:.2f}")
                
                st.divider()
                st.subheader("📝 전문가 분석 리포트")
                st.markdown(report) # GPT가 주는 마크다운 형식을 그대로 살림

        except Exception as e:
            # ViT 추론 자체가 실패한 경우 (모델 로드 문제 등)
            st.error(f"🚨 시스템 오류 발생: {str(e)}")
            print(f"CRITICAL ERROR: {str(e)}")