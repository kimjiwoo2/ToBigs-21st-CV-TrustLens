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
st.title("AI 생성 이미지 탐지 서비스")
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 상단 섹션: 분석 실행 전 이미지 확인
    st.divider()
    
    if "results" not in st.session_state:
        # 분석 전: 업로드된 이미지를 중앙에 작게 표시
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.image(image, caption="분석 대기 중...", use_container_width=True)
            btn = st.button("🚀 정밀 진단 시작", use_container_width=True)
    else:
        btn = False # 이미 결과가 있으면 버튼 비활성화 (필요 시)

    if btn:
        status_text = st.empty()
        try:
            status_text.info("🔍 모델 분석 및 리포트 작성 중...")
            results = vit_engine.predict(image)
            report = llm_engine.generate_report(results)
            st.session_state.results = (results, report) # 세션에 저장하여 리런 방지
            status_text.empty()
        except Exception as e:
            st.error(f"🚨 오류 발생: {str(e)}")

    # 결과 출력 섹션: 분석이 완료된 경우에만 출력
    if "results" in st.session_state:
        results, report = st.session_state.results
        
        # 1. 최상단: 핵심 지표 (Metric Cards)
        st.subheader("📊 종합 분석 대시보드")
        m1, m2, m3, m4 = st.columns(4)
        risk_labels = ["안전", "주의", "경고", "위험"]
        
        # 등급에 따른 색상 강조 (단순 텍스트)
        m1.metric("위험 등급", risk_labels[results['label']])
        m2.metric("SSIM (구조 유사도)", f"{results['ssim']:.4f}")
        m3.metric("LPIPS (지각 유사도)", f"{results['lpips']:.4f}")
        m4.metric("변형 강도", f"{results['strength']:.2f}")

        st.divider()

        # 2. 중간: 이미지 비교 (원본 vs 히트맵)
        st.subheader("🔍 시각적 근거 비교")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(image, caption="원본 이미지", use_container_width=True)
        with img_col2:
            st.image(results['heatmap'], caption="Attention Heatmap (위변조 의심 영역)", use_container_width=True)

        st.divider()

        # 3. 하단: 상세 전문가 리포트
        st.subheader("📝 전문가 심층 분석 리포트")
        st.info("LLM 모델이 생성한 포렌식 결과입니다.")
        st.markdown(report)
        
        # 다시하기 버튼
        if st.button("🔄 다른 이미지 분석하기"):
            del st.session_state.results
            st.rerun()