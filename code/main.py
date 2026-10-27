import streamlit as st
import pandas as pd
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="유동인구와 미세 기후 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. API 키 설정 (Streamlit Secrets 활용)
# -----------------------------------------------------------------------------
# Streamlit Cloud 환경에서는 앱 설정의 'Secrets' 메뉴에 아래 키를 입력해야 합니다.
# 로컬 환경에서는 프로젝트 폴더 내 .streamlit/secrets.toml 파일에 아래와 같이 저장합니다.
# [secrets]
# ENV_API_KEY = "여러분의_환경정보_API_키"
# POP_API_KEY = "여러분의_유동인구_API_키"

def load_api_keys():
    try:
        # 시크릿 변수명 명시: 환경정보 API, 유동인구 API
        env_key = st.secrets["ENV_API_KEY"]
        pop_key = st.secrets["POP_API_KEY"]
        return env_key, pop_key
    except KeyError as e:
        st.error(f"⚠️ 필수 API 키가 누락되었습니다: {e}")
        st.info("Streamlit Secrets 설정(.streamlit/secrets.toml)을 확인해 주세요.")
        st.stop() # 키가 없으면 앱 실행을 중단합니다.

env_api_key, pop_api_key = load_api_keys()

# -----------------------------------------------------------------------------
# 2. 메인 화면 구성
# -----------------------------------------------------------------------------
st.title("🚶‍♂️ 유동인구 밀집은 정말 미세 기후를 변화시킬까?")
st.markdown("""
이 대시보드는 **서울시 유동인구 데이터**와 **S-DoT 환경정보 데이터(온도, 습도, 소음)**를 
병합하여, 사람들의 밀집도가 국지적인 미세 기후에 미치는 영향을 회귀 분석을 통해 살펴봅니다.
""")

st.divider()

# 사이드바 구성
with st.sidebar:
    st.header("⚙️ 분석 설정")
    st.write("API Key 상태: **정상 로드됨 ✅**")
    st.caption("환경정보 및 유동인구 API 준비 완료")
    
    # 향후 구현할 필터링 옵션 예시
    selected_region = st.selectbox("분석 지역 선택", ["종로구", "강남구", "마포구"])
    selected_time = st.slider("시간대 필터링", 0, 23, (9, 18))

# -----------------------------------------------------------------------------
# 3. 데이터 로딩 및 분석 (Mock-up)
# -----------------------------------------------------------------------------
# 향후 실제 데이터를 병합하고 회귀 분석(Linear Regression) 모델을 적용할 공간입니다.
st.subheader("📊 데이터 탐색 및 회귀 분석 결과")
st.info("아직 실제 데이터가 연결되지 않았습니다. 현재는 기본 앱 뼈대만 출력됩니다.")

# 임시 데이터로 구조만 잡아두기
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="현재 선택된 지역", value=selected_region)
with col2:
    st.metric(label="분석 대상 시간", value=f"{selected_time[0]}시 ~ {selected_time[1]}시")
with col3:
    st.metric(label="핵심 분석 변수", value="방문자수 ➡️ 온도, 습도, 소음")

# TODO: 회귀 분석 결과 시각화 차트 추가 공간
st.write("---")
st.markdown("### 📈 (예정) 방문자수(VISITOR_COUNT) 변화에 따른 미세 기후 추세선")
st.line_chart(pd.DataFrame(np.random.randn(20, 2), columns=['방문자수(Scaled)', '온도 AVG_TP(Scaled)']))
