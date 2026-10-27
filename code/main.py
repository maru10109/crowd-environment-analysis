import streamlit as st
import pandas as pd
import numpy as np
import requests

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

    st.divider()
    st.markdown("**API 서비스명 설정**")
    st.caption("서울시 열린데이터광장 API 서비스명")
    env_service = st.text_input("환경정보 서비스명", value="IotVdata018")
    pop_service = st.text_input("유동인구 서비스명", value="SPOP_LOCAL_RESD_DONG")

# -----------------------------------------------------------------------------
# 3. 데이터 로딩 (서울시 공공데이터 API 연동)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600) # 1시간 동안 데이터를 캐싱하여 불필요한 API 호출 방지
def fetch_seoul_data(api_key, service_name, start_idx=1, end_idx=100):
    """서울시 열린데이터광장 API에서 JSON 데이터를 가져와 DataFrame으로 반환"""
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/{service_name}/{start_idx}/{end_idx}/"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # 정상적으로 데이터를 받아온 경우
        if service_name in data:
            return pd.DataFrame(data[service_name]['row'])
        else:
            # API 에러 메시지 출력
            error_msg = data.get('RESULT', {}).get('MESSAGE', '알 수 없는 에러 또는 잘못된 서비스명')
            st.error(f"[{service_name}] API 에러: {error_msg}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"[{service_name}] API 호출 중 오류 발생: {e}")
        return pd.DataFrame()

# 메인 화면 스피너 표시하며 데이터 로딩
with st.spinner("서울시 API에서 실시간 데이터를 불러오는 중입니다..."):
    df_env = fetch_seoul_data(env_api_key, env_service, 1, 100)
    df_pop = fetch_seoul_data(pop_api_key, pop_service, 1, 100)

# -----------------------------------------------------------------------------
# 4. 데이터 탐색 및 회귀 분석 결과
# -----------------------------------------------------------------------------
st.subheader("📊 데이터 탐색 및 회귀 분석 결과")

# 데이터가 모두 정상적으로 로드되었는지 확인
if not df_env.empty and not df_pop.empty:
    st.success("실제 데이터를 성공적으로 불러왔습니다! 🚀")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="현재 선택된 지역", value=selected_region)
    with col2:
        st.metric(label="분석 대상 시간", value=f"{selected_time[0]}시 ~ {selected_time[1]}시")
    with col3:
        st.metric(label="불러온 데이터 건수", value=f"환경 {len(df_env)}건 / 인구 {len(df_pop)}건")

    # 탭을 나누어 원본 데이터 미리보기 제공
    tab1, tab2 = st.tabs(["🍃 환경정보 데이터 (S-DoT)", "🚶 유동인구 데이터"])
    with tab1:
        st.dataframe(df_env.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df_pop.head(20), use_container_width=True)
        
    st.write("---")
    st.markdown("### 📈 데이터 시각화 및 상관관계 (진행 중)")
    st.caption("현재는 불러온 원본 데이터를 바탕으로 임시 차트를 렌더링합니다. (추후 병합 및 전처리 필요)")
    
    # 문자열로 들어온 수치형 데이터를 float으로 변환 시도 (차트를 그리기 위함)
    # 실제 사용하시는 API 규격(컬럼명)에 맞춰 추후 수정이 필요합니다.
    numeric_env = df_env.apply(pd.to_numeric, errors='ignore').select_dtypes(include=[np.number])
    
    if not numeric_env.empty:
        # 데이터의 편차가 커서 보기 어려울 수 있으므로 일부 컬럼만 선택하거나 정규화가 추후 필요합니다.
        st.line_chart(numeric_env)
    else:
        st.info("환경 데이터에서 즉시 시각화할 수치형 컬럼을 찾지 못했습니다. 전처리를 추가해주세요.")

else:
    st.warning("데이터를 불러오지 못했습니다. 사이드바의 **API 서비스명**이 맞는지 확인해 주세요.")
