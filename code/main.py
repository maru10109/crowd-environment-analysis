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
    
    # 필터링 옵션 (날짜, 지역, 시간)
    selected_date = st.date_input("날짜 선택", pd.to_datetime("today"))
    selected_region = st.selectbox("분석 지역 선택", ["은평구", "마포구", "종로구", "강남구", "서초구"])
    selected_time = st.slider("시간대 필터링 (시)", 0, 23, (9, 18))

# -----------------------------------------------------------------------------
# 3. 데이터 로딩 (서울시 공공데이터 API 연동)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600) # 1시간 동안 데이터를 캐싱하여 불필요한 API 호출 방지
def fetch_seoul_data(api_key, service_name, start_idx=1, end_idx=1000):
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

# 메인 화면 스피너 표시하며 데이터 로딩 (필터링을 위해 1000건 넉넉히 호출)
with st.spinner("서울시 API에서 실시간 데이터를 불러오는 중입니다..."):
    # IotVdata017: 환경정보, IotVdata018: 유동인구
    df_env = fetch_seoul_data(env_api_key, "IotVdata017", 1, 1000)
    df_pop = fetch_seoul_data(pop_api_key, "IotVdata018", 1, 1000)

# -----------------------------------------------------------------------------
# 4. 데이터 탐색 및 회귀 분석 결과
# -----------------------------------------------------------------------------
st.subheader("📊 데이터 탐색 및 회귀 분석 결과")

# 데이터가 모두 정상적으로 로드되었는지 확인
if not df_env.empty and not df_pop.empty:
    
    # 1. 환경 데이터 필터링 (CGG: 자치구, REG_DT: 등록일시)
    df_env['REG_DT'] = pd.to_datetime(df_env['REG_DT'], errors='coerce')
    mask_env = (
        (df_env['CGG'].str.contains(selected_region, na=False)) &
        (df_env['REG_DT'].dt.date == selected_date) &
        (df_env['REG_DT'].dt.hour >= selected_time[0]) &
        (df_env['REG_DT'].dt.hour <= selected_time[1])
    )
    df_env_filtered = df_env[mask_env].copy()

    # 2. 유동인구 데이터 필터링 (AUTONOMOUS_DISTRICT: 자치구, REG_DTTM: 등록일시)
    df_pop['REG_DTTM'] = pd.to_datetime(df_pop['REG_DTTM'], errors='coerce')
    mask_pop = (
        (df_pop['AUTONOMOUS_DISTRICT'].str.contains(selected_region, na=False)) &
        (df_pop['REG_DTTM'].dt.date == selected_date) &
        (df_pop['REG_DTTM'].dt.hour >= selected_time[0]) &
        (df_pop['REG_DTTM'].dt.hour <= selected_time[1])
    )
    df_pop_filtered = df_pop[mask_pop].copy()

    st.success("실제 데이터를 성공적으로 불러와 필터링을 적용했습니다! 🚀")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="현재 선택된 지역", value=selected_region)
    with col2:
        st.metric(label="분석 대상 날짜/시간", value=f"{selected_date} ({selected_time[0]}시~{selected_time[1]}시)")
    with col3:
        st.metric(label="필터링된 데이터 건수", value=f"환경 {len(df_env_filtered)}건 / 인구 {len(df_pop_filtered)}건")

    # 탭을 나누어 필터링된 원본 데이터 미리보기 제공
    tab1, tab2 = st.tabs(["🍃 환경정보 데이터 (S-DoT)", "🚶 유동인구 데이터"])
    with tab1:
        st.dataframe(df_env_filtered, use_container_width=True)
    with tab2:
        st.dataframe(df_pop_filtered, use_container_width=True)
        
    st.write("---")
    st.markdown("### 📈 환경 데이터 시각화 (온도, 습도, 소음)")
    
    if len(df_env_filtered) > 0:
        # 차트를 위해 문자열을 숫자(float)로 명시적 변환
        chart_data = df_env_filtered[['REG_DT', 'AVG_TP', 'AVG_HUM', 'AVG_NIS']].copy()
        chart_data['AVG_TP'] = pd.to_numeric(chart_data['AVG_TP'], errors='coerce')
        chart_data['AVG_HUM'] = pd.to_numeric(chart_data['AVG_HUM'], errors='coerce')
        chart_data['AVG_NIS'] = pd.to_numeric(chart_data['AVG_NIS'], errors='coerce')
        
        # 시간순 정렬 후 인덱스 설정
        chart_data = chart_data.sort_values('REG_DT').set_index('REG_DT').dropna()
        
        if not chart_data.empty:
            st.line_chart(chart_data)
        else:
            st.info("수치형 데이터로 변환할 수 있는 값이 부족합니다.")
    else:
        st.info("선택하신 조건(지역/날짜/시간)에 해당하는 데이터가 없습니다. 필터 조건을 넓혀보세요.")

else:
    st.warning("데이터를 불러오지 못했습니다. API 키가 정확한지 확인해 주세요.")
