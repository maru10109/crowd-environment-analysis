import streamlit as st
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="유동인구와 미세 기후 변화 분석", page_icon="🌡️", layout="wide")

# ---------------------------------------------------------------
# 1. API 키 (Streamlit Secrets: ENV_API_KEY, POP_API_KEY)
# ---------------------------------------------------------------
try:
    ENV_KEY = st.secrets["ENV_API_KEY"]
    POP_KEY = st.secrets["POP_API_KEY"]
except KeyError as e:
    st.error(f"필수 API 키가 없습니다: {e}")
    st.stop()

ENV_SERVICE = "IotVdata017"   # 환경정보 (SN, MSRMT_HR, CGG, AVG_TP ...)
POP_SERVICE = "IotVdata018"   # 유동인구 (SERIAL_NO, SENSING_TIME, VISITOR_COUNT ...)
PAGE = 1000                   # API 1회 최대 건수

# ---------------------------------------------------------------
# 2. API 함수
# ---------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def api_page(key, service, start, end):
    url = f"http://openapi.seoul.go.kr:8088/{key}/json/{service}/{start}/{end}/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    if service in j:
        total = int(j[service].get("list_total_count", 0))
        return total, pd.DataFrame(j[service].get("row", []))
    msg = j.get("RESULT", {}).get("MESSAGE", "알 수 없는 오류")
    raise RuntimeError(f"[{service}] {msg}")

def fetch_many(key, service, n_pages, from_end):
    """n_pages 페이지(1000건씩)를 한꺼번에 받아 합침. from_end=True면 뒤쪽부터."""
    total, _ = api_page(key, service, 1, 1)
    last_page = (total + PAGE - 1) // PAGE
    n_pages = min(n_pages, last_page)
    if from_end:
        pages = [last_page - i for i in range(n_pages)]
    else:
        pages = [1 + i for i in range(n_pages)]
    ranges = [((p - 1) * PAGE + 1, min(p * PAGE, total)) for p in pages]
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda r: api_page(key, service, r[0], r[1])[1], ranges))
    df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return total, df

# ---------------------------------------------------------------
# 3. 전처리
# ---------------------------------------------------------------
def prep_env(df):
    df = df.copy()
    df["MSRMT_HR"] = pd.to_datetime(df["MSRMT_HR"].str.replace("_", " "), errors="coerce")
    for c in ["AVG_TP", "AVG_HUM", "AVG_NIS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 비정상 센서값 제거 (예: 온도 -40.0)
    df.loc[(df["AVG_TP"] <= -30) | (df["AVG_TP"] >= 60), "AVG_TP"] = pd.NA
    df.loc[(df["AVG_HUM"] < 0) | (df["AVG_HUM"] > 100), "AVG_HUM"] = pd.NA
    return df.dropna(subset=["MSRMT_HR"])

# ---------------------------------------------------------------
# 4. 화면 / 사이드바
# ---------------------------------------------------------------
st.title("🚶‍♂️ 유동인구 밀집은 정말 미세 기후를 변화시킬까?")
st.caption("S-DoT 환경정보(온도·습도·소음)와 유동인구 데이터를 병합해 회귀 분석합니다.")

with st.sidebar:
    st.header("⚙️ 데이터 불러오기")
    n_pages = st.slider("불러올 페이지 수 (1페이지=1,000건)", 1, 30, 5)
    side = st.radio("가져올 위치", ["앞쪽(1번부터)", "뒤쪽(마지막부터)"],
                    help="어느 쪽이 최신 데이터인지는 불러온 뒤 '시간 범위'로 확인하세요.")
    load_btn = st.button("📥 API에서 데이터 불러오기", type="primary")

if load_btn:
    try:
        with st.spinner("API 호출 중..."):
            from_end = side.startswith("뒤쪽")
            env_total, env_raw = fetch_many(ENV_KEY, ENV_SERVICE, n_pages, from_end)
            pop_total, pop_raw = fetch_many(POP_KEY, POP_SERVICE, n_pages, from_end)
        st.session_state["env"] = prep_env(env_raw)
        st.session_state["pop"] = pop_raw
        st.session_state["totals"] = (env_total, pop_total)
    except Exception as e:
        st.error(f"API 호출 실패: {e}")

if "env" not in st.session_state:
    st.info("왼쪽 사이드바에서 **데이터 불러오기** 버튼을 눌러주세요.")
    st.stop()

df_env = st.session_state["env"]
df_pop = st.session_state["pop"]
env_total, pop_total = st.session_state["totals"]

# ---------------------------------------------------------------
# 5. 필터 (불러온 데이터 기준으로 선택지 생성)
# ---------------------------------------------------------------
tmin, tmax = df_env["MSRMT_HR"].min(), df_env["MSRMT_HR"].max()
st.success(f"환경 {len(df_env):,}건 로드 (전체 {env_total:,}건) · "
           f"유동인구 {len(df_pop):,}건 로드 (전체 {pop_total:,}건)")
st.write(f"📅 불러온 환경 데이터의 시간 범위: **{tmin} ~ {tmax}**")

with st.sidebar:
    st.header("🔎 필터")
    sel_date = st.date_input("날짜", value=tmax.date(), min_value=tmin.date(), max_value=tmax.date())
    districts = sorted(df_env["CGG"].dropna().unique())
    sel_districts = st.multiselect("자치구", districts, default=districts[:1])
    sel_hours = st.slider("시간대 (시)", 0, 23, (0, 23))

mask = (
    df_env["CGG"].isin(sel_districts)
    & (df_env["MSRMT_HR"].dt.date == sel_date)
    & (df_env["MSRMT_HR"].dt.hour.between(sel_hours[0], sel_hours[1]))
)
env_f = df_env[mask]

c1, c2, c3 = st.columns(3)
c1.metric("선택 자치구", ", ".join(sel_districts) or "-")
c2.metric("날짜/시간", f"{sel_date} {sel_hours[0]}~{sel_hours[1]}시")
c3.metric("필터된 환경 데이터", f"{len(env_f):,}건")

tab1, tab2 = st.tabs(["🍃 환경정보 (S-DoT)", "🚶 유동인구 (원본 확인용)"])
with tab1:
    st.dataframe(env_f[["SN", "MSRMT_HR", "CGG", "DONG", "AVG_TP", "AVG_HUM", "AVG_NIS"]],
                 use_container_width=True)
with tab2:
    st.write("컬럼:", list(df_pop.columns))   # 실제 컬럼명 확인용
    st.dataframe(df_pop.head(200), use_container_width=True)

st.markdown("### 📈 환경 데이터 시각화")
if env_f.empty:
    st.info("조건에 맞는 데이터가 없습니다. 위의 시간 범위 안에서 날짜를 고르거나, 불러올 페이지 수를 늘려보세요.")
else:
    metric = st.selectbox("지표", ["AVG_TP", "AVG_HUM", "AVG_NIS"])
    chart = (env_f.dropna(subset=[metric])
             .groupby("MSRMT_HR")[metric].mean())   # 시각별 센서 평균
    if chart.empty:
        st.info("이 지표는 선택 조건에서 유효한 값이 없습니다 (소음 센서가 없는 지역일 수 있어요).")
    else:
        st.line_chart(chart)
