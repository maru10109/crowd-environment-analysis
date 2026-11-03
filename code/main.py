import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="인파기상청", page_icon="🌡️", layout="wide")

# ---------------- API 키 (Streamlit Secrets) ----------------
try:
    ENV_KEY = st.secrets["ENV_API_KEY"]
    POP_KEY = st.secrets["POP_API_KEY"]
except KeyError as e:
    st.error(f"필수 API 키가 없습니다: {e}")
    st.stop()

ENV_SERVICE = "IotVdata017"   # 환경정보
POP_SERVICE = "IotVdata018"   # 유동인구
PAGE = 1000

DISTRICTS = {
    "강남구": "Gangnam-gu", "강동구": "Gangdong-gu", "강북구": "Gangbuk-gu",
    "강서구": "Gangseo-gu", "관악구": "Gwanak-gu", "광진구": "Gwangjin-gu",
    "구로구": "Guro-gu", "금천구": "Geumcheon-gu", "노원구": "Nowon-gu",
    "도봉구": "Dobong-gu", "동대문구": "Dongdaemun-gu", "동작구": "Dongjak-gu",
    "마포구": "Mapo-gu", "서대문구": "Seodaemun-gu", "서초구": "Seocho-gu",
    "성동구": "Seongdong-gu", "성북구": "Seongbuk-gu", "송파구": "Songpa-gu",
    "양천구": "Yangcheon-gu", "영등포구": "Yeongdeungpo-gu", "용산구": "Yongsan-gu",
    "은평구": "Eunpyeong-gu", "종로구": "Jongno-gu", "중구": "Jung-gu",
    "중랑구": "Jungnang-gu",
}

# ---------------- API 호출 ----------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch(key, service, district, date, max_pages):
    """district/date가 있으면 서버에서 필터링, 없으면 최신순 전체에서 max_pages만큼."""
    rows, start = [], 1
    for _ in range(max_pages):
        end = start + PAGE - 1
        url = f"http://openapi.seoul.go.kr:8088/{key}/json/{service}/{start}/{end}/"
        if district and date:
            url += f"{district}/{date}"
        try:
            j = requests.get(url, timeout=30).json()
        except Exception:
            # 예외 메시지에 URL(=API 키)이 들어 있어서 숨긴다
            raise RuntimeError(f"{service} 호출 실패(네트워크 또는 응답 형식 오류)")
        if service not in j:
            res = j.get("RESULT", {})
            if res.get("CODE") == "INFO-200":      # 해당 조건 데이터 없음
                break
            raise RuntimeError(f"{service}: {res.get('MESSAGE', '알 수 없는 오류')}")
        block = j[service]
        rows += block.get("row", [])
        if end >= int(block.get("list_total_count", 0)):
            break
        start = end + 1
    return pd.DataFrame(rows)

# ---------------- 전처리 ----------------
def prep_env(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["MSRMT_HR"].str.replace("_", " "), errors="coerce")
    for c in ["AVG_TP", "AVG_HUM", "AVG_NIS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[(df["AVG_TP"] <= -30) | (df["AVG_TP"] >= 60), "AVG_TP"] = pd.NA   # -40.0 같은 오류값
    df.loc[(df["AVG_HUM"] < 0) | (df["AVG_HUM"] > 100), "AVG_HUM"] = pd.NA
    df.loc[df["AVG_NIS"] <= 0, "AVG_NIS"] = pd.NA
    df["HOUR"] = df["time"].dt.floor("h")
    return (df.dropna(subset=["HOUR"])
              .groupby(["CGG", "DONG", "HOUR"], as_index=False)[["AVG_TP", "AVG_HUM", "AVG_NIS"]].mean())

def prep_pop(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["SENSING_TIME"].str.replace("_", " "), errors="coerce")
    df["VISITOR_COUNT"] = pd.to_numeric(df["VISITOR_COUNT"], errors="coerce")
    df["HOUR"] = df["time"].dt.floor("h")
    df = df.rename(columns={"AUTONOMOUS_DISTRICT": "CGG", "ADMINISTRATIVE_DISTRICT": "DONG"})
    return (df.dropna(subset=["HOUR"])
              .groupby(["CGG", "DONG", "HOUR"], as_index=False)["VISITOR_COUNT"].mean()
              .rename(columns={"VISITOR_COUNT": "VISITOR_MEAN"}))

# ---------------- 사이드바 ----------------
st.title("🚶 인파기상청")
st.caption("유동인구가 밀집하면 그 지역의 온도·습도·소음은 정말 변하는가")

with st.sidebar:
    st.header("⚙️ 조회 조건")
    ko = st.selectbox("자치구", list(DISTRICTS.keys()), index=list(DISTRICTS).index("마포구"))
    date = st.date_input("날짜", value=pd.Timestamp.today().date() - pd.Timedelta(days=1))
    btn = st.button("📥 API에서 데이터 불러오기", type="primary")

if btn:
    gu, d = DISTRICTS[ko], date.strftime("%Y-%m-%d")
    notice = ""
    try:
        with st.spinner("환경정보 API 호출 중..."):
            env_raw = fetch(ENV_KEY, ENV_SERVICE, gu, d, 10)
        with st.spinner("유동인구 API 호출 중..."):
            try:
                pop_raw = fetch(POP_KEY, POP_SERVICE, gu, d, 10)
            except RuntimeError:
                pop_raw = pd.DataFrame()
            if pop_raw.empty:   # 서버 필터가 안 먹으면 최신 데이터에서 앱이 직접 거름
                pop_raw = fetch(POP_KEY, POP_SERVICE, None, None, 15)
                pop_raw = pop_raw[(pop_raw["AUTONOMOUS_DISTRICT"] == gu) &
                                  (pop_raw["SENSING_TIME"].str.startswith(d))]
                notice = "유동인구는 서버 필터가 적용되지 않아 최신 데이터 중에서 골랐습니다. 최근 날짜가 아니면 결과가 비어 있을 수 있습니다."
        st.session_state.update(env_raw=env_raw, pop_raw=pop_raw, notice=notice, cond=(ko, d))
    except Exception as e:
        st.error(f"API 오류: {e}")
        st.stop()

if "env_raw" not in st.session_state:
    st.info("왼쪽에서 자치구와 날짜를 고르고 **데이터 불러오기**를 눌러주세요.")
    st.stop()

env_raw, pop_raw = st.session_state["env_raw"], st.session_state["pop_raw"]
ko, d = st.session_state["cond"]
if st.session_state["notice"]:
    st.warning(st.session_state["notice"])

c1, c2, c3 = st.columns(3)
c1.metric("조회 조건", f"{ko} · {d}")
c2.metric("환경정보 원본", f"{len(env_raw):,}건")
c3.metric("유동인구 원본", f"{len(pop_raw):,}건")

if env_raw.empty or pop_raw.empty:
    st.error("한쪽 데이터가 비어 있습니다. 다른 날짜나 자치구를 골라보세요.")
    st.stop()

env, pop = prep_env(env_raw), prep_pop(pop_raw)
merged = pd.merge(pop, env, on=["CGG", "DONG", "HOUR"], how="inner")
st.success(f"병합 결과: {len(merged):,}행 (행정동 × 시간 단위, 자치구·행정동·시간 기준)")

if merged.empty:
    st.error("병합된 행이 없습니다. 두 데이터의 행정동 이름이나 시간이 겹치는지 아래 원본을 확인하세요.")

t1, t2, t3 = st.tabs(["🔗 병합 데이터", "🍃 환경정보 원본", "🚶 유동인구 원본"])
with t1:
    st.dataframe(merged, use_container_width=True)
with t2:
    st.dataframe(env_raw[["SN", "MSRMT_HR", "RGN", "CGG", "DONG", "AVG_TP", "AVG_HUM", "AVG_NIS"]],
                 use_container_width=True)
with t3:
    st.dataframe(pop_raw, use_container_width=True)

if not merged.empty:
    st.subheader("📈 유동인구와 환경 수치")
    y = st.selectbox("환경 지표", ["AVG_TP", "AVG_HUM", "AVG_NIS"],
                     format_func=lambda c: {"AVG_TP": "온도(℃)", "AVG_HUM": "습도(%)", "AVG_NIS": "소음(dB)"}[c])
    plot = merged.dropna(subset=["VISITOR_MEAN", y])
    if len(plot) >= 3:
        st.scatter_chart(plot, x="VISITOR_MEAN", y=y)
        st.write(f"상관계수(피어슨): **{plot['VISITOR_MEAN'].corr(plot[y]):.3f}** (n={len(plot)})")
        st.caption("상관은 인과가 아닙니다. 낮에는 사람도 많고 기온도 높아서, 시간대 효과가 섞여 있을 수 있습니다.")
    else:
        st.info("이 지표는 유효한 데이터가 부족합니다. 소음은 거리(main_street) 센서에만 있는 경우가 많습니다.")
