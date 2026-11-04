import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="인파기상청", page_icon="🌡️", layout="wide")

# ---------------- API 키 (Streamlit Secrets) ----------------
try:
    ENV_KEY = st.secrets["ENV_API_KEY"]
    POP_KEY = st.secrets["POP_API_KEY"]
except KeyError as e:
    st.error(f"필수 API 키가 없습니다: {e}")
    st.stop()

ENV_SERVICE, POP_SERVICE, PAGE = "IotVdata017", "IotVdata018", 1000
MAX_DAYS = 14

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

# 화면 표시용 한글 이름
KO = {
    "MDL_NO": "모델번호", "SN": "시리얼", "MSRMT_HR": "측정시간", "RGN": "지역유형",
    "CGG": "자치구", "DONG": "행정동", "AVG_TP": "온도 평균(℃)", "AVG_HUM": "습도 평균(%)",
    "AVG_NIS": "소음 평균(dB)", "MODEL_NM": "모델명", "SERIAL_NO": "시리얼",
    "SENSING_TIME": "측정시간", "REGION": "지역유형", "AUTONOMOUS_DISTRICT": "자치구",
    "ADMINISTRATIVE_DISTRICT": "행정동", "VISITOR_COUNT": "방문자 수(명)",
    "REG_DTTM": "등록일시", "REG_DT": "등록일시", "HOUR": "시간(1시간 단위)",
    "VISITOR_MEAN": "평균 방문자 수(명)",
}
ENV_COLS = ["SN", "MSRMT_HR", "RGN", "CGG", "DONG", "AVG_TP", "AVG_HUM", "AVG_NIS"]
METRICS = {"AVG_TP": "온도(℃)", "AVG_HUM": "습도(%)", "AVG_NIS": "소음(dB)"}


def ko(df):
    return df.rename(columns=KO)


# ---------------- API 호출 (하루 단위로 호출해서 합침) ----------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_day(key, service, gu, day, max_pages):
    rows, start = [], 1
    for _ in range(max_pages):
        end = start + PAGE - 1
        url = f"http://openapi.seoul.go.kr:8088/{key}/json/{service}/{start}/{end}/"
        if gu and day:
            url += f"{gu}/{day}"
        try:
            j = requests.get(url, timeout=30).json()
        except Exception:
            raise RuntimeError(f"{service} 호출 실패(네트워크 또는 응답 형식 오류)")  # URL(키) 노출 방지
        if service not in j:
            res = j.get("RESULT", {})
            if res.get("CODE") == "INFO-200":
                break
            raise RuntimeError(f"{service}: {res.get('MESSAGE', '알 수 없는 오류')}")
        block = j[service]
        rows += block.get("row", [])
        if end >= int(block.get("list_total_count", 0)):
            break
        start = end + 1
    return pd.DataFrame(rows)


def fetch_range(key, service, gu, days, max_pages):
    def one(d):
        try:
            return fetch_day(key, service, gu, d, max_pages)
        except RuntimeError:
            return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=4) as ex:
        parts = list(ex.map(one, days))
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------- 전처리 ----------------
def clean_env(raw):
    """온도·습도·소음 이상값을 결측으로 바꾸고, 시간 단위로 집계. 처리 기록도 반환."""
    df = raw.replace("", np.nan).copy()
    df["time"] = pd.to_datetime(df["MSRMT_HR"].str.replace("_", " "), errors="coerce")
    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log = []
    rules = {
        "AVG_TP": (df["AVG_TP"] <= -30) | (df["AVG_TP"] >= 60),
        "AVG_HUM": (df["AVG_HUM"] < 0) | (df["AVG_HUM"] > 100),
        "AVG_NIS": df["AVG_NIS"] <= 0,
    }
    rule_text = {"AVG_TP": "-30℃ 이하 또는 60℃ 이상", "AVG_HUM": "0% 미만 또는 100% 초과",
                 "AVG_NIS": "0dB 이하"}
    for c, bad in rules.items():
        miss_before = int(df[c].isna().sum())
        n_bad = int(bad.sum())
        df.loc[bad, c] = np.nan
        log.append({"열": METRICS[c], "원래 빈 값": miss_before, "이상치 기준": rule_text[c],
                    "이상치 개수(빈 값 처리)": n_bad, "처리 후 빈 값": int(df[c].isna().sum())})
    df["HOUR"] = df["time"].dt.floor("h")
    agg = (df.dropna(subset=["HOUR"])
             .groupby(["CGG", "DONG", "HOUR"], as_index=False)[list(METRICS)].mean())
    return agg, pd.DataFrame(log)


def clean_pop(raw):
    df = raw.replace("", np.nan).copy()
    df["time"] = pd.to_datetime(df["SENSING_TIME"].str.replace("_", " "), errors="coerce")
    df["VISITOR_COUNT"] = pd.to_numeric(df["VISITOR_COUNT"], errors="coerce")
    n_neg = int((df["VISITOR_COUNT"] < 0).sum())
    df.loc[df["VISITOR_COUNT"] < 0, "VISITOR_COUNT"] = np.nan
    df["HOUR"] = df["time"].dt.floor("h")
    df = df.rename(columns={"AUTONOMOUS_DISTRICT": "CGG", "ADMINISTRATIVE_DISTRICT": "DONG"})
    agg = (df.dropna(subset=["HOUR"]).groupby(["CGG", "DONG", "HOUR"], as_index=False)["VISITOR_COUNT"]
             .mean().rename(columns={"VISITOR_COUNT": "VISITOR_MEAN"}))
    return agg, n_neg


def overview(df):
    d = df.replace("", np.nan)
    return pd.DataFrame({"열 이름": [KO.get(c, c) for c in d.columns],
                         "자료형": [str(t) for t in d.dtypes],
                         "빈 값 개수": d.isna().sum().values,
                         "빈 값 비율(%)": (d.isna().mean() * 100).round(1).values})


# ---------------- 사이드바 ----------------
st.title("🚶 인파기상청")
st.caption("유동인구가 밀집하면 그 지역의 온도·습도·소음은 정말 변하는가")

with st.sidebar:
    st.header("⚙️ 조회 조건")
    ko_gu = st.selectbox("자치구", list(DISTRICTS), index=list(DISTRICTS).index("마포구"))
    today = pd.Timestamp.today().date()
    rng = st.date_input("날짜 범위 (시작일 ~ 종료일)",
                        value=(today - pd.Timedelta(days=4), today - pd.Timedelta(days=1)),
                        max_value=today)
    st.caption(f"최대 {MAX_DAYS}일까지 조회할 수 있습니다.")
    btn = st.button("📥 API에서 데이터 불러오기", type="primary")

if btn:
    if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
        st.sidebar.error("달력에서 시작일과 종료일을 모두 선택하세요.")
        st.stop()
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(rng[0], rng[1])]
    if len(days) > MAX_DAYS:
        st.sidebar.error(f"{MAX_DAYS}일 이하로 선택하세요. (현재 {len(days)}일)")
        st.stop()
    gu, notice = DISTRICTS[ko_gu], ""
    try:
        with st.spinner(f"환경정보 API 호출 중 ({len(days)}일)..."):
            env_raw = fetch_range(ENV_KEY, ENV_SERVICE, gu, days, 5)
        with st.spinner(f"유동인구 API 호출 중 ({len(days)}일)..."):
            pop_raw = fetch_range(POP_KEY, POP_SERVICE, gu, days, 5)
            if pop_raw.empty:
                latest = fetch_day(POP_KEY, POP_SERVICE, None, None, 15)
                pop_raw = latest[(latest["AUTONOMOUS_DISTRICT"] == gu) &
                                 (latest["SENSING_TIME"].str[:10].isin(days))]
                notice = ("유동인구는 서버 필터가 적용되지 않아 최신 데이터에서 골랐습니다. "
                          "최근 날짜가 아니면 비어 있을 수 있습니다.")
        st.session_state.update(env_raw=env_raw, pop_raw=pop_raw, notice=notice,
                                cond=(ko_gu, days[0], days[-1], len(days)))
    except Exception as e:
        st.error(f"API 오류: {e}")
        st.stop()

if "env_raw" not in st.session_state:
    st.info("왼쪽에서 자치구와 날짜 범위를 고르고 **데이터 불러오기**를 눌러주세요.")
    st.stop()

env_raw, pop_raw = st.session_state["env_raw"], st.session_state["pop_raw"]
gu_name, d0, d1, n_days = st.session_state["cond"]
if st.session_state["notice"]:
    st.warning(st.session_state["notice"])
if env_raw.empty or pop_raw.empty:
    st.error("한쪽 데이터가 비어 있습니다. 다른 날짜나 자치구를 골라보세요.")
    st.stop()

env, env_log = clean_env(env_raw)
pop, n_neg = clean_pop(pop_raw)
merged = pd.merge(pop, env, on=["CGG", "DONG", "HOUR"], how="inner")

c1, c2, c3, c4 = st.columns(4)
c1.metric("조회 조건", f"{gu_name}")
c2.metric("기간", f"{d0} ~ {d1}", f"{n_days}일", delta_color="off")
c3.metric("환경정보 원본", f"{len(env_raw):,}행")
c4.metric("유동인구 원본", f"{len(pop_raw):,}행")

tab_ov, tab_clean, tab_graph, tab_raw = st.tabs(
    ["📋 데이터 개요", "🧹 결측치·이상치 처리", "📈 그래프와 발견", "🔗 병합·원본 데이터"])

# ---------------- 1. 데이터 개요 ----------------
with tab_ov:
    st.subheader("환경정보 (IotVdata017)")
    st.write(f"**{len(env_raw):,}행 × {env_raw.shape[1]}열**")
    st.dataframe(overview(env_raw), use_container_width=True, hide_index=True)
    st.subheader("유동인구 (IotVdata018)")
    st.write(f"**{len(pop_raw):,}행 × {pop_raw.shape[1]}열**")
    st.dataframe(overview(pop_raw), use_container_width=True, hide_index=True)
    st.caption("API가 빈 값을 빈 문자열로 주기 때문에, 빈 문자열을 빈 값으로 바꿔서 센 결과입니다. "
               "자료형은 API에서 받은 그대로(object=문자열)입니다.")

# ---------------- 2. 처리 기록 ----------------
final = merged.dropna(subset=["VISITOR_MEAN", *METRICS])
with tab_clean:
    st.subheader("처리 방법")
    st.markdown(
        "- **이상치**: 센서 오류로 보이는 값(예: 온도 -40.0℃)은 지우지 않고 **빈 값으로 바꿔서** 해당 열 계산에서만 제외\n"
        "- **결측치**: 평균 등으로 채우지 않음(없는 값을 만들지 않기 위해). 분석에 필요한 열이 비면 그 행만 제외\n"
        "- **시간 맞추기**: 두 데이터의 측정 분(分)이 달라 1시간 단위로 묶어 평균 → 자치구·행정동·시간으로 병합")
    st.dataframe(env_log, use_container_width=True, hide_index=True)
    st.caption(f"유동인구에서 음수 방문자 수: {n_neg}건 (빈 값 처리)")
    st.subheader("처리 전후 행 수")
    steps = pd.DataFrame({
        "단계": ["환경정보 원본", "환경정보 시간 단위 집계 후", "유동인구 원본",
                "유동인구 시간 단위 집계 후", "두 데이터 병합 후", "분석에 쓰는 행(4개 값 모두 있음)"],
        "행 수": [len(env_raw), len(env), len(pop_raw), len(pop), len(merged), len(final)]})
    st.dataframe(steps, use_container_width=True, hide_index=True)

# ---------------- 3. 그래프 ----------------
with tab_graph:
    if len(final) < 5:
        st.info("분석에 쓸 수 있는 행이 5개 미만입니다. 기간을 넓히거나 다른 자치구를 골라보세요.")
    else:
        st.caption(f"{gu_name} · {d0} ~ {d1} · 분석 행 {len(final):,}개 (행정동 × 시간)")

        # 그래프 1: 방문자 수 vs 환경 지표 산점도 + 추세선
        y = st.selectbox("그래프 1에서 볼 환경 지표", list(METRICS), format_func=METRICS.get)
        p = final.dropna(subset=[y])
        x = p["VISITOR_MEAN"]
        r = x.corr(p[y])
        slope, icpt = np.polyfit(x, p[y], 1)
        fig1 = px.scatter(p, x="VISITOR_MEAN", y=y, opacity=0.5,
                          labels={"VISITOR_MEAN": "평균 방문자 수 (명)", y: METRICS[y]},
                          title=f"그래프 1. 방문자 수와 {METRICS[y]}의 관계")
        xs = np.array([x.min(), x.max()])
        fig1.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", name="추세선",
                                  line=dict(color="red")))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"**찾아낸 것:** 방문자 수와 {METRICS[y]}의 상관계수는 **{r:.2f}**, "
                    f"방문자가 10명 늘 때 평균 **{slope*10:+.2f}** 변합니다. (n={len(p):,})")

        # 그래프 2: 시각대별 평균 방문자 수와 소음
        h = final.copy()
        h["시각"] = h["HOUR"].dt.hour
        hh = h.groupby("시각")[["VISITOR_MEAN", "AVG_NIS", "AVG_TP"]].mean().reset_index()
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=hh["시각"], y=hh["VISITOR_MEAN"], name="평균 방문자 수(명)",
                              opacity=0.5), secondary_y=False)
        fig2.add_trace(go.Scatter(x=hh["시각"], y=hh["AVG_NIS"], name="평균 소음(dB)",
                                  mode="lines+markers"), secondary_y=True)
        fig2.update_layout(title="그래프 2. 하루 중 시각별 방문자 수와 소음")
        fig2.update_xaxes(title_text="하루 중 시각 (시)", dtick=2)
        fig2.update_yaxes(title_text="평균 방문자 수 (명)", secondary_y=False)
        fig2.update_yaxes(title_text="평균 소음 (dB)", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)
        pk = hh.loc[hh["VISITOR_MEAN"].idxmax()]
        lo = hh.loc[hh["VISITOR_MEAN"].idxmin()]
        r2 = hh["VISITOR_MEAN"].corr(hh["AVG_NIS"])
        st.markdown(f"**찾아낸 것:** 방문자 수는 **{int(pk['시각'])}시**에 평균 {pk['VISITOR_MEAN']:.1f}명으로 "
                    f"가장 많고 **{int(lo['시각'])}시**에 {lo['VISITOR_MEAN']:.1f}명으로 가장 적습니다. "
                    f"시각별 평균으로 본 방문자 수와 소음의 상관계수는 **{r2:.2f}**입니다.")

        # 그래프 3: 방문자 수 구간별 환경 지표 상자그림
        q = final.dropna(subset=[y]).copy()
        try:
            q["구간"] = pd.qcut(q["VISITOR_MEAN"], 4, labels=["적음(하위25%)", "보통 이하", "보통 이상", "많음(상위25%)"])
            fig3 = px.box(q, x="구간", y=y,
                          labels={"구간": "방문자 수 구간", y: METRICS[y]},
                          title=f"그래프 3. 방문자 수 구간별 {METRICS[y]} 분포")
            st.plotly_chart(fig3, use_container_width=True)
            m = q.groupby("구간", observed=True)[y].mean()
            st.markdown(f"**찾아낸 것:** 방문자가 가장 적은 구간의 {METRICS[y]} 평균은 **{m.iloc[0]:.1f}**, "
                        f"가장 많은 구간은 **{m.iloc[-1]:.1f}** (차이 {m.iloc[-1]-m.iloc[0]:+.1f}).")
        except ValueError:
            st.info("방문자 수 값이 너무 비슷해 4개 구간으로 나눌 수 없습니다.")

        st.warning("상관은 인과가 아닙니다. 낮에는 사람도 많고 기온도 높아서 시간대 효과가 섞여 있습니다. "
                   "'찾아낸 것' 문장은 숫자를 자동 계산한 것이므로, 해석은 팀이 직접 써야 합니다.")

# ---------------- 4. 병합·원본 ----------------
with tab_raw:
    st.subheader("병합 데이터")
    st.dataframe(ko(merged), use_container_width=True)
    st.subheader("환경정보 원본")
    st.dataframe(ko(env_raw[ENV_COLS]), use_container_width=True)
    st.subheader("유동인구 원본")
    st.dataframe(ko(pop_raw), use_container_width=True)
