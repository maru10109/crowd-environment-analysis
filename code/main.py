import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
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
MAP_PATH = "data/sensor_map.csv"

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
KO = {
    "MDL_NO": "모델번호", "SN": "시리얼(환경)", "MSRMT_HR": "측정시간", "RGN": "지역유형",
    "CGG": "자치구", "DONG": "행정동", "AVG_TP": "온도 평균(℃)", "AVG_HUM": "습도 평균(%)",
    "AVG_NIS": "소음 평균(dB)", "MODEL_NM": "모델명", "SERIAL_NO": "시리얼(유동인구)",
    "SENSING_TIME": "측정시간", "REGION": "지역유형", "AUTONOMOUS_DISTRICT": "자치구",
    "ADMINISTRATIVE_DISTRICT": "행정동", "VISITOR_COUNT": "방문자 수(명)",
    "REG_DTTM": "등록일시", "REG_DT": "등록일시", "HOUR": "시간(1시간 단위)",
    "VISITOR_MEAN": "평균 방문자 수(명)", "N_READ": "측정 횟수",
}
ENV_COLS = ["SN", "MSRMT_HR", "RGN", "CGG", "DONG", "AVG_TP", "AVG_HUM", "AVG_NIS"]
METRICS = {"AVG_TP": "온도(℃)", "AVG_HUM": "습도(%)", "AVG_NIS": "소음(dB)"}


def ko(df):
    return df.rename(columns=KO)


def norm_id(s):
    return s.astype(str).str.strip().str.lstrip("0").str.upper()


def show(df, n=1000):
    if len(df) > n:
        st.caption(f"전체 {len(df):,}행 중 앞 {n:,}행만 표시합니다.")
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)


# ---------------- API (페이지 단위 병렬) ----------------
_session = requests.Session()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_page(key, service, gu, day, start, end):
    """한 페이지(최대 1000건). (총 건수, 행 목록)을 반환. 조건에 맞는 데이터가 없으면 (0, [])."""
    url = f"http://openapi.seoul.go.kr:8088/{key}/json/{service}/{start}/{end}/"
    if gu and day:
        url += f"{gu}/{day}"
    try:
        j = _session.get(url, timeout=20).json()
    except Exception:
        raise RuntimeError(f"{service} 호출 실패(네트워크 또는 응답 형식 오류)")  # 키 노출 방지
    if service not in j:
        res = j.get("RESULT", {})
        if res.get("CODE") == "INFO-200":
            return 0, []
        raise RuntimeError(f"{service}: {res.get('MESSAGE', '알 수 없는 오류')}")
    block = j[service]
    return int(block.get("list_total_count", 0)), block.get("row", [])


def fetch_all(key, service, gu, days, max_pages, progress=None):
    """모든 날짜의 1페이지를 병렬로 받고, 남은 페이지도 병렬로 받아 합친다."""
    rows = []
    # 1단계: 날짜별 첫 페이지
    with ThreadPoolExecutor(max_workers=12) as ex:
        firsts = list(ex.map(lambda d: (d, fetch_safe(key, service, gu, d, 1, PAGE)), days))
    todo = []
    for d, (total, r) in firsts:
        rows += r
        last = min((total + PAGE - 1) // PAGE, max_pages)
        for pg in range(2, last + 1):
            todo.append((d, (pg - 1) * PAGE + 1, pg * PAGE))
    # 2단계: 나머지 페이지 전부 동시에
    if todo:
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = [ex.submit(fetch_safe, key, service, gu, d, s, e) for d, s, e in todo]
            for f in futs:
                rows += f.result()[1]
    return pd.DataFrame(rows)


def fetch_safe(key, service, gu, day, start, end):
    try:
        return fetch_page(key, service, gu, day, start, end)
    except RuntimeError:
        return 0, []

@st.cache_data(show_spinner=False)
def load_map():
    if not os.path.exists(MAP_PATH):
        return None
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(MAP_PATH, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return None


# ---------------- 전처리: 환경정보 ----------------
@st.cache_data(show_spinner="환경정보 전처리 중...")
def clean_env(raw, hum100_missing, nis35_missing):
    steps, n_raw = [], len(raw)

    def add(stage, col, rule, action, n_val, n_sn):
        steps.append({"단계": stage, "대상": col, "기준": rule, "처리": action,
                      "해당 값 수": int(n_val), "관련 센서 수": n_sn})

    df = raw.replace("", np.nan).copy()

    df["time"] = pd.to_datetime(df["MSRMT_HR"].astype(str).str.replace("_", " "), errors="coerce")
    bad_t = df["time"].isna()
    add("1. 시간", "측정시간", "날짜로 해석 불가", "행 제외", bad_t.sum(), df.loc[bad_t, "SN"].nunique())
    df = df[~bad_t]

    dup = df.duplicated(["SN", "MSRMT_HR"], keep="first")
    add("2. 중복", "시리얼+측정시간", "같은 센서·같은 시각이 2번 이상", "첫 행만 유지", dup.sum(), df.loc[dup, "SN"].nunique())
    df = df[~dup].copy()
    n_after_dup = len(df)

    for c in METRICS:
        before = df[c].notna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        add("3. 형식", METRICS[c], "숫자로 변환 불가", "빈 값 처리", before - df[c].notna().sum(), "-")

    miss0 = {c: int(df[c].isna().sum()) for c in METRICS}
    for c in METRICS:
        add("4. 결측", METRICS[c], "센서가 값을 주지 않음(빈 값)", "채우지 않고 그대로 둠(없는 값을 만들지 않음)",
            miss0[c], df.loc[df[c].isna(), "SN"].nunique())

    bad = (df["AVG_TP"] == -40) | (df["AVG_TP"] < -30) | (df["AVG_TP"] > 50)
    add("5. 이상치(범위)", "온도", "-40℃(센서 오류 기본값) 또는 -30℃ 미만·50℃ 초과", "빈 값 처리",
        bad.sum(), df.loc[bad, "SN"].nunique())
    df.loc[bad, "AVG_TP"] = np.nan

    bad = (df["AVG_HUM"] < 0) | (df["AVG_HUM"] > 100)
    add("5. 이상치(범위)", "습도", "0% 미만 또는 100% 초과", "빈 값 처리", bad.sum(), df.loc[bad, "SN"].nunique())
    df.loc[bad, "AVG_HUM"] = np.nan

    bad = df["AVG_NIS"] <= 0
    add("5. 이상치(범위)", "소음", "0dB 이하", "빈 값 처리", bad.sum(), df.loc[bad, "SN"].nunique())
    df.loc[bad, "AVG_NIS"] = np.nan

    m = df["AVG_HUM"] == 100
    add("6. 의심 값", "습도", "정확히 100%", "결측 처리" if hum100_missing else "유지(분석에 사용)",
        m.sum(), df.loc[m, "SN"].nunique())
    if hum100_missing:
        df.loc[m, "AVG_HUM"] = np.nan
    m = df["AVG_NIS"] == 35
    add("6. 의심 값", "소음", "정확히 35dB(센서 측정 하한일 가능성)", "결측 처리" if nis35_missing else "유지(분석에 사용)",
        m.sum(), df.loc[m, "SN"].nunique())
    if nis35_missing:
        df.loc[m, "AVG_NIS"] = np.nan

    df = df.sort_values(["SN", "time"])
    med = (df.groupby("SN")["AVG_TP"]
             .rolling(5, center=True, min_periods=3).median()
             .reset_index(level=0, drop=True).reindex(df.index))
    sp = (df["AVG_TP"] - med).abs() > 5
    add("7. 이상치(튐)", "온도", "같은 센서 앞뒤 5개 중앙값과 5℃ 초과 차이", "빈 값 처리", sp.sum(), df.loc[sp, "SN"].nunique())
    df.loc[sp, "AVG_TP"] = np.nan

    miss1 = {c: int(df[c].isna().sum()) for c in METRICS}

    df["HOUR"] = df["time"].dt.floor("h")
    g = df.groupby(["SN", "CGG", "DONG", "HOUR"], as_index=False)
    agg = g[list(METRICS)].mean()
    agg["N_READ"] = g.size()["size"].values
    add("8. 집계", "전체", "센서별 1시간 평균(두 데이터의 측정 분이 다름)", f"{len(df):,}행 → {len(agg):,}행", len(df), df["SN"].nunique())

    sensors = (df.groupby("SN")[list(METRICS)].apply(lambda x: x.notna().mean() * 100)
                 .round(1).rename(columns=METRICS).reset_index().rename(columns={"SN": "센서(환경)"}))
    stat = {"n_raw": n_raw, "n_dedup": n_after_dup, "n_agg": len(agg), "miss0": miss0, "miss1": miss1}
    return agg, pd.DataFrame(steps), sensors, stat


# ---------------- 전처리: 유동인구 ----------------
@st.cache_data(show_spinner="유동인구 전처리 중...")
def clean_pop(raw):
    steps = []
    df = raw.replace("", np.nan).copy()
    df["time"] = pd.to_datetime(df["SENSING_TIME"].astype(str).str.replace("_", " "), errors="coerce")
    bad_t = df["time"].isna()
    steps.append(("시간 해석 불가", "행 제외", int(bad_t.sum())))
    df = df[~bad_t]
    dup = df.duplicated(["SERIAL_NO", "SENSING_TIME"], keep="first")
    steps.append(("같은 센서·같은 시각 중복", "첫 행만 유지", int(dup.sum())))
    df = df[~dup].copy()
    df["VISITOR_COUNT"] = pd.to_numeric(df["VISITOR_COUNT"], errors="coerce")
    steps.append(("방문자 수 빈 값/숫자 아님", "그대로 빈 값", int(df["VISITOR_COUNT"].isna().sum())))
    neg = df["VISITOR_COUNT"] < 0
    steps.append(("방문자 수 음수", "빈 값 처리", int(neg.sum())))
    df.loc[neg, "VISITOR_COUNT"] = np.nan
    q999 = df["VISITOR_COUNT"].quantile(0.999)
    hi = df["VISITOR_COUNT"] > q999
    steps.append((f"방문자 수 상위 0.1%(>{q999:.0f}명)", "실제 인파일 수 있어 유지(확인만)", int(hi.sum())))
    df["HOUR"] = df["time"].dt.floor("h")
    df = df.rename(columns={"AUTONOMOUS_DISTRICT": "CGG", "ADMINISTRATIVE_DISTRICT": "DONG"})
    agg = (df.groupby(["SERIAL_NO", "CGG", "DONG", "HOUR"], as_index=False)["VISITOR_COUNT"]
             .mean().rename(columns={"VISITOR_COUNT": "VISITOR_MEAN"}))
    steps.append(("센서별 1시간 평균 집계", f"{len(df):,}행 → {len(agg):,}행", len(df)))
    return agg, pd.DataFrame(steps, columns=["처리 대상", "처리", "해당 행 수"])


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
                        value=(today - pd.Timedelta(days=3), today - pd.Timedelta(days=1)),
                        min_value=today - pd.Timedelta(days=31), max_value=today)
    st.caption(f"Open API는 최근 한 달 자료만 제공합니다. 최대 {MAX_DAYS}일까지 조회.")
    btn = st.button("📥 API에서 데이터 불러오기", type="primary")
    st.header("🧹 의심 값 처리(팀 결정)")
    hum100_missing = st.checkbox("습도 100%를 결측으로 처리", value=False)
    nis35_missing = st.checkbox("소음 35dB를 결측으로 처리", value=False)
    st.caption("체크를 바꿔 결과가 얼마나 달라지는지 비교하고 log에 기록하세요. "
               "온도 -40℃는 항상 결측 처리합니다.")

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
        bar = st.progress(0, text="환경정보 불러오는 중...")
        env_raw = fetch_all(ENV_KEY, ENV_SERVICE, gu, days, 5)
        bar.progress(50, text="유동인구 불러오는 중...")
        pop_raw = fetch_all(POP_KEY, POP_SERVICE, gu, days, 5)
        if pop_raw.empty:
            bar.progress(80, text="유동인구 서버 필터가 안 되어 최신 데이터에서 고르는 중...")
            _, first = fetch_page(POP_KEY, POP_SERVICE, None, None, 1, PAGE)
            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = [ex.submit(fetch_safe, POP_KEY, POP_SERVICE, None, None,
                                  (i - 1) * PAGE + 1, i * PAGE) for i in range(1, 7)]
                latest = pd.DataFrame([r for f in futs for r in f.result()[1]])
            if not latest.empty:
                pop_raw = latest[(latest["AUTONOMOUS_DISTRICT"] == gu) &
                                 (latest["SENSING_TIME"].str[:10].isin(days))]
            notice = "유동인구는 서버 필터가 적용되지 않아 최신 데이터에서 골랐습니다."
        bar.progress(100, text="완료")
        bar.empty()
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

env, env_log, env_sensor, es = clean_env(env_raw, hum100_missing, nis35_missing)
pop, pop_log = clean_pop(pop_raw)

# ---------------- 병합 ----------------
mp = load_map()
merge_info, unmatched = {}, []
if mp is not None:
    env_ids, pop_ids = set(norm_id(env["SN"])), set(norm_id(pop["SERIAL_NO"]))

    def best(ids):
        sc = {c: len(ids & set(norm_id(mp[c].dropna()))) for c in mp.columns}
        return max(sc, key=sc.get)

    with st.sidebar:
        with st.expander("🔗 센서코드 대응표 열 선택", expanded=False):
            c_env = st.selectbox("환경정보 시리얼에 해당하는 열", mp.columns, index=list(mp.columns).index(best(env_ids)))
            c_pop = st.selectbox("유동인구 시리얼에 해당하는 열", mp.columns, index=list(mp.columns).index(best(pop_ids)))
    cmap = mp[[c_env, c_pop]].dropna().drop_duplicates()
    cmap.columns = ["ENV_ID", "POP_ID"]
    cmap["ENV_ID"], cmap["POP_ID"] = norm_id(cmap["ENV_ID"]), norm_id(cmap["POP_ID"])
    e2, p2 = env.assign(KEY=norm_id(env["SN"])), pop.assign(POP_ID=norm_id(pop["SERIAL_NO"]))
    p2 = p2.merge(cmap, on="POP_ID", how="left")
    unmatched = sorted(set(p2.loc[p2["ENV_ID"].isna(), "SERIAL_NO"]))
    p2 = p2.dropna(subset=["ENV_ID"]).rename(columns={"ENV_ID": "KEY"})
    merged = pd.merge(p2[["KEY", "SERIAL_NO", "HOUR", "VISITOR_MEAN"]],
                      e2[["KEY", "SN", "CGG", "DONG", "HOUR", *METRICS]], on=["KEY", "HOUR"], how="inner")
    merged = merged.drop(columns="KEY")
    how_merge = "센서코드 대응표로 센서끼리 연결 + 1시간 단위 시간"
    merge_info = {"환경정보 센서 수": len(env_ids), "유동인구 센서 수": len(pop_ids),
                  "대응표로 이어진 유동인구 센서": len(pop_ids) - len(unmatched),
                  "이어지지 않은 유동인구 센서": len(unmatched)}
else:
    st.info("병합 방식: 자치구 + 행정동 + 1시간 단위")
    pa = pop.groupby(["CGG", "DONG", "HOUR"], as_index=False)["VISITOR_MEAN"].mean()
    ea = env.groupby(["CGG", "DONG", "HOUR"], as_index=False)[list(METRICS)].mean()
    merged = pd.merge(pa, ea, on=["CGG", "DONG", "HOUR"], how="inner")
    how_merge = "행정동 + 1시간 단위 시간(대체 방식)"

final = merged.dropna(subset=["VISITOR_MEAN", *METRICS])

c1, c2, c3, c4 = st.columns(4)
c1.metric("조회 조건", gu_name)
c2.metric("기간", f"{d0} ~ {d1}", f"{n_days}일", delta_color="off")
c3.metric("환경정보 원본", f"{len(env_raw):,}행")
c4.metric("유동인구 원본", f"{len(pop_raw):,}행")

tab_ov, tab_clean, tab_merge, tab_graph, tab_raw = st.tabs(
    ["📋 데이터 개요", "🧹 결측치·이상치 처리", "🔗 병합 기록", "📈 그래프와 발견", "🗂 데이터 보기"])

with tab_ov:
    st.subheader("환경정보 (IotVdata017)")
    st.write(f"**{len(env_raw):,}행 × {env_raw.shape[1]}열**")
    st.dataframe(overview(env_raw), use_container_width=True, hide_index=True)
    st.subheader("유동인구 (IotVdata018)")
    st.write(f"**{len(pop_raw):,}행 × {pop_raw.shape[1]}열**")
    st.dataframe(overview(pop_raw), use_container_width=True, hide_index=True)
    st.caption("API가 빈 값을 빈 문자열로 주므로 빈 값으로 바꿔 센 결과입니다. 자료형은 API 원본 그대로(object=문자열)입니다.")

with tab_clean:
    st.subheader("환경정보 처리 단계")
    st.dataframe(env_log, use_container_width=True, hide_index=True)
    st.subheader("분석 열별 빈 값 변화 (센서별 10분 단위 행 기준)")
    st.dataframe(pd.DataFrame({
        "열": list(METRICS.values()),
        "처리 전 빈 값": [es["miss0"][c] for c in METRICS],
        "처리 후 빈 값": [es["miss1"][c] for c in METRICS],
        "새로 빈 값이 된 수(이상치)": [es["miss1"][c] - es["miss0"][c] for c in METRICS],
        "처리 후 빈 값 비율(%)": [round(es["miss1"][c] / max(es["n_dedup"], 1) * 100, 1) for c in METRICS]}),
        use_container_width=True, hide_index=True)
    st.subheader("센서별 값이 있는 비율(%)")
    st.dataframe(env_sensor, use_container_width=True, hide_index=True)
    st.caption("소음 비율이 0%인 센서는 소음 센서가 달리지 않은 것입니다. 그 센서는 소음 분석에서 자동으로 빠집니다.")
    st.subheader("유동인구 처리 단계")
    st.dataframe(pop_log, use_container_width=True, hide_index=True)
    st.subheader("처리 전후 행 수")
    st.dataframe(pd.DataFrame({
        "단계": ["환경정보 원본", "중복·시간오류 제거 후", "센서별 1시간 집계 후",
                "유동인구 원본", "유동인구 센서별 1시간 집계 후",
                "병합 후", "분석에 쓰는 행(방문자·온도·습도·소음 모두 있음)"],
        "행 수": [es["n_raw"], es["n_dedup"], es["n_agg"], len(pop_raw), len(pop), len(merged), len(final)]}),
        use_container_width=True, hide_index=True)

with tab_merge:
    st.write(f"**병합 방식:** {how_merge}")
    if merge_info:
        st.dataframe(pd.DataFrame(merge_info.items(), columns=["항목", "개수"]), use_container_width=True, hide_index=True)
        if unmatched:
            st.write("대응표로 이어지지 않아 분석에서 빠진 유동인구 센서:", ", ".join(unmatched))
    st.write(f"병합 전: 환경 {len(env):,}행 · 유동인구 {len(pop):,}행 → 병합 후 {len(merged):,}행 → 분석 행 {len(final):,}행")

with tab_graph:
    if len(final) < 5:
        st.info("분석에 쓸 수 있는 행이 5개 미만입니다. 기간을 넓히거나 다른 자치구를 골라보세요.")
    else:
        st.caption(f"{gu_name} · {d0} ~ {d1} · 분석 행 {len(final):,}개")
        y = st.selectbox("볼 환경 지표", list(METRICS), format_func=METRICS.get)
        p = final.dropna(subset=[y]).copy()
        p["시각"] = p["HOUR"].dt.hour
        x = p["VISITOR_MEAN"]
        r = x.corr(p[y])
        slope, icpt = np.polyfit(x, p[y], 1)
        xd = x - p.groupby("시각")["VISITOR_MEAN"].transform("mean")
        yd = p[y] - p.groupby("시각")[y].transform("mean")
        r_in = xd.corr(yd)
        slope_in = np.polyfit(xd, yd, 1)[0] if xd.std() > 0 else np.nan

        fig1 = px.scatter(p, x="VISITOR_MEAN", y=y, opacity=0.5,
                          labels={"VISITOR_MEAN": "평균 방문자 수 (명)", y: METRICS[y]},
                          title=f"그래프 1. 방문자 수와 {METRICS[y]}의 관계")
        xs = np.array([x.min(), x.max()])
        fig1.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines", name="추세선", line=dict(color="red")))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"**찾아낸 것:** 방문자 수와 {METRICS[y]}의 상관계수는 **{r:.3f}**, 방문자 10명 증가당 **{slope*10:+.2f}**. "
                    f"같은 시각끼리만 비교하면(시각별 평균을 뺀 값) 상관계수 **{r_in:.3f}**, 10명당 **{slope_in*10:+.2f}**입니다. (n={len(p):,})")

        hh = p.groupby("시각")[["VISITOR_MEAN", y]].mean().reset_index()
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=hh["시각"], y=hh["VISITOR_MEAN"], name="평균 방문자 수(명)", mode="lines+markers"), secondary_y=False)
        fig2.add_trace(go.Scatter(x=hh["시각"], y=hh[y], name=f"평균 {METRICS[y]}", mode="lines+markers"), secondary_y=True)
        fig2.update_layout(title=f"그래프 2. 하루 중 시각별 방문자 수와 {METRICS[y]}")
        fig2.update_xaxes(title_text="하루 중 시각 (시)", dtick=2)
        fig2.update_yaxes(title_text="평균 방문자 수 (명)", secondary_y=False)
        fig2.update_yaxes(title_text=f"평균 {METRICS[y]}", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)
        pk_v, pk_y = hh.loc[hh["VISITOR_MEAN"].idxmax()], hh.loc[hh[y].idxmax()]
        st.markdown(f"**찾아낸 것:** 방문자 수는 **{int(pk_v['시각'])}시**({pk_v['VISITOR_MEAN']:.1f}명), "
                    f"{METRICS[y]}은 **{int(pk_y['시각'])}시**({pk_y[y]:.1f})에 가장 높습니다. "
                    f"시각별 평균끼리의 상관계수는 **{hh['VISITOR_MEAN'].corr(hh[y]):.3f}**입니다.")

        q = p.copy()
        try:
            q["구간"] = pd.qcut(q["VISITOR_MEAN"], 4, labels=["적음(하위25%)", "보통 이하", "보통 이상", "많음(상위25%)"])
            fig3 = px.box(q, x="구간", y=y, labels={"구간": "방문자 수 구간", y: METRICS[y]},
                          title=f"그래프 3. 방문자 수 구간별 {METRICS[y]} 분포")
            st.plotly_chart(fig3, use_container_width=True)
            m = q.groupby("구간", observed=True)[y].mean()
            st.markdown(f"**찾아낸 것:** 방문자가 가장 적은 구간의 {METRICS[y]} 평균은 **{m.iloc[0]:.1f}**, "
                        f"가장 많은 구간은 **{m.iloc[-1]:.1f}** (차이 {m.iloc[-1]-m.iloc[0]:+.1f}).")
        except ValueError:
            st.info("방문자 수 값이 너무 비슷해 4개 구간으로 나눌 수 없습니다.")

with tab_raw:
    if st.checkbox("원본·병합 표 보기 (느릴 수 있음)"):
        st.subheader("병합 데이터")
        show(ko(merged))
        st.subheader("환경정보 원본")
        show(ko(env_raw[ENV_COLS]))
        st.subheader("유동인구 원본")
        show(ko(pop_raw))
