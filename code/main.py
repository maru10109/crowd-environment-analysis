import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="서울 도시데이터 머신러닝 분석",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ 서울 도시데이터 머신러닝 분석")
st.caption("서울 열린데이터광장의 S-DoT 환경정보와 유동인구 자료를 활용한 데이터 분석·머신러닝 실험")

with st.expander("📥 데이터 다운로드 방법", expanded=False):
    st.markdown(
        """
        1. [S-DoT 환경정보](https://data.seoul.go.kr/dataList/OA-15969/S/1/datasetView.do)에 접속합니다.
        2. 파일 내려받기에서 원하는 기간의 `S-DoT_NATURE_...csv` 파일을 내려받습니다.
        3. [S-DoT 유동인구 측정 정보](https://data.seoul.go.kr/dataList/OA-15964/S/1/datasetView.do)에 접속합니다.
        4. `S-DoT_WALK_...csv` 파일을 내려받습니다.
        5. 아래 각 탭에서 해당 CSV 파일을 업로드합니다.

        **팁:** 먼저 1주일 분량 CSV 파일로 시험해 보세요. 연간 ZIP 파일은 압축을 풀고 CSV를 선택해야 합니다.
        """
    )

st.info(
    "두 자료는 측정 주기와 센서 구성이 다릅니다. 이 앱은 각 데이터셋을 별도로 분석하며, "
    "위치·시간 키가 실제로 일치한다고 확인되지 않은 데이터를 임의로 합치지 않습니다."
)

def read_csv_safely(uploaded_file):
    """Try common encodings; load CSV and normalize column names."""
    errors = []
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            # Remove completely empty rows/columns
            df = df.dropna(how="all").dropna(axis=1, how="all")
            return df
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise ValueError("CSV를 읽지 못했습니다. 파일이 CSV인지 확인해 주세요.\n" + "\n".join(errors[-2:]))

def guess_datetime_columns(df):
    result = []
    for col in df.columns:
        name = str(col).lower()
        if any(word in name for word in ["일시", "시간", "날짜", "측정일", "date", "time", "dtm"]):
            result.append(col)
    return result

def parse_datetime_feature(df, col):
    dt = pd.to_datetime(df[col], errors="coerce")
    if dt.notna().mean() < 0.5:
        # Try compact timestamps and strings with separators removed
        compact = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
            trial = pd.to_datetime(compact, format=fmt, errors="coerce")
            if trial.notna().mean() > dt.notna().mean():
                dt = trial
    return dt

def prepare_features(df, selected_features, datetime_features):
    X = df[selected_features].copy()
    for col in datetime_features:
        if col not in X.columns:
            continue
        dt = parse_datetime_feature(X, col)
        X[col + "_연도"] = dt.dt.year
        X[col + "_월"] = dt.dt.month
        X[col + "_요일"] = dt.dt.dayofweek
        X[col + "_시각"] = dt.dt.hour
        X[col + "_분"] = dt.dt.minute
        X = X.drop(columns=[col])
    # Convert numeric-like strings to numeric when mostly convertible
    for col in X.columns:
        if X[col].dtype == "object":
            converted = pd.to_numeric(X[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
            if converted.notna().mean() >= 0.85:
