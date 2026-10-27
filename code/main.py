    y = working[target].copy()

    if X.shape[1] == 0:
        st.error("학습에 사용할 수 있는 입력 특징이 없습니다. 다른 특징 열을 선택해 주세요.")
        return

    # Keep a fixed test set; vary only the amount of training data.
    stratify = None
    if task == "분류":
        counts = y.value_counts()
        if y.nunique() > 1 and counts.min() >= 2:
            stratify = y
    try:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    if len(X_train_full) < 10 or len(X_test) < 5:
        st.error("학습 또는 테스트 데이터가 부족합니다.")
        return

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    model = make_model(task, model_name, numeric_cols, categorical_cols)

    # Compare several training-data fractions on the same held-out test set.
    fractions = [0.1, 0.3, 0.5, 0.8, 1.0]
    rows = []
    fitted_for_selected = None
    selected_train_n = max(5, int(len(X_train_full) * train_percent / 100))
    selected_train_n = min(selected_train_n, len(X_train_full))

    for frac in fractions:
        n = max(5, int(len(X_train_full) * frac))
        n = min(n, len(X_train_full))
        sample_indices = np.linspace(0, len(X_train_full) - 1, n, dtype=int)
        X_sub = X_train_full.iloc[sample_indices]
        y_sub = y_train_full.iloc[sample_indices]
        current_model = make_model(task, model_name, numeric_cols, categorical_cols)
        try:
            current_model.fit(X_sub, y_sub)
            pred = current_model.predict(X_test)
            if task == "회귀":
                mae = mean_absolute_error(y_test, pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                r2 = r2_score(y_test, pred) if len(y_test) > 1 else np.nan
                rows.append({
                    "학습 데이터 비율": f"{int(frac*100)}%",
                    "학습 행 수": n,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R²": r2
                })
            else:
                rows.append({
                    "학습 데이터 비율": f"{int(frac*100)}%",
                    "학습 행 수": n,
                    "정확도": accuracy_score(y_test, pred),
                    "정밀도(가중)": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "재현율(가중)": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1(가중)": f1_score(y_test, pred, average="weighted", zero_division=0)
                })
        except Exception as e:
            rows.append({"학습 데이터 비율": f"{int(frac*100)}%", "학습 행 수": n, "오류": str(e)[:150]})

    # Fit final model using chosen fraction
    sample_indices = np.linspace(0, len(X_train_full) - 1, selected_train_n, dtype=int)
    X_selected = X_train_full.iloc[sample_indices]
    y_selected = y_train_full.iloc[sample_indices]
    try:
        fitted_for_selected = make_model(task, model_name, numeric_cols, categorical_cols)
        fitted_for_selected.fit(X_selected, y_selected)
        predictions = fitted_for_selected.predict(X_test)
    except Exception as e:
        st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("사용 가능한 행 수", f"{len(working):,}")
    m2.metric("학습 행 수", f"{len(X_selected):,}")
    m3.metric("테스트 행 수", f"{len(X_test):,}")
    m4.metric("입력 특징 수", f"{X.shape[1]}")

    st.subheader("📊 학습 데이터 양에 따른 성능 비교")
    results = pd.DataFrame(rows)
    st.dataframe(results, use_container_width=True, hide_index=True)

    if task == "회귀":
        fig, ax = plt.subplots()
        ax.plot(results["학습 데이터 비율"], results["MAE"], marker="o", label="MAE")
        ax.plot(results["학습 데이터 비율"], results["RMSE"], marker="o", label="RMSE")
        ax.set_xlabel("Training data used")
        ax.set_ylabel("Error (lower is better)")
        ax.set_title("Training data amount vs error")
        ax.legend()
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mean_absolute_error(y_test, predictions):,.3f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, predictions)):,.3f}")
        c3.metric("R²", f"{r2_score(y_test, predictions):.3f}")

        compare = pd.DataFrame({"실제값": np.asarray(y_test), "예측값": predictions})
        st.subheader("실제값과 예측값 비교")
        fig2, ax2 = plt.subplots()
        ax2.scatter(compare["실제값"], compare["예측값"], alpha=0.5)
        minv = min(compare["실제값"].min(), compare["예측값"].min())
        maxv = max(compare["실제값"].max(), compare["예측값"].max())
        ax2.plot([minv, maxv], [minv, maxv], linestyle="--")
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.set_title("Actual vs predicted")
        ax2.grid(True, alpha=0.25)
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption("MAE·RMSE는 낮을수록 좋고, R²는 일반적으로 1에 가까울수록 예측 성능이 좋습니다.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("정확도", f"{accuracy_score(y_test, predictions):.3f}")
        c2.metric("정밀도(가중)", f"{precision_score(y_test, predictions, average='weighted', zero_division=0):.3f}")
        c3.metric("재현율(가중)", f"{recall_score(y_test, predictions, average='weighted', zero_division=0):.3f}")
        c4.metric("F1(가중)", f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.3f}")

        fig, ax = plt.subplots()
        ax.plot(results["학습 데이터 비율"], results["정확도"], marker="o")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Training data used")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training data amount vs accuracy")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
        cm = confusion_matrix(y_test.astype(str), pd.Series(predictions).astype(str), labels=labels)
        fig2, ax2 = plt.subplots(figsize=(max(5, len(labels) * 0.6), max(4, len(labels) * 0.5)))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
            ax=ax2, xticks_rotation=45, colorbar=False
        )
        ax2.set_title("Confusion matrix")
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("🔎 테스트 데이터 예측 결과")
    output = pd.DataFrame({"실제값": np.asarray(y_test), "예측값": predictions})
    st.dataframe(output.head(200), use_container_width=True)
    st.download_button(
        "예측 결과 CSV 다운로드",
        data=output.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{prefix}_prediction_results.csv",
        mime="text/csv"
    )

    with st.expander("분석 방법과 주의사항"):
        st.markdown(
            """
            - 테스트 데이터는 학습 데이터와 분리해 두고, 학습 데이터의 양만 바꾸어 비교합니다.
            - 센서 ID나 위치 정보는 범주형 특징으로 처리할 수 있습니다.
            - 시간 열은 연도·월·요일·시각·분 특징으로 변환합니다.
            - **주의:** 무작위 분할은 미래 예측 상황을 완벽하게 재현하지 않습니다. 실제 미래 예측이 목적이라면 시간순으로 과거를 학습, 이후 기간을 테스트하는 방식으로 바꾸는 것이 더 적절합니다.
            - 유동인구 센서값은 서울시 설명에 따르면 정확한 전체 인구수를 뜻하지 않으므로, 결과를 실제 인구수로 단정하지 마세요.
            """
        )

def dataset_tab(label, target_hint, task_default="회귀"):
    uploaded = st.file_uploader(
        f"{label} CSV 파일 업로드",
        type=["csv"],
        key=f"upload_{label}"
    )
    if uploaded is None:
        st.markdown(f"먼저 위 데이터 다운로드 안내에서 **{label} CSV**를 내려받아 업로드해 주세요.")
        return

    try:
        df = read_csv_safely(uploaded)
    except Exception as e:
        st.error(str(e))
        return

    if df.empty:
        st.error("파일에 데이터가 없습니다.")
        return

    st.success(f"데이터를 불러왔습니다: {len(df):,}행 × {len(df.columns):,}열")
    with st.expander("데이터 미리보기", expanded=False):
