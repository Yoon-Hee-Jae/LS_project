# VSCode를 쓰신다면, 상단 메뉴에서 [Terminal] → [New Terminal] 클릭

# 아래 명령어 한 번만 입력:
# streamlit run app.py

# 이후에는 터미널을 닫지 말고, 코드를 수정하면
# Streamlit이 자동 새로고침(Hot Reload) 해줍니다!
# → 저장(ctrl+s)만 해도 웹이 자동 업데이트됩니다.


import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib # [추가] 모델 로딩
import os     # [추가] 파일 경로
import time   # [추가] 시뮬레이션
import lightgbm
import xgboost
import catboost


# -----------------------------
# [추가] 예측 모델/함수
# -----------------------------
MODEL_DIR = "models"

# 모델/인코더 로딩 (앱 실행 시 한 번만)
@st.cache_resource
def load_models_and_encoders():
    """models 폴더에서 11개의 .pkl 파일을 모두 로드합니다."""
    try:
        models = {}
        # 1. 인코더 로드 (3개)
        models["le_job"] = joblib.load(os.path.join(MODEL_DIR, "le_job.pkl"))
        models["le_band"] = joblib.load(os.path.join(MODEL_DIR, "le_band.pkl"))
        models["le_tj"] = joblib.load(os.path.join(MODEL_DIR, "le_tj.pkl"))

        # 2. Stage 1 모델 로드 (5개)
        s1_targets = [
            "s1_전력사용량.pkl", "s1_지상무효전력량.pkl", "s1_진상무효전력량.pkl",
            "s1_지상역률.pkl", "s1_진상역률.pkl"
        ]
        models["s1_model_map"] = {}
        target_map = { # 파일명 -> 원본 타겟명
            "s1_전력사용량.pkl": "전력사용량(kWh)",
            "s1_지상무효전력량.pkl": "지상무효전력량(kVarh)",
            "s1_진상무효전력량.pkl": "진상무효전력량(kVarh)",
            "s1_지상역률.pkl": "지상역률(%)",
            "s1_진상역률.pkl": "진상역률(%)"
        }
        for fname in s1_targets:
            target_name = target_map[fname]
            models["s1_model_map"][target_name] = joblib.load(os.path.join(MODEL_DIR, fname))

        # 3. Stage 2 모델 로드 (3개)
        models["s2_lgb"] = joblib.load(os.path.join(MODEL_DIR, "s2_lgb.pkl"))
        models["s2_xgb"] = joblib.load(os.path.join(MODEL_DIR, "s2_xgb.pkl"))
        models["s2_cat"] = joblib.load(os.path.join(MODEL_DIR, "s2_cat.pkl"))
        
        return models
    except FileNotFoundError:
        st.error(f"'{MODEL_DIR}' 폴더 또는 모델 파일(.pkl)을 찾을 수 없습니다.")
        st.error("먼저 train_and_save_models.py를 실행하여 모델을 생성해주세요.")
        return None

# 전처리 함수 (train_and_save_models.py와 동일)
REF_DATE = pd.Timestamp("2024-10-24")
def adjust_hour(dt):
    if pd.isna(dt): return np.nan
    return (dt.hour - 1) % 24 if dt.minute == 0 else dt.hour
def band_of_hour(h):
    if (22 <= h <= 23) or (0 <= h <= 7): return "경부하"
    elif 16 <= h <= 21: return "최대부하"
    else: return "중간부하"

def enrich(df):
    df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    df["월"] = df["측정일시"].dt.month
    df["일"] = df["측정일시"].dt.day
    df["요일"] = df["측정일시"].dt.dayofweek
    df["시간"] = df["측정일시"].apply(adjust_hour)
    df["주말여부"] = (df["요일"]>=5).astype(int)
    df["겨울여부"] = df["월"].isin([11,12,1,2]).astype(int)
    df["period_flag"] = (df["측정일시"] >= REF_DATE).astype(int)
    df["sin_time"] = np.sin(2*np.pi*df["시간"]/24)
    df["cos_time"] = np.cos(2*np.pi*df["시간"]/24)
    df["부하구분"] = df["시간"].apply(band_of_hour)
    return df

def add_pf(df):
    df["유효역률(%)"] = df[["지상역률(%)","진상역률(%)"]].max(axis=1)
    df["역률_패널티율"] = (90 - df["유효역률(%)"]).clip(lower=0)*0.01
    df["역률_보상율"]   = (df["유효역률(%)"] - 90).clip(lower=0)*0.005
    df["역률_조정요율"] = df["역률_보상율"] - df["역률_패널티율"]
    return df

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="전기요금 분석", layout="wide", # wide, centered
)

# ---------------------------------
# 사이드바 (메뉴)
# ---------------------------------
with st.sidebar:
    st.title("전기요금 분석")
    
    # 페이지 선택 라디오 버튼
    page = st.radio(
        "페이지 이동",
        ["실시간 전기요금 분석", "과거 전력사용량 분석"],
        label_visibility="collapsed" # 라벨 숨기기
    )
    
    st.divider() # 구분선

# ---------------------------------
# 1. 실시간 전기요금 분석 페이지 (수정)
# ---------------------------------
if page == "실시간 전기요금 분석":
    st.title(" 12월 전기요금 실시간 예측 시뮬레이션")

    # 1. 모델 로드
    models = load_models_and_encoders()
    
    if models: # 모델 로딩에 성공한 경우
        
        # --- [수정] 시뮬레이션 제어 버튼 ---
        col1, col2 = st.columns([1, 1])
        with col1:
            # [수정] 버튼 텍스트 변경
            if st.button("▶️ 시작/재개"):
                
                # [수정] Resume 로직 추가
                # 'current_index'가 0보다 크면 (즉, 중지된 적이 있으면) '재개'
                if 'current_index' in st.session_state and st.session_state.current_index > 0:
                    st.session_state.simulation_running = True
                
                # 'current_index'가 0이거나 없으면 (즉, 처음 시작이면) '초기화 후 시작'
                else:
                    try:
                        # 3. 데이터 로드 (train.csv는 Lag 생성용, test.csv는 예측 대상용)
                        train_df = pd.read_csv("./data/train.csv")
                        test_df = pd.read_csv("./data/test.csv")
                        
                        # [추가] 안정적인 클리핑을 위해 train 데이터로 경계값 계산
                        clipping_low, clipping_high = np.percentile(train_df["전기요금(원)"], [0.2, 99.8])
                        st.session_state.clipping_bounds = (clipping_low, clipping_high)

                    except FileNotFoundError as e:
                        st.error(f"데이터 파일({e.filename})을 찾을 수 없습니다. './data/' 폴더에 train.csv, test.csv가 필요합니다.")
                        st.stop()
                    
                    # 재귀 생성을 위한 11월 마지막 24시간 이력
                    train_df = enrich(train_df).sort_values("측정일시").reset_index(drop=True)
                    last24 = train_df[["측정일시","전력사용량(kWh)"]].tail(24).copy()
                    
                    # --- Session State 초기화 (Hard Reset) ---
                    st.session_state.simulation_running = True
                    st.session_state.current_index = 0
                    st.session_state.test_df = test_df # 전체 test.csv 저장
                    st.session_state.history = list(last24["전력사용량(kWh)"].values.astype(float)) # Lag 이력
                    st.session_state.predictions = [] # 예측 결과(DataFrame) 저장 리스트
                    st.session_state.total_bill = 0.0
                    st.session_state.total_usage = 0.0
                    st.session_state.errors = []
                    # st.session_state.last_shap_fig = None # (SHAP 제거)
        
        with col2:
            if st.button("⏹️ 중지"):
                # [수정] 중지 버튼은 상태만 변경, 데이터는 유지
                st.session_state.simulation_running = False

        # --- [수정] 동적 컨텐츠를 위한 Placeholders ---
        st.subheader("12월 예측 집계")
        metric_cols = st.columns(2)
        total_bill_metric = metric_cols[0].empty()
        total_usage_metric = metric_cols[1].empty()

        st.subheader("현재 예측")
        latest_time_placeholder = st.empty()
        latest_pred_placeholder = st.empty()
        
        # [수정] SHAP 레이아웃 제거
        st.subheader("12월 시간대별 예측 요금 추이")
        chart_placeholder = st.empty()
        # shap_placeholder 제거
        
        # --- [추가] 세션 상태 초기화 (최초 실행 시) ---
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False

        # --- [수정] 메인 시뮬레이션 루프 ---
        if st.session_state.simulation_running:
            # 1. 시뮬레이션 상태 유효성 검사
            if 'test_df' not in st.session_state or 'history' not in st.session_state:
                st.error("시뮬레이션 상태가 초기화되지 않았습니다. '시작' 버튼을 다시 눌러주세요.")
                st.session_state.simulation_running = False
            
            # 2. 예측할 데이터가 남았는지 확인
            elif st.session_state.current_index < len(st.session_state.test_df):
                # 2-1. 현재 행(row) 가져오기
                row_df = st.session_state.test_df.iloc[[st.session_state.current_index]].copy()
                
                # 2-2. 전처리 (Enrich)
                row_df = enrich(row_df)
                
                # 2-3. 인코딩 (로드한 인코더 사용)
                try:
                    row_df["작업유형_encoded"] = models["le_job"].transform(row_df["작업유형"].astype(str))
                    row_df["부하구분_encoded"] = models["le_band"].transform(row_df["부하구분"].astype(str))
                    row_df["시간_작업유형"] = row_df["시간"].astype(str)+"_"+row_df["작업유형_encoded"].astype(str)
                    row_df["시간_작업유형_encoded"]  = models["le_tj"].transform(row_df["시간_작업유형"])
                except ValueError as e:
                    # 인코딩 오류 발생 시 (예: train에 없던 작업유형)
                    st.session_state.errors.append(f"인코딩 오류 (Index {st.session_state.current_index}): {e}")
                    st.session_state.current_index += 1
                    st.rerun() # 다음 행으로 즉시 이동
                
                # 2-4. Stage 1 예측
                feat_s1 = ["월","일","요일","시간","주말여부","겨울여부","period_flag", "sin_time","cos_time","작업유형_encoded","부하구분_encoded","시간_작업유형_encoded"]
                targets_s1 = ["전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)", "지상역률(%)","진상역률(%)"]
                
                for tgt in targets_s1:
                    m = models["s1_model_map"][tgt]
                    row_df[tgt] = m.predict(row_df[feat_s1])
                
                # 2-5. 유효역률 파생
                row_df = add_pf(row_df)
                
                # 2-6. 재귀 Lag/Rolling 생성
                kwh_pred = row_df["전력사용량(kWh)"].values[0] # S1에서 예측된 사용량
                hist = st.session_state.history
                
                row_df["kwh_lag24"] = hist[-24] if len(hist)>=24 else np.nan
                arr = np.array(hist[-24:]) if len(hist)>=24 else np.array(hist)
                row_df["kwh_roll24_mean"] = arr.mean() if arr.size>0 else np.nan
                row_df["kwh_roll24_std"]  = arr.std()  if arr.size>1 else 0.0
                
                # 2-7. Stage 2 예측 (앙상블)
                feat_s2 = feat_s1 + ["전력사용량(kWh)","지상무효전력량(kVarh)","진상무효전력량(kVarh)", "지상역률(%)","진상역률(%)","유효역률(%)","역률_조정요율", "kwh_lag24","kwh_roll24_mean","kwh_roll24_std"]
                X_te = row_df[feat_s2].copy()
                
                pred_lgb = np.expm1(models["s2_lgb"].predict(X_te))
                pred_xgb = np.expm1(models["s2_xgb"].predict(X_te))
                pred_cat = np.expm1(models["s2_cat"].predict(X_te))
                
                pred_te = (0.5 * pred_lgb + 0.3 * pred_xgb + 0.2 * pred_cat)[0]
                
                # 2-8. 이상치 안정화(클리핑)
                low, high = st.session_state.clipping_bounds
                pred_te = np.clip(pred_te, low, high)
                
                row_df["예측요금(원)"] = pred_te
                
                # 2-9. 상태 업데이트
                st.session_state.history.append(kwh_pred) # 히스토리에 현재 예측 *사용량* 추가
                st.session_state.predictions.append(row_df) # 결과 리스트에 현재 row_df 추가
                st.session_state.total_bill += pred_te
                st.session_state.total_usage += kwh_pred
                st.session_state.current_index += 1

                # 2-10. UI 업데이트 (Placeholders)
                total_bill_metric.metric("12월 누적 예상 전기요금", f"{st.session_state.total_bill:,.0f} 원")
                total_usage_metric.metric("12월 누적 예상 전력사용량", f"{st.session_state.total_usage:,.0f} kWh")
                
                latest_time_placeholder.write(f"**측정일시:** {row_df['측정일시'].iloc[0]}")
                latest_pred_placeholder.write(f"**예측요금:** `{pred_te:,.0f} 원` | **예측사용량:** `{kwh_pred:,.2f} kWh`")

                # 2-11. Chart Update (최종 수정)
                results_df = pd.concat(st.session_state.predictions)
            
                if not results_df.empty:
                
                    # [수정] 24시간 윈도우 로직 (최종)
                    first_time = results_df['측정일시'].iloc[0]
                    latest_time = results_df['측정일시'].iloc[-1]
                
                    # 24시간이 경과했는지 확인
                    if latest_time < (first_time + pd.Timedelta(hours=24)):
                        # [Phase 1] 24시간 미만:
                        # X축을 고정하지 않고, 데이터에 맞게 자동 스케일링 (전체 화면 채우기)
                        x_axis = alt.X('측정일시:T', title='측정일시')
                    else:
                        # [Phase 2] 24시간 초과:
                        # X축이 최신 시간을 따라 24시간 윈도우로 슬라이딩
                        start_domain = latest_time - pd.Timedelta(hours=24)
                        end_domain = latest_time
                        x_axis = alt.X('측정일시:T',
                                        title='측정일시',
                                        scale=alt.Scale(domain=[start_domain, end_domain])
                                    )
                
                    chart = alt.Chart(results_df).mark_line().encode(
                        x=x_axis, # [수정] 동적 X축 할당
                        y=alt.Y('예측요금(원):Q', title='예측요금 (원)'),
                        tooltip=['측정일시', alt.Tooltip('예측요금(원)', format=',.0f')]
                    ).interactive(bind_y=False) # Y축 줌 방지
                
                    chart_placeholder.altair_chart(chart, use_container_width=True)
                
                # [수정] 2-11.5 SHAP Plot Update (제거)
                
                # 2-12. Loop (1.5초 대기 후 rerun)
                time.sleep(1.5)
                st.rerun()

            else:
                # 3. 시뮬레이션 완료
                st.session_state.simulation_running = False
                st.success("✅ 12월 전체 예측 시뮬레이션 완료!")
                if st.session_state.errors:
                    st.warning("일부 데이터에서 인코딩 오류가 발생했습니다:")
                    st.json(st.session_state.errors)

        # --- [수정] 시뮬레이션 비활성 시 (초기/중지/완료) ---
        elif 'predictions' in st.session_state and st.session_state.predictions:
            # 시뮬레이션이 완료되었거나 중지된 경우, 최종 결과 표시
            total_bill_metric.metric("12월 누적 예상 전기요금", f"{st.session_state.total_bill:,.0f} 원")
            total_usage_metric.metric("12월 누적 예상 전력사용량", f"{st.session_state.total_usage:,.0f} kWh")

            # [수정] Sliding Window 적용 (최종본)
            results_df = pd.concat(st.session_state.predictions)

            if not results_df.empty:

                # [수정] 24시간 윈도우 로직 수정
                first_time = results_df['측정일시'].iloc[0]
                latest_time = results_df['측정일시'].iloc[-1]

                # 24시간이 경과했는지 확인
                if latest_time < (first_time + pd.Timedelta(hours=24)):
                    # [Phase 1] 24시간 미만:
                    # X축을 0~24시간으로 고정
                    start_domain = first_time
                    end_domain = first_time + pd.Timedelta(hours=24)
                else:
                    # [Phase 2] 24시간 초과:
                    # X축이 최신 시간을 따라 24시간 윈도우로 슬라이딩
                    start_domain = latest_time - pd.Timedelta(hours=24)
                    end_domain = latest_time
                
                first_time = results_df['측정일시'].iloc[0]
                latest_time = results_df['측정일시'].iloc[-1]

                # 24시간이 경과했는지 확인
                if latest_time < (first_time + pd.Timedelta(hours=24)):
                    # [Phase 1] 24시간 미만:
                    # [수정] scale을 고정하지 않고, '실행 중'일 때와 동일하게 자동 스케일링
                    x_axis = alt.X('측정일시:T', title='측정일시')
                else:
                    # [Phase 2] 24시간 초과:
                    # X축이 최신 시간을 따라 24시간 윈도우로 슬라이딩
                    start_domain = latest_time - pd.Timedelta(hours=24)
                    end_domain = latest_time
                    x_axis = alt.X('측정일시:T',
                                    title='측정일시',
                                    scale=alt.Scale(domain=[start_domain, end_domain])
                                )
                
                # [수정] 차트 생성 로직을 밖으로 빼고 x_axis 변수 사용
                chart = alt.Chart(results_df).mark_line().encode(
                    x=x_axis, # [수정] 위에서 정의된 x_axis 변수 사용
                    y=alt.Y('예측요금(원):Q', title='예측요금 (원)'),
                    tooltip=['측정일시', alt.Tooltip('예측요금(원)', format=',.0f')]
                ).interactive(bind_y=False) # Y축 줌 방지
                
                chart_placeholder.altair_chart(chart, use_container_width=True)
            
                # 상세 데이터 expander (이 부분은 동일)
                with st.expander("12월 예측 상세 데이터 보기 (최종)"):
                    st.dataframe(results_df[[ # 여기는 전체 df 표시
                        "측정일시", "작업유형", "전력사용량(kWh)", "유효역률(%)", "예측요금(원)"
                    ]].style.format({
                    "전력사용량(kWh)": "{:,.2f}",
                    "유효역률(%)": "{:,.2f}",
                    "예측요금(원)": "{:,.0f}"
                    }))
        else:
            # 시뮬레이션 시작 전 (초기 상태)
            total_bill_metric.metric("12월 누적 예상 전기요금", "0 원")
            total_usage_metric.metric("12월 누적 예상 전력사용량", "0 kWh")
            latest_time_placeholder.info("시뮬레이션을 시작해주세요.")

# ---------------------------------
# 2. 과거 전력사용량 분석 페이지
# ---------------------------------
elif page == "과거 전력사용량 분석":
    st.title("과거 전력사용량 분석 (1월 ~ 11월)")
    st.write("학습(Train) 데이터인 과거 11개월간의 전력 사용량 및 관련 데이터를 분석합니다.")

    # --- 실제 데이터 로드 ---
    @st.cache_data  # 데이터 로딩 및 처리를 캐시하여 속도 향상
    def load_data(filepath="./data/train.csv"):
        try:
            df = pd.read_csv(filepath)
            df['측정일시'] = pd.to_datetime(df['측정일시'])
            df['월'] = df['측정일시'].dt.month
            df['일'] = df['측정일시'].dt.day
            df['시간'] = df['측정일시'].dt.hour
            df['날짜'] = df['측정일시'].dt.date
            df['연월'] = df['측정일시'].dt.to_period('M').astype(str)
            return df
        except FileNotFoundError:
            st.error(f"'{filepath}' 파일을 찾을 수 없습니다. 'app.py'와 같은 위치에 파일을 두었는지 확인해주세요.")
            return None

    df = load_data()

    if df is not None:
        st.subheader("전체 기간(1~11월) 개요")

        # --- 전체 기간 요약 지표 (기존과 동일) ---
        total_usage = df['전력사용량(kWh)'].sum()
        total_bill = df['전기요금(원)'].sum()
        avg_hourly_usage = df['전력사용량(kWh)'].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric(label="총 전력사용량", value=f"{total_usage:,.0f} kWh")
        col2.metric(label="총 전기요금", value=f"{total_bill:,.0f} 원")
        col3.metric(label="평균 시간당 사용량", value=f"{avg_hourly_usage:,.2f} kWh")
        st.divider()

        # --- 월별/기간별 집계 데이터 (기존과 동일) ---
        monthly_summary = df.groupby('월').agg(
            total_usage=('전력사용량(kWh)', 'sum'),
            total_bill=('전기요금(원)', 'sum'),
            avg_usage=('전력사용량(kWh)', 'mean')
        ).reset_index()
        min_date = df['측정일시'].min().date()
        max_date = df['측정일시'].max().date()

        st.subheader("기간별 상세 분석")
        col_left, col_right = st.columns(2)

        # --- 입력 위젯 (기존과 동일) ---
        with col_left:
            st.write("#### 분석 기간 선택 (차트 시작 범위)") # 설명 변경
            filter_type = st.radio( "분석 기간 선택 방식:", ["월별 선택", "기간별 선택"], horizontal=True, index=0)

            analysis_df = pd.DataFrame()
            analysis_title = ""
            delta_usage_str = None
            delta_bill_str = None
            delta_usage_color = "off"
            delta_bill_color = "off"

            if filter_type == "월별 선택":
                month_list = sorted(df['월'].unique())
                selected_month = st.selectbox("분석할 월을 선택하세요:", month_list, format_func=lambda x: f"{x}월")
                analysis_df = df[df['월'] == selected_month]
                analysis_title = f"{selected_month}월"
                # ... (delta 계산 로직은 기존과 동일) ...
                if selected_month > 1:
                        prev_month_summary = monthly_summary[monthly_summary['월'] == (selected_month - 1)]
                        if not prev_month_summary.empty:
                                current_val_usage = monthly_summary[monthly_summary['월'] == selected_month]['total_usage'].values[0]
                                prev_val_usage = prev_month_summary['total_usage'].values[0]
                                delta_usage = int(current_val_usage - prev_val_usage)
                                current_val_bill = monthly_summary[monthly_summary['월'] == selected_month]['total_bill'].values[0]
                                prev_val_bill = prev_month_summary['total_bill'].values[0]
                                delta_bill = int(current_val_bill - prev_val_bill)
                                delta_usage_str = f"{delta_usage:+,} kWh"
                                delta_usage_color = "inverse"
                                delta_bill_str = f"{delta_bill:+,} 원"
                                delta_bill_color = "inverse"

            elif filter_type == "기간별 선택":
                selected_range = st.date_input("분석할 기간을 선택하세요:", [min_date, max_date], min_value=min_date, max_value=max_date)
                if isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
                    start_date, end_date = selected_range
                    analysis_df = df[(df['날짜'] >= start_date) & (df['날짜'] <= end_date)]
                    analysis_title = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
                else:
                    st.warning("시작일과 종료일을 모두 선택해주세요.")
                    analysis_df = pd.DataFrame(columns=df.columns) # 빈 DF
                    analysis_title = "기간 미선택"

        # --- [수정] 선택 기간 요약 지표 (오른쪽 컬럼, 기존과 동일) ---
        with col_right:
             st.write(f"#### {analysis_title} 주요 지표")
             if not analysis_df.empty:
                 current_total_usage = analysis_df['전력사용량(kWh)'].sum()
                 current_total_bill = analysis_df['전기요금(원)'].sum()
                 current_avg_usage = analysis_df['전력사용량(kWh)'].mean()
                 current_total_carbon = analysis_df['탄소배출량(tCO2)'].sum()
                 # ... (st.metric 4개 표시 로직은 기존과 동일) ...
                 row1_col1, row1_col2 = st.columns(2)
                 with row1_col1: st.metric(label=f"{analysis_title} 총 사용량", value=f"{current_total_usage:,.0f} kWh", delta=delta_usage_str, delta_color=delta_usage_color)
                 with row1_col2: st.metric(label=f"{analysis_title} 총 전기요금", value=f"{current_total_bill:,.0f} 원", delta=delta_bill_str, delta_color=delta_bill_color)
                 row2_col1, row2_col2 = st.columns(2)
                 with row2_col1: st.metric(label=f"{analysis_title} 평균 시간당 사용량", value=f"{current_avg_usage:,.0f} kWh")
                 with row2_col2: st.metric(label=f"{analysis_title} 총 탄소 배출량", value=f"{current_total_carbon:,.2f} tCO2")
             else:
                 st.warning(f"선택된 '{analysis_title}' 기간에 대한 데이터가 없어 지표를 계산할 수 없습니다.")
                 # ... (빈 metric 4개 표시 로직은 기존과 동일) ...
                 row1_col1, row1_col2 = st.columns(2)
                 row1_col1.metric(f"{analysis_title} 총 사용량", "0 kWh")
                 row1_col2.metric(f"{analysis_title} 총 전기요금", "0 원")
                 row2_col1, row2_col2 = st.columns(2)
                 row2_col1.metric(f"{analysis_title} 평균 시간당 사용량", "0 kWh")
                 row2_col2.metric(f"{analysis_title} 총 탄소 배출량", "0 tCO2")

        st.divider()

        # --- [신규] 선택된 기간의 시작/종료 날짜 결정 (차트 도메인용) ---
        if not analysis_df.empty:
            # Timestamp 타입으로 가져와야 시간 정보 포함 가능
            selected_start_dt = analysis_df['측정일시'].min()
            selected_end_dt = analysis_df['측정일시'].max()
            # 기간별 선택 시 종료일 다음날 0시까지 포함하도록 조정 (선택한 날짜 전체 포함)
            if filter_type == "기간별 선택" and isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
                # [수정] 종료일을 선택한 날짜의 23:59:59로 설정
                selected_end_dt = pd.Timestamp(selected_range[1] + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1) 
                # [수정] 시작일을 선택한 날짜의 00:00:00으로 설정
                selected_start_dt = pd.Timestamp(selected_range[0])

        else: # analysis_df가 비어있거나 초기 상태일 경우 전체 기간으로 설정
            selected_start_dt = df['측정일시'].min()
            selected_end_dt = df['측정일시'].max()
            analysis_title = "전체 기간" # 제목 업데이트
            if not (filter_type == "월별 선택" or (filter_type == "기간별 선택" and isinstance(selected_range, (list, tuple)) and len(selected_range) == 2)):
                st.info("기간을 선택하면 해당 구간을 확대하여 보여줍니다. (현재 전체 기간 표시 중)")


        # --- [신규] 차트 생성 함수 정의 ---
        
        # [!!!!] <<<--- 여기가 수정된 함수입니다 [!!!!]
        def create_daily_chart(full_df, start_dt, end_dt):
            # 1. 전체 기간(full_df)을 기준으로 일별 집계
            daily_summary = full_df.groupby('날짜').agg(
                total_usage=('전력사용량(kWh)', 'sum'),
                total_bill=('전기요금(원)', 'sum')
            ).reset_index()
            
            # 2. [중요] '날짜'(date) 컬럼을 'Timestamp'(datetime)로 변환
            #    그래야 'start_dt', 'end_dt' (Timestamp)와 타입이 일치합니다.
            daily_summary['날짜'] = pd.to_datetime(daily_summary['날짜'])

            # 3. Altair가 이중축을 그릴 수 있도록 데이터 포맷 변경 (Melt)
            daily_summary_melted = daily_summary.melt(
                var_name='범주', value_name='값', id_vars=['날짜'], value_vars=['total_usage', 'total_bill']
            )
            daily_summary_melted['범주'] = daily_summary_melted['범주'].map({
                'total_usage': '총 사용량 (kWh)', 'total_bill': '총 전기요금 (원)'
            })
            
            # 4. 기본 차트 정의 (데이터는 1~11월 전체)
            base = alt.Chart(daily_summary_melted).encode(
                # 5. [중요] X축의 'scale'에 'domain'을 지정하여 
                #    선택된 기간(start_dt, end_dt)을 "초기 뷰"로 설정
                x=alt.X('날짜:T', 
                        axis=alt.Axis(title='날짜', format='%Y-%m-%d'),
                        scale=alt.Scale(domain=[start_dt, end_dt])), # <--- 이 부분이 창(Window) 역할
                color=alt.Color('범주:N', legend=alt.Legend(title=None, orient='top-left', fillColor='white', padding=5)),
                tooltip=['날짜', '범주', alt.Tooltip('값', title='값', format=',.0f')]
            )
            
            # 6. 이중 축을 위한 두 개의 라인 생성
            usage_line = base.mark_line(point=alt.MarkConfig(opacity=0.3, size=10)).encode(
                y=alt.Y('값:Q', title='총 사용량 (kWh)')
            ).transform_filter(alt.datum.범주 == '총 사용량 (kWh)')
            
            bill_line = base.mark_line(point=alt.MarkConfig(opacity=0.3, size=10), color='darkorange').encode(
                y=alt.Y('값:Q', title='총 전기요금 (원)')
            ).transform_filter(alt.datum.범주 == '총 전기요금 (원)')
            
            # 7. 차트 합치기, Y축 분리, X축만 interactive하게 설정
            return alt.layer(usage_line, bill_line).resolve_scale(
                y='independent'
            ).interactive(bind_y=False) # X축 줌/패닝 활성화

        def create_hourly_comparison_chart(full_df, analysis_df_for_avg, title_for_avg):
            overall_hourly_avg = full_df.groupby('시간')['전력사용량(kWh)'].mean().reset_index()
            overall_hourly_avg['구분'] = '전체 평균 (1-11월)'
            if not analysis_df_for_avg.empty:
                hourly_avg = analysis_df_for_avg.groupby('시간')['전력사용량(kWh)'].mean().reset_index()
                hourly_avg['구분'] = f'{title_for_avg} 평균'
                combined_hourly = pd.concat([overall_hourly_avg, hourly_avg])
            else:
                combined_hourly = overall_hourly_avg # 선택 기간 데이터 없으면 전체 평균만 표시

            area = alt.Chart(combined_hourly).mark_area(opacity=0.3, color='lightgray').encode(
                x=alt.X('시간:Q', axis=alt.Axis(title='시간 (0-23시)')),
                y=alt.Y('전력사용량(kWh):Q', title='평균 전력사용량 (kWh)'),
                tooltip=[alt.Tooltip('시간', format='d'), alt.Tooltip('전력사용량(kWh)', format='.2f', title='평균 사용량'), '구분']
            ).transform_filter(alt.datum.구분 == '전체 평균 (1-11월)')
            line = alt.Chart(combined_hourly).mark_line(point=True, color='steelblue').encode(
                x='시간:Q', y='전력사용량(kWh):Q',
                tooltip=[alt.Tooltip('시간', format='d'), alt.Tooltip('전력사용량(kWh)', format='.2f', title='평균 사용량'), '구분']
            ).transform_filter(alt.datum.구분 == f'{title_for_avg} 평균')

            # analysis_df가 비어있으면 line은 그려지지 않음 (transform_filter에 의해)
            return alt.layer(area, line).interactive(bind_y=False) # interactive(bind_y=False) 대신 사용

        def create_pf_chart(full_df, pf_col_name, time_filter_expr, threshold, color, title_time, start_dt, end_dt):
            # 시간 필터링 및 유효값 필터링
            pf_data = full_df[full_df.eval(time_filter_expr) & (full_df[pf_col_name] > 0)].copy()

            if pf_data.empty: return None # 데이터 없으면 None 반환

            line = alt.Chart(pf_data).mark_line(
                point=alt.MarkConfig(opacity=0.3, size=10), color=color
            ).encode(
                x=alt.X('측정일시:T', title='측정일시', axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45),
                        scale=alt.Scale(domain=[start_dt, end_dt])), # 시간 정보 포함된 datetime 사용
                y=alt.Y(f'{pf_col_name}:Q', title=f'{pf_col_name.split("(")[0]} (%)', scale=alt.Scale(zero=False, padding=0.1)),
                tooltip=[alt.Tooltip('측정일시', format="%Y-%m-%d %H:%M"), f'{pf_col_name}']
            ).interactive(bind_y=False) # interactive(bind_y=False) 대신 사용
            rule = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(
                color=color, strokeDash=[5, 5], size=2 # 기준선 색상을 라인 색상과 맞춤 (red/blue 대신)
            ).encode(
                y='threshold:Q', tooltip=[alt.Tooltip('threshold', title='기준치')]
            )
            return line + rule


        # --- [수정] 트렌드 시각화 섹션 (함수 호출) ---
        st.write(f"#### {analysis_title} 트렌드 (전체 기간 데이터 표시)")
        col1_viz, col2_viz = st.columns(2)
        with col1_viz:
            st.write(f"**일별 사용량 및 요금** (초기 표시: {analysis_title})")
            # [중요] full_df(df)와 선택된 날짜 범위(selected_start_dt, selected_end_dt) 전달
            daily_chart = create_daily_chart(df, selected_start_dt, selected_end_dt)
            st.altair_chart(daily_chart, use_container_width=True)
        with col2_viz:
            st.write(f"**시간대별 평균 사용량** ({analysis_title} vs 전체 평균)")
            # 시간대별 비교는 analysis_df가 필요
            hourly_chart = create_hourly_comparison_chart(df, analysis_df, analysis_title)
            st.altair_chart(hourly_chart, use_container_width=True)

        st.divider()

        # --- [수정] 역률 상세 분석 섹션 (함수 호출) ---
        st.subheader(f"{analysis_title} 역률 상세 분석 (전체 기간 데이터 표시)")
        col1_sec3, col2_sec3 = st.columns(2)
        with col1_sec3:
            st.write(f"**지상역률(%) 추이 (09-23시)** (초기 표시: {analysis_title})")
            lagging_chart = create_pf_chart(
                full_df=df, pf_col_name='지상역률(%)',
                time_filter_expr='(시간 >= 9) & (시간 <= 23)', # 필터 표현식 전달
                threshold=90.0, color='darkorange', title_time='09-23시',
                start_dt=selected_start_dt, end_dt=selected_end_dt
            )
            if lagging_chart:
                st.altair_chart(lagging_chart, use_container_width=True)
                # Metric 계산 (analysis_df 기준)
                lagging_data_selected = analysis_df[
                    (analysis_df['시간'] >= 9) & (analysis_df['시간'] <= 23) & (analysis_df['지상역률(%)'] > 0)
                ]
                if not lagging_data_selected.empty:
                    below_90 = (lagging_data_selected['지상역률(%)'] < 90).sum()
                    total_lagging_obs = len(lagging_data_selected)
                    percent_below = (below_90 / total_lagging_obs) * 100 if total_lagging_obs > 0 else 0
                    st.metric(label="90% 미만 측정 비율 (패널티 구간)", value=f"{percent_below:.1f} %",
                                help=f"{analysis_title} 기간(09-23시) 중 {below_90} / {total_lagging_obs} 회")
                else:
                    st.metric(label="90% 미만 측정 비율 (패널티 구간)", value="N/A", help=f"{analysis_title} 기간(09-23시) 데이터 없음")
            else:
                st.info("전체 기간(09-23시)에 유효한 지상역률 데이터가 없습니다.")

        with col2_sec3:
            st.write(f"**진상역률(%) 추이 (23-09시)** (초기 표시: {analysis_title})")
            leading_chart = create_pf_chart(
                full_df=df, pf_col_name='진상역률(%)',
                time_filter_expr='(시간 >= 23) | (시간 < 9)', # 필터 표현식 전달
                threshold=95.0, color='steelblue', title_time='23-09시',
                start_dt=selected_start_dt, end_dt=selected_end_dt
            )
            if leading_chart:
                st.altair_chart(leading_chart, use_container_width=True)
                # Metric 계산 (analysis_df 기준)
                leading_data_selected = analysis_df[
                    ((analysis_df['시간'] >= 23) | (analysis_df['시간'] < 9)) & (analysis_df['진상역률(%)'] > 0)
                ]
                if not leading_data_selected.empty:
                    below_95 = (leading_data_selected['진상역률(%)'] < 95).sum()
                    total_leading_obs = len(leading_data_selected)
                    percent_below = (below_95 / total_leading_obs) * 100 if total_leading_obs > 0 else 0
                    st.metric(label="95% 미만 측정 비율 (패널티 구간)", value=f"{percent_below:.1f} %",
                                help=f"{analysis_title} 기간(23-09시) 중 {below_95} / {total_leading_obs} 회")
                else:
                    st.metric(label="95% 미만 측정 비율 (패널티 구간)", value="N/A", help=f"{analysis_title} 기간(23-09시) 데이터 없음")
            else:
                st.info("전체 기간(23-09시)에 유효한 진상역률 데이터가 없습니다.")

        st.divider()

        # --- 전체 기간 월별 트렌드 (기존과 동일) ---
        st.write("#### 전체 기간 월별 트렌드 (1~11월)")
        # (이 부분은 월별 비교이므로 X축 도메인 설정 불필요)
        chart_data = monthly_summary # .reset_index() 불필요
        monthly_summary_melted = chart_data.melt(
            var_name='범주',
            value_name='값',
            id_vars=['월'],
            value_vars=['total_usage', 'total_bill'] # <-- 이 인수들을 다시 추가
        )
        monthly_summary_melted['범주'] = monthly_summary_melted['범주'].map({
            'total_usage': '총 사용량 (kWh)',
            'total_bill': '총 전기요금 (원)'
        })
        base_monthly = alt.Chart(monthly_summary_melted).encode(
            x=alt.X('월:O', axis=alt.Axis(title='월', labelAngle=0, labelExpr="datum.value + '월'")),
            color=alt.Color('범주:N', legend=alt.Legend(title=None, orient='top-right', fillColor='white', padding=5)),
            tooltip=['월', '범주', alt.Tooltip('값', title='값', format=',.0f')]
        ).interactive(bind_y=False) # bind_y=False 제거
        usage_line_monthly = base_monthly.transform_filter(alt.datum.범주 == '총 사용량 (kWh)').mark_line(point=True).encode(y=alt.Y('값:Q', title='총 사용량 (kWh)'))
        bill_line_monthly = base_monthly.transform_filter(alt.datum.범주 == '총 전기요금 (원)').mark_line(point=True).encode(y=alt.Y('값:Q', title='총 전기요G (원)'))
        dual_axis_chart = alt.layer(usage_line_monthly, bill_line_monthly).resolve_scale(y='independent')
        st.altair_chart(dual_axis_chart, use_container_width=True)

        # --- 상세 데이터 Expander (기존과 동일, analysis_df 사용) ---
        with st.expander(f"Dataframe: {analysis_title} 상세 데이터 보기"):
            if not analysis_df.empty:
                st.dataframe(analysis_df) # 선택된 기간의 데이터만 보여줌
            else:
                st.write(f"선택된 '{analysis_title}' 기간에 데이터가 없습니다. 전체 데이터를 보려면 기간을 다시 선택하세요.")

    else: # df is None
        st.error("데이터 파일을 로드할 수 없습니다.") # 기존 오류 메시지 유지