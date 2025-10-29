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
    page_title="전기요금 예측 프로젝트",
    page_icon="💡", # 아이콘
    layout="wide", # wide, centered
)

# ---------------------------------
# 사이드바 (메뉴)
# ---------------------------------
with st.sidebar:
    st.title("💡 전기요금 예측 프로젝트")
    st.write("11월까지의 데이터를 기반으로 12월 전기요금을 예측합니다.")
    
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
    st.title("⚡ 12월 전기요금 실시간 예측 시뮬레이션")
    st.write("1-2초마다 12월(test.csv)의 다음 시간대 데이터를 받아 실시간으로 요금을 예측합니다.")

    # 1. 모델 로드
    models = load_models_and_encoders()
    
    if models: # 모델 로딩에 성공한 경우
        
        # --- [수정] 시뮬레이션 제어 버튼 ---
        col1, col2 = st.columns([1, 1])
        with col1:
            # [수정] 버튼 텍스트 변경
            if st.button("▶️ 12월 실시간 예측 시작"):
                
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
        st.subheader("🗓️ 12월 예측 집계")
        metric_cols = st.columns(2)
        total_bill_metric = metric_cols[0].empty()
        total_usage_metric = metric_cols[1].empty()

        st.subheader("⏱️ 현재 예측")
        latest_time_placeholder = st.empty()
        latest_pred_placeholder = st.empty()
        
        # [수정] SHAP 레이아웃 제거
        st.subheader("📈 12월 시간대별 예측 요금 추이 (최근 1일)")
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

                # 2-11. Chart Update (Request 1: 최근 1일치)
                results_df = pd.concat(st.session_state.predictions)
                display_df = results_df.tail(96) # 최근 96개 (1일치) 데이터만
                
                chart = alt.Chart(display_df).mark_line().encode(
                    x=alt.X('측정일시:T', title='측정일시'),
                    y=alt.Y('예측요금(원):Q', title='예측요금 (원)'),
                    tooltip=['측정일시', alt.Tooltip('예측요금(원)', format=',.0f')]
                ).interactive()
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
            
            # [수정] Request 1: 최근 1일치 표시
            results_df = pd.concat(st.session_state.predictions)
            display_df = results_df.tail(96) # 최근 96개 (1일치) 데이터만
            
            chart = alt.Chart(display_df).mark_line().encode(
                x=alt.X('측정일시:T', title='측정일시'),
                y=alt.Y('예측요금(원):Q', title='예측요금 (원)'),
                tooltip=['측정일시', alt.Tooltip('예측요금(원)', format=',.0f')]
            ).interactive()
            chart_placeholder.altair_chart(chart, use_container_width=True)
            
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
    st.title("📊 과거 전력사용량 분석 (1월 ~ 11월)")
    st.write("학습(Train) 데이터인 과거 11개월간의 전력 사용량 및 관련 데이터를 분석합니다.")

    # --- 실제 데이터 로드 ---
    @st.cache_data  # 데이터 로딩 및 처리를 캐시하여 속도 향상
    def load_data(filepath="./data/train.csv"): # 경로를 "train.csv"로 수정 (app.py와 같은 위치 기준)
        try:
            df = pd.read_csv(filepath)
            df['측정일시'] = pd.to_datetime(df['측정일시'])
            df['월'] = df['측정일시'].dt.month
            df['일'] = df['측정일시'].dt.day
            df['시간'] = df['측정일시'].dt.hour
            # 월별 집계를 위해 '연-월' 컬럼 추가
            df['연월'] = df['측정일시'].dt.to_period('M').astype(str)
            return df
        except FileNotFoundError:
            st.error(f"'{filepath}' 파일을 찾을 수 없습니다. 'app.py'와 같은 위치에 파일을 두었는지 확인해주세요.")
            return None

    df = load_data()

    if df is not None:
        st.subheader("1. 전체 기간(1~11월) 개요")

        total_usage = df['전력사용량(kWh)'].sum()
        total_bill = df['전기요금(원)'].sum()
        avg_hourly_usage = df['전력사용량(kWh)'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric(label="총 전력사용량", value=f"{total_usage:,.0f} kWh")
        col2.metric(label="총 전기요금", value=f"{total_bill:,.0f} 원")
        col3.metric(label="평균 시간당 사용량", value=f"{avg_hourly_usage:,.2f} kWh")

        st.divider()

        st.subheader("2. 월별 상세 분석")
        
        # --- 월별 집계 데이터 생성 ---
        monthly_summary = df.groupby('월').agg(
            total_usage=('전력사용량(kWh)', 'sum'),
            total_bill=('전기요금(원)', 'sum'),
            avg_usage=('전력사용량(kWh)', 'mean')
        ).reset_index()

        # --- 월 선택 ---
        month_list = sorted(df['월'].unique())
        selected_month = st.selectbox(
            "분석할 월을 선택하세요:", 
            month_list, 
            format_func=lambda x: f"{x}월" # 1 -> 1월
        )

        # --- 선택된 월의 데이터 필터링 ---
        month_df = df[df['월'] == selected_month]
        
        # --- 지난달 데이터 필터링 ---
        prev_month_df = pd.DataFrame() # 빈 데이터프레임으로 초기화
        delta_usage = None
        delta_bill = None

        if selected_month > 1: # 1월이 아닐 경우
            prev_month_summary = monthly_summary[monthly_summary['월'] == (selected_month - 1)]
            if not prev_month_summary.empty:
                current_val_usage = monthly_summary[monthly_summary['월'] == selected_month]['total_usage'].values[0]
                prev_val_usage = prev_month_summary['total_usage'].values[0]
                delta_usage = int(current_val_usage - prev_val_usage)
                current_val_bill = monthly_summary[monthly_summary['월'] == selected_month]['total_bill'].values[0]
                prev_val_bill = prev_month_summary['total_bill'].values[0]
                delta_bill = int(current_val_bill - prev_val_bill) # float로 명시적 변환

        # [수정] 2페이지 델타 포맷팅을 위한 로직 추가
        delta_usage_str = None
        delta_bill_str = None
        
        # 델타 색상 결정 (inverse: +는 빨강, -는 초록)
        delta_usage_color = "inverse" if delta_usage is None or delta_usage >= 0 else "normal"
        delta_bill_color = "inverse" if delta_bill is None or delta_bill >= 0 else "normal"

        if delta_usage is not None:
            # 쉼표(,) 포맷팅 및 단위 추가, 부호(+) 명시
            delta_usage_str = f"{delta_usage:+,} kWh"
        
        if delta_bill is not None:
            # 쉼표(,) 포맷팅 및 단위 추가, 부호(+) 명시
            delta_bill_str = f"{delta_bill:+,} 원"

        # --- 선택한 월의 지표 표시 (지난달과 비교) ---
        st.write(f"#### 📈 {selected_month}월 주요 지표 (지난달 대비)")
        col1, col2, col3 = st.columns(3)
        
        current_month_stats = monthly_summary[monthly_summary['월'] == selected_month]
        
        col1.metric(
            label=f"{selected_month}월 총 사용량", 
            value=f"{current_month_stats['total_usage'].values[0]:,.0f} kWh",
            delta=delta_usage_str, # [수정] 포맷팅된 문자열 전달
            delta_color=delta_usage_color # [수정] 수동으로 계산된 색상 전달
        )
        
        col2.metric(
            label=f"{selected_month}월 총 전기요금", 
            value=f"{current_month_stats['total_bill'].values[0]:,.0f} 원",
            delta=delta_bill_str, # [수정] 포맷팅된 문자열 전달
            delta_color=delta_bill_color # [수정] 수동으로 계산된 색상 전달
        )

        col3.metric(
            label=f"{selected_month}월 평균 시간당 사용량", 
            value=f"{current_month_stats['avg_usage'].values[0]:,.0f} kWh" # [수정] 소수점 제거
        )
        
        st.divider()

        # --- 월별 시각화 ---
        st.write("#### 📊 월별 트렌드 시각화")
        
        col1_viz, col2_viz = st.columns(2)

        with col1_viz:
            st.write(f"**{selected_month}월 일별 사용량 및 요금 (이중축)**")
            
            # 1. 일별 데이터 집계
            daily_summary = month_df.groupby('일').agg(
                total_usage=('전력사용량(kWh)', 'sum'),
                total_bill=('전기요금(원)', 'sum')
            ).reset_index()

            # [수정] 2. 데이터 Melt (범주/레전드 생성)
            # [오류 수정] '일'을 위치 인자로 전달하지 않고, id_vars 키워드 인자만 사용
            daily_summary_melted = daily_summary.melt(
                var_name='범주',
                value_name='값',
                id_vars=['일'], # '일' 컬럼을 기준으로 melt
                value_vars=['total_usage', 'total_bill']
            )
            daily_summary_melted['범주'] = daily_summary_melted['범주'].map({
                'total_usage': '총 사용량 (kWh)',
                'total_bill': '총 전기요금 (원)'
            })
            
            # 3. Altair 이중축 차트 (Melted data 기반)
            base = alt.Chart(daily_summary_melted).encode(
                x=alt.X('일:Q', axis=alt.Axis(title='일', format='d')),
                color=alt.Color('범주:N', title='범주'), # 범주(Legend) 생성
                tooltip=['일', '범주', alt.Tooltip('값', title='값', format=',.2f')] # 툴팁 소수점 둘째자리
            ).interactive()

            # 사용량 (kWh) - Y축1
            usage_line = base.transform_filter(
                alt.datum.범주 == '총 사용량 (kWh)'
            ).mark_line(point=True).encode(
                y=alt.Y('값:Q', title='총 사용량 (kWh)')
            )
            
            # 전기요금 (원) - Y축2
            bill_line = base.transform_filter(
                alt.datum.범주 == '총 전기요금 (원)'
            ).mark_line(point=True).encode(
                y=alt.Y('값:Q', title='총 전기요금 (원)')
            )

            # 4. 차트 결합 (Layer)
            dual_axis_daily_chart = alt.layer(usage_line, bill_line).resolve_scale(
                y='independent' # Y축을 독립적으로 사용
            )
            
            # 5. Streamlit에 표시
            st.altair_chart(dual_axis_daily_chart, use_container_width=True)


        with col2_viz:
            st.write(f"**{selected_month}월 시간대별 전력 사용량 (평균)**")
            # 시간대별 평균
            hourly_avg = month_df.groupby('시간')['전력사용량(kWh)'].mean()
            st.line_chart(hourly_avg)
            
        st.write("**전체 기간 월별 총 사용량 및 전기요금 비교 (이중축)**") 
        
        chart_data = monthly_summary.reset_index()

        # [수정] 1. 데이터 Melt (범주/레전드 생성)
        # [오류 수정] '월'을 위치 인자로 전달하지 않고, id_vars 키워드 인자만 사용
        monthly_summary_melted = chart_data.melt(
            var_name='범주',
            value_name='값',
            id_vars=['월'], # '월' 컬럼을 기준으로 melt
            value_vars=['total_usage', 'total_bill']
        )
        monthly_summary_melted['범주'] = monthly_summary_melted['범주'].map({
            'total_usage': '총 사용량 (kWh)',
            'total_bill': '총 전기요금 (원)'
        })

        # 2. Altair 이중축 차트 (Melted data 기반)
        base_monthly = alt.Chart(monthly_summary_melted).encode(
            x=alt.X('월:O', axis=alt.Axis(title='월', labelAngle=0, labelExpr="datum.value + '월'")),
            color=alt.Color('범주:N', title='범주'), # 범주(Legend) 생성
            tooltip=['월', '범주', alt.Tooltip('값', title='값', format=',.2f')] # 툴팁 소수점 둘째자리
        ).interactive()
        
        # 3. 사용량 (Line) - Y축1
        usage_line_monthly = base_monthly.transform_filter(
            alt.datum.범주 == '총 사용량 (kWh)'
        ).mark_line(point=True).encode(
            y=alt.Y('값:Q', title='총 사용량 (kWh)')
        )

        # 4. 전기요금 (Line) - Y축2
        bill_line_monthly = base_monthly.transform_filter(
            alt.datum.범주 == '총 전기요금 (원)'
        ).mark_line(point=True).encode(
            y=alt.Y('값:Q', title='총 전기요금 (원)')
        )

        # 5. 이중축 차트 결합 (Line + Line)
        dual_axis_chart = alt.layer(usage_line_monthly, bill_line_monthly).resolve_scale(
            y='independent' # Y축을 독립적으로 설정
        )

        st.altair_chart(dual_axis_chart, use_container_width=True)

        
        st.divider()
        
        st.subheader(f"3. {selected_month}월 상세 분석") 
        
        col1_sec3, col2_sec3 = st.columns(2) # 2열 레이아웃 생성

        with col1_sec3:
            st.write(f"**{selected_month}월 작업 유형별 전력 사용량 (Pie Chart)**")
            
            # Pie Chart 데이터 준비
            work_type_usage = month_df.groupby('작업유형')['전력사용량(kWh)'].sum().reset_index()
            work_type_usage = work_type_usage.rename(columns={'전력사용량(kWh)': '사용량'})
            # 비율 계산
            work_type_usage['percent'] = (work_type_usage['사용량'] / work_type_usage['사용량'].sum())

            # Altair Pie Chart
            base = alt.Chart(work_type_usage).encode(
               theta=alt.Theta("사용량:Q", stack=True)
            ).properties(title=f'{selected_month}월 작업 유형별 사용량')

            # 파이 차트 부분
            pie = base.mark_arc(outerRadius=120, innerRadius=0).encode(
                color=alt.Color("작업유형:N"), # 작업유형별 색상
                order=alt.Order("사용량", sort="descending"), # 큰 순서대로 정렬
                tooltip=["작업유형", 
                         alt.Tooltip("사용량", format=",.2f", title="사용량(kWh)"), 
                         alt.Tooltip("percent", title="비율", format=".1%")]
            )

            # 텍스트 (비율)
            text = base.mark_text(radius=140).encode(
                text=alt.Text("percent", format=".1%"),
                order=alt.Order("사용량", sort="descending"),
                color=alt.value("black")  # 텍스트 색상
            )
            
            chart_pie = pie + text
            st.altair_chart(chart_pie, use_container_width=True)

        with col2_sec3:
            st.write(f"**{selected_month}월 작업 유형별 탄소 배출량 (Bar Chart)**")
            
            # Bar Chart 데이터 준비
            work_type_carbon = month_df.groupby('작업유형')['탄소배출량(tCO2)'].sum().reset_index()
            work_type_carbon = work_type_carbon.rename(columns={'탄소배출량(tCO2)': '총탄소배출량'})

            # Altair Bar Chart
            chart_carbon = alt.Chart(work_type_carbon).mark_bar().encode(
                x=alt.X('작업유형:N', title='작업 유형'),
                y=alt.Y('총탄소배출량:Q', title='총 탄소 배출량 (tCO2)'),
                color='작업유형:N', # 작업유형별 색상
                tooltip=['작업유형', alt.Tooltip('총탄소배출량', title='총 배출량 (tCO2)', format=',.2f')] 
            ).interactive()
            
            st.altair_chart(chart_carbon, use_container_width=True)
        

        # --- 상세 데이터 ---
        with st.expander(f"Dataframe: {selected_month}월 상세 데이터 보기"):
            st.dataframe(month_df)

