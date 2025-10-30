# VSCode를 쓰신다면, 상단 메뉴에서 [Terminal] → [New Terminal] 클릭

# 아래 명령어 한 번만 입력:
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time 
from fpdf import FPDF
from datetime import datetime
import io 

# -----------------------------
# [삭제] 예측 모델/함수 섹션
# (모두 삭제됨)
# -----------------------------

# -----------------------------
# [유지] 전처리 및 PDF 생성 함수
# -----------------------------
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

# [유지] create_comparison_table_data 함수 (PDF 생성에 필요)
def create_comparison_table_data(train_df, results_df):
    if train_df is None or results_df.empty:
        return pd.DataFrame() 
        
    try:
        # 1. 지난 달 (11월) 평균
        nov_df = train_df[train_df['월'] == 11].copy()
        nov_hourly_avg = nov_df.groupby('시간')['전기요금(원)'].mean()

        # 2. 어제 (Yesterday)
        latest_datetime = results_df['측정일시'].iloc[-1]
        latest_date = latest_datetime.date()
        yesterday_date = latest_date - pd.Timedelta(days=1)
        
        yesterday_df = results_df[results_df['측정일시'].dt.date == yesterday_date]
        
        if yesterday_df.empty:
            # 12월 1일인 경우, 어제(11월 30일)는 train_df에 있음
            yesterday_df = train_df[train_df['측정일시'].dt.date == yesterday_date]
            if not yesterday_df.empty:
                yesterday_hourly = yesterday_df.groupby('시간')['전기요금(원)'].mean()
            else:
                yesterday_hourly = pd.Series(dtype=float)
        else:
            # 12월 2일 이후인 경우, 어제는 results_df에 있음
            yesterday_hourly = yesterday_df.groupby('시간')['예측요금(원)'].mean()
        
        # 3. 오늘 (Today - 현재까지)
        today_df = results_df[results_df['측정일시'].dt.date == latest_date]
        today_hourly = today_df.groupby('시간')['예측요금(원)'].mean()

        # 4. DataFrame으로 취합
        comp_df = pd.DataFrame({
            "11월 평균": nov_hourly_avg,
            "어제": yesterday_hourly,
            "오늘": today_hourly
        }).reindex(range(24))
        
        # 5. 전일 대비 차이 계산
        comp_df["전일 대비"] = comp_df["오늘"] - comp_df["어제"].fillna(0)
        
        return comp_df.fillna(np.nan)
        
    except Exception as e:
        st.error(f"비교 테이블 데이터 생성 중 오류 발생: {e}")
        return pd.DataFrame()

# [유지] 폰트 경로 및 train.csv 로드 함수
FONT_PATH_REGULAR = "www/fonts/NanumGothic-Regular.ttf" 
FONT_PATH_BOLD = "www/fonts/NanumGothic-Bold.ttf"

@st.cache_data
def load_train_data():
    """train_.csv를 로드하고 기본 전처리를 수행합니다. (앱 실행 시 1회)"""
    try:
        train_df = pd.read_csv("./data/train_.csv") 
        train_df = enrich(train_df) 
        return train_df
    except FileNotFoundError:
        st.error("data/train_.csv 파일을 찾을 수 없습니다. 시뮬레이션 및 PDF 생성이 불가합니다.")
        return None
    except KeyError:
        st.error("data/train_.csv 파일에 '전기요금(원)' 컬럼이 없습니다. 비교 데이터 생성에 필요합니다.")
        return None

# [유지] PDF 생성 함수
def generate_bill_pdf(report_data, comparison_df=None): 
    try:
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.add_font('Nanum', '', FONT_PATH_REGULAR, uni=True)
        pdf.add_font('Nanum', 'B', FONT_PATH_BOLD, uni=True)
        pdf.set_font('Nanum', '', 10)
        
        # [!!!] 3. (날짜 헤더 추가) [!!!]
        # PDF 생성 함수 상단에서 날짜 헤더 문자열 미리 만들기
        yesterday_header = f"어제 ({report_data.get('yesterday_str', '')})"
        today_header = f"오늘 ({report_data.get('today_str', '')})"
        
        # --- 1~4. 상단 정보 (기존과 동일) ---
        pdf.set_font_size(18)
        pdf.cell(0, 15, "12월 실시간 예측 전기요금 명세서", border=1, ln=1, align='C')
        pdf.ln(3) 
        
        pdf.set_font_size(12)
        pdf.cell(0, 8, " [ 예측 고객 정보 ]", border='B', ln=1)
        col_width = pdf.w / 2 - 12 
        pdf.cell(col_width, 8, "고객명: LS 청주공장", border=0)
        pdf.cell(col_width, 8, f"청구서 발행일: {report_data['report_date'].strftime('%Y-%m-%d')}", border=0, ln=1)
        start_str = report_data['period_start'].strftime('%Y-%m-%d %H:%M')
        end_str = report_data['period_end'].strftime('%Y-%m-%d %H:%M')
        pdf.multi_cell(0, 6, f"예측 기간: {start_str} ~ {end_str}", border=0, ln=1)
        pdf.ln(3) 

        pdf.set_fill_color(240, 240, 240) 
        pdf.set_font_size(14)
        pdf.cell(40, 12, "총 예측 요금", border=1, align='C', fill=True)
        pdf.set_font_size(16)
        pdf.cell(0, 12, f"{report_data['total_bill']:,.0f} 원", border=1, ln=1, align='R')
        pdf.ln(3) 

        # --- 5. 세부 내역 (기존과 동일) ---
        pdf.set_font_size(12)
        pdf.cell(0, 8, " [ 예측 세부 내역 ]", border='B', ln=1)
        
        pdf.set_font_size(11)
        pdf.set_fill_color(240, 240, 240)
        header_h = 8
        w1, w2, w3, w4 = 45, 50, 50, 45 
        pdf.cell(w1, header_h, "항목 (부하구분)", border=1, align='C', fill=True)
        pdf.cell(w2, header_h, "예측 사용량 (kWh)", border=1, align='C', fill=True)
        pdf.cell(w3, header_h, "예측 요금 (원)", border=1, align='C', fill=True)
        pdf.cell(w4, header_h, "요금/사용량 (원/kWh)", border=1, ln=1, align='C', fill=True) 
        
        pdf.set_font_size(10)
        bands = ['경부하', '중간부하', '최대부하']
        for band in bands:
            usage = report_data['usage_by_band'].get(band, 0.0)
            bill = report_data['bill_by_band'].get(band, 0.0)
            cost_per_kwh = bill / usage if usage > 0 else 0.0 
            
            pdf.cell(w1, header_h, band, border=1, align='C')
            pdf.cell(w2, header_h, f"{usage:,.2f}", border=1, align='R')
            pdf.cell(w3, header_h, f"{bill:,.0f}", border=1, align='R')
            pdf.cell(w4, header_h, f"{cost_per_kwh:,.1f}", border=1, ln=1, align='R')
            
        pdf.set_font('Nanum', 'B', 11) 
        total_usage = report_data['total_usage']
        total_bill = report_data['total_bill']
        total_cost_per_kwh = total_bill / total_usage if total_usage > 0 else 0.0
        
        pdf.cell(w1, header_h, "합계", border=1, align='C', fill=True)
        pdf.cell(w2, header_h, f"{total_usage:,.2f}", border=1, align='R', fill=True)
        pdf.cell(w3, header_h, f"{total_bill:,.0f}", border=1, align='R', fill=True)
        pdf.cell(w4, header_h, f"{total_cost_per_kwh:,.1f}", border=1, ln=1, align='R', fill=True)
        
        pdf.ln(5) 
        
        # --- [!!!] 6. 주요 요금 결정 지표 (2번 요청사항 반영) [!!!] ---
        pdf.set_font('Nanum', '', 12)
        pdf.cell(0, 8, " [ 주요 요금 결정 지표 (예측) ]", border='B', ln=1)
        pdf.ln(1)
        
        start_y = pdf.get_y()
        col_width = 95 
        
        # --- 1. 왼쪽 컬럼 (기본요금) ---
        pdf.set_x(10) 
        pdf.set_font('Nanum', 'B', 10)
        pdf.multi_cell(col_width, 7, "1. 기본요금 (Demand Charge) 지표", border=0, align='L')
        
        pdf.set_font('Nanum', '', 9)
        peak_kw = report_data.get('peak_demand_kw', 0)
        peak_time = report_data.get('peak_demand_time', pd.NaT)
        peak_time_str = peak_time.strftime('%Y-%m-%d %H:%M') if pd.notna(peak_time) else "N/A"
        
        # [!!!] 2. (최저 수요전력 추가) [!!!]
        min_kw = report_data.get('min_demand_kw', 0)
        min_time = report_data.get('min_demand_time', pd.NaT)
        min_time_str = min_time.strftime('%Y-%m-%d %H:%M') if pd.notna(min_time) else "N/A"
        
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 12월 최대 요금적용전력: {peak_kw:,.2f} kW", border=0, align='L')
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 최대치 발생일시: {peak_time_str}", border=0, align='L')
        
        # [!!!] 2. (최저 수요전력 추가) - PDF에 그리기 [!!!]
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 12월 최저 요금적용전력: {min_kw:,.2f} kW", border=0, align='L')
        pdf.set_x(10)
        pdf.multi_cell(col_width, 6, f"  - 최저치 발생일시: {min_time_str}", border=0, align='L')
        
        end_y_left = pdf.get_y()

        # --- 2. 오른쪽 컬럼 (역률요금) ---
        pdf.set_y(start_y) 
        pdf.set_x(10 + col_width) 

        pdf.set_font('Nanum', 'B', 10)
        pdf.multi_cell(col_width, 7, "2. 역률요금 (Power Factor) 지표", border=0, align='L')
        
        pdf.set_font('Nanum', '', 9)
        avg_day_pf = report_data.get('avg_day_pf', 0)
        penalty_d_h = report_data.get('penalty_day_hours', 0)
        bonus_d_h = report_data.get('bonus_day_hours', 0)
        avg_night_pf = report_data.get('avg_night_pf', 0)
        penalty_n_h = report_data.get('penalty_night_hours', 0)
        
        pdf.set_x(10 + col_width)
        pdf.multi_cell(col_width, 6, f"  - 주간(09-23시) 평균 지상역률: {avg_day_pf:.2f} %", border=0, align='L')
        pdf.set_x(10 + col_width)
        pdf.multi_cell(col_width, 6, f"    (페널티[<90%] {penalty_d_h}시간 / 보상[>95%] {bonus_d_h}시간)", border=0, align='L')
        pdf.set_x(10 + col_width)
        pdf.multi_cell(col_width, 6, f"  - 야간(23-09시) 평균 진상역률: {avg_night_pf:.2f} %", border=0, align='L')
        pdf.set_x(10 + col_width)
        pdf.multi_cell(col_width, 6, f"    (페널티[<95%] {penalty_n_h}시간)", border=0, align='L')
        
        end_y_right = pdf.get_y() 

        pdf.set_y(max(end_y_left, end_y_right))
        pdf.ln(5) 
        # --- [!!!] (끝) 새 섹션 수정 완료 [!!!] ---

        # --- [!!!] 7. 시간대별 요금 비교 (표) (3번 요청사항 반영) [!!!] ---
        pdf.set_font('Nanum', '', 12)
        pdf.cell(0, 8, " [ 시간대별 요금 비교 (단위: 원) ]", border='B', ln=1)
        pdf.ln(1) 

        if comparison_df is not None and not comparison_df.empty:
            pdf.set_font('Nanum', '', 8) 
            cell_h = 6 
            w_time = 12 
            w_nov = 21 
            w_yes = 21 
            w_tod = 21 
            w_diff = 20 
            
            # [!!!] 3. (날짜 헤더 추가) [!!!]
            # draw_header 함수가 위에서 정의한 yesterday_header, today_header 변수를 사용
            def draw_header(start_x):
                pdf.set_font('Nanum', 'B', 8)
                pdf.set_x(start_x)
                pdf.cell(w_time, cell_h, "시간", 1, 0, 'C', 1)
                pdf.cell(w_nov, cell_h, "11월 평균", 1, 0, 'C', 1)
                pdf.cell(w_yes, cell_h, yesterday_header, 1, 0, 'C', 1) # 수정됨
                pdf.cell(w_tod, cell_h, today_header, 1, 0, 'C', 1)     # 수정됨
                pdf.cell(w_diff, cell_h, "전일 대비", 1, 0, 'C', 1)

            start_y = pdf.get_y() 
            draw_header(10) 
            pdf.set_y(start_y) 
            draw_header(10 + 95) 
            pdf.ln(cell_h) 
            
            # (이하 테이블 내용 그리는 코드는 동일)
            pdf.set_font('Nanum', '', 8)
            def fmt(val, is_diff=False):
                if pd.isna(val): return "-"
                prefix = "+" if is_diff and val > 0 else ""
                return f"{prefix}{val:,.0f}"

            for i in range(12): 
                row_left = comparison_df.iloc[i]
                pdf.set_x(10)
                pdf.cell(w_time, cell_h, str(i), 1, 0, 'C')
                pdf.cell(w_nov, cell_h, fmt(row_left["11월 평균"]), 1, 0, 'R')
                pdf.cell(w_yes, cell_h, fmt(row_left["어제"]), 1, 0, 'R')
                pdf.cell(w_tod, cell_h, fmt(row_left["오늘"]), 1, 0, 'R')
                pdf.cell(w_diff, cell_h, fmt(row_left["전일 대비"], True), 1, 0, 'R')

                row_right = comparison_df.iloc[i + 12]
                pdf.set_x(10 + 95)
                pdf.cell(w_time, cell_h, str(i + 12), 1, 0, 'C')
                pdf.cell(w_nov, cell_h, fmt(row_right["11월 평균"]), 1, 0, 'R')
                pdf.cell(w_yes, cell_h, fmt(row_right["어제"]), 1, 0, 'R')
                pdf.cell(w_tod, cell_h, fmt(row_right["오늘"]), 1, 0, 'R')
                pdf.cell(w_diff, cell_h, fmt(row_right["전일 대비"], True), 1, 0, 'R')
                
                pdf.ln(cell_h) 
            
            pdf.ln(3) 

        else:
            pdf.set_font_size(10)
            pdf.cell(0, 10, "비교 데이터를 생성할 수 없습니다 (데이터 부족 또는 오류).", border=1, ln=1, align='C')
            pdf.ln(3)
            
        # --- 8. 하단 안내문 (기존과 동일) ---
        pdf.set_font_size(9)
        pdf.multi_cell(0, 5, 
            "* 본 명세서는 '12월 전기요금 실시간 예측 시뮬레이션'을 통해 생성된 예측값이며, "
            "실제 청구되는 요금과 다를 수 있습니다.\n"
            "* 예측 모델: LightGBM, XGBoost, CatBoost 앙상블 모델",
            border=1, align='L'
        )

        # 8. PDF 결과물 반환
        return bytes(pdf.output())

    except FileNotFoundError:
        st.error(f"PDF 생성 오류: 폰트 파일('{FONT_PATH_REGULAR}' 등)을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"PDF 생성 중 알 수 없는 오류 발생: {e}")
        return None

# -----------------------------
# 페이지 기본 설정 (유지)
# -----------------------------
st.set_page_config(
    page_title="전기요금 분석", layout="wide", 
)

# ---------------------------------
# 사이드바 (메뉴) (유지)
# ---------------------------------
with st.sidebar:
    st.title("전기요금 분석")
    
    page = st.radio(
        "페이지 이동",
        ["실시간 전기요금 분석", "과거 전력사용량 분석"],
        label_visibility="collapsed" 
    )
    
    st.divider() 

# ---------------------------------
# 1. 실시간 전기요금 분석 페이지 (수정됨)
# ---------------------------------
if page == "실시간 전기요금 분석":
    st.title(" 12월 전기요금 실시간 예측 시뮬레이션")

    train_df = load_train_data() # 캐시된 train_df 로드 (PDF 비교용)
    
    if (train_df is not None): 
        
        # --- 시뮬레이션 제어 버튼 ---
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("▶️ 시작/재개"):
                if 'current_index' in st.session_state and st.session_state.current_index > 0:
                    st.session_state.simulation_running = True
                
                else: # '처음 시작'
                    try:
                        test_df = pd.read_csv("./data/predicted_test_data.csv")
                        test_df["측정일시"] = pd.to_datetime(test_df["측정일시"])
                        
                    except FileNotFoundError as e:
                        st.error(f"데이터 파일(./data/predicted_test_data.csv)을 찾을 수 없습니다.")
                        st.error("먼저 1단계(학습 스크립트)를 실행하여 예측 파일을 생성해주세요.")
                        st.stop()
                    
                    st.session_state.simulation_running = True
                    st.session_state.current_index = 0
                    st.session_state.test_df = test_df 
                    st.session_state.predictions = [] 
                    st.session_state.total_bill = 0.0
                    st.session_state.total_usage = 0.0
                    st.session_state.errors = []
        
        with col2:
            if st.button("⏹️ 중지"):
                st.session_state.simulation_running = False

        # --- 동적 컨텐츠를 위한 Placeholders (유지) ---
        st.subheader("12월 예측 집계")
        metric_cols = st.columns(2)
        total_bill_metric = metric_cols[0].empty()
        total_usage_metric = metric_cols[1].empty()

        st.subheader("현재 예측")
        latest_time_placeholder = st.empty()
        latest_pred_placeholder = st.empty()
        
        st.subheader("12월 시간대별 예측 요금 추이")
        chart_placeholder = st.empty()
        
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False

        # --- 메인 시뮬레이션 루프 ---
        if st.session_state.simulation_running:
            if 'test_df' not in st.session_state:
                st.error("시뮬레이션 상태가 초기화되지 않았습니다. '시작' 버튼을 다시 눌러주세요.")
                st.session_state.simulation_running = False
            
            elif st.session_state.current_index < len(st.session_state.test_df):
                
                row_df = st.session_state.test_df.iloc[[st.session_state.current_index]].copy()
                
                # [!!!] 예측 로직 없음 - 미리 계산된 값 읽기 [!!!]
                pred_te = row_df["예측요금(원)"].values[0]
                kwh_pred = row_df["전력사용량(kWh)"].values[0] 
                current_time = row_df['측정일시'].iloc[0] 
                
                # 상태 업데이트
                st.session_state.predictions.append(row_df) 
                st.session_state.total_bill += pred_te
                st.session_state.total_usage += kwh_pred
                st.session_state.current_index += 1

                # UI 업데이트 (Placeholders)
                total_bill_metric.metric("12월 누적 예상 전기요금", f"{st.session_state.total_bill:,.0f} 원")
                total_usage_metric.metric("12월 누적 예상 전력사용량", f"{st.session_state.total_usage:,.0f} kWh")
                
                # [!!!] 1. (버그 수정) [!!!]
                # '현재 예측' 섹션 실시간 업데이트
                latest_time_placeholder.write(f"**측정일시:** {current_time}")
                latest_pred_placeholder.write(f"**예측요금:** `{pred_te:,.0f} 원` | **예측사용량:** `{kwh_pred:,.2f} kWh`")

                # Chart Update (유지)
                results_df = pd.concat(st.session_state.predictions)
            
                if not results_df.empty:
                    first_time = results_df['측정일시'].iloc[0]
                    latest_time = results_df['측정일시'].iloc[-1]
                
                    if latest_time < (first_time + pd.Timedelta(hours=24)):
                        x_axis = alt.X('측정일시:T', title='측정일시')
                    else:
                        start_domain = latest_time - pd.Timedelta(hours=24)
                        end_domain = latest_time
                        x_axis = alt.X('측정일시:T',
                                        title='측정일시',
                                        scale=alt.Scale(domain=[start_domain, end_domain])
                                    )
                
                    chart = alt.Chart(results_df).mark_line().encode(
                        x=x_axis, 
                        y=alt.Y('예측요금(원):Q', title='예측요금 (원)'),
                        tooltip=['측정일시', alt.Tooltip('예측요금(원)', format=',.0f')]
                    ).interactive(bind_y=False) 
                
                    chart_placeholder.altair_chart(chart, use_container_width=True)
                
                # Loop (속도 조절)
                time.sleep(1) 
                st.rerun()

            else:
                # 시뮬레이션 완료
                st.session_state.simulation_running = False
                st.success("✅ 12월 전체 예측 시뮬레이션 완료!")
                st.rerun() # 완료 후 '비활성' 섹션 로직을 타기 위해 rerun

        # --- [!!!] 시뮬레이션 비활성 시 (초기/중지/완료) (수정됨) [!!!] ---
        elif 'predictions' in st.session_state and st.session_state.predictions:
            # Metric 표시
            total_bill_metric.metric("12월 누적 예상 전기요금", f"{st.session_state.total_bill:,.0f} 원")
            total_usage_metric.metric("12월 누적 예상 전력사용량", f"{st.session_state.total_usage:,.0f} kWh")
            
            results_df = pd.concat(st.session_state.predictions)

            if not results_df.empty:
                
                # [!!!] 1. (버그 수정) [!!!]
                # 시뮬레이션 중지/완료 시 '현재 예측' 섹션에 "최종" 데이터 표시
                latest_row = results_df.iloc[-1]
                latest_time = latest_row['측정일시']
                latest_bill = latest_row['예측요금(원)']
                latest_kwh = latest_row['전력사용량(kWh)']
                latest_time_placeholder.write(f"**최종 측정일시:** {latest_time}")
                latest_pred_placeholder.write(f"**최종 예측요금:** `{latest_bill:,.0f} 원` | **최종 예측사용량:** `{latest_kwh:,.2f} kWh`")
                
                # --- PDF 다운로드 버튼 로직 ---
                usage_by_band = results_df.groupby('부하구분')['전력사용량(kWh)'].sum()
                bill_by_band = results_df.groupby('부하구분')['예측요금(원)'].sum()
                
                # --- [!!!] 2 & 3번 요청사항 반영 (데이터 집계) [!!!] ---
                
                # 1. 기본요금 (Demand Charge) 지표
                peak_demand_kw = results_df['요금적용전력_kW'].max()
                peak_demand_time = pd.NaT 
                if not pd.isna(peak_demand_kw):
                     peak_demand_time = results_df.loc[results_df['요금적용전력_kW'].idxmax()]['측정일시']
                
                # [!!!] 2. (최저 수요전력 추가) [!!!]
                min_demand_kw = results_df['요금적용전력_kW'].min()
                min_demand_time = pd.NaT
                if not pd.isna(min_demand_kw):
                     min_demand_time = results_df.loc[results_df['요금적용전력_kW'].idxmin()]['측정일시']

                # 2. 역률 (Power Factor) 지표
                daytime_df = results_df[results_df['주간여부'] == 1]
                nighttime_df = results_df[results_df['야간여부'] == 1]

                avg_day_pf = daytime_df['지상역률_주간클립'].mean() if not daytime_df.empty else 0
                penalty_day_hours = len(daytime_df[daytime_df['지상역률_주간클립'] < 90])
                bonus_day_hours = len(daytime_df[daytime_df['지상역률_주간클립'] > 95])
                
                avg_night_pf = nighttime_df['진상역률(%)'].mean() if not nighttime_df.empty else 0
                penalty_night_hours = len(nighttime_df[nighttime_df['진상역률(%)'] < 95])

                # [!!!] 3. (날짜 헤더 추가) [!!!]
                # PDF 헤더에 사용할 날짜 문자열 추출
                yesterday_str = (results_df['측정일시'].iloc[-1].date() - pd.Timedelta(days=1)).strftime('%m-%d')
                today_str = results_df['측정일시'].iloc[-1].date().strftime('%m-%d')
                
                # --- report_data 딕셔너리에 모든 값 추가 ---
                report_data = {
                    "total_bill": st.session_state.total_bill,
                    "total_usage": st.session_state.total_usage,
                    "period_start": results_df['측정일시'].min(),
                    "period_end": results_df['측정일시'].iloc[-1],
                    "report_date": datetime.now(),
                    "usage_by_band": usage_by_band.to_dict(),
                    "bill_by_band": bill_by_band.to_dict(),
                    
                    "peak_demand_kw": peak_demand_kw,
                    "peak_demand_time": peak_demand_time,
                    "min_demand_kw": min_demand_kw,       # 2번
                    "min_demand_time": min_demand_time, # 2번
                    
                    "avg_day_pf": avg_day_pf,
                    "penalty_day_hours": penalty_day_hours,
                    "bonus_day_hours": bonus_day_hours,
                    "avg_night_pf": avg_night_pf,
                    "penalty_night_hours": penalty_night_hours,
                    
                    "yesterday_str": yesterday_str, # 3번
                    "today_str": today_str        # 3번
                }
                
                # 비교 테이블 데이터 생성
                comparison_df = create_comparison_table_data(train_df, results_df)
                
                # PDF 생성 함수 호출
                pdf_bytes = generate_bill_pdf(report_data, comparison_df=comparison_df)
                
                if pdf_bytes:
                    st.download_button(
                        label="📄 예측 요금 명세서 PDF 다운로드",
                        data=pdf_bytes,
                        file_name=f"predicted_bill_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf"
                    )
                
                st.divider() 

                # --- 중지/완료 시 차트 표시 로직 (유지) ---
                first_time = results_df['측정일시'].iloc[0]
                latest_time = results_df['측정일시'].iloc[-1]

                if latest_time < (first_time + pd.Timedelta(hours=24)):
                    x_axis = alt.X('측정일시:T', title='측정일시')
                else:
                    start_domain = latest_time - pd.Timedelta(hours=24)
                    end_domain = latest_time
                    x_axis = alt.X('측정일시:T',
                                    title='측정일시',
                                    scale=alt.Scale(domain=[start_domain, end_domain])
                                )
                
                chart = alt.Chart(results_df).mark_line().encode(
                    x=x_axis, 
                    y=alt.Y('예측요금(원):Q', title='예측요금 (원)'),
                    tooltip=['측정일시', alt.Tooltip('예측요금(원)', format=',.0f')]
                ).interactive(bind_y=False) 
                
                chart_placeholder.altair_chart(chart, use_container_width=True)
            
                # 상세 데이터 expander
                with st.expander("12월 예측 상세 데이터 보기 (최종)"):
                    display_cols = ["측정일시", "작업유형", "전력사용량(kWh)", "예측요금(원)"]
                    if "유효역률(%)" in results_df.columns:
                         display_cols.insert(3, "유효역률(%)")

                    st.dataframe(results_df[display_cols].style.format({
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
# 2. 과거 전력사용량 분석 페이지 (수정 없음)
# ---------------------------------
elif page == "과거 전력사용량 분석":
    # --- (이 페이지의 코드는 이전과 동일합니다) ---
    st.title("과거 전력사용량 분석 (1월 ~ 11월)")

    @st.cache_data 
    def load_data(filepath="./data/train_.csv"):
        try:
            df = pd.read_csv(filepath)
            df['측정일시'] = pd.to_datetime(df['측정일시'])
            df['월'] = df['측정일시'].dt.month
            df['일'] = df['측정일시'].dt.day
            df['시간'] = df['측정일시'].dt.hour
            df['날짜'] = df['측정일시'].dt.date
            df['연월'] = df['측정일시'].dt.to_period('M').astype(str)
            
            if '탄소배출량(tCO2)' not in df.columns:
                df['탄소배출량(tCO2)'] = (df['전력사용량(kWh)'] / 1000) * 0.45 
                
            return df
        except FileNotFoundError:
            st.error(f"'{filepath}' 파일을 찾을 수 없습니다. ./data/ 폴더를 확인해주세요.")
            return None

    df = load_data()

    if df is not None:
        st.subheader("전체 기간(1~11월) 개요")

        total_usage = df['전력사용량(kWh)'].sum()
        total_bill = df['전기요금(원)'].sum()
        avg_hourly_usage = df['전력사용량(kWh)'].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric(label="총 전력사용량", value=f"{total_usage:,.0f} kWh")
        col2.metric(label="총 전기요금", value=f"{total_bill:,.0f} 원")
        col3.metric(label="평균 시간당 사용량", value=f"{avg_hourly_usage:,.2f} kWh")
        st.divider()

        monthly_summary = df.groupby('월').agg(
            total_usage=('전력사용량(kWh)', 'sum'),
            total_bill=('전기요금(원)', 'sum'),
            avg_usage=('전력사용량(kWh)', 'mean')
        ).reset_index()
        min_date = df['측정일시'].min().date()
        max_date = df['측정일시'].max().date()

        st.subheader("기간별 상세 분석")
        col_left, col_right = st.columns(2)

        with col_left:
            st.write("#### 분석 기간 선택 (차트 시작 범위)")
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
                    analysis_df = pd.DataFrame(columns=df.columns) 
                    analysis_title = "기간 미선택"

        with col_right:
            st.write(f"#### {analysis_title} 주요 지표")
            if not analysis_df.empty:
                current_total_usage = analysis_df['전력사용량(kWh)'].sum()
                current_total_bill = analysis_df['전기요금(원)'].sum()
                current_avg_usage = analysis_df['전력사용량(kWh)'].mean()
                current_total_carbon = analysis_df['탄소배출량(tCO2)'].sum()
                
                row1_col1, row1_col2 = st.columns(2)
                with row1_col1: st.metric(label=f"{analysis_title} 총 사용량", value=f"{current_total_usage:,.0f} kWh", delta=delta_usage_str, delta_color=delta_usage_color)
                with row1_col2: st.metric(label=f"{analysis_title} 총 전기요금", value=f"{current_total_bill:,.0f} 원", delta=delta_bill_str, delta_color=delta_bill_color)
                row2_col1, row2_col2 = st.columns(2)
                with row2_col1: st.metric(label=f"{analysis_title} 평균 시간당 사용량", value=f"{current_avg_usage:,.0f} kWh")
                with row2_col2: st.metric(label=f"{analysis_title} 총 탄소 배출량", value=f"{current_total_carbon:,.2f} tCO2")
            else:
                st.warning(f"선택된 '{analysis_title}' 기간에 대한 데이터가 없어 지표를 계산할 수 없습니다.")
                row1_col1, row1_col2 = st.columns(2)
                row1_col1.metric(f"{analysis_title} 총 사용량", "0 kWh")
                row1_col2.metric(f"{analysis_title} 총 전기요금", "0 원")
                row2_col1, row2_col2 = st.columns(2)
                row2_col1.metric(f"{analysis_title} 평균 시간당 사용량", "0 kWh")
                row2_col2.metric(f"{analysis_title} 총 탄소 배출량", "0 tCO2")

        st.divider()

        if not analysis_df.empty:
            selected_start_dt = analysis_df['측정일시'].min()
            selected_end_dt = analysis_df['측정일시'].max()
            if filter_type == "기간별 선택" and isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
                selected_end_dt = pd.Timestamp(selected_range[1] + pd.Timedelta(days=1)) - pd.Timedelta(seconds=1) 
                selected_start_dt = pd.Timestamp(selected_range[0])
        else: 
            selected_start_dt = df['측정일시'].min()
            selected_end_dt = df['측정일시'].max()
            analysis_title = "전체 기간" 
            if not (filter_type == "월별 선택" or (filter_type == "기간별 선택" and isinstance(selected_range, (list, tuple)) and len(selected_range) == 2)):
                st.info("기간을 선택하면 해당 구간을 확대하여 보여줍니다. (현재 전체 기간 표시 중)")

        def create_daily_chart(full_df, start_dt, end_dt):
            daily_summary = full_df.groupby('날짜').agg(
                total_usage=('전력사용량(kWh)', 'sum'),
                total_bill=('전기요금(원)', 'sum')
            ).reset_index()
            daily_summary['날짜'] = pd.to_datetime(daily_summary['날짜'])
            daily_summary_melted = daily_summary.melt(
                var_name='범주', value_name='값', id_vars=['날짜'], value_vars=['total_usage', 'total_bill']
            )
            daily_summary_melted['범주'] = daily_summary_melted['범주'].map({
                'total_usage': '총 사용량 (kWh)', 'total_bill': '총 전기요금 (원)'
            })
            base = alt.Chart(daily_summary_melted).encode(
                x=alt.X('날짜:T', 
                        axis=alt.Axis(title='날짜', format='%Y-%m-%d'),
                        scale=alt.Scale(domain=[start_dt, end_dt])), 
                color=alt.Color('범주:N', legend=alt.Legend(title=None, orient='top-left', fillColor='white', padding=5)),
                tooltip=['날짜', '범주', alt.Tooltip('값', title='값', format=',.0f')]
            )
            usage_line = base.mark_line(point=alt.MarkConfig(opacity=0.3, size=10)).encode(
                y=alt.Y('값:Q', title='총 사용량 (kWh)')
            ).transform_filter(alt.datum.범주 == '총 사용량 (kWh)')
            bill_line = base.mark_line(point=alt.MarkConfig(opacity=0.3, size=10), color='darkorange').encode(
                y=alt.Y('값:Q', title='총 전기요금 (원)')
            ).transform_filter(alt.datum.범주 == '총 전기요금 (원)')
            return alt.layer(usage_line, bill_line).resolve_scale(
                y='independent'
            ).interactive(bind_y=False) 

        def create_hourly_comparison_chart(full_df, analysis_df_for_avg, title_for_avg):
            overall_hourly_avg = full_df.groupby('시간')['전력사용량(kWh)'].mean().reset_index()
            overall_hourly_avg['구분'] = '전체 평균 (1-11월)'
            if not analysis_df_for_avg.empty:
                hourly_avg = analysis_df_for_avg.groupby('시간')['전력사용량(kWh)'].mean().reset_index()
                hourly_avg['구분'] = f'{title_for_avg} 평균'
                combined_hourly = pd.concat([overall_hourly_avg, hourly_avg])
            else:
                combined_hourly = overall_hourly_avg
            area = alt.Chart(combined_hourly).mark_area(opacity=0.3, color='lightgray').encode(
                x=alt.X('시간:Q', axis=alt.Axis(title='시간 (0-23시)')),
                y=alt.Y('전력사용량(kWh):Q', title='평균 전력사용량 (kWh)'),
                tooltip=[alt.Tooltip('시간', format='d'), alt.Tooltip('전력사용량(kWh)', format='.2f', title='평균 사용량'), '구분']
            ).transform_filter(alt.datum.구분 == '전체 평균 (1-11월)')
            line = alt.Chart(combined_hourly).mark_line(point=True, color='steelblue').encode(
                x='시간:Q', y='전력사용량(kWh):Q',
                tooltip=[alt.Tooltip('시간', format='d'), alt.Tooltip('전력사용량(kWh)', format='.2f', title='평균 사용량'), '구분']
            ).transform_filter(alt.datum.구분 == f'{title_for_avg} 평균')
            return alt.layer(area, line).interactive(bind_y=False)

        def create_pf_chart(full_df, pf_col_name, time_filter_expr, threshold, color, title_time, start_dt, end_dt):
            pf_data = full_df[full_df.eval(time_filter_expr) & (full_df[pf_col_name] > 0)].copy()
            if pf_data.empty: return None 
            line = alt.Chart(pf_data).mark_line(
                point=alt.MarkConfig(opacity=0.3, size=10), color=color
            ).encode(
                x=alt.X('측정일시:T', title='측정일시', axis=alt.Axis(format="%m-%d %H:%M", labelAngle=-45),
                        scale=alt.Scale(domain=[start_dt, end_dt])), 
                y=alt.Y(f'{pf_col_name}:Q', title=f'{pf_col_name.split("(")[0]} (%)', scale=alt.Scale(zero=False, padding=0.1)),
                tooltip=[alt.Tooltip('측정일시', format="%Y-%m-%d %H:%M"), f'{pf_col_name}']
            ).interactive(bind_y=False) 
            rule = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(
                color=color, strokeDash=[5, 5], size=2 
            ).encode(
                y='threshold:Q', tooltip=[alt.Tooltip('threshold', title='기준치')]
            )
            return line + rule

        st.write(f"#### {analysis_title} 트렌드 (전체 기간 데이터 표시)")
        col1_viz, col2_viz = st.columns(2)
        with col1_viz:
            st.write(f"**일별 사용량 및 요금**")
            daily_chart = create_daily_chart(df, selected_start_dt, selected_end_dt)
            st.altair_chart(daily_chart, use_container_width=True)
        with col2_viz:
            st.write(f"**시간대별 평균 사용량** ({analysis_title} vs 전체 평균)")
            hourly_chart = create_hourly_comparison_chart(df, analysis_df, analysis_title)
            st.altair_chart(hourly_chart, use_container_width=True)

        st.divider()

        st.subheader(f"{analysis_title} 역률 상세 분석 (전체 기간 데이터 표시)")
        col1_sec3, col2_sec3 = st.columns(2)
        with col1_sec3:
            st.write(f"**지상역률(%) 추이 (09-23시)**")
            lagging_chart = create_pf_chart(
                full_df=df, pf_col_name='지상역률(%)',
                time_filter_expr='(시간 >= 9) & (시간 <= 23)', 
                threshold=90.0, color='darkorange', title_time='09-23시',
                start_dt=selected_start_dt, end_dt=selected_end_dt
            )
            if lagging_chart:
                st.altair_chart(lagging_chart, use_container_width=True)
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
            st.write(f"**진상역률(%) 추이 (23-09시)**")
            leading_chart = create_pf_chart(
                full_df=df, pf_col_name='진상역률(%)',
                time_filter_expr='(시간 >= 23) | (시간 < 9)', 
                threshold=95.0, color='steelblue', title_time='23-09시',
                start_dt=selected_start_dt, end_dt=selected_end_dt
            )
            if leading_chart:
                st.altair_chart(leading_chart, use_container_width=True)
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

        # --- 전체 기간 월별 트렌드 (오타 수정됨) ---
        st.write("#### 전체 기간 월별 트렌드 (1~11월)")
        chart_data = monthly_summary 
        monthly_summary_melted = chart_data.melt(
            var_name='범주',
            value_name='값',
            id_vars=['월'],
            value_vars=['total_usage', 'total_bill'] 
        )
        monthly_summary_melted['범주'] = monthly_summary_melted['범주'].map({
            'total_usage': '총 사용량 (kWh)',
            'total_bill': '총 전기요금 (원)'
        })
        
        base_monthly = alt.Chart(monthly_summary_melted).encode(
            x=alt.X('월:O', axis=alt.Axis(title='월', labelAngle=0, labelExpr="datum.value + '월'")),
            color=alt.Color('범주:N', legend=alt.Legend(title=None, orient='top-right', fillColor='white', padding=5)),
            tooltip=['월', '범주', alt.Tooltip('값', title='값', format=',.0f')]
        )
        # .interactive(bind_y=False) # <-- 삭제됨 (범주형 축)
        
        usage_line_monthly = base_monthly.transform_filter(
            alt.datum.범주 == '총 사용량 (kWh)'
        ).mark_line(point=True).encode(
            y=alt.Y('값:Q', title='총 사용량 (kWh)')
        )
        
        bill_line_monthly = base_monthly.transform_filter(
            alt.datum.범주 == '총 전기요금 (원)' 
        ).mark_line(point=True).encode(
            y=alt.Y('값:Q', title='총 전기요금 (원)') 
        )
        
        dual_axis_chart = alt.layer(usage_line_monthly, bill_line_monthly).resolve_scale(y='independent')
        st.altair_chart(dual_axis_chart, use_container_width=True)

        with st.expander(f"Dataframe: {analysis_title} 상세 데이터 보기"):
            if not analysis_df.empty:
                st.dataframe(analysis_df) 
            else:
                st.write(f"선택된 '{analysis_title}' 기간에 데이터가 없습니다. 전체 데이터를 보려면 기간을 다시 선택하세요.")

    else: # df is None
        st.error("데이터 파일을 로드할 수 없습니다.")