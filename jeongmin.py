import pandas as pd
import numpy as np


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.info()
train.head()

train.columns
train = train.drop(columns='id')
train.columns

len(train['작업유형'].unique())

train.tail(10)

# ================ 4시간씩 묶기 ====================
# 2. '1,2,3,4' '5,6,7,8' ... 단위로 그룹핑하기 위한 키 생성
# np.arange(len(df)) -> [0, 1, 2, 3, 4, 5, 6, 7, ...]
# // 4 (정수 나누기) -> [0, 0, 0, 0, 1, 1, 1, 1, ...]
# 이 [0, 0, 0, 0, 1, 1, 1, 1] 값을 기준으로 groupby를 실행합니다.
grouper = np.arange(len(train)) // 4

# 3. 새로운 컬럼 생성
# groupby(grouper)로 4행씩 그룹을 묶습니다.
# ['전력사용량(kWh)'] 컬럼을 선택합니다.
# .transform('sum')을 사용하여 그룹별 합계를 구하고, 
# 그 합계를 원래의 모든 행(4개 행)에 동일하게 매핑합니다.
train['시간단위_총_전력사용량(kWh)'] = train.groupby(grouper)['전력사용량(kWh)'].transform('sum')

# 4. 결과 확인 (처음 10개 행)
print("--- 4행 단위로 합계가 추가된 데이터 (Head 10) ---")
# 'id', '측정일시', '전력사용량(kWh)', '시간단위_총_전력사용량(kWh)' 컬럼만 선택하여 명확하게 보여줍니다.
print(train[['측정일시', '전력사용량(kWh)', '시간단위_총_전력사용량(kWh)']].head(10))

train.loc[train['시간단위_총_전력사용량(kWh)']>=300]


train


# ============= 2단 예측 (점수 더 낮게나옴) ====================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings('ignore')

print("--- 2단계 예측 모델링 시작 ---")

# --- [공통] 훈련 데이터 로드 및 전처리 ---
train_file_path = './data/train.csv'
if not os.path.exists(train_file_path):
    print(f"오류: {train_file_path}를 찾을 수 없습니다.")
else:
    try:
        train = pd.read_csv(train_file_path)

        # 1. 공통 피처 엔지니어링
        train['측정일시'] = pd.to_datetime(train['측정일시'])
        train['hour'] = train['측정일시'].dt.hour
        train['dayofweek'] = train['측정일시'].dt.dayofweek
        train['month'] = train['측정일시'].dt.month

        # LabelEncoder 'le' 객체 생성 (test.csv에 재사용해야 함)
        le = LabelEncoder()
        train['작업유형_encoded'] = le.fit_transform(train['작업유형'])

        # 2. 피처 그룹 정의
        # test.csv와 공통된 피처 (모델 1의 X)
        common_features = ['hour', 'dayofweek', 'month', '작업유형_encoded']
        
        # test.csv에 없어서 예측해야 할 피처 (모델 1의 Y)
        features_to_predict = [
            '전력사용량(kWh)', 
            '지상무효전력량(kVarh)', 
            '진상무효전력량(kVarh)', 
            '지상역률(%)', 
            '진상역률(%)'
        ]
        
        # 최종 타겟 (모델 2의 Y)
        final_target = '전기요금(원)'
        
        # 모델 2가 사용할 모든 피처 (모델 2의 X)
        final_features = common_features + features_to_predict

        print("--- [단계 1/4] 훈련 데이터 및 피처 그룹 정의 완료 ---")

        # --- [모델 1] 피처 예측 모델들 학습 ---
        print("--- [단계 2/4] '모델 1' (피처 예측 모델) 5개 학습 시작 ---")
        
        models_stage1 = {} # 5개의 피처 예측 모델을 저장할 딕셔너리

        for feature in features_to_predict:
            print(f"  -> '{feature}' 예측 모델 학습 중...")
            
            X_train_s1 = train[common_features]
            y_train_s1 = train[feature]
            
            # 모델 1은 objective='regression' (L2 loss, RMSE) 사용
            model_s1 = lgb.LGBMRegressor(
                objective='regression_l2', 
                n_estimators=300, # 피처 예측은 최종 예측보다 단순하게
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            model_s1.fit(X_train_s1, y_train_s1, categorical_feature=common_features)
            
            models_stage1[feature] = model_s1 # 딕셔너리에 저장

        print("--- '모델 1' 학습 완료 ---")

        # --- [모델 2] 최종 요금 예측 모델 학습 ---
        print("--- [단계 3/4] '모델 2' (최종 요금 예측 모델) 학습 시작 ---")

        X_train_s2 = train[final_features] # train의 모든 피처 (실제 측정값) 사용
        y_train_s2 = train[final_target]

        # 모델 2는 objective='regression_l1' (MAE) 사용
        model_stage2 = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=1000, # 최종 모델은 더 강력하게
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        model_stage2.fit(X_train_s2, y_train_s2, categorical_feature=common_features)

        print("--- '모델 2' 학습 완료 ---")

        # --- [최종 예측] test.csv에 2단계 모델 적용 ---
        print("--- [단계 4/4] 'test.csv' 로드 및 2단계 예측 시작 ---")
        
        test_file_path = './data/test.csv'
        if not os.path.exists(test_file_path):
            print(f"오류: {test_file_path}를 찾을 수 없습니다.")
        else:
            test = pd.read_csv(test_file_path)
            test_ids = test['id'].copy()

            # 1. test 데이터에 공통 피처 생성
            test['측정일시'] = pd.to_datetime(test['측정일시'])
            test['hour'] = test['측정일시'].dt.hour
            test['dayofweek'] = test['측정일시'].dt.dayofweek
            test['month'] = test['측정일시'].dt.month
            test['작업유형_encoded'] = le.transform(test['작업유형']) # 훈련시 사용한 'le' 적용

            X_test_common = test[common_features]

            # 2. [모델 1 적용] test.csv의 빈 피처 채우기
            print("  -> '모델 1' 적용: 누락된 5개 피처 예측/생성 중...")
            for feature, model in models_stage1.items():
                predicted_values = model.predict(X_test_common)
                test[feature] = predicted_values # test 데이터프레임에 예측된 피처 추가

            # 3. [모델 2 적용] 최종 전기요금 예측
            print("  -> '모델 2' 적용: 최종 전기요금 예측 중...")
            X_test_s2 = test[final_features] # 예측된 피처가 포함된 전체 피처
            final_predictions = model_stage2.predict(X_test_s2)
            final_predictions[final_predictions < 0] = 0 # 0 미만 값 보정

            # 4. 결과 저장
            submission = pd.DataFrame({
                'id': test_ids,
                '전기요금(원)': final_predictions
            })

            output_filename = 'submission.csv'
            submission.to_csv(output_filename, index=False)

            print(f"\n--- 2단계 예측 완료! {output_filename} 파일로 저장되었습니다. ---")
            print(submission.head())

    except Exception as e:
        print(f"전체 프로세스 중 오류 발생: {e}")






# ============== 작업 유형별 전력사용량 분포 그래프 ==============
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

# --- [단계 1/3] 데이터 로드 ---
print("--- [단계 1/3] 데이터 로드 ('train.csv') ---")
try:
    train = pd.read_csv('./data/train.csv')
    print(f"로드 성공 (Shape: {train.shape})")
except FileNotFoundError:
    print("오류: 'train.csv' 파일을 찾을 수 없습니다.")
    # 파일이 없으면 중단
    raise

# --- [단계 2/3] 겹치는 히스토그램 생성 ---
print("--- [단계 2/3] 겹치는 히스토그램 플롯 생성 ---")
plt.figure(figsize=(12, 7))

# seaborn의 histplot을 사용하여 겹치는 히스토그램 생성
# data: 사용할 데이터프레임
# x: X축에 사용할 컬럼
# hue: '작업유형' 별로 색상 구분
# kde: True -> 커널 밀도 추정 곡선(라인) 추가
# multiple: "layer" -> 레이어를 겹치게 표시 (요청하신 "겹치게")
# stat: "density" -> Y축을 빈도(count)가 아닌 밀도(density)로 표시
# common_norm: False -> 각 hue(작업유형)별로 독립적으로 정규화 (각 곡선의 면적이 1이 됨)
sns.histplot(
    data=train,
    x='전력사용량(kWh)',
    hue='작업유형',
    kde=True,
    multiple="layer",
    stat="density",
    common_norm=False,
    palette="tab10" # 색상 팔레트
)

plt.title('작업유형별 전력사용량(kWh) 분포 (겹침)', fontsize=16)
plt.xlabel('전력사용량(kWh)', fontsize=12)
plt.ylabel('밀도 (Density)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6) # 그리드 추가
plt.legend(title='작업유형') # 범례 제목 설정
plt.tight_layout() # 레이아웃 최적화
plt.show()



# ============== 작업유형 패턴 분류 ==============
import pandas as pd
import numpy as np
import os

# --- [단계 1] 데이터 로드 및 '일별 패턴 시그니처' 생성 ---
print("--- [단계 1/4] 데이터 로드 및 '일별 패턴' 생성 ---")
try:
    # (주의) 사용자가 제공한 경로 './data/train.csv' 대신 VM의 루트 경로 'train.csv' 사용
    train = pd.read_csv('./data/train.csv')
except FileNotFoundError:
    print("오류: 'train.csv' 파일을 찾을 수 없습니다.")
    raise

# '측정일시'를 datetime으로 변환
train['측정일시'] = pd.to_datetime(train['측정일시'])
# '날짜' 컬럼 생성
train['date'] = train['측정일시'].dt.date

# (중요) 날짜와 시간 순서대로 정렬
train = train.sort_values(by=['date', '측정일시'])

# 각 날짜(date)별로 '작업유형'을 순서대로 '_'(언더바)로 연결
print("날짜별로 96개의 '작업유형'을 묶어 고유 패턴 생성 중...")
daily_signatures = train.groupby('date')['작업유형'].apply(lambda x: '_'.join(x))

# --- [단계 2] 패턴별 빈도수 및 'Pattern_ID' 맵 생성 ---
print("--- [단계 2/4] 고유 패턴별 ID 맵 생성 ---")
pattern_counts = daily_signatures.value_counts().reset_index()
pattern_counts.columns = ['Pattern_Signature', 'Day_Count']
pattern_counts = pattern_counts.sort_values(by='Day_Count', ascending=False)
pattern_counts['Pattern_ID'] = [f'Pattern {i+1}' for i in range(len(pattern_counts))]

total_unique_patterns = len(pattern_counts)
print(f"분석 결과: 총 {total_unique_patterns}개의 고유한 일별 패턴이 발견되었습니다.")

# (핵심) 2개의 맵(Map) 생성
# 1. Signature -> Pattern_ID 맵 (예: 'Light_..._Max' -> 'Pattern 1')
signature_to_id_map = pd.Series(
    pattern_counts.Pattern_ID.values, 
    index=pattern_counts.Pattern_Signature
).to_dict()

# 2. Date -> Pattern_ID 맵 (예: '2024-01-02' -> 'Pattern 1')
#    daily_signatures (Date -> Signature)를 위 1번 맵으로 변환
date_to_id_map = daily_signatures.map(signature_to_id_map)

# --- [단계 3] 원본 train 데이터에 'Pattern_ID' 컬럼 추가 ---
print("--- [단계 3/4] 원본 train 데이터에 'Pattern_ID' 컬럼 매핑 ---")

# train 데이터의 'date' 컬럼을 사용해 'date_to_id_map'을 매핑
train['Pattern_ID'] = train['date'].map(date_to_id_map)

# --- [단계 4] 결과 확인 및 저장 ---
print("--- [단계 4/4] 결과 확인 및 CSV 저장 ---")

print("\n--- 'Pattern_ID' 컬럼 추가 확인 (Head) ---")
# 'date' 컬럼은 임시로 사용했으므로 최종 결과에서는 제외할 수 있습니다.
# 여기서는 확인을 위해 포함하여 출력
print(train[['id', '측정일시', '작업유형', 'date', 'Pattern_ID']].head())

print("\n--- 'Pattern_ID' 컬럼 분포 확인 ---")
print(train['Pattern_ID'].value_counts())

train.loc[train['Pattern_ID']=='Pattern 1','date'].unique()
train.loc[train['Pattern_ID']=='Pattern 2','date'].unique()
train.loc[train['Pattern_ID']=='Pattern 3','date'].unique()





# ============== 패턴별 전기요금 분포 박스플롯 ==============
df = train.copy()

# --- [단계 2] 플롯 순서 정렬 ---
print("--- [단계 2/4] 'Pattern_ID' 순서 정렬 ---")
# 'Pattern 1', 'Pattern 10', 'Pattern 2' 순서가 아닌,
# 'Pattern 1', 'Pattern 2', 'Pattern 3' ... 순서로 정렬하기 위해
# ID에서 숫자를 추출하여 정렬 키(Key)를 생성합니다.


# 'Pattern_ID'가 null인 행이 있다면 제거
df = df.dropna(subset=['Pattern_ID'])

# 'Pattern 1' -> 1, 'Pattern 2' -> 2 추출
df['Pattern_Num'] = df['Pattern_ID'].str.extract(r'(\d+)').astype(int)

# 정렬된 순서 리스트 생성 (예: ['Pattern 1', 'Pattern 2', ...])
order_list = df.drop_duplicates(subset=['Pattern_ID', 'Pattern_Num']) \
               .sort_values(by='Pattern_Num')['Pattern_ID'] \
               .tolist()

print(f"플롯 순서: {order_list}")

# --- [단계 3] 박스 플롯(Box Plot) 시각화 ---
print("--- [단계 3/4] 박스 플롯 생성 중 ---")
plt.figure(figsize=(12, 7))

sns.boxplot(
    data=df,
    x='Pattern_ID',     # X축: 범주형 (패턴 ID)
    y='전기요금(원)',     # Y축: 수치형 (전기요금)
    order=order_list,   # 위에서 정렬한 순서대로 플롯
    palette='muted'
)

plt.title('일별 작업유형 패턴(Pattern_ID)별 전기요금 분포', fontsize=16)
plt.xlabel('패턴 ID (빈도순)', fontsize=12)
plt.ylabel('전기요금(원) (15분 단위)', fontsize=12)
plt.xticks(rotation=45, ha='right') # ID가 많을 경우 겹치지 않게 45도 회전
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()







# =========== 작업 유형별 전력사용량/전기요금 분포 박스플롯 ===========
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

# --- [단계 1] 데이터 로드 ---
print("--- [단계 1/3] 데이터 로드 ('train.csv') ---")
try:
    train = pd.read_csv('./data/train.csv')
    print(f"로드 성공 (Shape: {train.shape})")
except FileNotFoundError:
    print("오류: 'train.csv' 파일을 찾을 수 없습니다.")
    raise

# --- [단계 2] 박스 플롯(Box Plot) 시각화 ---
print("--- [단계 2/3] 박스 플롯 생성 중 ---")
plt.figure(figsize=(10, 6))

# 부하 순서대로 X축 정렬
plot_order = ['Light_Load', 'Medium_Load', 'Maximum_Load']

sns.boxplot(
    data=train,
    x='작업유형',       # X축: 범주형 (작업유형)
    y='전력사용량(kWh)', # Y축: 수치형 (전력사용량)
    order=plot_order,   # 부하가 낮은 순 -> 높은 순으로 정렬
    palette='pastel'    # 색상 팔레트
)

# sns.boxplot(
#     data=train,
#     x='작업유형',       # X축: 범주형 (작업유형)
#     y='전기요금(원)', # Y축: 수치형 (전력사용량)
#     order=plot_order,   # 부하가 낮은 순 -> 높은 순으로 정렬
#     palette='pastel'    # 색상 팔레트
# )

plt.title('작업유형별 전력사용량(kWh) 분포', fontsize=16)
plt.xlabel('작업유형', fontsize=12)
plt.ylabel('전력사용량(kWh)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




# ============== 패턴3랑 나머지 사이 전력사용량 통계검정 ==============
import pandas as pd
from scipy import stats
import os

df = train.copy()

# 그룹 1: 'Pattern 3'에 속하는 날들의 전기요금
group_p3 = df[df['Pattern_ID'] == 'Pattern 3']['전기요금(원)'].dropna()

# 그룹 2: 'Pattern 3'이 아닌 날들(즉, Pattern 1, 2)의 전기요금
group_not_p3 = df[df['Pattern_ID'] != 'Pattern 3']['전기요금(원)'].dropna()

print(f"  -> Pattern 3 그룹 샘플 수: {len(group_p3)}")
print(f"  -> Not Pattern 3 그룹 샘플 수: {len(group_not_p3)}")

# --- [단계 3] 만-위트니 U 검정 수행 ---
print("\n--- [단계 3/3] 만-위트니 U 검정 (양측 검정) 수행 ---")

# alternative='two-sided' : 두 그룹이 '다른지' 검증 (기본값)
statistic, pvalue = stats.mannwhitneyu(group_p3, group_not_p3, alternative='two-sided')

print(f"검정 통계량 (U statistic): {statistic}")
print(f"P-value: {pvalue}")

print("\n--- 결과 해석 ---")
if pvalue < 0.05:
    print(f"P-value ({pvalue:.2e})가 0.05보다 매우 작습니다.")
    print("결론: 귀무가설 기각. 두 그룹(Pattern 3 vs Not Pattern 3)의 전기요금 분포는")
    print("      **통계적으로 매우 유의미하게 다릅니다.**")
else:
    print(f"P-value ({pvalue})가 0.05보다 큽니다.")
    print("결론: 귀무가설 채택. 두 그룹 간 통계적으로 유의미한 차이가 없습니다.")

# --- (심화) 단측 검정 ---
print("\n--- (심화) 단측 검정 ('Pattern 3'이 더 적은지) ---")
# alternative='less' : group_p3가 group_not_p3보다 '더 적은지' 검증
stat_less, p_less = stats.mannwhitneyu(group_p3, group_not_p3, alternative='less')

print(f"P-value (less): {p_less}")
if p_less < 0.05:
    print("결론: 'Pattern 3'의 전기요금이 'Not Pattern 3'보다 **통계적으로 유의미하게 적습니다.**")






train.loc[train['Pattern_ID']=='Pattern 3', '전력사용량(kWh)'].describe()
train.loc[(train['Pattern_ID']=='Pattern 3') & (train['전력사용량(kWh)']>8), '전력사용량(kWh)'].count()
train.loc[(train['Pattern_ID']!='Pattern 3') & (train['작업유형']=='Light_Load'), '전력사용량(kWh)'].describe()


train.loc[(train['Pattern_ID']=='Pattern 3') & ((train['id']%95)==0), '전력사용량(kWh)'].describe()





# ============== 패턴3 전날 패턴의 분포 시각화 ==============
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

df = train.copy()
# 'Pattern 3'인 날짜(date 객체) 목록을 가져옴
pattern_3_dates = date_to_id_map[date_to_id_map == 'Pattern 3'].index

# 각 날짜에서 하루(1 day)를 뺀 '전날' 목록 생성
previous_days = [d - timedelta(days=1) for d in pattern_3_dates]

# '전날' 목록에서 train 데이터에 존재하는 날짜만 필터링
# (예: 1월 1일의 전날인 2023-12-31은 train 데이터에 없으므로 제외)
valid_previous_days = [d for d in previous_days if d in date_to_id_map.index]

print(f"'Pattern 3'은 총 {len(pattern_3_dates)}일 발견되었습니다.")
print(f"그중 유효한 '전날' 데이터 {len(valid_previous_days)}일을 분석합니다.")

# --- [단계 4] '전날'들의 패턴 분포 집계 ---
print("--- [단계 4/5] '전날'들의 패턴 분포 집계 ---")

# 'date_to_id_map'에서 '전날' 목록에 해당하는 패턴 ID들만 조회
previous_day_patterns = date_to_id_map.loc[valid_previous_days]

# 조회된 패턴 ID들의 빈도수 집계
result_distribution = previous_day_patterns.value_counts().reset_index()
result_distribution.columns = ['Previous_Day_Pattern', 'Count']

print("\n[분석 결과]")
print("'Pattern 3'(휴무일)의 전날은 다음과 같은 패턴들이었습니다:")
print(result_distribution)

# --- [단계 5] 시각화 (바 차트) ---
print("--- [단계 5/5] 시각화 생성 및 저장 ---")
plt.figure(figsize=(10, 6))
sns.barplot(
    data=result_distribution,
    x='Previous_Day_Pattern',
    y='Count',
    palette='Set2'
)
plt.title("'Pattern 3' (휴무일)의 바로 전날 패턴 분포", fontsize=16)
plt.xlabel('전날의 패턴 ID', fontsize=12)
plt.ylabel('일수 (Days)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()




# ============== 패턴1과 패턴2 사이 통계검정 ==============
import pandas as pd
from scipy import stats
import os
df = train.copy()

# --- [단계 2] 'Pattern 1'과 'Pattern 2' 그룹으로 데이터 분리 ---
print("--- [단계 2/3] 데이터를 'Pattern 1'과 'Pattern 2' 그룹으로 분리 ---")

if 'Pattern_ID' not in df.columns:
    print("오류: 'Pattern_ID' 컬럼이 파일에 없습니다.")
else:
    # 그룹 1: 'Pattern 1'에 속하는 날들의 전력사용량
    group_p1 = df[df['Pattern_ID'] == 'Pattern 1']['전력사용량(kWh)'].dropna()

    # 그룹 2: 'Pattern 2'에 속하는 날들의 전력사용량
    group_p2 = df[df['Pattern_ID'] == 'Pattern 2']['전력사용량(kWh)'].dropna()

    if len(group_p1) == 0 or len(group_p2) == 0:
        print("오류: 'Pattern 1' 또는 'Pattern 2' 데이터가 없습니다.")
    else:
        print(f"  -> Pattern 1 그룹 샘플 수: {len(group_p1)}")
        print(f"  -> Pattern 2 그룹 샘플 수: {len(group_p2)}")

        # --- [단계 3] 만-위트니 U 검정 수행 ---
        print("\n--- [단계 3/3] 만-위트니 U 검정 (Pattern 1 vs Pattern 2) 수행 ---")

        # alternative='two-sided' : 두 그룹이 '다른지' 검증 (양측 검정)
        statistic, pvalue = stats.mannwhitneyu(group_p1, group_p2, alternative='two-sided')

        print(f"검정 통계량 (U statistic): {statistic}")
        print(f"P-value: {pvalue}")

        print("\n--- 결과 해석 ---")
        if pvalue < 0.05:
            print(f"P-value ({pvalue:.2e})가 0.05보다 매우 작습니다.")
            print("결론: 귀무가설 기각. 'Pattern 1'과 'Pattern 2'의 전력사용량 분포는")
            print("      **통계적으로 매우 유의미하게 다릅니다.**")
        else:
            print(f"P-value ({pvalue})가 0.05보다 큽니다.")
            print("결론: 귀무가설 채택. 두 그룹 간 통계적으로 유의미한 차이가 없습니다.")