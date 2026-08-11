import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── 1. 원시 데이터 로드 ─────────────────────────────────────────────────────
df = pd.read_csv('ecofarm_merged_dataset.csv')
print(f"[로드] Shape: {df.shape} | 기간: {df['date'].min()} ~ {df['date'].max()}")

# ── 2. 결측값 처리 ──────────────────────────────────────────────────────────
# 응급 미발생 레코드(96.18%)는 emergency_types, max_severity가 NaN → 'none'으로 채움
df['emergency_types'] = df['emergency_types'].fillna('none')
df['max_severity']    = df['max_severity'].fillna('none')
# avg_response_time_min: 응급 발생 시에만 존재하는 값 → NaN 유지 (단계별 모델에서 별도 처리)

print(f"[결측값] emergency_types/max_severity → 'none' 처리 완료")
print(f"         avg_response_time_min 결측 {df['avg_response_time_min'].isnull().sum()}건 (정상 — 응급 미발생)")

# ── 3. 타입 변환 ────────────────────────────────────────────────────────────
df['date'] = pd.to_datetime(df['date'])
for col in ['is_rain', 'is_heatwave', 'is_coldwave']:
    df[col] = df[col].astype(int)

# ── 4. 날짜 파생 피처 ───────────────────────────────────────────────────────
df['month']      = df['date'].dt.month
df['dayofweek']  = df['date'].dt.dayofweek   # 0=월, 6=일
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# ── 5. 범주형 인코딩 ─────────────────────────────────────────────────────────
# day_type: 이진
df['day_type_enc'] = df['day_type'].map({'평일': 0, '휴일': 1})

# season: 순서형 (여름 > 가을 > 봄 > 겨울 — 발생률 기준)
df['season_enc'] = df['season'].map({'겨울': 1, '봄': 2, '가을': 3, '여름': 4})

# facility_type: 원-핫
df = pd.get_dummies(df, columns=['facility_type'], prefix='fac', dtype=int)

# ── 6. 타깃 인코딩 (2·3단계 모델용) ─────────────────────────────────────────
# [2단계] 응급 유형 — LabelEncoder (none=미발생 포함)
le_type = LabelEncoder()
df['emergency_type_enc'] = le_type.fit_transform(df['emergency_types'])

type_map = dict(zip(le_type.classes_, le_type.transform(le_type.classes_)))
print(f"\n[응급 유형 인코딩] {type_map}")

# [3단계] 중증도 — 순서형
severity_map = {'none': 0, '경증': 1, '중등증': 2, '중증': 3}
df['severity_enc'] = df['max_severity'].map(severity_map)
print(f"[중증도 인코딩]   none=0, 경증=1, 중등증=2, 중증=3")

# ── 7. 피처 엔지니어링 ──────────────────────────────────────────────────────
# 기상 × 밀도 상호작용
df['heat_density']    = df['is_heatwave'] * df['density']
df['rain_density']    = df['is_rain']      * df['density']
# cold_density 제외 — 데이터 기간 내 한파 발생 0건으로 상수 피처

# 시간/날씨 복합 위험
df['holiday_summer']  = ((df['day_type_enc'] == 1) & (df['season'] == '여름')).astype(int)
df['is_peak_hour']    = df['hour'].isin([13, 15]).astype(int)

# 밀도 기반
df['is_high_density']     = (df['density'] > 0.6).astype(int)
df['density_risk_ratio']  = (df['density'] / df['base_risk']).round(4)
df['weighted_risk']       = (df['density'] * df['base_risk']).round(6)

# 공간 피처 — 파크 입구(주차장/매표소 F06: x=5, y=5) 기준 거리
df['dist_from_gate'] = np.sqrt((df['x'] - 5)**2 + (df['y'] - 5)**2).round(3)

# 파크 중심(전체 시설 중심) 기준 거리
cx = df.groupby('facility_id')['x'].first().mean()  # 20.83
cy = df.groupby('facility_id')['y'].first().mean()  # 27.50
df['dist_from_center'] = np.sqrt((df['x'] - cx)**2 + (df['y'] - cy)**2).round(3)

print(f"\n[피처 엔지니어링] 9개 피처 추가:")
new_feats = ['heat_density','rain_density','holiday_summer',
             'is_peak_hour','is_high_density','density_risk_ratio','weighted_risk',
             'dist_from_gate','dist_from_center']
for f in new_feats:
    print(f"  {f}: min={df[f].min():.3f}, mean={df[f].mean():.3f}, max={df[f].max():.3f}")

# ── 8. 시계열 분할 검증 ─────────────────────────────────────────────────────
print(f"\n[시계열 분할]")
splits = {
    'Train': df[df['date'] < '2026-05-01'],
    'Val'  : df[(df['date'] >= '2026-05-01') & (df['date'] < '2026-07-01')],
    'Test' : df[df['date'] >= '2026-07-01'],
}
for name, subset in splits.items():
    n = len(subset)
    pos = subset['emergency_occurred'].sum()
    print(f"  {name}: {n:,}건 ({n/len(df)*100:.1f}%) | 응급 {pos}건 ({pos/n*100:.2f}%)")

# ── 9. 컬럼 역할 요약 출력 ──────────────────────────────────────────────────
FEATURE_COLS = [
    # 시간
    'hour', 'month', 'dayofweek', 'is_peak_hour',
    # 날씨
    'is_rain', 'temperature', 'is_heatwave',
    # 시설
    'capacity', 'x', 'y', 'base_risk',
    'fac_camping', 'fac_experience', 'fac_gate', 'fac_observatory', 'fac_playground',
    # 방문객
    'visitor_count', 'density',
    # 인코딩
    'day_type_enc', 'season_enc',
    # 엔지니어링
    'heat_density', 'rain_density',
    'holiday_summer', 'is_high_density',
    'density_risk_ratio', 'weighted_risk',
    'dist_from_gate', 'dist_from_center',
    # 제외: is_weekend (day_type_enc와 상관 0.93 중복)
    # 제외: is_coldwave, cold_density (기간 내 한파 0건 — 상수)
]

print(f"\n[컬럼 요약]")
print(f"  공통 피처: {len(FEATURE_COLS)}개")
print(f"  [1단계] 타깃: emergency_occurred  |  누수 제외: emergency_count, emergency_types, max_severity, avg_response_time_min")
print(f"  [2단계] 타깃: emergency_type_enc  |  대상: emergency_occurred==1 ({df['emergency_occurred'].sum()}건)")
print(f"  [3단계] 타깃: severity_enc         |  대상: emergency_occurred==1 ({df['emergency_occurred'].sum()}건)")

# ── 10. 불필요 컬럼 제거 후 저장 ────────────────────────────────────────────
# 누수 컬럼 제거 (사고 발생 후에야 알 수 있는 정보)
df = df.drop(columns=['emergency_count', 'emergency_types', 'max_severity'])
# 상수 피처 제거 (기간 내 한파 0건)
df = df.drop(columns=['is_coldwave', 'cold_density'], errors='ignore')
# 중복 피처 제거 (day_type_enc와 상관 0.93)
df = df.drop(columns=['is_weekend'])

output_path = 'ecofarm_preprocessed_master.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 저장 완료: {output_path}")
print(f"   Shape: {df.shape}")
print(f"   컬럼 수: {len(df.columns)}개")
print(f"   결측값: {df.drop(columns=['avg_response_time_min']).isnull().sum().sum()}개 (avg_response_time_min 제외)")
