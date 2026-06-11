import pandas as pd

# 1. 데이터 로드 (시작 데이터 정제)
data = pd.read_csv("data/raw/survey.csv")

# 컬럼명 변경 (단순화)
data.columns = ['Timestamp', 'Type', 'N1', 'N2', 'N3', 'N4', 'N5']

# 텍스트로 된 점수를 숫자로 변환하는 함수
def text_to_score(text):
    if pd.isna(text): return 0
    if '+2' in str(text): return 2
    if '+1' in str(text): return 1
    if '-1' in str(text): return -1
    if '-2' in str(text): return -2
    return 0

# 변환 적용
for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
    data[col] = data[col].apply(text_to_score)

# 투자자 타입 단순화
data['Type'] = data['Type'].apply(lambda x: 'Short' if '단기' in str(x) else 'Long')

# 집단별 평균 구하기
means = data.groupby('Type')[['N1', 'N2', 'N3', 'N4', 'N5']].mean()

# K-NSI 계산 함수 정의
def calculate_k_nsi(row):
    pos_score = 0
    neg_score = 0
    
    for val in row:
        if val > 0:
            pos_score += val
        elif val < 0:
            neg_score += abs(val)
            
    k_nsi = ((pos_score - neg_score) / (pos_score + neg_score + 1.0)) * 100 + 100
    return k_nsi

# 최종 지수 산출
means['K-NSI'] = means.apply(calculate_k_nsi, axis=1)
print(means[['K-NSI']])