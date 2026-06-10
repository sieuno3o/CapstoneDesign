import pandas as pd
import numpy as np
import re

# =====================================================================
# 1. 설문조사 파일(survey.csv) 로드 및 가중치(평균) 계산 함수
# =====================================================================
def calculate_survey_weights(survey_path):
    # CSV 로드
    df = pd.read_csv(survey_path)
    
    # [해결 포인트] 뒤에 오는 코드와 꼬이지 않도록 명확하게 한 줄로 끝맺음합니다.
    # survey.csv의 열 개수(7개)에 맞춰 정확하게 이름을 강제 매핑합니다.
    df.columns = ['Timestamp', 'Type', 'N1', 'N2', 'N3', 'N4', 'N5']
    
    # 텍스트 응답(예: "+2 (적극 매수)")에서 숫자만 추출하는 함수 
    def extract_score(text):
        if pd.isna(text): 
            return 0.0
        # 정규식을 이용해 +, -, 숫자가 포함된 패턴을 찾습니다. (예: +2, -1, 0)
        match = re.search(r'([+-]?\d+)', str(text))
        if match:
            return float(match.group(1))
        return 0.0

    # N1~N5 컬럼의 데이터를 모두 숫자로 변환합니다 [cite: 2]
    for col in ['N1', 'N2', 'N3', 'N4', 'N5']:
        df[col] = df[col].apply(extract_score)
        
    # '단기투자자'와 '장기투자자' 텍스트를 통일성 있게 매핑합니다.
    df['Type'] = df['Type'].apply(lambda x: 'Short' if '단기' in str(x) else 'Long')
    
    # 집단별 평균값(가중치) 산출
    weights = df.groupby('Type')[['N1', 'N2', 'N3', 'N4', 'N5']].mean()
    
    print("=== 설문조사 기반 도출된 가중치(문항별 평균값) ===")
    print(weights)
    print("-" * 50)
    
    return weights.loc['Short'].to_dict(), weights.loc['Long'].to_dict()

# =====================================================================
# 2. 뉴스 카운트 파일과 가중치를 결합해 일별 K-NSI를 산출하는 함수
# =====================================================================
def generate_daily_k_nsi(news_path, w_short, w_long):
    # 매일 크롤링한 뉴스 건수 데이터 로드
    df_news = pd.read_csv(news_path)
    
    # [해결 포인트] 열 이름 앞뒤에 혹시 모를 공백이 있다면 깔끔하게 제거합니다.
    df_news.columns = df_news.columns.str.strip()
    
    # 디버깅용: 현재 데이터프레임에 열들이 잘 존재하는지 확인 출력
    # print("현재 뉴스 파일의 열 이름들:", df_news.columns.tolist())
    
    short_nsi_list = []
    long_nsi_list = []
    
    # 하루치 데이터씩 순회하며 공식 적용
    for idx, row in df_news.iterrows():
        try:
            # 변수를 읽어올 때 에러가 나면 몇 번째 줄에서 왜 났는지 알려주도록 설정
            n1 = row['N1']
            n2 = row['N2']
            n3 = row['N3']
            n4 = row['N4']
            n5 = row['N5']
        except KeyError as e:
            raise KeyError(f"데이터프레임에 {e} 열이 존재하지 않습니다. 실제 열 이름들을 확인해주세요: {df_news.columns.tolist()}")
        
        # ---------------------------------------------------
        # [단기 투자자 K-NSI] N1, N2, N4는 음수 / N3, N5는 양수 가중치 적용
        # ---------------------------------------------------
        pos_s = (n3 * w_short['N3']) + (n5 * w_short['N5'])
        neg_s = (n1 * abs(w_short['N1'])) + (n2 * abs(w_short['N2'])) + (n4 * abs(w_short['N4']))
        nsi_s = ((pos_s - neg_s) / (pos_s + neg_s + 1.0)) * 100 + 100
        short_nsi_list.append(nsi_s)
        
        # ---------------------------------------------------
        # [장기 투자자 K-NSI] N1~N5 모든 가중치가 양수이므로 전량 Positive 배정
        # ---------------------------------------------------
        pos_l = (n1 * w_long['N1']) + (n2 * w_long['N2']) + (n3 * w_long['N3']) + (n4 * w_long['N4']) + (n5 * w_long['N5'])
        neg_l = 0
        nsi_l = ((pos_l - neg_l) / (pos_l + neg_l + 1.0)) * 100 + 100
        long_nsi_list.append(nsi_l)
        
    # 결과 시계열 데이터프레임 구축
    df_news['K_NSI_Short'] = short_nsi_list
    df_news['K_NSI_Long'] = long_nsi_list
    
    return df_news[['date', 'K_NSI_Short', 'K_NSI_Long']]