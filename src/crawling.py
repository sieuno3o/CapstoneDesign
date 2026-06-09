import time
from datetime import datetime, timedelta
import pandas as pd
import requests
import re

# ==========================================
# [중요] 발급받은 네이버 API 키를 여기에 입력하세요!
# ==========================================
NAVER_CLIENT_ID = "ClaD1mTIODD0xCGIyFGV"
NAVER_CLIENT_SECRET = "qfXMm50SS9"

# 💡 금융 뉴스 껍데기 단어 차단 리스트
stop_words = [
    '증시', '시장', '코스피', '코스닥', '주가', '뉴스', '기사', '게시판', '특징주', '네이버', '금융', 
    '출발', '국내', '개장', '시황', '종합', '마감', '오전', '오후', '해외', '외인', '기관', '개인', '투자자',
    '뉴욕증시', '미국', '뉴욕', '나스닥', '다우', '유럽', '아시아', '글로벌', '하루', '이틀', '연속', '닷새',
    '등', '및', '위해', '이유', '속', '줄', '중', '것', '이', '그', '대해', '다시', '이번', '올해', '내년',
    '상승', '하락', '반등', '폭락', '호조', '부진', '급등', '급락', '랠리', '쇼크', '돌파', '위기', '우려',
    '전망', '분석', '대비', '지수', '거래일', '기대', '기대감', '영향', '때문', '우려에', '속보'
]

def crawl_pure_top_keywords():
    raw_titles = []
    today = datetime.now()
    date_list = [(today - timedelta(days=i)) for i in range(60)] # 두 달치
    
    print(f"📅 [정밀 사건 키워드 추출] 두 달치({len(date_list)}일) 메인 헤드라인 뉴스를 수집합니다...")
    
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    for target_date in date_list:
        search_query = f"증시 {target_date.strftime('%Y.%m.%d')}"
        
        params = {
            "query": search_query,
            "display": 100,
            "start": 1,
            "sort": "sim" # 💡 네이버가 공인한 그날의 가장 핫한 뉴스 순 정렬
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                items = response.json().get("items", [])
                for item in items:
                    title = item["title"].replace("<b>", "").replace("</b>", "").replace("&quot;", '"').strip()
                    raw_titles.append(title)
            time.sleep(0.1)
        except:
            continue
            
    return raw_titles

if __name__ == "__main__":
    titles = crawl_pure_top_keywords()
    
    if titles:
        word_counts = {}
        for title in titles:
            # 순수 단어 분리
            words = re.findall(r'[가-힣a-zA-Z0-9]{2,}', title)
            for word in words:
                # 조사 잘라내기 정제
                word = re.sub(r'(이|가|은|는|를|에|으로|로|에서|의|과|와|도|만|에 따른|포함)$', '', word)
                if len(word) >= 2 and word not in stop_words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                    
        # 💡 상위 20개 진짜 알맹이 키워드만 정렬해서 추출
        top_20 = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        print("\n" + "=" * 60)
        print("📊 [최근 두 달간 대한민국 증시를 뒤흔든 진짜 메인 사건 키워드 TOP 20]")
        print("=" * 60)
        
        # 💡 "result : " 포맷 출력
        results = [f"{word}({count}회)" for word, count in top_20]
        print(f"result : 핵심 키워드 명단 -> {', '.join(results)}")
        print("=" * 60)
        print("💡 위 단어들을 바탕으로 '투자 심리 영향도 설문조사' 항목을 구성하면 완벽합니다.")
        
    else:
        print("❌ 데이터를 가져오지 못했습니다.")