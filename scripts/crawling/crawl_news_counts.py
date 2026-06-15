# -*- coding: utf-8 -*-
"""
crawl_news_counts_ex.py
-----------------------
매 영업일별로 5대 뉴스 카테고리(N1~N5)에 해당하는 Daum 뉴스 기사 건수를 수집하여
data/sentiment/macro_news_counts_90d.csv 파일로 내보내는 크롤링 스크립트입니다.
"""

import os
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
import re
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Daum 뉴스 검색 설정
BASE_URL = "https://search.daum.net/search"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# 5대 뉴스 카테고리 정규식 및 쿼리 설정
CATEGORIES = {
    "N1": {
        "desc": "이란 및 중동전쟁 위기 고조",
        "queries": ["이란 이스라엘", "중동 전쟁", "호르무즈 해협"],
        "keywords": re.compile(r"이란|중동전쟁|이스라엘.*이란|이란.*이스라엘|중동.*위기|중동.*충돌|호르무즈")
    },
    "N2": {
        "desc": "원·달러 환율 및 달러 강세 급등",
        "queries": ["환율 급등", "달러 강세", "원달러 환율"],
        "keywords": re.compile(r"환율.*급등|달러.*강세|고환율|환율.*상승|외환시장")
    },
    "N3": {
        "desc": "중동 지역 휴전 및 협상 타결 소식",
        "queries": ["중동 휴전", "이스라엘 휴전", "가자지구 협상"],
        "keywords": re.compile(r"휴전|협상.*타결|평화.*협상|휴전.*합의")
    },
     "N4": {
        "desc": "트럼프 및 대통령의 정치적 발언과 정책",
        "queries": ["트럼프 정책", "대통령 발언", "행정명령 관세"],
        "keywords": re.compile(r"트럼프|대통령.*발언|대통령.*정책|관세|대선|행정명령")
    },
    "N5": {
        "desc": "반도체 및 AI 산업 성장/수출 호조",
        "queries": ["반도체 수출", "AI 산업 성장", "HBM 수출"],
        "keywords": re.compile(r"반도체.*수출|AI.*산업|반도체.*성장|AI.*반도체|HBM.*수출|반도체.*호조")
    }
}

SAVE_PATH = str(BASE_DIR / "data/sentiment/macro_news_counts_90d.csv")

def fetch_daum_news_texts(query: str, date_str: str) -> list:
    """Daum에서 특정 일자의 뉴스 검색 결과를 2페이지(20개 기사)까지 순회하며 기사 제목을 가져옵니다."""
    texts = []
    
    # 🌟 변경 포인트 1: 1페이지와 2페이지를 순서대로 도는 루프 추가
    for page in range(1, 3):  
        params = {
            "w": "news",
            "q": query,
            "DA": "STC",
            "period": "u",
            "sd": f"{date_str}000000",
            "ed": f"{date_str}235959",
            "p": page  # 🌟 변경 포인트 2: URL에 페이지 번호(p) 파라미터 매핑
        }
        
        try:
            response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                page_texts = []
                for tag in soup.select("ul.c-list-basic li div.item-title a, ul.list_news li div.wrap_cont a"):
                    page_texts.append(tag.get_text(strip=True))
                
                # 🌟 변경 포인트 3: 다음 페이지에 기사가 아예 없다면 (검색 결과 끝) 루프 탈출
                if not page_texts:
                    break
                    
                texts.extend(page_texts)
        except Exception as e:
            print(f"  [오류] 검색 요청 중 예외 발생 ({query}, {date_str}, Page {page}): {e}")
            
        # 🌟 변경 포인트 4: 페이지 전환 시 포털 차단을 막기 위해 0.3~0.5초 대기
        time.sleep(random.uniform(0.3, 0.5))
        
    return texts

def count_matching_texts(texts: list, pattern) -> int:
    """기사 텍스트 리스트 중 정규식 패턴과 매치되는 건수를 계산합니다."""
    count = 0
    for text in texts:
        if pattern.search(text):
            count += 1
    return count

def date_range(start: datetime, end: datetime):
    """영업일(월~금) 날짜 제너레이터"""
    cur = start
    while cur <= end:
        if cur.weekday() < 5:  # 0:월 ~ 4:금
            yield cur
        cur += timedelta(days=1)

def crawl_all():
    print("=" * 60)
    print("   Daum 뉴스 수집 및 K-NSI 카운팅 크롤러 가동   ")
    print("=" * 60)
    
    # 2026년 2월 16일부터 2026년 4월 15일까지 60일치 (43거래일)
    start_date = datetime(2026, 2, 16)
    end_date = datetime(2026, 4, 15)
    
    dates = list(date_range(start_date, end_date))
    records = []
    total = len(dates)
    
    for i, d in enumerate(dates):
        date_param = d.strftime("%Y%m%d")      # Daum 기간 필터용
        date_key   = d.strftime("%Y-%m-%d")    # 저장용
        row = {"date": date_key}
        
        print(f"[{i+1}/{total}] {date_key} 뉴스 수집 중...")
        
        for cat_name, cat_info in CATEGORIES.items():
            total_count = 0
            
            for q in cat_info["queries"]:
                texts = fetch_daum_news_texts(q, date_param)
                matched = count_matching_texts(texts, cat_info["keywords"])
                total_count += matched
                time.sleep(random.uniform(0.3, 0.6))  # 디레이팅 방지
                
            row[cat_name] = total_count
            print(f"  {cat_name} ({cat_info['desc'][:10]}..): {total_count}건")
            
        records.append(row)
        time.sleep(random.uniform(0.5, 1.0))
        
    df = pd.DataFrame(records, columns=["date", "N1", "N2", "N3", "N4", "N5"])
    df = df.sort_values("date").reset_index(drop=True)
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print(f"[DONE] 크롤링 완료! 저장 경로: {SAVE_PATH}")
    print("=" * 60)
    print(df.tail(10).to_string(index=False))
    return df

if __name__ == "__main__":
    crawl_all()
