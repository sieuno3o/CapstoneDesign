import csv
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

BASE_URL = "https://finance.naver.com/news/mainnews.naver"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://finance.naver.com/",
}

session = requests.Session()
session.headers.update(HEADERS)


def fetch_news_page(page: int, section_id: str = "101", section_id2: str = "258", mode: str = "LSS2D") -> str:
    params = {
        "mode": mode,
        "section_id": section_id,
        "section_id2": section_id2,
        "page": str(page),
    }
    response = session.get(BASE_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.text


def parse_news_items(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, str]] = []

    for li in soup.select("ul.newsList li"):
        title_tag = li.select_one("dd.articleSubject a")
        date_tag = li.select_one("span.wdate, span.date")
        if title_tag is None or date_tag is None:
            continue

        title = title_tag.get_text(strip=True)
        date_text = date_tag.get_text(strip=True)
        items.append({"title": title, "date": date_text})

    return items


def parse_date(date_text: str) -> Optional[datetime]:
    try:
        return date_parser.parse(date_text)
    except (ValueError, OverflowError):
        return None


def crawl_naver_finance_news(
    years: int = 5,
    max_pages: int = 2000,
    section_id: str = "101",
    section_id2: str = "258",
    mode: str = "LSS2D",
) -> List[Dict[str, str]]:
    end_date = datetime.today()
    threshold = end_date - timedelta(days=years * 365)
    page = 1
    news_items: List[Dict[str, str]] = []
    seen_keys = set()

    while page <= max_pages:
        html = fetch_news_page(page, section_id=section_id, section_id2=section_id2, mode=mode)
        page_items = parse_news_items(html)
        if not page_items:
            break

        reached_older = False
        for item in page_items:
            item_date = parse_date(item["date"])
            if item_date is None:
                continue

            if item_date < threshold:
                reached_older = True
                continue

            key = (item["title"], item["date"])
            if key in seen_keys:
                continue

            seen_keys.add(key)
            news_items.append({"date": item["date"], "title": item["title"]})

        if reached_older:
            break

        page += 1
        time.sleep(0.5)

    return news_items


def save_news_to_csv(news_items: List[Dict[str, str]], csv_path: str) -> None:
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["date", "title"])
        writer.writeheader()
        writer.writerows(news_items)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="네이버 금융 뉴스 제목과 날짜를 크롤링하여 CSV로 저장합니다.")
    parser.add_argument("--output", "-o", default="naver_finance_news.csv", help="저장할 CSV 파일 경로")
    parser.add_argument("--years", type=int, default=5, help="최근 몇 년간 뉴스까지 수집할지 지정")
    parser.add_argument("--section-id", default="101", help="네이버 금융 뉴스 section_id")
    parser.add_argument("--section-id2", default="258", help="네이버 금융 뉴스 section_id2")
    parser.add_argument("--max-pages", type=int, default=2000, help="최대 크롤링 페이지 수")
    args = parser.parse_args()

    articles = crawl_naver_finance_news(
        years=args.years,
        max_pages=args.max_pages,
        section_id=args.section_id,
        section_id2=args.section_id2,
    )
    save_news_to_csv(articles, args.output)
    print(f"Saved {len(articles)} articles to {args.output}")
