import os
import pandas as pd

# raw 데이터 폴더 경로
data_dir = "data/rawdata"

# 폴더 안 csv 파일 목록 가져오기
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

print("=" * 60)
print("전체 CSV 데이터 검수 시작")
print("=" * 60)

for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)

    print("\n" + "=" * 60)
    print(f"파일명: {file_name}")
    print("=" * 60)

    try:
        # CSV 읽기
        df = pd.read_csv(file_path)

        # Date 컬럼이 있으면 날짜형 변환
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # 기본 정보 출력
        print("\n컬럼명")
        print(df.columns.tolist())

        print("\n데이터 크기 (행, 열)")
        print(df.shape)

        if "Date" in df.columns:
            print("\n날짜 범위")
            print(f"{df['Date'].min()} ~ {df['Date'].max()}")

        print("\n결측치 개수")
        print(df.isnull().sum())

        print("\n상위 3행")
        print(df.head(3))

        print("\n기초 통계량")
        print(df.describe(include="all"))

    except Exception as e:
        print(f"\n오류 발생: {e}")

print("\n" + "=" * 60)
print("전체 CSV 데이터 검수 완료")
print("=" * 60)