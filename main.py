from src.train import train_all_models

def main():
    print("4개 모델 비교 프로젝트 시작")
    print("================================")
    print("1. [실행됨] 고전적 모델 (ARIMA 파이프라인)")
    print("2. [대기중] AI 기반 모델")
    print("3. [대기중] 심리 지수만 이용한 모델")
    print("4. [대기중] AI + 심리 지수 결합 최종 모델")
    print("================================\n")
    
    # 1. 고전적 모델(ARIMA) 30개 종목 순회 파이프라인 실행
    train_all_models()


if __name__ == "__main__":
    main()