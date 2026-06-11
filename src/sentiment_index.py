import pandas as pd


def build_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    candidate_cols = [
        "news_sentiment",
        "comment_sentiment",
        "community_sentiment",
        "volume_change",
        "position_change",
    ]

    existing_cols = [col for col in candidate_cols if col in result.columns]

    if not existing_cols:
        raise ValueError("심리지수 생성에 사용할 컬럼이 없습니다.")

    result["sentiment_index"] = result[existing_cols].mean(axis=1)
    return result