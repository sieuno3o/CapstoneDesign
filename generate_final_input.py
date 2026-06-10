from src.final_input import calculate_survey_weights, generate_daily_k_nsi


def main():
    survey_path = "data/raw/survey.csv"
    news_path = "data/raw/macro_news_counts_90d.csv"
    output_path = "final_input.csv"

    short_w, long_w = calculate_survey_weights(survey_path)
    df_final = generate_daily_k_nsi(news_path, short_w, long_w)
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Generated final_input.csv")
    print(df_final.head())


if __name__ == "__main__":
    main()
