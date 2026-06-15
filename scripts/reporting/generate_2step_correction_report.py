from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]
from string import Template
import pandas as pd

METRICS_PATH = BASE_DIR / "results/metrics/2step_psychological_correction_summary.csv"
FIGURES_DIR  = BASE_DIR / "results/figures/final_model"
OUTPUT_PATH  = FIGURES_DIR / "2step_correction_report.html"
SUMMARY_IMAGE = FIGURES_DIR / "2step_correction_rmse_summary.png"

ANALYSIS_SECTION = """
<section class="card">
    <div class="section-title"><span></span><h3>프로젝트 분석 요약</h3></div>
    <p>아래 내용은 2-step 심리 보정 모델 결과를 바탕으로 한 기업별 해석과 결론입니다.</p>
    <div>
        <h4>1. 반도체 및 소부장 기업군 분석</h4>
        <ul>
            <li><strong>삼성전자</strong>: 심리 지수 반영 시 RMSE/MAE 증가, R² 및 방향 정확도 감소. 대형주는 글로벌 자금 흐름과 패시브 자금 영향이 더 크며, 뉴스 심리는 노이즈로 작용할 수 있습니다.</li>
            <li><strong>SK하이닉스</strong>: R²가 0에 가까워졌으나 ANN의 방향 정확도는 40%에서 53.3%로 개선되었습니다. K-NSI는 방향성을 잡는 데 긍정적인 역할을 한 것으로 해석됩니다.</li>
            <li><strong>원익IPS</strong>: RF/ANN 모두 RMSE, MAE, MAPE 개선. 방향 정확도는 최대 60%까지 상승했고 R²도 개선되었습니다. 장비주는 전방 산업 뉴스와 심리 변화에 민감하게 반응합니다.</li>
            <li><strong>동진쎄미켐</strong>: 기존 가격 모델(Extended RF)이 이미 우수하여 심리 지수 반영 후 오차가 소폭 증가했습니다. 기업 고유의 기술적 요인이 더 큰 영향을 미쳤을 가능성이 있습니다.</li>
        </ul>
        <h4>2. 방위산업 기업군 분석</h4>
        <ul>
            <li><strong>한화에어로스페이스</strong>: RF 모델에서 RMSE, MAE 감소, R²이 0.37에서 0.49로 상승, 방향 정확도 60% 개선. 지정학적 리스크 뉴스가 주가에 직결된다고 볼 수 있습니다.</li>
            <li><strong>LIG넥스원</strong>: 기존 Extended RF RMSE가 약 11만에서 7.4만으로 크게 감소, R²도 음수에서 양수로 개선되었습니다. K-NSI가 대외 리스크 역할을 했습니다.</li>
            <li><strong>SNT다이나믹스</strong>: 모든 오차 지표가 최저 수준으로 떨어지고 방향 정확도가 86.6%/80%에 근접, R² 0.63까지 상승. 뉴스 심리에 가장 강하게 반응한 기업입니다.</li>
            <li><strong>퍼스텍</strong>: 기존 모델의 R²가 이미 매우 높아 심리 지수 추가 시 오차가 증가했습니다. 소형 테마주는 기술적 변수와 수급이 더 중요할 수 있습니다.</li>
        </ul>
        <h4>결론 요약</h4>
        <ol>
            <li>대형주보다 원익IPS, 한화에어로스페이스, LIG넥스원, SNT다이나믹스 같은 중대형 우량 섹터주에서 K-NSI 보정 효과가 가장 뛰어났습니다.</li>
            <li>방산주는 대외 리스크/정책 뉴스에, 반도체 장비주는 호재 뉴스에 민감하게 반응하며 K-NSI 인과성이 확인되었습니다.</li>
            <li>시가총액과 섹터 특성에 따라 심리 지수의 가치가 달라지며, 모든 종목에 동일하게 적용하기보다는 섹터별 차별적 활용이 필요합니다.</li>
        </ol>
    </div>
</section>
"""

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2-Step Psychological Correction Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+KR:wght@300;400;500;700;900&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(17, 24, 39, 0.78);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #8b5cf6;
            --secondary: #06b6d4;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --success: #10b981;
            --danger: #ef4444;
            --accent: #38bdf8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Outfit', 'Noto Sans KR', sans-serif;
            line-height: 1.7;
            padding: 40px 24px;
            background-image:
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.18) 0px, transparent 40%),
                radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.15) 0px, transparent 40%),
                radial-gradient(at 50% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%);
            background-attachment: fixed;
        }

        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 50px; }
        header .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            border-radius: 999px;
            border: 1px solid rgba(139, 92, 246, 0.25);
            padding: 8px 18px;
            background: rgba(139, 92, 246, 0.12);
            color: var(--secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 18px;
        }
        header h1 {
            font-size: 2.8rem;
            line-height: 1.1;
            margin-bottom: 18px;
            background: linear-gradient(90deg, #b490ff, #38bdf8, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        header p {
            color: var(--text-muted);
            max-width: 860px;
            margin: 0 auto;
            font-size: 1.05rem;
        }

        .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }
        @media (max-width: 960px) { .grid-2 { grid-template-columns: 1fr; } }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 28px;
            padding: 32px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
            position: relative;
            overflow: hidden;
        }
        .card h2 { font-size: 1.6rem; margin-bottom: 16px; }
        .card p { color: var(--text-muted); }
        .metric-table { width: 100%; border-collapse: collapse; margin-top: 18px; }
        .metric-table th, .metric-table td { padding: 14px 16px; border: 1px solid rgba(255, 255, 255, 0.08); text-align: center; }
        .metric-table th { background: rgba(255, 255, 255, 0.04); font-weight: 600; color: var(--text-main); }
        .metric-table tbody tr:hover { background: rgba(255, 255, 255, 0.04); }
        .figure-card { margin-top: 20px; border-radius: 24px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
        .figure-card img { width: 100%; display: block; }
        .summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; margin-top: 30px; }
        @media (max-width: 960px) { .summary-grid { grid-template-columns: 1fr; } }
        .summary-card {
            border-radius: 22px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 24px;
        }
        .summary-card strong { display: block; color: var(--text-muted); margin-bottom: 10px; font-size: 0.95rem; }
        .summary-card span { display: block; font-size: 2rem; font-weight: 700; color: #ffffff; }
        .section-title { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
        .section-title span { width: 10px; height: 38px; display: inline-block; background: linear-gradient(180deg, var(--primary), var(--secondary)); border-radius: 999px; }
        .section-title h3 { margin: 0; font-size: 1.4rem; }
        .image-caption { margin-top: 10px; color: var(--text-muted); font-size: 0.95rem; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <div class="badge">Final Model Report</div>
        <h1>Benchmark + 심리지수 결합 모델 분석 보고서</h1>
        <p>이 보고서는 <strong>train_2step_correction.py</strong> 실행 결과로 생성된 지표와 그림을 기반으로 작성되었습니다. 각 기업별 기존 가격 모델(Extended)과 심리지수 결합 보정 모델의 성능을 한눈에 확인할 수 있습니다.</p>
    </header>

    <section class="card">
        <div class="section-title"><span></span><h3>전체 요약</h3></div>
        <p>기존 Benchmark 예측과 심리지수 결합 보정 모델의 RMSE 개선 여부를 중심으로 성능을 비교합니다.</p>
        <div class="summary-grid">
            <div class="summary-card"><strong>대상 기업 수</strong><span>$company_count</span></div>
            <div class="summary-card"><strong>평균 RMSE 개선률</strong><span>$avg_rmse_improvement%</span></div>
            <div class="summary-card"><strong>생성된 이미지</strong><span>$image_count</span></div>
        </div>
        <div class="figure-card">
            <img src="$summary_image" alt="2-Step Correction RMSE Summary">
            <div class="image-caption">기존 모델과 심리지수 결합 보정 모델의 RMSE 비교 요약</div>
        </div>
    </section>

    <section class="card">
        <div class="section-title"><span></span><h3>세부 지표 테이블</h3></div>
        $summary_table
    </section>

    $analysis_section

    $company_sections
</div>
</body>
</html>
""")


MODEL_MAPPING = {}

def format_metric(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_summary_table(df: pd.DataFrame) -> str:
    header = "<table class=\"metric-table\"><thead><tr><th>Company</th><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>MBE</th><th>R²</th><th>Directional Accuracy</th></tr></thead><tbody>"
    rows = []
    for _, row in df.iterrows():
        model_name = MODEL_MAPPING.get(row['Model'], row['Model'])
        rows.append(
            f"<tr><td>{row['Company']}</td><td>{model_name}</td><td>{format_metric(row['rmse'])}</td><td>{format_metric(row['mae'])}</td><td>{format_metric(row['mape'])}</td><td>{format_metric(row['mbe'])}</td><td>{format_metric(row['r2'])}</td><td>{format_metric(row['direction_accuracy'])}</td></tr>"
        )
    footer = "</tbody></table>"
    return header + "".join(rows) + footer


def build_company_section(df: pd.DataFrame, company: str) -> str:
    subset = df[df['Company'] == company]
    if subset.empty:
        return ""

    section = [
        f"<section class=\"card\"><div class=\"section-title\"><span></span><h3>{company.replace('_', ' ').title()}</h3></div>",
        "<p>기존 가격 모델과 심리지수 결합 보정 모델의 가격 예측 성능을 비교한 기업별 상세 결과입니다.</p>",
        "<div class=\"figure-card\">"
    ]

    image_name = f"{company}_2step_correction.png"
    if (FIGURES_DIR / image_name).exists():
        section.append(f"<img src=\"{image_name}\" alt=\"{company} 2-step correction chart\">")
        section.append(f"<div class=\"image-caption\">{company.replace('_', ' ').title()} 모델 비교 차트</div>")
    section.append("</div>")

    section.append("<table class=\"metric-table\"><thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>MBE</th><th>R²</th><th>Directional Accuracy</th></tr></thead><tbody>")
    for _, row in subset.iterrows():
        model_name = MODEL_MAPPING.get(row['Model'], row['Model'])
        section.append(
            f"<tr><td>{model_name}</td><td>{format_metric(row['rmse'])}</td><td>{format_metric(row['mae'])}</td><td>{format_metric(row['mape'])}</td><td>{format_metric(row['mbe'])}</td><td>{format_metric(row['r2'])}</td><td>{format_metric(row['direction_accuracy'])}</td></tr>"
        )
    section.append("</tbody></table>")
    section.append("</section>")
    return "\n".join(section)


def main():
    df = pd.read_csv(METRICS_PATH)
    df['Company'] = df['Company'].astype(str)
    df['Model'] = df['Model'].astype(str)

    company_count = df['Company'].nunique()
    baseline_rmse = df[df['Model'] == 'Benchmark'].groupby('Company')['rmse'].mean()
    corrected_rmse = df[df['Model'] == 'Benchmark + 심리지수 결합모델'].groupby('Company')['rmse'].mean()
    avg_rmse_improvement = ((baseline_rmse - corrected_rmse) / baseline_rmse * 100).mean()
    image_count = sum((FIGURES_DIR / f"{company}_2step_correction.png").exists() for company in df['Company'].unique())

    summary_table = build_summary_table(df)
    company_sections = [build_company_section(df, company) for company in sorted(df['Company'].unique())]

    template_text = HTML_TEMPLATE.template.replace('$analysis_section', ANALYSIS_SECTION)
    html = Template(template_text).substitute(
        summary_table=summary_table,
        company_count=company_count,
        avg_rmse_improvement=f"{avg_rmse_improvement:.2f}",
        image_count=image_count,
        summary_image=SUMMARY_IMAGE.name,
        company_sections='\n'.join(company_sections)
    )

    OUTPUT_PATH.write_text(html, encoding='utf-8')
    print(f"HTML report generated: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
