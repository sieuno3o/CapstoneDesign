from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]
from string import Template
import pandas as pd

METRICS_PATH = BASE_DIR / "results/metrics/2step_psychological_correction_summary.csv"
FIGURES_DIR  = BASE_DIR / "results/figures/final_model"
OUTPUT_PATH  = FIGURES_DIR / "2step_psychological_correction_report.html"

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2-Step Psychological Correction Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b1220;
            --card: rgba(15, 23, 42, 0.88);
            --border: rgba(148, 163, 184, 0.18);
            --primary: #8b5cf6;
            --accent: #38bdf8;
            --text: #e2e8f0;
            --muted: #94a3b8;
            --success: #34d399;
            --danger: #f87171;
        }
        * { box-sizing: border-box; }
        body { background: radial-gradient(circle at top left, rgba(139,92,246,0.15), transparent 35%), radial-gradient(circle at bottom right, rgba(56,189,248,0.12), transparent 30%), var(--bg); color: var(--text); font-family: 'Outfit', 'Noto Sans KR', sans-serif; margin: 0; padding: 36px; }
        a { color: var(--accent); text-decoration: none; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 40px; }
        header h1 { font-size: 3rem; line-height: 1.05; letter-spacing: -0.04em; margin-bottom: 14px; }
        header p { color: var(--muted); font-size: 1.05rem; max-width: 860px; margin: 0 auto; }
        .badge { display: inline-flex; align-items: center; gap: 10px; border-radius: 999px; border: 1px solid rgba(139,92,246,0.2); padding: 10px 18px; color: var(--accent); margin-bottom: 20px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 28px; padding: 28px; box-shadow: 0 24px 80px rgba(10, 15, 28, 0.25); }
        .card p { color: var(--muted); line-height: 1.7; }
        .metric-table { width: 100%; border-collapse: collapse; margin-top: 18px; }
        .metric-table th, .metric-table td { padding: 12px 14px; border: 1px solid rgba(148, 163, 184, 0.12); text-align: center; }
        .metric-table th { background: rgba(255, 255, 255, 0.04); font-weight: 600; }
        .metric-table tbody tr:hover { background: rgba(255, 255, 255, 0.03); }
        .figure-card { border-radius: 24px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); background: rgba(15,23,42,0.92); margin-top: 20px; }
        .figure-card img { width: 100%; display: block; }
        .section-title { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
        .section-title span { width: 10px; height: 38px; display: inline-block; background: linear-gradient(180deg, var(--primary), var(--accent)); border-radius: 999px; }
        .section-title h3 { margin: 0; font-size: 1.35rem; }
        .summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 32px; }
        .summary-card { padding: 20px; border-radius: 22px; background: rgba(255,255,255,0.04); border: 1px solid rgba(148, 163, 184, 0.1); }
        .summary-card strong { display: block; font-size: 1rem; color: var(--muted); margin-bottom: 10px; }
        .summary-card div { font-size: 1.75rem; font-weight: 700; color: white; }
        .small-text { color: var(--muted); font-size: 0.95rem; }
        .image-caption { margin-top: 10px; color: var(--muted); font-size: 0.95rem; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <div class="badge">2-Step Psychological Correction Report</div>
        <h1>기존 주가 예측 + 심리 보정 모델 성능 리포트</h1>
        <p>Baseline RF/ANN 예측과 심리 보정 후 최종 성능을 기업별로 비교합니다. 이 보고서는 final_model 폴더의 이미지와 지표를 기반으로 생성되었습니다.</p>
    </header>
    <section class="card">
        <div class="section-title"><span></span><h3>전체 모델 성능 요약</h3></div>
        <p>다음 표는 각 기업별 baseline 및 corrected 모델의 RMSE/MAE/MAPE/MBE/R²/Directional Accuracy를 정리한 전체 요약입니다.</p>
        $summary_table
    </section>
    <section class="card summary-card-grid">
        <div class="summary-grid">
            <div class="summary-card">
                <strong>기업 수</strong>
                <div>$company_count</div>
                <div class="small-text">Baseline/Corrected 모델 비교</div>
            </div>
            <div class="summary-card">
                <strong>평균 개선 RMSE</strong>
                <div>$avg_rmse_improvement%</div>
                <div class="small-text">Baseline 대비 Corrected RMSE 평균 개선률</div>
            </div>
        </div>
    </section>
    $company_sections
</div>
</body>
</html>
""")


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
        rows.append(
            f"<tr><td>{row['Company']}</td><td>{row['Model']}</td><td>{format_metric(row['rmse'])}</td><td>{format_metric(row['mae'])}</td><td>{format_metric(row['mape'])}</td><td>{format_metric(row['mbe'])}</td><td>{format_metric(row['r2'])}</td><td>{format_metric(row['direction_accuracy'])}</td></tr>"
        )
    footer = "</tbody></table>"
    return header + "".join(rows) + footer


def build_company_section(df: pd.DataFrame, company: str) -> str:
    subset = df[df['Company'] == company]
    if subset.empty:
        return ''

    section = [
        f"<section class=\"card\"><div class=\"section-title\"><span></span><h3>{company.replace('_', ' ').title()}</h3></div>",
        "<p>Baseline / Corrected 모델 결과를 비교한 지표와 최종 시각화를 확인하세요.</p>",
        "<div class=\"figure-card\">"
    ]
    image_name = f"{company}_2step_correction.png"
    if (FIGURES_DIR / image_name).exists():
        section.append(f"<img src=\"{image_name}\" alt=\"{company} 2step correction chart\">")
        section.append(f"<div class=\"image-caption\">{company.replace('_', ' ').title()} 최종 모델 비교</div>")
    section.append("</div>")
    section.append("<table class=\"metric-table\"><thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>MAPE</th><th>MBE</th><th>R²</th><th>Directional Accuracy</th></tr></thead><tbody>")
    for _, row in subset.iterrows():
        section.append(
            f"<tr><td>{row['Model']}</td><td>{format_metric(row['rmse'])}</td><td>{format_metric(row['mae'])}</td><td>{format_metric(row['mape'])}</td><td>{format_metric(row['mbe'])}</td><td>{format_metric(row['r2'])}</td><td>{format_metric(row['direction_accuracy'])}</td></tr>"
        )
    section.append("</tbody></table>")
    section.append("</section>")
    return "\n".join(section)


def main():
    df = pd.read_csv(METRICS_PATH)
    df['Company'] = df['Company'].astype(str)
    df['Model'] = df['Model'].astype(str)

    company_count = df['Company'].nunique()
    baseline_rmse = df[df['Model'].isin(['Baseline RF', 'Baseline ANN'])].groupby('Company')['rmse'].mean()
    corrected_rmse = df[df['Model'].isin(['Corrected RF', 'Corrected ANN'])].groupby('Company')['rmse'].mean()
    avg_rmse_improvement = ((baseline_rmse - corrected_rmse) / baseline_rmse * 100).mean()

    summary_table = build_summary_table(df)
    company_sections = [build_company_section(df, company) for company in sorted(df['Company'].unique())]

    html = HTML_TEMPLATE.substitute(
        summary_table=summary_table,
        company_count=company_count,
        avg_rmse_improvement=f"{avg_rmse_improvement:.2f}",
        company_sections='\n'.join(company_sections)
    )

    OUTPUT_PATH.write_text(html, encoding='utf-8')
    print(f"HTML report generated: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
