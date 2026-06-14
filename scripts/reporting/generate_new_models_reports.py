# -*- coding: utf-8 -*-
import os
import base64
import pandas as pd
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "individual_reports"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create reports directory
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES_KR = {
    "samsung_electronics": "삼성전자",
    "sk_hynix": "SK하이닉스",
    "wonik_ips": "원익IPS",
    "dongjin_semichem": "동진쎄미켐",
    "hanwha_aerospace": "한화에어로스페이스",
    "lig_nex1": "LIG넥스원",
    "snt_dynamics": "SNT다이내믹스",
    "firstec": "퍼스텍"
}

def generate_report(csv_name, baseline_name, model_name, title, output_name, desc_baseline, desc_model, chart_suffix):
    csv_path = METRICS_DIR / csv_name
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    df_rounded = df.copy()
    df_rounded['rmse'] = df_rounded['rmse'].round(2)
    df_rounded['mae'] = df_rounded['mae'].round(2)
    df_rounded['mape'] = df_rounded['mape'].round(2)
    df_rounded['direction_accuracy'] = (df_rounded['direction_accuracy'] * 100).round(2)

    # Calculate overall improvement
    try:
        baseline_rmse = df[df['Model'] == baseline_name].groupby('Company')['rmse'].mean()
        corrected_rmse = df[df['Model'] == model_name].groupby('Company')['rmse'].mean()
        avg_rmse_improvement = ((baseline_rmse - corrected_rmse) / baseline_rmse * 100).mean()
    except Exception as e:
        print(f"Error calculating RMSE improvement: {e}")
        avg_rmse_improvement = 0.0

    company_count = df['Company'].nunique()
    
    # We will embed the chart for lig_nex1 as a sample
    sample_chart_path = FIGURES_DIR / f"final_model/lig_nex1_{chart_suffix}.png"
    if not sample_chart_path.exists():
        # Fallback to direct hybrid path
        sample_chart_path = FIGURES_DIR / f"lig_nex1_{chart_suffix}.png"
        
    base64_chart = ""
    if sample_chart_path.exists():
        with open(sample_chart_path, "rb") as image_file:
            base64_chart = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        print(f"Warning: Sample chart {sample_chart_path} missing.")

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #10b981;
            --primary-dark: #059669;
            --secondary: #3b82f6;
            --background: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #10b981;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', 'Malgun Gothic', sans-serif; background-color: var(--background); color: var(--text-main); line-height: 1.6; padding: 40px 20px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 2px solid var(--border); }}
        h1 {{ font-family: 'Outfit', sans-serif; font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }}
        .subtitle {{ font-size: 1.1rem; color: var(--text-muted); }}
        .card {{ background-color: var(--card-bg); border-radius: 16px; box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05); border: 1px solid var(--border); padding: 30px; margin-bottom: 30px; }}
        h2 {{ font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 600; color: #1e293b; margin-bottom: 20px; display: flex; align-items: center; border-left: 5px solid var(--primary); padding-left: 12px; }}
        h3 {{ font-family: 'Outfit', sans-serif; font-size: 1.1rem; font-weight: 600; color: var(--primary-dark); margin-top: 15px; margin-bottom: 8px; }}
        p {{ margin-bottom: 15px; color: #334155; font-size: 0.98rem; }}
        .formula-box {{ background: #f8fafc; border: 1px dashed var(--primary); padding: 20px; border-radius: 12px; margin: 20px 0; font-family: 'Courier New', Courier, monospace; font-size: 0.95rem; overflow-x: auto; color: #1e293b; font-weight: 500; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; margin-top: 20px; }}
        .summary-card {{ background-color: #f8fafc; border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; }}
        .summary-card strong {{ display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 5px; text-transform: uppercase; }}
        .summary-card span {{ font-size: 1.8rem; font-weight: 700; color: var(--text-main); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95rem; }}
        th {{ background-color: #0f172a; color: #ffffff; text-align: left; padding: 12px 15px; font-weight: 500; }}
        th:first-child {{ border-top-left-radius: 8px; }} th:last-child {{ border-top-right-radius: 8px; }}
        td {{ padding: 12px 15px; border-bottom: 1px solid var(--border); }}
        tr:hover {{ background-color: #f8fafc; }}
        .model-highlight {{ background-color: #ecfeff !important; font-weight: 600; }}
        .model-bench {{ background-color: #f0fdfa; color: #0f766e; }}
        .badge-win {{ display: inline-block; background-color: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }}
        .chart-container {{ text-align: center; margin-top: 20px; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid var(--border); }}
        footer {{ text-align: center; margin-top: 50px; color: var(--text-muted); font-size: 0.85rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">{desc_baseline} 대비 {desc_model}의 성능 분석 보고서</p>
        </header>

        <div class="card">
            <h2>1. 예측 성능 요약 대시보드</h2>
            <div class="summary-grid">
                <div class="summary-card"><strong>대상 기업 수</strong><span>{company_count}개</span></div>
                <div class="summary-card"><strong>평균 RMSE 개선률</strong><span>{avg_rmse_improvement:+.2f}%</span></div>
                <div class="summary-card"><strong>비교 기준</strong><span>{baseline_name}</span></div>
            </div>
        </div>

        <div class="card">
            <h2>2. 8개 종목 모델별 실험 결과 비교표</h2>
            <table>
                <thead>
                    <tr>
                        <th>종목명 (Company)</th>
                        <th>예측 모델 (Model)</th>
                        <th>RMSE (원)</th>
                        <th>MAE (원)</th>
                        <th>MAPE (%)</th>
                        <th>방향성 정확도 (Acc)</th>
                        <th>비고 (Performance)</th>
                    </tr>
                </thead>
                <tbody>
"""
    current_company = ""
    for idx, row in df_rounded.iterrows():
        comp = row['Company']
        comp_display = COMPANIES_KR.get(comp, comp)
        model_raw = row['Model']
        
        # Only show baseline and the target model to keep the report focused
        if model_raw not in [baseline_name, model_name]:
            continue
            
        rmse = f"{row['rmse']:,.2f}"
        mae = f"{row['mae']:,.2f}"
        mape = f"{row['mape']:.2f}%"
        acc = f"{row['direction_accuracy']:.2f}%"
        
        tr_class = ' class="model-highlight"' if model_raw == model_name else ' class="model-bench"'
            
        win_badge = ""
        if model_raw == model_name:
            # Find baseline row for this company
            baseline_rows = df[(df['Company'] == comp) & (df['Model'] == baseline_name)]
            if len(baseline_rows) > 0:
                baseline_row = baseline_rows.iloc[0]
                if df.loc[idx]['rmse'] < baseline_row['rmse']:
                    win_badge = '<span class="badge-win">오차 개선 완료!</span>'
                elif df.loc[idx]['direction_accuracy'] > baseline_row['direction_accuracy']:
                    win_badge = '<span class="badge-win">방향성 개선!</span>'
                
        comp_cell = ""
        if comp != current_company:
            comp_cell = f'<td rowspan="2" style="font-weight: 600; background: #fff; border-right: 1px solid var(--border);">{comp_display}</td>'
            current_company = comp
            
        html_content += f"""
                    <tr{tr_class}>
                        {comp_cell}
                        <td>{model_raw}</td>
                        <td>{rmse}</td>
                        <td>{mae}</td>
                        <td>{mape}</td>
                        <td>{acc}</td>
                        <td>{win_badge}</td>
                    </tr>"""

    html_content += f"""
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>3. 예측 비교 시각화 (LIG넥스원 샘플)</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{base64_chart}" alt="LIG Nex1 Chart" />
                <div class="image-caption">LIG넥스원 가격 예측 시계열</div>
            </div>
        </div>

        <footer>
            <p>© 2026 Capstone Design - K-NSI Project Group. All rights reserved.</p>
        </footer>
    </div>
</body>
</html>
"""
    out_path = REPORTS_DIR / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated: {out_path}")

def organize_existing_reports():
    # Move all *_report.html files from results to results/individual_reports except final_summary_report.html
    count = 0
    for file in RESULTS_DIR.glob("*_report.html"):
        if file.name == "final_summary_report.html":
            continue
        dest = REPORTS_DIR / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file.name} -> {REPORTS_DIR.name}/")
        count += 1
    print(f"Moved {count} existing HTML reports.")

def main():
    # 1. Extended RF (Model 12)
    generate_report(
        csv_name="2step_psychological_correction_summary.csv",
        baseline_name="Extended RF",
        model_name="Extended RF + 심리지수 결합모델",
        title="Extended RF + 심리지수 결합 모델 분석 보고서 (Model 12)",
        output_name="extended_rf_sentiment_report.html",
        desc_baseline="확장 변수 10개 기반 RF",
        desc_model="K-NSI 선형 보정",
        chart_suffix="2step_correction"
    )
    
    # 2. Extended ANN (Model 13)
    generate_report(
        csv_name="2step_psychological_correction_summary.csv",
        baseline_name="Extended ANN",
        model_name="Extended ANN + 심리지수 결합모델",
        title="Extended ANN + 심리지수 결합 모델 분석 보고서 (Model 13)",
        output_name="extended_ann_sentiment_report.html",
        desc_baseline="확장 변수 10개 기반 ANN",
        desc_model="K-NSI 선형 보정",
        chart_suffix="2step_correction"
    )
    
    # 3. Direct Hybrid RF (Model 14)
    generate_report(
        csv_name="direct_hybrid_summary.csv",
        baseline_name="Extended RF",
        model_name="Direct Hybrid RF",
        title="Direct Hybrid RF 모델 분석 보고서 (Model 14)",
        output_name="direct_hybrid_rf_report.html",
        desc_baseline="확장 변수 10개 단독",
        desc_model="결정변수+심리변수 12개 직접 결합",
        chart_suffix="direct_hybrid_prediction"
    )
    
    # 4. Direct Hybrid ANN (Model 15)
    generate_report(
        csv_name="direct_hybrid_summary.csv",
        baseline_name="Extended ANN",
        model_name="Direct Hybrid ANN",
        title="Direct Hybrid ANN 모델 분석 보고서 (Model 15)",
        output_name="direct_hybrid_ann_report.html",
        desc_baseline="확장 변수 10개 단독",
        desc_model="결정변수+심리변수 12개 직접 결합",
        chart_suffix="direct_hybrid_prediction"
    )

    organize_existing_reports()

if __name__ == "__main__":
    main()
