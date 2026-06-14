# -*- coding: utf-8 -*-
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import base64
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = BASE_DIR / "results" / "metrics"
REPORTS_DIR = BASE_DIR / "results" / "individual_reports"
FIGURES_DIR = BASE_DIR / "results" / "figures" / "final_model"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_korean_fonts = [
    "AppleGothic",
    "Apple SD Gothic Neo",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Malgun Gothic",
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font in _korean_fonts:
    if _font in _available_fonts:
        matplotlib.rc("font", family=_font)
        break
matplotlib.rcParams["axes.unicode_minus"] = False

COMPANIES_KR = {
    "samsung_electronics": "삼성전자",
    "sk_hynix": "SK하이닉스",
    "wonik_ips": "원익IPS",
    "dongjin_semichem": "동진쎄미켐",
    "hanwha_aerospace": "한화에어로스페이스",
    "lig_nex1": "LIG넥스원",
    "snt_dynamics": "SNT다이내믹스",
    "firstec": "퍼스텍",
}

MODEL_CONFIGS = [
    {
        "model_no": "12",
        "model_name": "Extended RF + 심리지수 결합모델",
        "baseline_name": "Extended RF",
        "source_csv": "2step_psychological_correction_summary.csv",
        "method": "2-Step Correction",
    },
    {
        "model_no": "13",
        "model_name": "Extended ANN + 심리지수 결합모델",
        "baseline_name": "Extended ANN",
        "source_csv": "2step_psychological_correction_summary.csv",
        "method": "2-Step Correction",
    },
    {
        "model_no": "14",
        "model_name": "Direct Hybrid RF",
        "baseline_name": "Extended RF",
        "source_csv": "direct_hybrid_summary.csv",
        "method": "Direct Feature Concat",
    },
    {
        "model_no": "15",
        "model_name": "Direct Hybrid ANN",
        "baseline_name": "Extended ANN",
        "source_csv": "direct_hybrid_summary.csv",
        "method": "Direct Feature Concat",
    },
]


def load_metrics(csv_name: str) -> pd.DataFrame:
    path = METRICS_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(f"필수 결과 CSV가 없습니다: {path}")
    return pd.read_csv(path)


def build_comparison_rows() -> pd.DataFrame:
    cache = {}
    rows = []
    for config in MODEL_CONFIGS:
        csv_name = config["source_csv"]
        if csv_name not in cache:
            cache[csv_name] = load_metrics(csv_name)
        df = cache[csv_name]

        baseline = df[df["Model"] == config["baseline_name"]].copy()
        target = df[df["Model"] == config["model_name"]].copy()
        merged = baseline.merge(target, on="Company", suffixes=("_baseline", "_model"))
        if merged.empty:
            raise ValueError(
                f"{config['source_csv']}에서 {config['baseline_name']} 또는 "
                f"{config['model_name']} 결과를 찾지 못했습니다."
            )

        for _, row in merged.iterrows():
            rmse_change = (row["rmse_baseline"] - row["rmse_model"]) / row["rmse_baseline"] * 100
            mae_change = (row["mae_baseline"] - row["mae_model"]) / row["mae_baseline"] * 100
            mape_change = row["mape_baseline"] - row["mape_model"]
            da_change = (row["direction_accuracy_model"] - row["direction_accuracy_baseline"]) * 100
            rows.append({
                "Model No": f"Model {config['model_no']}",
                "Method": config["method"],
                "Company": COMPANIES_KR.get(row["Company"], row["Company"]),
                "Baseline": config["baseline_name"],
                "Target Model": config["model_name"],
                "Baseline RMSE": row["rmse_baseline"],
                "Model RMSE": row["rmse_model"],
                "RMSE Change (%)": rmse_change,
                "Baseline MAE": row["mae_baseline"],
                "Model MAE": row["mae_model"],
                "MAE Change (%)": mae_change,
                "MAPE Change (%p)": mape_change,
                "DA Change (%p)": da_change,
                "Model MAPE (%)": row["mape_model"],
                "Model DA (%)": row["direction_accuracy_model"] * 100,
            })
    return pd.DataFrame(rows)


def build_average_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    summary = comparison_df.groupby(["Model No", "Method", "Target Model"], as_index=False).agg({
        "Baseline RMSE": "mean",
        "Model RMSE": "mean",
        "RMSE Change (%)": "mean",
        "Baseline MAE": "mean",
        "Model MAE": "mean",
        "MAE Change (%)": "mean",
        "MAPE Change (%p)": "mean",
        "DA Change (%p)": "mean",
        "Model MAPE (%)": "mean",
        "Model DA (%)": "mean",
    })
    return summary


def format_table(df: pd.DataFrame, detail: bool = False) -> str:
    formatted = df.copy()
    money_cols = ["Baseline RMSE", "Model RMSE", "Baseline MAE", "Model MAE"]
    pct_cols = ["RMSE Change (%)", "MAE Change (%)", "MAPE Change (%p)", "DA Change (%p)", "Model MAPE (%)", "Model DA (%)"]

    for col in money_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:,.2f}")
    for col in pct_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x: f"{x:+.2f}" if "Change" in col else f"{x:.2f}")

    if not detail:
        formatted = formatted[[
            "Model No", "Method", "Target Model",
            "Baseline RMSE", "Model RMSE", "RMSE Change (%)",
            "Model MAPE (%)", "Model DA (%)",
        ]]
    else:
        formatted = formatted[[
            "Model No", "Company", "Baseline", "Target Model",
            "Baseline RMSE", "Model RMSE", "RMSE Change (%)",
            "Model MAE", "Model MAPE (%)", "Model DA (%)",
        ]]
    return formatted.to_html(index=False, classes="summary-table", escape=False)


def make_chart(avg_df: pd.DataFrame) -> Path:
    labels = avg_df["Model No"].tolist()
    baseline_rmse = avg_df["Baseline RMSE"].to_numpy()
    model_rmse = avg_df["Model RMSE"].to_numpy()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    ax.bar(x - width / 2, baseline_rmse, width, label="Baseline", color="#64748b")
    ax.bar(x + width / 2, model_rmse, width, label="Model 12-15", color="#2563eb")
    ax.set_title("Models 12-15 Average RMSE Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("RMSE (KRW)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    for i, value in enumerate(model_rmse):
        ax.text(i + width / 2, value + max(model_rmse) * 0.015, f"{value:,.0f}", ha="center", fontsize=9)

    fig.tight_layout()
    chart_path = FIGURES_DIR / "model_12_15_rmse_summary.png"
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def image_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_report():
    comparison_df = build_comparison_rows()
    avg_df = build_average_summary(comparison_df)
    chart_uri = image_to_data_uri(make_chart(avg_df))

    total_companies = comparison_df["Company"].nunique()
    best_row = avg_df.loc[avg_df["Model RMSE"].idxmin()]
    avg_table = format_table(avg_df, detail=False)
    detail_table = format_table(comparison_df, detail=True)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>모델 12-15 최종 요약 리포트</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #0f172a;
            --muted: #64748b;
            --border: #e2e8f0;
            --primary: #2563eb;
            --accent: #10b981;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 36px 20px;
            font-family: "Malgun Gothic", "Apple SD Gothic Neo", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}
        .container {{ max-width: 1180px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 32px; }}
        h1 {{ margin: 0 0 8px; font-size: 2.2rem; color: #1e293b; }}
        .subtitle {{ margin: 0; color: var(--muted); }}
        .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 26px;
            margin-bottom: 24px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
        }}
        h2 {{
            margin: 0 0 18px;
            padding-left: 12px;
            border-left: 5px solid var(--primary);
            font-size: 1.35rem;
        }}
        .kpis {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
        }}
        .kpi {{
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            background: #f8fafc;
        }}
        .kpi strong {{ display: block; color: var(--muted); font-size: 0.86rem; }}
        .kpi span {{ display: block; margin-top: 6px; font-size: 1.45rem; font-weight: 700; }}
        .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        th {{ background: #0f172a; color: white; text-align: left; padding: 11px 12px; white-space: nowrap; }}
        td {{ border-top: 1px solid var(--border); padding: 10px 12px; white-space: nowrap; }}
        tr:nth-child(even) td {{ background: #f8fafc; }}
        img {{ max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 10px; }}
        .note {{ color: var(--muted); margin-top: 12px; font-size: 0.94rem; }}
        @media (max-width: 760px) {{ .kpis {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>모델 12-15 최종 요약 리포트</h1>
            <p class="subtitle">학습은 target_next_return, 평가는 target_next_close 기준으로 통일</p>
        </header>

        <section class="card">
            <h2>요약</h2>
            <div class="kpis">
                <div class="kpi"><strong>대상 종목</strong><span>{total_companies}개</span></div>
                <div class="kpi"><strong>비교 모델</strong><span>4개</span></div>
                <div class="kpi"><strong>평균 RMSE 최저</strong><span>{best_row["Model No"]}</span></div>
            </div>
            <p class="note">RMSE 변화율은 기준 모델 대비 낮아지면 양수, 높아지면 음수입니다.</p>
        </section>

        <section class="card">
            <h2>모델별 평균 성과</h2>
            <div class="table-wrap">{avg_table}</div>
        </section>

        <section class="card">
            <h2>평균 RMSE 비교 차트</h2>
            <img src="{chart_uri}" alt="Models 12-15 Average RMSE Chart">
        </section>

        <section class="card">
            <h2>8개 종목 상세 비교</h2>
            <div class="table-wrap">{detail_table}</div>
        </section>
    </div>
</body>
</html>
"""

    output_path = REPORTS_DIR / "final_summary_12_15_report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"[저장] {output_path}")


if __name__ == "__main__":
    generate_report()
