# -*- coding: utf-8 -*-
import os
import json
import pandas as pd

def generate_report():
    project_dir = r"C:\Users\User\Desktop\CapstoneDesign-K-NSI-real\CapstoneDesign-K-NSI"
    csv_path = os.path.join(project_dir, "results", "metrics", "2step_psychological_correction_summary.csv")
    
    html_project_path = os.path.join(project_dir, "results", "final_summary_report.html")
    html_desktop_path = r"C:\Users\User\Desktop\K-NSI_주가예측결합모델_최종종합보고서.html"
    
    # 1. Read CSV metrics
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
    df = pd.read_csv(csv_path)
    
    companies_kr = {
        "samsung_electronics": "삼성전자",
        "sk_hynix": "SK하이닉스",
        "wonik_ips": "원익IPS",
        "dongjin_semichem": "동진쎄미켐",
        "hanwha_aerospace": "한화에어로스페이스",
        "lig_nex1": "LIG넥스원",
        "snt_dynamics": "SNT다이내믹스",
        "firstec": "퍼스텍"
    }

    # Helper function to get row metrics
    def get_metrics(df_sub, model_name):
        row = df_sub[df_sub['Model'] == model_name]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "rmse": r["rmse"],
            "mae": r["mae"],
            "mape": r["mape"],
            "acc": r["direction_accuracy"] * 100
        }

    # Generate comparison tables data dynamically
    # Use "vs 심리결합" as requested by the user
    comparison_rows = ""
    for comp in sorted(df['Company'].unique()):
        comp_display = companies_kr.get(comp, comp)
        df_comp = df[df['Company'] == comp]
        
        comparisons = [
            ("Benchmark vs 심리결합", "Benchmark", "Benchmark + 심리지수 결합모델"),
            ("ARIMA vs 심리결합", "ARIMA", "ARIMA + 심리지수 결합모델"),
            ("Existing RF vs 심리결합", "Existing RF", "Existing RF + 심리지수 결합모델"),
            ("Existing ANN vs 심리결합", "Existing ANN", "Existing ANN + 심리지수 결합모델")
        ]
        
        comp_cell = f'<td rowspan="4" style="font-weight: 700; background: #f8fafc; border-right: 1px solid #e2e8f0; text-align: center;">{comp_display}</td>'
        
        for idx, (label, base, corr) in enumerate(comparisons):
            m_base = get_metrics(df_comp, base)
            m_corr = get_metrics(df_comp, corr)
            
            if not m_base or not m_corr:
                continue
                
            rmse_imp = ((m_base["rmse"] - m_corr["rmse"]) / m_base["rmse"] * 100)
            mape_imp = (m_base["mape"] - m_corr["mape"])
            acc_imp = (m_corr["acc"] - m_base["acc"])
            
            # Format improvements (Blue for positive improvement, Red for negative degradation)
            rmse_badge = f'<span style="color: {"#2563eb" if rmse_imp > 0 else "#ef4444"}; font-weight:600;">{rmse_imp:+.2f}%</span>'
            mape_badge = f'<span style="color: {"#2563eb" if mape_imp > 0 else "#ef4444"}; font-weight:600;">{mape_imp:+.2f}%p</span>'
            
            if acc_imp > 0:
                acc_badge = f'<span style="color: #2563eb; font-weight:600;">{acc_imp:+.2f}%p</span>'
            elif acc_imp < 0:
                acc_badge = f'<span style="color: #ef4444; font-weight:600;">{acc_imp:+.2f}%p</span>'
            else:
                acc_badge = f'<span style="color: #64748b; font-weight:600;">{acc_imp:+.2f}%p</span>'
            
            comp_col = comp_cell if idx == 0 else ""
            
            comparison_rows += f"""
            <tr>
                {comp_col}
                <td style="font-weight: 500;">{label}</td>
                <td>{m_base["rmse"]:,.1f} &rarr; {m_corr["rmse"]:,.1f}</td>
                <td>{rmse_badge}</td>
                <td>{m_base["mape"]:.2f}% &rarr; {m_corr["mape"]:.2f}%</td>
                <td>{mape_badge}</td>
                <td>{m_base["acc"]:.2f}% &rarr; {m_corr["acc"]:.2f}%</td>
                <td>{acc_badge}</td>
            </tr>"""

    # Prepare Chart.js data
    companies_sorted = sorted(df['Company'].unique())
    company_labels = [companies_kr.get(c, c) for c in companies_sorted]
    
    models = [
        "Benchmark", "Benchmark + 심리지수 결합모델",
        "ARIMA", "ARIMA + 심리지수 결합모델",
        "Existing RF", "Existing RF + 심리지수 결합모델",
        "Extended RF",
        "Existing ANN", "Existing ANN + 심리지수 결합모델",
        "Extended ANN"
    ]
    
    colors = [
        "#64748b", # Benchmark (Slate Gray)
        "#1e40af", # Benchmark + 심리지수 결합모델 (Deep Blue)
        "#a1a1aa", # ARIMA (Light Gray)
        "#d97706", # ARIMA + 심리지수 결합모델 (Amber)
        "#93c5fd", # Existing RF (Light Blue)
        "#2563eb", # Existing RF + 심리지수 결합모델 (Blue)
        "#10b981", # Extended RF (Emerald Green)
        "#fca5a5", # Existing ANN (Light Red)
        "#dc2626", # Existing ANN + 심리지수 결합모델 (Red)
        "#8b5cf6"  # Extended ANN (Purple)
    ]
    
    datasets = []
    for model, color in zip(models, colors):
        model_data = []
        for comp in companies_sorted:
            row = df[(df['Company'] == comp) & (df['Model'] == model)]
            if not row.empty:
                model_data.append(float(row.iloc[0]['rmse']))
            else:
                model_data.append(0.0)
        datasets.append({
            "label": model,
            "data": model_data,
            "backgroundColor": color,
            "borderColor": color,
            "borderWidth": 1
        })
        
    datasets_json = json.dumps(datasets, ensure_ascii=False)
    company_labels_json = json.dumps(company_labels, ensure_ascii=False)

    # 4. Create HTML content with a premium dashboard style and Chart.js
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>K-NSI 주가예측결합모델 최종종합분석보고서</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <!-- Chart.js CDN for interactive visual chart -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #4f46e5;
            --primary-dark: #3730a3;
            --secondary: #a855f7;
            --background: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #2563eb; /* Blue highlight for positive improvements */
            --danger: #ef4444;  /* Red for negative/degradations */
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', 'Malgun Gothic', sans-serif;
            background-color: var(--background);
            color: var(--text-main);
            line-height: 1.6;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 25px;
            border-bottom: 2px solid var(--border);
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }}
        
        .subtitle {{
            font-size: 1.2rem;
            color: var(--text-muted);
            font-weight: 500;
        }}
        
        .card {{
            background-color: var(--card-bg);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
            border: 1px solid var(--border);
            padding: 35px;
            margin-bottom: 35px;
        }}
        
        h2 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            border-left: 6px solid var(--primary);
            padding-left: 15px;
        }}
        
        h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--primary-dark);
            margin-top: 25px;
            margin-bottom: 10px;
        }}
        
        p {{
            margin-bottom: 16px;
            color: #334155;
            font-size: 1rem;
        }}
        
        ul, ol {{
            margin-bottom: 20px;
            padding-left: 25px;
            color: #334155;
            font-size: 0.98rem;
        }}
        
        li {{
            margin-bottom: 10px;
        }}
        
        /* Directory Structure visual styling (Modified to Pre-wrap style) */
        .dir-tree {{
            background-color: #0f172a;
            color: #e2e8f0;
            padding: 25px;
            border-radius: 16px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            margin: 20px 0;
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.08);
            line-height: 1.6;
            white-space: pre-wrap; /* Keeps indentation and breaks lines vertically */
        }}
        .dir-tree .folder {{ color: #38bdf8; font-weight: 600; }}
        .dir-tree .file {{ color: #34d399; }}
        .dir-tree .comment {{ color: #64748b; font-style: italic; }}

        /* Table Styling */
        .table-responsive {{
            width: 100%;
            overflow-x: auto;
            margin: 25px 0;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
            text-align: left;
        }}
        th {{
            background-color: #0f172a;
            color: #ffffff;
            padding: 14px 16px;
            font-weight: 600;
            border-bottom: 2px solid var(--border);
        }}
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            color: #334155;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            text-align: center;
        }}
        .badge-primary {{ background-color: rgba(79, 70, 229, 0.1); color: var(--primary); border: 1px solid rgba(79, 70, 229, 0.2); }}
        .badge-success {{ background-color: rgba(37, 99, 235, 0.1); color: #2563eb; border: 1px solid rgba(37, 99, 235, 0.2); }}
        
        .chart-container {{
            text-align: center;
            margin-top: 25px;
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid var(--border);
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.02);
        }}
        
        /* Notice block for interpretation */
        .alert-box {{
            background-color: #f0f7ff;
            border-left: 5px solid #2563eb;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
        }}
        .alert-box p {{
            margin-bottom: 0;
            color: #1e3a8a;
            font-size: 0.98rem;
        }}
        
        footer {{
            text-align: center;
            margin-top: 60px;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            padding-top: 25px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>K-NSI 주가예측결합모델 최종종합분석보고서</h1>
            <p class="subtitle">투자자 설문 가중치 뉴스 심리지수(K-NSI) 기반 4대 예측군의 2단계 선형보정 성능 종합평가</p>
        </header>

        <!-- 1. ZIP File Map & Directory Structure -->
        <div class="card">
            <h2>1. 압축폴더 내 모델 파일 및 코드 경로 지도</h2>
            <p>바탕화면의 <code>CapstoneDesign-K-NSI-real.zip</code> 압축폴더를 해제했을 때의 세로형 트리 구조와 모델 7번부터 11번까지 우리가 구축한 소스 코드 및 결과 보고서 파일의 경로 매핑입니다.</p>
            
            <pre class="dir-tree"><span class="folder">CapstoneDesign-K-NSI/</span>
├── <span class="folder">configs/</span>
│   └── <span class="file">settings.py</span>                  <span class="comment"># 프로젝트 기본 환경 변수 및 경로 지정</span>
├── <span class="folder">data/</span>
│   ├── <span class="folder">raw/</span>                      <span class="comment"># 8개사의 5개년 주가 OHLCV 원본 데이터 (CSV)</span>
│   └── <span class="folder">sentiment/</span>                <span class="comment"># 90일간의 뉴스 데이터 및 최종 NSI 결합 파일 (final_input.csv)</span>
├── <span class="folder">results/</span>
│   ├── <span class="folder">figures/final_model/</span>       <span class="comment"># 종목별 2단계 시계열 보정 비교 시각화 이미지</span>
│   ├── <span class="folder">metrics/</span>                  <span class="comment"># 2step_psychological_correction_summary.csv (평가지표 종합 CSV)</span>
│   ├── <span class="file">arima_sentiment_report.html</span>  <span class="comment"># Model 9 전용 개별 성능 보고서 (아리마_심리지수결합_성능보고서.html)</span>
│   ├── <span class="file">benchmark_sentiment_report.html</span><span class="comment"># Model 8 전용 개별 성능 보고서 (벤치마크_심리지수결합_성능보고서.html)</span>
│   ├── <span class="file">existing_ann_sentiment_report.html</span><span class="comment"># Model 11 전용 개별 성능 보고서 (existing ann+심리지수결합.html)</span>
│   ├── <span class="file">existing_rf_sentiment_report.html</span> <span class="comment"># Model 10 전용 개별 성능 보고서 (existing rf+심리지수결합.html)</span>
│   └── <span class="file">sentiment_only_report.html</span>    <span class="comment"># Model 7 전용 개별 성능 보고서 (심리단독모델_성능보고서.html)</span>
├── <span class="folder">scripts/</span>
│   ├── <span class="folder">reporting/</span>
│   │   ├── <span class="file">generate_existing_ann_hybrid_report.py</span> <span class="comment"># ANN 결합 HTML 생성기</span>
│   │   ├── <span class="file">generate_existing_rf_hybrid_report.py</span>  <span class="comment"># RF 결합 HTML 생성기</span>
│   │   └── <span class="file">generate_final_summary_report.py</span>       <span class="comment"># 최종 종합 보고서 생성기 (현재 파일)</span>
│   └── <span class="folder">training/</span>
│       └── <span class="file">train_2step_correction.py</span>  <span class="comment"># 모델 8~11번 전체를 학습 및 2단계 선형보정 연산 스크립트</span>
└── <span class="folder">src/</span>                              <span class="comment"># 모델링, 전처리, 평가 관련 공통 모듈 함수 폴더</span>
    └── <span class="file">sentiment_only_model.py</span>       <span class="comment"># Model 7 심리단독모델 코어 모듈</span></pre>

            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>모델 번호</th>
                            <th>예측 모델명 (Model Name)</th>
                            <th>구현 및 학습 코드 경로</th>
                            <th>결과 산출물 경로</th>
                            <th>구현 특징</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><span class="badge badge-primary">Model 7</span></td>
                            <td><strong>심리단독모델 (Sentiment Only)</strong></td>
                            <td><code>src/sentiment_only_model.py</code><br> 및 <code>scripts/reporting/generate_sentiment_report.py</code></td>
                            <td><code>results/sentiment_only_report.html</code></td>
                            <td>과거 가격 데이터를 완전히 제외하고 뉴스 심리지수(K-NSI) 변수만을 학습하여 선형 회귀로 종가를 예측하는 베이스라인 모델</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-success">Model 8</span></td>
                            <td><strong>Benchmark + 심리지수 결합모델</strong></td>
                            <td><code>scripts/training/train_2step_correction.py</code></td>
                            <td><code>results/benchmark_sentiment_report.html</code></td>
                            <td>Benchmark(전일 종가 Naive 모델)의 예측 가격에 K-NSI 지수 변수를 2단계 선형 회귀로 보정한 모델</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-success">Model 9</span></td>
                            <td><strong>ARIMA + 심리지수 결합모델</strong></td>
                            <td><code>scripts/training/train_2step_correction.py</code></td>
                            <td><code>results/arima_sentiment_report.html</code></td>
                            <td>전통 시계열 ARIMA 모델의 다단계 장기 예측 가격에 K-NSI 변수를 선형 회귀로 보정하여 누적 발산을 억제한 결합 모델</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-success">Model 10</span></td>
                            <td><strong>Existing RF + 심리지수 결합모델</strong></td>
                            <td><code>scripts/training/train_2step_correction.py</code></td>
                            <td><code>results/existing_rf_sentiment_report.html</code></td>
                            <td>기존 기술적 피처 7개만 사용한 Random Forest 모델의 가격 예측치에 K-NSI 변수를 선형 결합하여 변곡점을 교정한 모델</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-success">Model 11</span></td>
                            <td><strong>Existing ANN + 심리지수 결합모델</strong></td>
                            <td><code>scripts/training/train_2step_correction.py</code></td>
                            <td><code>results/existing_ann_sentiment_report.html</code></td>
                            <td>기존 기술적 피처 7개만 사용한 인공신경망(ANN) 모델의 가격 예측치에 K-NSI 변수를 선형 결합하여 방향성(Acc)을 보정한 모델</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 2. Side-by-Side Model Comparison -->
        <div class="card">
            <h2>2. 4대 오리지널 베이스라인 vs 심리지수 결합 모델 일대일 성능 비교</h2>
            <p>각 종목별로 베이스라인 모델(Benchmark, ARIMA, Existing RF, Existing ANN)과 각각의 K-NSI 뉴스 심리지수 결합 모델의 실제 가격 오차 개선률(RMSE, MAPE) 및 방향성 정확도(Acc) 변화를 보여주는 통합 비교표입니다.</p>
            
            <div class="alert-box">
                <p>
                    <strong>💡 수치 판독 가이드 (개선율 부호 및 색상 설명)</strong><br>
                    • <strong><span style="color: #2563eb; font-weight:700;">파란색 (+)</span></strong>: 오차(RMSE, MAPE)가 감소하거나 방향성 정확도(Acc)가 상승한 <strong>성능 개선</strong> 상태입니다.<br>
                    • <strong><span style="color: #ef4444; font-weight:700;">빨간색 (-)</span></strong>: 결합 모델의 오차가 늘어났거나 방향성 정확도가 하락한 <strong>성능 저하</strong> 상태입니다.<br>
                    • 베이스라인이 이미 극단적으로 우수한 안정적 자산에서는 심리지수의 매일 변동성이 과보정(Over-correction) 노이즈로 작용하여 오차가 일부 늘어날 수 있습니다. (자세한 원인은 아래 3번 및 5번 항목 참조)
                </p>
            </div>

            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>분석 대상 기업</th>
                            <th>일대일 대조군</th>
                            <th>RMSE 변화 (원)</th>
                            <th>RMSE 개선률 (%)</th>
                            <th>MAPE 변화 (%)</th>
                            <th>MAPE 개선률 (%p)</th>
                            <th>방향성 정확도 (Acc) 변화</th>
                            <th>방향성 개선률 (%p)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {comparison_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 3. Qualitative Analysis and Conclusions -->
        <div class="card">
            <h2>3. 주요 대조군별 성능 해석 및 K-NSI 심리지수의 영향</h2>
            
            <h3>① ARIMA vs ARIMA + 심리결합 (Model 9)</h3>
            <p>가장 극적인 성능 향상을 보인 대조군입니다. ARIMA 모델은 학습 기간 이후 미래로 가면서 재학습을 거치지 않는 다단계 장기 예측(Multi-step ahead Forecast) 시, 오차가 급격하게 누적되면서 가격 예측선이 실제 주가 수준에서 어마어마하게 이탈합니다.</p>
            <p>여기에 매 영업일 뉴스 심리의 흐름을 반영한 K-NSI 변수를 결합하여 2단계 보정을 수행하자, <strong>예측선의 축을 실제 주가 가격대 근처로 잡아당겨 주는 강력한 '안정제(Stabilizer/Anchor)' 역할</strong>을 훌륭히 완수하였습니다. 그 결과 <strong>LIG넥스원은 RMSE가 36.5만 원에서 6.0만 원으로 무려 83.41% 감소</strong>하였고, <strong>한화에어로스페이스 역시 34.0만 원에서 6.9만 원으로 79.51%의 오차가 크게 상쇄</strong>되었습니다.</p>
            
            <h3>② Benchmark vs Benchmark + 심리결합 (Model 8)</h3>
            <p>Benchmark(전일 종가 Naive 모델)는 하루 전날 주가를 그대로 추종하기 때문에 일별 가격 변동폭이 작고 안정적인 주식(삼성전자, SK하이닉스 등)에서는 그 자체로 매우 강력한 베이스라인입니다.</p>
            <p>따라서 이미 오차가 좁혀진 상태에서 K-NSI 변수를 활용해 보정을 주입하면, 뉴스 심리지수가 가진 매일의 변동성 노이즈가 과보정(Over-correction) 요소로 유입되어 RMSE 오차가 증가(삼성전자: 9,037원 &rarr; 13,072원)하는 양상을 보였습니다. 이는 가격 모멘텀이 극도로 정체된 우량 자산에서는 심리지수 보정이 노이즈가 될 수 있음을 의미합니다.</p>
            
            <h3>③ Existing RF vs Existing RF + 심리결합 (Model 10)</h3>
            <p>기존 Random Forest는 과거 기술적 피처 7개만 활용하기 때문에 종가의 장기 흐름을 과도하게 평균화하는 특징이 있습니다. 여기에 심리지수가 결합되면서 일별 등락의 감성 모멘텀 신호가 공급되었고, 가격 수준의 미세 오차(RMSE) 조정보다는 <strong>방향성을 예측하는 능력(Acc)을 비약적으로 보조</strong>했습니다.</p>
            <p>실제로 **SNT다이내믹스의 경우 방향성 정확도가 42.86%에서 71.43%로 28.57%p 폭등**하였으며, **원익IPS 역시 42.86%에서 57.14%로 14.28%p 향상**되는 뚜렷한 변곡점 가이드 효과를 보였습니다.</p>
            
            <h3>④ Existing ANN vs Existing ANN + 심리결합 (Model 11)</h3>
            <p>인공신경망(ANN) 예측치 역시 비선형 공간을 통계적으로 근사하므로 주가 예측 범위가 매우 가변적입니다. 이를 2단계 다중 선형 회귀(Linear Regression)를 사용해 교정하려고 시도할 때, 신경망 출력값 특유의 잔차 왜곡이 선형 가중치 산정을 어긋나게 만들면서 대부분 종목에서 RMSE가 늘어나는 성능 저하를 초래했습니다.</p>
            <p>그럼에도 불구하고 방향성 흐름 판별(Acc) 지표만큼은 **원익IPS에서 기존 21.43%에서 결합 후 57.14%로 35.71%p 대폭 상향**되는 등 심리 변수가 일일 매수/매도 세력의 실질 변곡점 신호 역할을 적극 수행했음을 시사합니다.</p>
        </div>

        <!-- 4. Summary Chart Visual Presentation (Replaced with Dynamic Chart.js) -->
        <div class="card">
            <h2>4. 모델별 RMSE 비교 시각화 (8개사 종합 인터랙티브 차트)</h2>
            <p>훈련 및 평가 프로세스 전체를 완료한 후 최종 산출된 8개 기업 전체 모델별 RMSE(예측 오차) 종합 분석 차트입니다.</p>
            
            <div class="chart-container" style="position: relative; height:500px; width:100%;">
                <canvas id="rmseChart"></canvas>
            </div>
            
            <div style="text-align: center; margin-top: 15px; font-size: 0.9rem; color: var(--text-muted); font-weight: 500;">
                💡 <strong>팁</strong>: 범례의 특정 모델명을 클릭하면 해당 막대를 화면에서 켜거나 꺼서 겹쳐진 수치를 한눈에 조절할 수 있습니다.<br>
                마우스를 개별 막대 그래프에 가져가면 해당 모델명과 정확한 RMSE 수치(원)가 말풍선(Tooltip)으로 표시됩니다.
            </div>
        </div>

        <!-- 5. Final Conclusions and Limitations -->
        <div class="card">
            <h2>5. 종합 결론 및 향후 보완점 (연구의 의의와 한계)</h2>
            
            <h3>① 본 연구의 주요 의의 및 시사점</h3>
            <ul>
                <li><strong>비정형 금융 감성 데이터의 보완력 실증</strong>: 단순 주가 시계열 가격 데이터만으로는 규명하기 어려운 시장 참여자들의 심리적 동요(K-NSI 뉴스 심리지수)를 모델링의 통제 변수로 도입함으로써, 뉴스 미디어가 주식 시장 가격 형성에 주는 단기 임팩트를 수학적으로 연계 및 증명하였습니다.</li>
                <li><strong>시계열 오차 발산 통제 (Stabilizer로서의 가치)</strong>: 학습되지 않은 미래 시계열 장기 예측에서 속절없이 발산하는 ARIMA 계열 모델의 한계를 매일 NSI 지수를 통한 선형 보정으로 통제하여 성능을 80% 가깝게 깎아내는 <strong>강력한 시계열 결합 안정제(Anchor)</strong>로 활용 가능합니다.</li>
                <li><strong>의사결정 변곡점의 조타수</strong>: 등락 예측 모멘텀 방향(Acc)을 기계학습/AI 모델에 피드백해 주어 <strong>SNT다이내믹스(RF 결합) +28.57%p, 원익IPS(ANN 결합) +35.71%p</strong> 등 실제 방향성 트레이딩 배팅 시 훌륭한 필터 신호로 사용될 실무적 가치를 입증했습니다.</li>
            </ul>
            
            <h3>② 본 연구의 한계점 및 향후 보완점</h3>
            <ul>
                <li><strong>단순 선형 회귀(Linear Regression) 보정의 수학적 한계</strong>: 비선형적 패턴을 이미 강하게 모사하고 있는 Existing ANN(Model 11) 등의 딥러닝 출력값을 2단계 선형 방정식으로 보정하면서 가격 스케일 왜곡과 잔차 오차가 겹쳐 증폭되었습니다. <strong>이를 보완하기 위해서는 차기 모델 설계 시 2단계 결합(보정) 학습 모형 자체를 선형 회귀가 아닌 XGBoost, LightGBM 또는 소형 MLP(다층 신경망)와 같은 비선형 메타 예측기(Non-linear Meta-learner)로 적용해야 합니다.</strong></li>
                <li><strong>종목별 변동성 기반 가중 보정 설계 (과보정 해결)</strong>: 삼성전자, SK하이닉스 벤치마크처럼 예측 편차가 극소화된 우량주에서는 요동치는 매일의 뉴스 심리가 불필요한 노이즈로 작용하여 오차를 증가시켰습니다. 따라서 종목 고유의 베타(Beta) 또는 역사적 변동성 크기에 비례하여 K-NSI의 보정률(Correction Factor) 강도를 동적으로 감쇠하는 적응형 보정(Adaptive Correction) 아키텍처 연구가 보완되어야 합니다.</li>
                <li><strong>감성 지수의 시차 피처(Lagged Features) 미반영</strong>: 매일의 감성 변곡점이 시장 가격에 반영되는 시차(Lag)의 뉘앙스를 정밀화하기 위해, 당일 지수 외에 1~3 영업일 전의 뉴스 심리 지연 변수를 다차원으로 활용하는 추가 분석이 보완될 필요가 있습니다.</li>
            </ul>
        </div>

        <footer>
            <p>© 2026 Capstone Design - K-NSI Project Group. All rights reserved.</p>
        </footer>
    </div>

    <!-- ChartJS Initialization Script -->
    <script>
        const ctx = document.getElementById('rmseChart').getContext('2d');
        const chartData = {{
            labels: {company_labels_json},
            datasets: {datasets_json}
        }};
        
        new Chart(ctx, {{
            type: 'bar',
            data: chartData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{
                            font: {{
                                family: "'Inter', sans-serif",
                                size: 11,
                                weight: '500'
                            }},
                            boxWidth: 10,
                            usePointStyle: true
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleFont: {{
                            family: "'Inter', sans-serif",
                            size: 13,
                            weight: 'bold'
                        }},
                        bodyFont: {{
                            family: "'Inter', sans-serif",
                            size: 12
                        }},
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                if (context.parsed.y !== null) {{
                                    label += Math.round(context.parsed.y).toLocaleString() + ' 원';
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{
                            display: false
                        }},
                        ticks: {{
                            font: {{
                                family: "'Inter', sans-serif",
                                weight: '600'
                            }}
                        }}
                    }},
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'RMSE 오차 (원)',
                            font: {{
                                family: "'Inter', sans-serif",
                                weight: '600'
                            }}
                        }},
                        ticks: {{
                            font: {{
                                family: "'Inter', sans-serif"
                            }},
                            callback: function(value) {{
                                return (value / 1000) + 'k';
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save to project
    with open(html_project_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved project report: {html_project_path}")
    
    # Save to desktop
    with open(html_desktop_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved desktop report: {html_desktop_path}")

if __name__ == "__main__":
    generate_report()
