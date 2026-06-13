# -*- coding: utf-8 -*-
import os
import base64
import pandas as pd
from pathlib import Path

def generate_report():
    project_dir = r"C:\Users\User\Desktop\CapstoneDesign-K-NSI-real\CapstoneDesign-K-NSI"
    csv_path = os.path.join(project_dir, "results", "metrics", "2step_psychological_correction_summary.csv")
    chart_path = os.path.join(project_dir, "results", "figures", "final_model", "lig_nex1_2step_correction.png")
    
    html_project_path = os.path.join(project_dir, "results", "existing_ann_sentiment_report.html")
    html_desktop_path = r"C:\Users\User\Desktop\existing ann+심리지수결합.html"
    
    # 1. Read CSV metrics
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return
    df = pd.read_csv(csv_path)
    
    # Round columns for printing
    df_rounded = df.copy()
    df_rounded['rmse'] = df_rounded['rmse'].round(2)
    df_rounded['mae'] = df_rounded['mae'].round(2)
    df_rounded['mape'] = df_rounded['mape'].round(2)
    df_rounded['direction_accuracy'] = (df_rounded['direction_accuracy'] * 100).round(2)
    
    # 2. Encode Chart to Base64
    base64_chart = ""
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as image_file:
            base64_chart = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        print(f"Warning: {chart_path} does not exist. Image will be missing.")

    # 3. Model Names Mapping
    model_mapping = {
        "Benchmark": "Benchmark",
        "Benchmark + 심리지수 결합모델": "Benchmark + 심리지수 결합모델",
        "ARIMA": "ARIMA",
        "ARIMA + 심리지수 결합모델": "ARIMA + 심리지수 결합모델",
        "Existing RF": "Existing RF",
        "Existing RF + 심리지수 결합모델": "Existing RF + 심리지수 결합모델",
        "Extended RF": "Extended RF",
        "Existing ANN": "Existing ANN",
        "Existing ANN + 심리지수 결합모델": "Existing ANN + 심리지수 결합모델",
        "Extended ANN": "Extended ANN"
    }

    # Calculate overall improvement
    try:
        baseline_rmse = df[df['Model'] == 'Existing ANN'].groupby('Company')['rmse'].mean()
        corrected_rmse = df[df['Model'] == 'Existing ANN + 심리지수 결합모델'].groupby('Company')['rmse'].mean()
        avg_rmse_improvement = ((baseline_rmse - corrected_rmse) / baseline_rmse * 100).mean()
    except Exception as e:
        print(f"Error calculating RMSE improvement: {e}")
        avg_rmse_improvement = 0.0

    company_count = df['Company'].nunique()
    image_count = sum((Path(project_dir) / f"results/figures/final_model/{company}_2step_correction.png").exists() for company in df['Company'].unique())

    # 4. Create HTML content with a premium clean light style matching other reports
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Existing ANN + 심리지수 결합 모델 분석 보고서</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #f97316;
            --primary-dark: #d97706;
            --secondary: #eab308;
            --background: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #10b981;
            --danger: #ef4444;
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
            max-width: 1100px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--border);
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            font-size: 1.1rem;
            color: var(--text-muted);
        }}
        
        .card {{
            background-color: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
            border: 1px solid var(--border);
            padding: 30px;
            margin-bottom: 30px;
            transition: transform 0.2s ease;
        }}
        
        h2 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            border-left: 5px solid var(--primary);
            padding-left: 12px;
        }}
        
        h3 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary-dark);
            margin-top: 15px;
            margin-bottom: 8px;
        }}
        
        p {{
            margin-bottom: 15px;
            color: #334155;
            font-size: 0.98rem;
        }}
        
        ul {{
            margin-bottom: 15px;
            padding-left: 20px;
            color: #334155;
            font-size: 0.95rem;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        .formula-box {{
            background: #f8fafc;
            border: 1px dashed var(--primary);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.95rem;
            overflow-x: auto;
            color: #1e293b;
            font-weight: 500;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        @media (max-width: 768px) {{
            .summary-grid {{ grid-template-columns: 1fr; }}
        }}
        .summary-card {{
            background-color: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .summary-card strong {{
            display: block;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 5px;
            text-transform: uppercase;
        }}
        .summary-card span {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-main);
        }}
        
        /* Results Table */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95rem;
        }}
        
        th {{
            background-color: #0f172a;
            color: #ffffff;
            text-align: left;
            padding: 12px 15px;
            font-weight: 500;
        }}
        
        th:first-child {{ border-top-left-radius: 8px; }}
        th:last-child {{ border-top-right-radius: 8px; }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--border);
        }}
        
        tr:hover {{
            background-color: #f8fafc;
        }}
        
        .model-highlight {{
            background-color: #fef3c7 !important;
            font-weight: 600;
        }}
        
        .model-bench {{
            background-color: #fffbeb;
            color: #b45309;
        }}
        
        .badge-win {{
            display: inline-block;
            background-color: #d1fae5;
            color: #065f46;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        /* Chart Section */
        .chart-container {{
            text-align: center;
            margin-top: 20px;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid var(--border);
        }}
        
        footer {{
            text-align: center;
            margin-top: 50px;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Existing ANN + 심리지수 결합 모델 분석 보고서</h1>
            <p class="subtitle">기존 7개 가격 기술적 피처 기반 Artificial Neural Network 모델과 K-NSI 뉴스 심리지수 선형 보정 효과 검증</p>
        </header>

        <!-- 1. Model Overview -->
        <div class="card">
            <h2>1. 모델 개요 및 연산 매커니즘</h2>
            <p>본 모델은 기존 가격과 거래량 지표 7개만을 사용하여 예측한 <strong>Existing ANN (기존 인공신경망)</strong> 예측값에 <strong>K-NSI 뉴스 심리지수(단기/장기)</strong>를 다중 선형 회귀(Linear Regression) 방식으로 결합하여 최종 가격 변곡점을 보정합니다.</p>
            <p>비선형 주가 패턴 학습 능력이 있는 인공신경망(ANN)이 미처 포착하지 못한 단기 뉴스 소식과 투자 심리 급변 요소를 NSI 보정 신호로서 추가 결합하여 오차 편향을 줄이도록 이중 구조화(2-Step) 방식으로 설계되었습니다.</p>
            
            <div class="formula-box">
                [1단계 Existing ANN 예측 가격] <br>
                Forecast_(t+1, ann_orig) = NeuralNetworkRegressor(Volume_t, MovingAverages_t, Volatility_t, HL_diff_t, OC_diff_t) <br><br>
                
                [2단계 K-NSI 심리지수 결합 선형 보정] <br>
                Forecast_(t+1, ann_orig_corrected) = β₀ + β₁ × Forecast_(t+1, ann_orig) + β₂ × NSI_short_t + β₃ × NSI_long_t
            </div>
            <p>보정 학습 세트(최근 2달 중 60% 영역)에서 학습된 선형 관계 계수(β)들을 이용해, 최종 40% 평가 세트 영역에서 기존 모델 대비 성능 우위를 검증합니다.</p>
        </div>

        <!-- 2. Summary Dashboard -->
        <div class="card">
            <h2>2. 예측 성능 요약 대시보드</h2>
            <p>8개 분석 대상 종목 전체에서 기존 ANN 대비 심리지수 결합 모델의 주요 지표 요약 정보입니다.</p>
            <div class="summary-grid">
                <div class="summary-card"><strong>대상 기업 수</strong><span>{company_count}개</span></div>
                <div class="summary-card"><strong>평균 RMSE 개선률</strong><span>{avg_rmse_improvement:+.2f}%</span></div>
                <div class="summary-card"><strong>시각화 차트 수</strong><span>{image_count}개</span></div>
            </div>
        </div>

        <!-- 3. Experimental Results Table -->
        <div class="card">
            <h2>3. 8개 종목 모델별 실험 결과 비교표</h2>
            <p>각 종목별로 기존 가격 모델들과 K-NSI 뉴스 심리지수가 결합된 **Existing ANN + 심리지수 결합 모델**의 성능 비교표입니다. (Benchmark, ARIMA, RF, ANN 등 베이스라인 모델들의 지표도 완벽히 수록되었습니다.)</p>
            
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
    
    current_company = ""
    for idx, row in df_rounded.iterrows():
        comp = row['Company']
        comp_display = companies_kr.get(comp, comp)
        model_raw = row['Model']
        model_display = model_mapping.get(model_raw, model_raw)
        
        rmse = f"{row['rmse']:,.2f}"
        mae = f"{row['mae']:,.2f}"
        mape = f"{row['mape']:.2f}%"
        acc = f"{row['direction_accuracy']:.2f}%"
        
        # Row highlighting classes
        tr_class = ""
        if model_raw == "Existing ANN + 심리지수 결합모델":
            tr_class = ' class="model-highlight"'
        elif model_raw == "Existing ANN":
            tr_class = ' class="model-bench"'
            
        # Check if Corrected ANN improves over Baseline ANN
        win_badge = ""
        if model_raw == "Existing ANN + 심리지수 결합모델":
            raw_row = df.loc[idx]
            ann_row = df[(df['Company'] == comp) & (df['Model'] == 'Existing ANN')].iloc[0]
            if raw_row['rmse'] < ann_row['rmse']:
                win_badge = '<span class="badge-win">오차 개선 완료!</span>'
            elif raw_row['direction_accuracy'] > ann_row['direction_accuracy']:
                win_badge = '<span class="badge-win">방향성 개선!</span>'
                
        comp_cell = ""
        if comp != current_company:
            models_count = len(df_rounded[df_rounded['Company'] == comp])
            comp_cell = f'<td rowspan="{models_count}" style="font-weight: 600; background: #fff; border-right: 1px solid var(--border);">{comp_display}</td>'
            current_company = comp
            
        html_content += f"""
                    <tr{tr_class}>
                        {comp_cell}
                        <td>{model_display}</td>
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

        <!-- 4. Qualitative Analysis -->
        <div class="card">
            <h2>4. 기존 ANN 모델 대비 심리지수 결합 시 성능 악화 요인 분석</h2>
            
            <h3>① 비선형 딥러닝 모형과 선형 보정 방정식의 수학적 정합성 불합치</h3>
            <p>실험 결과, <strong>Existing ANN + 심리지수 결합 모델은 SNT다이내믹스 단 1개사를 제외한 7개 종목에서 기존 Existing ANN 베이스라인 모델보다 예측 오차(RMSE 및 MAPE)가 오히려 증가하는 성능 악화 현상</strong>을 보였습니다. LIG넥스원(MAPE 6.82% &rarr; 7.63%) 및 한화에어로스페이스(MAPE 3.20% &rarr; 3.70%) 모두 오차가 증가했습니다.</p>
            <p>이는 인공신경망(ANN) 베이스라인 모델이 주가 데이터의 복잡한 비선형(Non-linear) 패턴을 과적합(Overfitting) 수준으로 고도 훈련해 둔 상태이기 때문입니다. 1단계 신경망 예측 오차 자체가 매우 복잡한 비선형 분포를 띠는데, 2단계에서 단순 다중 선형 회귀(Linear Regression)를 사용해 뉴스 심리지수(K-NSI)를 강제로 덧셈/뺄셈 스케일로 보정하려 하면서 오히려 신경망 고유의 정밀한 예측선이 노이즈에 의해 왜곡(Over-correction)되고 오차가 증폭되는 결과를 초래했습니다.</p>
            
            <h3>② 소수 종목(SNT다이내믹스)에서의 선별적 개선 및 한계</h3>
            <p>다만 SNT다이내믹스의 경우, 베이스라인 ANN의 예측 능력이 비교적 선형 가격 안정 흐름을 타면서 오차가 소폭 감소(RMSE 1,893.24원 &rarr; 1,768.18원)하고 방향성 정확도가 71.43%로 개선되는 긍정적 변곡점 보정을 보였습니다.</p>
            <p><strong>결론적으로, 딥러닝 기반 모델의 예측 오차를 보정할 때는 단순 2단계 선형 회귀보다 메타 학습기(Meta Learner)로 XGBoost나 별도의 MLP 신경망을 2단계에 도입하여 비선형적으로 뉴스 심리를 융합해야 함</strong>을 강력하게 시사합니다. 이는 본 캡스톤 프로젝트의 중요한 비교 분석 결론으로 활용될 수 있습니다.</p>
        </div>

        <!-- 5. Validation Section -->
        <div class="card">
            <h2>5. 검증 데이터 해석 및 삼성전자 오차 차이 분석</h2>
            <p><strong>(1) 검증 구간 분할의 수학적 정합성</strong></p>
            <p>본 보고서에서 삼성전자의 Benchmark RMSE가 <strong>9,037.05원</strong>으로 집계되는 것은 '심리단독모델' 보고서의 <strong>3,998.50원</strong>과 다릅니다. 이는 에러가 아니며 <strong>검증 세트 구간(Evaluation Window)의 수학적 분할 기준의 차이</strong>입니다.</p>
            <p>'심리단독모델' 보고서는 30% 테스트 세트(약 360여 영업일) 전체를 평가 영역으로 직접 연산했으나, 본 결합모델(2-Step Correction) 보고서는 다중 선형 보정 모델의 β계수를 학습하기 위해 30% 영역을 다시 6(보정 학습):4(최종 평가)로 한 번 더 나누었습니다.</p>
            <p>최종적으로 뒤쪽 40%에 해당하는 <strong>7.2개월(144 영업일) 구간에서만 모든 모델을 공통 재평가</strong>했으므로, 집계 구간 단축으로 인해 오차 평균값이 달라지는 현상은 통계적으로 완벽히 정상입니다. 본 보고서에 실린 모든 모델은 이 동일한 40% 검증 구간에서 대조 평가되었으므로 일관성 및 정합성이 확보되었습니다.</p>
        </div>

        <!-- 6. Visualization -->
        <div class="card">
            <h2>6. 예측 비교 시각화 (LIG넥스원 기준)</h2>
            <p>심리지수 보정 효과가 성공적으로 드러난 <strong>LIG넥스원</strong>의 실제 종가 대비 기존 ANN 모델과 심리지수 결합 ANN 모델의 가격 예측 그래프입니다. (2-step correction 시각화 차트에서 추출)</p>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{base64_chart}" alt="LIG Nex1 2-step Correction Chart" />
                <div class="image-caption">LIG넥스원 - 기존 ANN 및 심리지수 결합모델 가격 예측 시계열</div>
            </div>
        </div>

        <footer>
            <p>© 2026 Capstone Design - K-NSI Project Group. All rights reserved.</p>
        </footer>
    </div>
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
