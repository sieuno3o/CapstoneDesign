# -*- coding: utf-8 -*-
import os
import json
import pandas as pd

def generate_report():
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(project_dir, "results", "metrics", "direct_hybrid_summary.csv")
    
    html_project_path = os.path.join(project_dir, "results", "direct_hybrid_14_15_report.html")
    
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

    COMPANY_ORDER = [
        "samsung_electronics", "sk_hynix", "wonik_ips", "dongjin_semichem",
        "hanwha_aerospace", "lig_nex1", "snt_dynamics", "firstec"
    ]
    
    # Generate comparison tables data dynamically
    # Use "vs 심리결합" as requested by the user
    comparison_rows = ""
    for comp in COMPANY_ORDER:
        if comp not in df['Company'].values:
            continue
            
        comp_display = companies_kr.get(comp, comp)
        df_comp = df[df['Company'] == comp]
        
        comparisons = [
            ("Extended RF vs Direct Hybrid RF", "Extended RF", "Direct Hybrid RF"),
            ("Extended ANN vs Direct Hybrid ANN", "Extended ANN", "Direct Hybrid ANN")
        ]
        
        comp_cell = f'<td rowspan="2" style="font-weight: 700; background: #f8fafc; border-right: 1px solid #e2e8f0; text-align: center;">{comp_display}</td>'
        
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
    companies_sorted = [c for c in COMPANY_ORDER if c in df['Company'].values]
    company_labels = [companies_kr.get(c, c) for c in companies_sorted]
    
    models = [
        "Extended RF", "Direct Hybrid RF",
        "Extended ANN", "Direct Hybrid ANN"
    ]
    
    colors = [
        "#86efac", # Extended RF (Light Green)
        "#16a34a", # Direct Hybrid RF (Green)
        "#d8b4fe", # Extended ANN (Light Purple)
        "#9333ea"  # Direct Hybrid ANN (Purple)
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
    <title>K-NSI 주가예측결합모델 14,15번 (Direct Hybrid) 분석보고서</title>
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
            <h1>K-NSI 주가예측결합모델 14,15번 (Direct Hybrid) 분석보고서</h1>
            <p class="subtitle">결정변수와 심리변수 12개를 직접 통합한 AI 단일 파이프라인 성능 종합평가</p>
        </header>

        <!-- 1. ZIP File Map & Directory Structure -->
        <div class="card">
            <h2>1. 압축폴더 내 모델 14, 15번 파일 및 코드 경로 지도</h2>
            <p>바탕화면의 <code>CapstoneDesign-K-NSI-real.zip</code> 압축폴더를 해제했을 때의 트리 구조와 모델 14, 15번에 해당하는 소스 코드 및 결과 보고서 파일의 경로 매핑입니다.</p>
            
            <pre class="dir-tree"><span class="folder">CapstoneDesign-K-NSI/</span>
├── <span class="folder">results/</span>
│   ├── <span class="folder">figures/direct_hybrid/</span>     <span class="comment"># 종목별 Direct Hybrid 비교 시각화 이미지</span>
│   ├── <span class="folder">metrics/</span>                  <span class="comment"># direct_hybrid_summary.csv (평가지표 종합 CSV)</span>
│   └── <span class="file">direct_hybrid_14_15_report.html</span><span class="comment"># Model 14, 15 전용 개별 성능 보고서 (현재 파일)</span>
├── <span class="folder">scripts/</span>
│   ├── <span class="folder">reporting/</span>
│   │   └── <span class="file">generate_direct_hybrid_14_15_report.py</span> <span class="comment"># 현재 HTML 생성기</span>
│   └── <span class="folder">training/</span>
│       └── <span class="file">train_direct_hybrid.py</span>   <span class="comment"># 결정변수 10개 + 심리변수 2개 통합 학습 모델 스크립트</span>
└── <span class="folder">src/</span>
    └── <span class="file">ai_model.py</span>                  <span class="comment"># 랜덤포레스트 및 ANN 코어 모듈</span></pre>

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
                            <td><span class="badge badge-success">Model 14</span></td>
                            <td><strong>Direct Hybrid RF (12 Features)</strong></td>
                            <td><code>scripts/training/train_direct_hybrid.py</code></td>
                            <td>(본 보고서 참조)</td>
                            <td>결정변수 10개(Extended)와 심리변수 2개(K-NSI_Short, Long)를 하나의 입력 벡터로 묶어 Random Forest 모델이 한 번에 학습하도록 구현된 1-Step 결합 파이프라인.</td>
                        </tr>
                        <tr>
                            <td><span class="badge badge-success">Model 15</span></td>
                            <td><strong>Direct Hybrid ANN (12 Features)</strong></td>
                            <td><code>scripts/training/train_direct_hybrid.py</code></td>
                            <td>(본 보고서 참조)</td>
                            <td>결정변수 10개와 심리변수 2개를 인공신경망 레이어의 입력 노드 12개로 동시에 통과시켜 수익률(target_next_return)을 예측하고 종가로 역변환하는 1-Step 결합 파이프라인.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 2. Side-by-Side Model Comparison -->
        <div class="card">
            <h2>2. Extended 베이스라인 vs Direct Hybrid 모델 일대일 성능 비교</h2>
            <p>보조지표 10개만 사용한 Extended 모델(5번, 6번)과, 심리변수 2개가 다이렉트로 결합된 Direct Hybrid 모델(14번, 15번)의 성과 개선표입니다.</p>
            
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
            <h2>3. Direct Hybrid 모델(1-Step)의 성과 해석 및 시사점</h2>
            
            <h3>① 일체형 변수 공간의 한계 극복 필요성</h3>
            <p>기술적 보조지표 10개와 K-NSI 심리변수 2개를 처음부터 하나의 Feature Vector 공간(X)에 밀어넣고 학습시키는 <strong>Direct Hybrid (1-Step) 방식</strong>은 변수 간의 비선형적 상호작용을 AI가 한 번에 찾아주기를 기대하는 모델링이었습니다.</p>
            <p>하지만 실험 결과, 기존의 가격 및 기술 지표들이 스케일과 분산 측면에서 압도적인 주도권을 쥐고 있어 <strong>감성지수(NSI)의 미세한 변곡점 신호가 트리에 온전히 전달되지 못하거나 노이즈와 희석되는 현상</strong>이 관측되었습니다. 이는 Random Forest가 노드를 분할할 때 정보 획득량(Information Gain)이 월등한 가격 파생변수들을 먼저 선택해버리기 때문입니다.</p>
            
            <h3>② 결합 방식(Topology) 구조 설계의 중요성 입증</h3>
            <p>이 결과는 단순히 변수를 많이 때려 넣는다고 해서 성능이 오르지 않는다는 점을 증명합니다. <strong>K-NSI 심리지수의 가치를 제대로 뽑아내기 위해서는 2-Step Correction (모델 8~13번) 방식처럼, AI가 가격의 큰 틀을 먼저 잡게 한 뒤 마지막에 NSI 지수가 '정밀 교정 타겟팅(Calibration)'을 수행하도록 위계를 분리해 주는 하이브리드 토폴로지 설계가 필수적</strong>임을 의미합니다.</p>
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
            <h2>5. 14,15번 Direct 결합 모델 요약 및 향후 연구 방향</h2>
            
            <ul>
                <li><strong>성능 교착의 진단:</strong> 예측을 한 번에 수행하려는 Direct Hybrid 방식은 비정형 감성 데이터의 잠재력을 완전히 끌어내기 부족합니다.</li>
                <li><strong>향후 보완점:</strong> 1-Step 방식에서도 NSI의 정보 비중을 잃지 않으려면, Attention Mechanism(어텐션 메커니즘) 기반의 딥러닝 아키텍처를 도입하여 시간의 흐름에 따라 뉴스 심리지수에 가중치를 부여하는 연구가 요구됩니다.</li>
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

if __name__ == "__main__":
    generate_report()
