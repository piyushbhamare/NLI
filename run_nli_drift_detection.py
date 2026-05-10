#!/usr/bin/env python3
"""
NLI DRIFT DETECTION - Analyze nli_sosd_results.csv with CombinedDriftDetector
Saves: nli_drift_analysis.csv + drift_plots.png
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory for drift_detection.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drift_detection import CombinedDriftDetector  # [file:1158]

print("=" * 80)
print("NLI DRIFT DETECTION ANALYSIS (Phase 5)")
print("=" * 80)

# Load NLI baseline results
df = pd.read_csv('nli_sosd_results.csv')
print("✅ Loaded NLI results:", df.shape)
print(df[['Dataset', 'Keys', 'NLI ns', 'NLI Speedup']].round(2))

# Initialize drift detector
detector = CombinedDriftDetector()
print("\n✅ Drift detector initialized (EWMA+PSI+Autoencoder)")

# Baseline: Use Books 1M as stable reference
baseline_speedups = df[(df['Dataset'] == 'Books') & (df['Keys'] == 1000000)]['NLI Speedup'].values
if len(baseline_speedups) == 0:
    baseline_speedups = df['NLI Speedup'].iloc[:3].values  # Fallback
print(f"📊 Baseline (Books 1M): {baseline_speedups[0]:.2f}x speedup")

# Test scenarios
tests = {
    "Stable (Books)": df[df['Dataset'] == 'Books']['NLI Speedup'].values,
    "Stable (Facebook)": df[df['Dataset'] == 'Facebook']['NLI Speedup'].values,
    "Mixed (All Datasets)": df['NLI Speedup'].values,
    "Gradual Drift (Simulated)": df['NLI Speedup'].values * 0.75,  # 25% degradation
    "Sudden Shift (Simulated)": np.random.uniform(1.0, 1.8, len(df))  # Random shift
}

results = []
drift_events = []

print("\n🔬 RUNNING DRIFT DETECTION TESTS")
print("-" * 50)

for test_name, test_data in tests.items():
    detector.reset()  # Reset for each test
    detector.fit_baseline(test_data[:3])  # Fit on first 3 points
    
    # Simulate sequential detection
    scores = []
    for i in range(len(test_data)):
        errors = np.random.normal(50, 10, 1000)  # Simulated prediction errors
        is_drift, score, methods = detector.detect_drift(test_data[:i+1], errors)
        scores.append(score)
        
        if is_drift:
            detector.trigger_repair("REFIT")
            drift_events.append(f"{test_name}: Drift at step {i+1}, score={score:.3f}")
    
    avg_score = np.mean(scores)
    final_score = scores[-1]
    is_final_drift = final_score > 0.4
    
    results.append({
        'Test': test_name,
        'Final_Score': final_score,
        'Avg_Score': avg_score,
        'Drift_Detected': is_final_drift,
        'Max_Score': max(scores),
        'Repair_Events': len([s for s in scores if s > 0.4])
    })
    
    status = "✅ STABLE" if not is_final_drift else "🚨 DRIFT"
    print(f"{test_name:20} | Score: {final_score:.3f} | {status} | Repairs: {len([s for s in scores if s > 0.4])}")

# Save results
drift_df = pd.DataFrame(results)
drift_df.to_csv('nli_drift_analysis.csv', index=False)
print(f"\n✅ Results saved: nli_drift_analysis.csv")

# Plot drift scores
plt.figure(figsize=(12, 6))
for i, (test_name, test_data) in enumerate(tests.items()):
    detector.reset()
    detector.fit_baseline(test_data[:3])
    scores = []
    for data in np.array_split(test_data, max(1, len(test_data)//10)):
        errors = np.random.normal(50, 10, 1000)
        _, score, _ = detector.detect_drift(data, errors)
        scores.append(score)
    
    plt.plot(scores, label=test_name, marker='o', linewidth=2)

plt.axhline(y=0.4, color='r', linestyle='--', label='Drift Threshold')
plt.xlabel('Detection Steps')
plt.ylabel('Drift Score (0-1)')
plt.title('NLI Drift Detection: Real + Simulated Scenarios')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('nli_drift_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📈 Plot saved: nli_drift_plots.png")
print("\n🎉 NLI DRIFT ANALYSIS COMPLETE!")
print(f"📄 Files created:")
print(f"   → nli_drift_analysis.csv")
print(f"   → nli_drift_plots.png")
print("\nNext: python generate_research_paper_plots.py --include-drift")
