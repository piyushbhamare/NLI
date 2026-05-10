"""
Neural Learned Index (NLI) - Publication Plots
NLI 🥇 vs ALEX vs PGM vs RMI vs B-Tree
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12

# Load NLI results
df = pd.read_csv('nli_sosd_results.csv')
print("=" * 80)
print("GENERATING NLI PUBLICATION PLOTS (5-Model Comparison)")
print("=" * 80)

# Figure 1: NLI Speedup (vs B-Tree)
plt.figure(figsize=(10, 6))
for dataset in df['Dataset'].unique():
    subset = df[df['Dataset'] == dataset]
    plt.plot(subset['Keys'], subset['NLI Speedup'], marker='o', linewidth=3, 
             label=dataset, markersize=8)
plt.xscale('log')
plt.xlabel('Dataset Size (keys)', fontsize=14)
plt.ylabel('NLI Speedup vs B-Tree', fontsize=14)
plt.title('Neural Learned Index Speedup (5.98x avg)', fontsize=16, fontweight='bold')
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='B-Tree baseline')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_nli_speedup.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ fig1_nli_speedup.png")

# Figure 2: 5-Model Latency Bar Chart (1M keys)
plt.figure(figsize=(12, 7))
df_1m = df[df['Keys'] == 1000000]
models = ['B-Tree ns', 'RMI ns', 'PGM ns', 'ALEX ns', 'NLI ns']
x = np.arange(len(df_1m))
width = 0.18

for i, model in enumerate(models):
    plt.bar(x + i*width - 2*width, df_1m[model], width, 
            label=model.replace(' ns', '').replace('B-Tree', 'B-Tree'), alpha=0.85)
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Read Latency (ns)', fontsize=14)
plt.title('NLI vs State-of-the-Art (1M keys)', fontsize=16, fontweight='bold')
plt.xticks(x - 2*width, df_1m['Dataset'], fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig2_nli_latency.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ fig2_nli_latency.png")

# Figure 3: 5-Model Heatmap
plt.figure(figsize=(10, 6))
model_cols = ['B-Tree ns', 'RMI ns', 'PGM ns', 'ALEX ns', 'NLI ns']
pivot_data = df.pivot(index='Dataset', columns='Keys', values='NLI ns')
plt.imshow(pivot_data.values, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
plt.colorbar(label='NLI Read Latency (ns)')
plt.xticks(ticks=range(len(pivot_data.columns)), labels=[f'{int(c/1e6)}M' for c in pivot_data.columns])
plt.yticks(ticks=range(len(pivot_data.index)), labels=pivot_data.index)
plt.title('Neural Learned Index Latency Heatmap', fontsize=16, fontweight='bold')
plt.xlabel('Dataset Size')
plt.ylabel('Dataset')
plt.tight_layout()
plt.savefig('fig3_nli_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ fig3_nli_heatmap.png")

print("\n" + "=" * 80)
print("NLI PUBLICATION PLOTS COMPLETE 🥇")
print("=" * 80)
