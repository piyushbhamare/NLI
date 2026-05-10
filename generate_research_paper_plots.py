"""
Publication-Quality Research Paper Plots Generator
Generates all figures needed for a learned index research paper
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Create output directory
os.makedirs('paper_plots', exist_ok=True)

# Load results
df = pd.read_csv('nli_sosd_results.csv')

print("=" * 80)
print("GENERATING PUBLICATION-QUALITY PLOTS FOR RESEARCH PAPER")
print("=" * 80)

# Define color palette (colorblind-friendly)
colors = {
    'B-Tree': '#E69F00',  # Orange
    'ALEX': '#56B4E9',    # Sky Blue
    'PGM': '#009E73',     # Green
    'RMI': '#F0E442',     # Yellow
    'NLI': '#D55E00'      # Vermillion (Red-Orange)
}

# ============================================================================
# FIGURE 1: Query Latency Comparison (Main Result)
# ============================================================================
print("\n[1/6] Generating Figure 1: Query Latency Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
datasets = df['Dataset'].unique()

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    data = df[df['Dataset'] == dataset]
    
    x = np.arange(len(data))
    width = 0.15
    
    models = ['B-Tree', 'ALEX', 'NLI']  # Focus on competitive models
    
    for i, model in enumerate(models):
        values = data[f'{model} ns'].values
        ax.bar(x + i * width, values, width, label=model, color=colors[model])
    
    ax.set_xlabel('Dataset Size (keys)')
    ax.set_ylabel('Query Latency (ns)')
    ax.set_title(f'{dataset} Dataset')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['100K', '1M', '10M'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('paper_plots/fig1_latency_comparison.pdf', bbox_inches='tight')
plt.savefig('paper_plots/fig1_latency_comparison.png', bbox_inches='tight')
print("   ✅ Saved: fig1_latency_comparison.pdf/.png")
plt.close()

# ============================================================================
# FIGURE 2: Speedup over B-Tree Baseline
# ============================================================================
print("\n[2/6] Generating Figure 2: Speedup Analysis...")

fig, ax = plt.subplots(figsize=(8, 4))

for dataset in datasets:
    data = df[df['Dataset'] == dataset]
    x = data['Keys'].values / 1e6  # Convert to millions
    speedup = data['NLI Speedup'].values
    ax.plot(x, speedup, marker='o', linewidth=2, markersize=8, 
            label=dataset, alpha=0.8)

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
           label='B-Tree Baseline', alpha=0.7)
ax.set_xlabel('Dataset Size (Million Keys)')
ax.set_ylabel('Speedup vs B-Tree (×)')
ax.set_title('NLI Performance Speedup over B-Tree')
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('paper_plots/fig2_speedup_analysis.pdf', bbox_inches='tight')
plt.savefig('paper_plots/fig2_speedup_analysis.png', bbox_inches='tight')
print("   ✅ Saved: fig2_speedup_analysis.pdf/.png")
plt.close()

# ============================================================================
# FIGURE 3: All Models Performance Comparison (Heatmap)
# ============================================================================
print("\n[3/6] Generating Figure 3: Performance Heatmap...")

# Create performance matrix (relative to B-Tree)
models = ['ALEX', 'PGM', 'RMI', 'NLI']
heatmap_data = []
labels = []

for dataset in datasets:
    for size in df['Keys'].unique():
        row_data = df[(df['Dataset'] == dataset) & (df['Keys'] == size)]
        if len(row_data) > 0:
            btree_latency = row_data['B-Tree ns'].values[0]
            speedups = [btree_latency / row_data[f'{m} ns'].values[0] for m in models]
            heatmap_data.append(speedups)
            labels.append(f"{dataset}\n{size//1000}K")

heatmap_data = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=5)

ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(models)
ax.set_yticklabels(labels)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Speedup vs B-Tree (×)', rotation=270, labelpad=20)

# Add text annotations
for i in range(len(labels)):
    for j in range(len(models)):
        text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_title('Performance Heatmap: Speedup over B-Tree')
plt.tight_layout()
plt.savefig('paper_plots/fig3_performance_heatmap.pdf', bbox_inches='tight')
plt.savefig('paper_plots/fig3_performance_heatmap.png', bbox_inches='tight')
print("   ✅ Saved: fig3_performance_heatmap.pdf/.png")
plt.close()

# ============================================================================
# FIGURE 4: Scalability Analysis (Log-Log Plot)
# ============================================================================
print("\n[4/6] Generating Figure 4: Scalability Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    data = df[df['Dataset'] == dataset]
    
    x = data['Keys'].values
    
    for model in ['B-Tree', 'ALEX', 'NLI']:
        y = data[f'{model} ns'].values
        ax.plot(x, y, marker='o', linewidth=2, markersize=8,
                label=model, color=colors[model], alpha=0.8)
    
    ax.set_xlabel('Dataset Size (keys)')
    ax.set_ylabel('Query Latency (ns)')
    ax.set_title(f'{dataset} Scalability')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')

plt.tight_layout()
plt.savefig('paper_plots/fig4_scalability.pdf', bbox_inches='tight')
plt.savefig('paper_plots/fig4_scalability.png', bbox_inches='tight')
print("   ✅ Saved: fig4_scalability.pdf/.png")
plt.close()

# ============================================================================
# FIGURE 5: Model Comparison Bar Chart (Average Performance)
# ============================================================================
print("\n[5/6] Generating Figure 5: Average Performance Comparison...")

fig, ax = plt.subplots(figsize=(8, 4))

models = ['B-Tree', 'ALEX', 'PGM', 'RMI', 'NLI']
avg_latencies = []

for model in models:
    # Calculate geometric mean (better for ratio data)
    latencies = df[f'{model} ns'].values
    avg_latencies.append(np.exp(np.mean(np.log(latencies))))

x_pos = np.arange(len(models))
bars = ax.bar(x_pos, avg_latencies, color=[colors[m] for m in models], 
              alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.0f} ns',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Index Structure')
ax.set_ylabel('Average Query Latency (ns)')
ax.set_title('Average Performance Across All Datasets (Geometric Mean)')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('paper_plots/fig5_average_performance.pdf', bbox_inches='tight')
plt.savefig('paper_plots/fig5_average_performance.png', bbox_inches='tight')
print("   ✅ Saved: fig5_average_performance.pdf/.png")
plt.close()

# ============================================================================
# FIGURE 6: Summary Statistics Table (as image)
# ============================================================================
print("\n[6/6] Generating Figure 6: Summary Statistics Table...")

# Calculate summary statistics
summary_data = []
for model in ['B-Tree', 'ALEX', 'PGM', 'RMI', 'NLI']:
    latencies = df[f'{model} ns'].values
    summary_data.append([
        model,
        f"{np.mean(latencies):.1f}",
        f"{np.median(latencies):.1f}",
        f"{np.min(latencies):.1f}",
        f"{np.max(latencies):.1f}",
        f"{np.std(latencies):.1f}"
    ])

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=summary_data,
                colLabels=['Model', 'Mean (ns)', 'Median (ns)', 'Min (ns)', 'Max (ns)', 'Std Dev (ns)'],
                cellLoc='center',
                loc='center',
                colColours=['lightgray']*6)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the model column
for i in range(1, len(summary_data) + 1):
    model = summary_data[i-1][0]
    table[(i, 0)].set_facecolor(colors[model])
    table[(i, 0)].set_alpha(0.3)

plt.title('Summary Statistics: Query Latency Across All Tests', pad=20, fontsize=12, weight='bold')
plt.savefig('paper_plots/fig6_summary_table.pdf', bbox_inches='tight')
plt.savefig('paper_plots/fig6_summary_table.png', bbox_inches='tight')
print("   ✅ Saved: fig6_summary_table.pdf/.png")
plt.close()

# ============================================================================
# Generate LaTeX Table for Paper
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING LATEX TABLE")
print("=" * 80)

latex_output = r"""\begin{table}[h]
\centering
\caption{Query Latency Comparison (ns) - NLI vs Baseline Methods}
\label{tab:results}
\begin{tabular}{lrrrrr}
\hline
\textbf{Dataset} & \textbf{Size} & \textbf{B-Tree} & \textbf{ALEX} & \textbf{RMI} & \textbf{NLI} \\
\hline
"""

for _, row in df.iterrows():
    size_str = f"{row['Keys']:,}".replace(',', '\\,')
    latex_output += f"{row['Dataset']} & {size_str} & "
    latex_output += f"{row['B-Tree ns']:.1f} & {row['ALEX ns']:.1f} & "
    latex_output += f"{row['RMI ns']:.0f} & \\textbf{{{row['NLI ns']:.1f}}} \\\\\n"

latex_output += r"""\hline
\end{tabular}
\end{table}
"""

with open('paper_plots/results_table.tex', 'w') as f:
    f.write(latex_output)

print("\n✅ Saved: results_table.tex")
print("\n" + latex_output)

# ============================================================================
# Generate Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY REPORT")
print("=" * 80)

print("\n📊 KEY FINDINGS:")
print("-" * 80)

# Best speedup
best_speedup = df.loc[df['NLI Speedup'].idxmax()]
print(f"✓ Best NLI Speedup: {best_speedup['NLI Speedup']:.2f}×")
print(f"  Dataset: {best_speedup['Dataset']}, Size: {best_speedup['Keys']:,} keys")

# Average speedup
avg_speedup = df['NLI Speedup'].mean()
print(f"\n✓ Average NLI Speedup: {avg_speedup:.2f}×")

# Win rate
wins = (df['NLI ns'] < df['B-Tree ns']).sum()
total = len(df)
print(f"\n✓ NLI wins against B-Tree: {wins}/{total} tests ({100*wins/total:.1f}%)")

# Best absolute latency
best_latency = df['NLI ns'].min()
best_idx = df['NLI ns'].idxmin()
print(f"\n✓ Best NLI Latency: {best_latency:.1f} ns")
print(f"  Dataset: {df.loc[best_idx, 'Dataset']}, Size: {df.loc[best_idx, 'Keys']:,} keys")

print("\n" + "=" * 80)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\n📁 Output directory: paper_plots/")
print("\n📄 Generated files:")
print("   • fig1_latency_comparison.pdf/.png")
print("   • fig2_speedup_analysis.pdf/.png")
print("   • fig3_performance_heatmap.pdf/.png")
print("   • fig4_scalability.pdf/.png")
print("   • fig5_average_performance.pdf/.png")
print("   • fig6_summary_table.pdf/.png")
print("   • results_table.tex (LaTeX table)")
print("\n✅ Ready for publication!")
print("=" * 80)