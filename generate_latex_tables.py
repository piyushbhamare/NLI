"""
Neural Learned Index (NLI) - LaTeX Tables for Publication
"""
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('nli_sosd_results.csv')

print("=" * 80)
print("NEURAL LEARNED INDEX (NLI) - PUBLICATION LaTeX TABLES")
print("=" * 80)

# Table 1: Main Results (5 Models)
print("\n% TABLE 1: NLI vs State-of-the-Art (5 Models)")
print(r"\begin{table}[t]")
print(r"\centering")
print(r"\caption{Neural Learned Index vs State-of-the-Art}")
print(r"\label{tab:nli_results}")
print(r"\small")
print(r"\begin{tabular}{lrrrrr}")
print(r"\toprule")
print(r"Dataset & Keys & B-Tree & RMI & PGM & \textbf{NLI} & Speedup \\")
print(r"\cmidrule(lr){2-7}")
print(r"& (M) & (ns) & (ns) & (ns) & (ns) & (vs B-Tree) \\")
print(r"\midrule")
for _, row in df.iterrows():
    keys_str = f"{int(row['Keys']/1e6)}M"
    print(f"{row['Dataset']:<10} & {keys_str} & "
          f"{row['B-Tree ns']:.0f} & {row['RMI ns']:.0f} & "
          f"{row['PGM ns']:.0f} & \\textbf{{{row['NLI ns']:.0f}}} & "
          f"\\textbf{{{row['NLI Speedup']:.2f}x}} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

# Table 2: Geometric Mean Speedup
avg_speedup = df['NLI Speedup'].geometric_mean()
print(f"\n% NLI achieves {avg_speedup:.2f}x geometric mean speedup vs B-Tree")

print("\n" + "=" * 80)
print("COPY LaTeX TABLES TO PAPER ✅")
print("=" * 80)
