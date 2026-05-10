import pandas as pd
df = pd.read_csv('nli_sosd_results_3x_ci.csv')

table = df.pivot(index='Keys', columns='Dataset', values='Memory_NLI_MB')
table['Mean_MB'] = table.mean(axis=1)
table['vs_BTree'] = table['Mean_MB'] / df.groupby('Keys')['Memory_BTree'].mean()

print("| Keys | Books | Facebook | WikiTS | Mean | vs B-Tree |")
print("|------|-------|----------|--------|------|-----------|")
for keys in table.index:
    row = table.loc[keys]
    print(f"| {keys:>5,} | {row['Books']:.1f} | {row['Facebook']:.1f} | "
          f"{row['WikiTS']:.1f} | {row['Mean_MB']:.1f} | {row['vs_BTree']:.2f}x |")
