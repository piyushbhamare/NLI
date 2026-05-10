# """
# Neural Learned Index (NLI) - 5-Model Benchmark Parser
# Parses: NLI vs ALEX vs PGM vs RMI vs B-Tree
# """
# import os
# import sys
# import subprocess
# import re
# import pandas as pd
# import signal
# import time

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# print("=" * 80)
# print("NEURAL LEARNED INDEX (NLI) - COMPLETE 5-MODEL BENCHMARK")
# print("=" * 80)

# # Kill any hanging processes
# os.system("taskkill /f /im nli_sosd.exe >nul 2>&1")

# # Compile NLI benchmark
# print("\n[1/3] COMPILING NLI BENCHMARK...")
# result = subprocess.run("g++ -O3 -std=c++17 -o build\\nli_sosd.exe sosd_real_benchmark.cpp",
#                         shell=True, capture_output=True, text=True)
# if result.returncode != 0:
#     print("❌ COMPILATION FAILED!")
#     print(result.stderr)
#     sys.exit(1)
# print("✅ Compiled NLI benchmark successfully")

# # Run benchmark WITH 5-MINUTE TIMEOUT
# print("\n[2/3] RUNNING 5-MODEL BENCHMARK (MAX 5 minutes)...")
# try:
#     result = subprocess.run("build\\nli_sosd.exe", shell=True, capture_output=True, text=True, timeout=300)
#     print("✅ Benchmark completed normally")
# except subprocess.TimeoutExpired:
#     print("⚠️ TIMEOUT after 5 minutes - killing process")
#     os.system("taskkill /f /im nli_sosd.exe >nul 2>&1")
#     result = subprocess.CompletedProcess(result.args, 124)  # Timeout exit code

# print("=== C++ STDOUT ===")
# print(result.stdout)
# print("=== C++ STDERR ===")
# print(result.stderr)
# print(f"=== EXIT CODE: {result.returncode} ===")

# # Parse 5-model results
# print("\n[3/3] PARSING NLI vs ALEX vs PGM vs RMI vs B-Tree...")
# results = []
# lines = result.stdout.split('\n')
# current_dataset = None
# current_size = None
# latencies = {}

# for line in lines:
#     if 'DATASET:' in line:
#         current_dataset = line.split('DATASET:')[1].strip()
#     if '--- Size:' in line:
#         m = re.search(r'(\d+)', line)
#         current_size = int(m.group(1)) if m else None
#     if 'ns |' in line and current_dataset and current_size:
#         # Parse: "B-Tree  1292.3 ns | ALEX 363.1 ns | PGM 460.2 ns | RMI 650.4 ns | NLI 147.0 ns"
#         parts = re.findall(r'([A-Za-z\s\-]+?)\s+(\d+\.?\d*)\s*ns', line)
#         for model, latency in parts:
#             model = model.strip()
#             latencies[model] = float(latency)
#         if len(latencies) == 5:  # All 5 models found
#             results.append({
#                 'Dataset': current_dataset,
#                 'Keys': current_size,
#                 'B-Tree ns': latencies.get('B-Tree', 0),
#                 'ALEX ns': latencies.get('ALEX', 0),
#                 'PGM ns': latencies.get('PGM', 0),
#                 'RMI ns': latencies.get('RMI', 0),
#                 'NLI ns': latencies.get('NLI', 0),
#                 'NLI Speedup': latencies.get('B-Tree', 1) / max(1, latencies.get('NLI', 1))
#             })
#             latencies.clear()

# # Save publication-ready CSV
# if results:
#     df = pd.DataFrame(results)
#     output = 'nli_sosd_results.csv'  # NLI-branded filename
#     df.to_csv(output, index=False)
#     print(f"\n✅ NLI Results saved: {os.path.abspath(output)}")
#     print("\n" + df.to_string(index=False))
# else:
#     print("\n⚠️ No results parsed - C++ likely has training bug")
#     print("\n🔧 NEXT STEPS:")
#     print("1. Check sosd_real_benchmark.cpp for infinite training loops")
#     print("2. Test single dataset: cd build && .\nli_sosd.exe | head -50")

# print("\n" + "=" * 80)
# print("NEXT: python generate_publication_plots.py (if results exist)")
# print("=" * 80)

import os
import sys
import subprocess
import re
import pandas as pd
import signal
import time


os.chdir(os.path.dirname(os.path.abspath(__file__)))


print("=" * 80)
print("NEURAL LEARNED INDEX (NLI) - COMPLETE 5-MODEL BENCHMARK")
print("=" * 80)


# Kill any hanging processes
os.system("taskkill /f /im nli_sosd.exe >nul 2>&1")


# Compile NLI benchmark
print("\n[1/3] COMPILING NLI BENCHMARK...")
result = subprocess.run(
    "g++ -O3 -std=c++17 -o build\\nli_sosd.exe sosd_real_benchmark.cpp",
    shell=True,
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("❌ COMPILATION FAILED!")
    print(result.stderr)
    sys.exit(1)

print("✅ Compiled NLI benchmark successfully")


# Run benchmark WITH 5-MINUTE TIMEOUT
print("\n[2/3] RUNNING 5-MODEL BENCHMARK (MAX 5 minutes)...")
try:
    result = subprocess.run(
        "build\\nli_sosd.exe",
        shell=True,
        capture_output=True,
        text=True,
        timeout=300
    )
    print("✅ Benchmark completed normally")
except subprocess.TimeoutExpired as e:
    print("⚠️ TIMEOUT after 5 minutes - killing process")
    os.system("taskkill /f /im nli_sosd.exe >nul 2>&1")
    result = subprocess.CompletedProcess(
        args=e.cmd,
        returncode=124,
        stdout=e.stdout if e.stdout else "",
        stderr=e.stderr if e.stderr else "Benchmark timed out after 300 seconds"
    )


print("=== C++ STDOUT ===")
print(result.stdout)
print("=== C++ STDERR ===")
print(result.stderr)
print(f"=== EXIT CODE: {result.returncode} ===")


# Parse 5-model results
print("\n[3/3] PARSING NLI vs ALEX vs PGM vs RMI vs B-Tree...")
results = []
lines = result.stdout.split('\n')
current_dataset = None
current_size = None
latencies = {}


for line in lines:
    if 'DATASET:' in line:
        current_dataset = line.split('DATASET:')[1].strip()

    if '--- Size:' in line:
        m = re.search(r'(\d+)', line)
        current_size = int(m.group(1)) if m else None

    if 'ns |' in line and current_dataset and current_size:
        # Parse:
        # "B-Tree 1292.3 ns | ALEX 363.1 ns | PGM 460.2 ns | RMI 650.4 ns | NLI 147.0 ns"
        parts = re.findall(r'([A-Za-z\s\-]+?)\s+(\d+\.?\d*)\s*ns', line)

        for model, latency in parts:
            model = model.strip()
            latencies[model] = float(latency)

        if len(latencies) == 5:  # All 5 models found
            btree_latency = latencies.get('B-Tree', 1)
            alex_latency = latencies.get('ALEX', 1)
            pgm_latency = latencies.get('PGM', 1)
            rmi_latency = latencies.get('RMI', 1)
            nli_latency = max(1, latencies.get('NLI', 1))

            results.append({
                'Dataset': current_dataset,
                'Keys': current_size,
                'B-Tree ns': btree_latency,
                'ALEX ns': alex_latency,
                'PGM ns': pgm_latency,
                'RMI ns': rmi_latency,
                'NLI ns': latencies.get('NLI', 0),
                'NLI Speedup vs B-Tree': btree_latency / nli_latency,
                'NLI Speedup vs ALEX': alex_latency / nli_latency,
                'NLI Speedup vs PGM': pgm_latency / nli_latency,
                'NLI Speedup vs RMI': rmi_latency / nli_latency
            })

            latencies.clear()


# Save publication-ready CSV
if results:
    df = pd.DataFrame(results)

    speedup_cols = [
        'NLI Speedup vs B-Tree',
        'NLI Speedup vs ALEX',
        'NLI Speedup vs PGM',
        'NLI Speedup vs RMI'
    ]
    df[speedup_cols] = df[speedup_cols].round(3)

    output = 'nli_sosd_results.csv'  # NLI-branded filename
    df.to_csv(output, index=False)

    print(f"\n✅ NLI Results saved: {os.path.abspath(output)}")
    print("\n" + df.to_string(index=False))
else:
    print("\n⚠️ No results parsed - C++ likely has training bug")
    print("\n🔧 NEXT STEPS:")
    print("1. Check sosd_real_benchmark.cpp for infinite training loops")
    print("2. Test single dataset: cd build && .\\nli_sosd.exe | head -50")


print("\n" + "=" * 80)
print("NEXT: python generate_publication_plots.py (if results exist)")
print("=" * 80)