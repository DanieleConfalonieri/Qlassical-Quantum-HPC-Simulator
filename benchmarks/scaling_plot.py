import pandas as pd
import matplotlib.pyplot as plt
import io

# Data from Karolina HPC Cluster
strong_csv = """Threads,RealTime_ms
1,112635
2,57582
4,29294
8,15178
16,7823
32,5289
64,5117
112,4791"""

weak_csv = """Shift,Threads,Qubits,RealTime_ms
0,1,22,5792
1,2,23,6640
2,4,24,7279
3,8,25,7598
4,16,26,7891
5,32,27,10674
6,64,28,21685"""

df_strong = pd.read_csv(io.StringIO(strong_csv))
df_weak = pd.read_csv(io.StringIO(weak_csv))

# Compute Speedup (Strong Scaling)
# Speedup = T(1) / T(N)
t1_strong = df_strong['RealTime_ms'].iloc[0]
df_strong['Speedup'] = t1_strong / df_strong['RealTime_ms']
df_strong['Ideal_Speedup'] = df_strong['Threads']

# Plotting Setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: Strong Scaling (Speedup curve)
ax1.plot(df_strong['Threads'], df_strong['Speedup'], marker='o', linewidth=2, color='#1f77b4', label='Measured Speedup')
ax1.plot(df_strong['Threads'], df_strong['Ideal_Speedup'], linestyle='--', color='gray', label='Ideal (Linear) Speedup')

# Memory wall
saturation_x = df_strong['Threads'].iloc[5:]
saturation_y = df_strong['Speedup'].iloc[5:]
ax1.plot(saturation_x, saturation_y, color='red', linewidth=2, label='Memory Bandwidth Saturation')

ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=2)
ax1.set_xticks(df_strong['Threads'])
ax1.set_xticklabels(df_strong['Threads'])
ax1.set_xlabel('Number of OpenMP Threads', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.set_title('Strong Scaling (Fixed Workload: 26 Qubits)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)

# Plot B: Weak Scaling (Execution Time curve)
ax2.plot(df_weak['Threads'], df_weak['RealTime_ms'], marker='s', linewidth=2, color='#d62728', label='Execution Time')

# Ideal weak scaling
ideal_weak_time = df_weak['RealTime_ms'].iloc[0]
ax2.axhline(ideal_weak_time, linestyle='--', color='gray', label='Ideal Weak Scaling')

ax2.set_xscale('log', base=2)
ax2.set_xticks(df_weak['Threads'])
xticklabels = [f"{t}T\n({q}Q)" for t, q in zip(df_weak['Threads'], df_weak['Qubits'])]
ax2.set_xticklabels(xticklabels)

ax2.set_xlabel('Threads (and proportional Qubits)', fontsize=12)
ax2.set_ylabel('Execution Time (ms)', fontsize=12)
ax2.set_title('Weak Scaling (Proportional Workload)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)

plt.suptitle('Qlassical Simulator: Hardware Profiling on Karolina', fontsize=16, y=1.02)
plt.tight_layout()

# Save high-res for the thesis
plt.savefig('hpc_scaling_results.png', dpi=300, bbox_inches='tight')
print("Plot successfully generated and saved as 'hpc_scaling_results.png'")
plt.show()