import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import matplotlib.lines as mlines

# Create figure with larger size for better readability
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Enhanced color palette
colors = {
    'data_layer': '#E8F4F8',
    'synthetic': '#D6EAF8',
    'sosd': '#AED6F1',
    'workload': '#E8E8E8',
    'btree': '#E8F5E9',
    'nli': '#FFF3E0',
    'drift': '#FCE4EC',
    'eval': '#E0F7FA',
    'perf': '#D0F0C0',
    'legend_bg': '#FFFFFF'
}

def draw_box(x, y, w, h, text, color, ax, fontsize=11, title=None, bullets=None):
    """Draw a rounded rectangle with text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='#333333', linewidth=2)
    ax.add_patch(box)
    
    # Title
    if title:
        ax.text(x + w/2, y + h - 2.5, title, fontsize=fontsize+3, weight='bold',
                ha='center', va='top', family='sans-serif')
        start_y = y + h - 6
    else:
        start_y = y + h - 2.5
    
    # Main text or bullets
    if bullets:
        for i, bullet in enumerate(bullets):
            ax.text(x + 1.5, start_y - i*2.8, f"• {bullet}", fontsize=fontsize,
                    ha='left', va='top', wrap=True, family='sans-serif')
    else:
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize+1, weight='normal',
                ha='center', va='center', wrap=True, family='sans-serif')

def draw_arrow(x1, y1, x2, y2, ax, label='', color='black', width=2.5, style='->', label_offset=(0,0)):
    """Draw arrow with optional label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=25,
                           linewidth=width, color=color, zorder=2)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1]
        ax.text(mid_x, mid_y, label, fontsize=10, ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='gray'))

# ===== DATA LAYER =====
ax.text(50, 98, 'Layer 1: Data & Workload Generation', fontsize=16, weight='bold', 
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor=colors['data_layer'], 
                 edgecolor='#333333', linewidth=2))

# Synthetic Workloads
draw_box(5, 82, 21, 13, '', colors['synthetic'], ax, fontsize=10,
         title='Synthetic Workloads\nPhase 3-4 Experiments',
         bullets=['Dataset sizes: 10K, 100K, 1M keys', 
                  'Distributions: Uniform, Skewed, Sequential',
                  'Controlled drift test scenarios'])

# Real SOSD Datasets
draw_box(28, 82, 21, 13, '', colors['sosd'], ax, fontsize=10,
         title='Real SOSD Datasets',
         bullets=['Books: 200M Amazon records', 
                  'Wikipedia: 200M edit timestamps',
                  'Facebook: 200M user IDs', 
                  'Test sizes: 100K, 1M, 10M keys'])

# Workload Generator
draw_box(51, 82, 20, 13, '', colors['workload'], ax, fontsize=10,
         title='Workload Generator',
         bullets=['Decompress binary datasets',
                  'Generate 100K random point queries',
                  'Normalize to uint64 format'])

# Arrows from data sources to workload generator
draw_arrow(15.5, 82, 15.5, 77, ax, width=3)
draw_arrow(38.5, 82, 52, 77, ax, width=3)

# ===== INDEX & DETECTION LAYER =====
ax.text(50, 75, 'Layer 2: Index Structures & Drift Detection', fontsize=16, 
        weight='bold', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#EEEEEE', 
                 edgecolor='#333333', linewidth=2))

# B-tree Baseline
draw_box(5, 57, 16, 13, '', colors['btree'], ax, fontsize=10,
         title='B-tree Baseline\n(std::map)',
         bullets=['O(log n) search complexity',
                  'Correctness oracle',
                  'Performance baseline'])

# NLI Ensemble - larger box
draw_box(23, 57, 26, 13, '', colors['nli'], ax, fontsize=10,
         title='Neural Learned Index\nEnsemble')

# Router model inside NLI
ax.text(36, 67, 'Linear Router Model', fontsize=11, weight='bold',
        ha='center', bbox=dict(boxstyle='round,pad=0.4', 
                              facecolor='#FFE0B2', edgecolor='#FF9800', linewidth=1.5))

# Sub-indexes with larger boxes
ax.text(27, 62, 'ALEX\n70%', fontsize=10, ha='center', weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCCBC', 
                 edgecolor='#FF7043', linewidth=1.5))
ax.text(36, 62, 'PGM\n20%', fontsize=10, ha='center', weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCCBC', 
                 edgecolor='#FF7043', linewidth=1.5))
ax.text(45, 62, 'RMI\n10%', fontsize=10, ha='center', weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCCBC', 
                 edgecolor='#FF7043', linewidth=1.5))

# Arrows from workload generator to indexes
draw_arrow(61, 82, 13, 70, ax, label='Build', width=2.5)
draw_arrow(61, 82, 36, 70, ax, label='Build', width=2.5)

# Query arrows from indexes downward
draw_arrow(13, 57, 25, 49, ax, label='Queries', width=2)
draw_arrow(36, 57, 36, 49, ax, label='Queries', width=2)

# Drift Detection & Auto-Repair - wider box
draw_box(10, 30, 55, 17, '', colors['drift'], ax, fontsize=10,
         title='Combined Drift Detection & Auto-Repair')

# Detection methods boxes inside
ax.text(20, 43, 'EWMA\nWeight: 20%\nError Trends', fontsize=10, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8BBD0', 
                 edgecolor='#E91E63', linewidth=1.5))
ax.text(36, 43, 'PSI\nWeight: 60%\n★ PRIMARY ★', fontsize=10, ha='center', weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8BBD0', 
                 edgecolor='#C2185B', linewidth=2))
ax.text(52, 43, 'Autoencoder\nWeight: 20%\nNon-linear', fontsize=10, ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8BBD0', 
                 edgecolor='#E91E63', linewidth=1.5))

# Drift threshold and strategies
ax.text(36, 38, 'Drift Score Combiner → Threshold: 0.40', fontsize=11,
        ha='center', weight='bold', style='italic')
ax.text(14, 35, 'Auto-Repair Strategies:', fontsize=11, ha='left', weight='bold')
ax.text(14, 33, '• REFIT: Retrain models (gradual drift)', fontsize=10, ha='left')
ax.text(14, 31, '• SPLIT: Rebuild partitions (sudden shift)', fontsize=10, ha='left')

# Feedback arrow (curved red) from drift back to NLI
arc = FancyArrowPatch((58, 38), (48, 60), connectionstyle="arc3,rad=.4",
                      arrowstyle='->', mutation_scale=25, linewidth=3.5, 
                      color='#D32F2F', zorder=3)
ax.add_patch(arc)
ax.text(64, 49, 'Model\nUpdate', fontsize=11, color='#D32F2F', weight='bold', 
        ha='center', style='italic')

# ===== EVALUATION LAYER =====
ax.text(82, 75, 'Layer 3: Evaluation', fontsize=16, weight='bold',
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor=colors['eval'], 
                 edgecolor='#333333', linewidth=2))

# Benchmark Orchestration
draw_box(70, 57, 24, 13, '', colors['eval'], ax, fontsize=10,
         title='Benchmark\nOrchestration',
         bullets=['Execute queries on indexes',
                  'Measure latency & memory',
                  'Compute speedup ratios'])

# Results Storage
draw_box(70, 40, 24, 11, '', colors['eval'], ax, fontsize=10,
         title='Results Storage',
         bullets=['Performance metrics (CSV)',
                  'Drift analysis logs',
                  'Benchmark statistics'])

# Publication Outputs
draw_box(70, 26, 24, 11, '', colors['eval'], ax, fontsize=10,
         title='Publication Outputs',
         bullets=['Speedup: 1.8x - 5.98x',
                  'Performance comparison tables',
                  'Drift visualization plots'])

# Arrows in evaluation flow
draw_arrow(82, 57, 82, 51, ax, width=3)
draw_arrow(82, 40, 82, 37, ax, width=3)

# Arrows from components to evaluation
draw_arrow(21, 63, 70, 63, ax, label='Latency', width=2.5)
draw_arrow(50, 33, 70, 45, ax, label='Drift Scores', width=2.5)

# ===== FOOTER =====
# Key Performance box
draw_box(5, 10, 23, 11, '', colors['perf'], ax, fontsize=10,
         title='Key Performance Metrics',
         bullets=['Speedup: 1.8x - 5.98x vs B-tree',
                  'Memory: 40-60% reduction',
                  'Latency: Sub-microsecond',
                  'All drift tests pass'])

# Legend box
draw_box(62, 10, 30, 11, '', colors['legend_bg'], ax, fontsize=9,
         title='Legend')

# Legend arrows
arrow1 = mlines.Line2D([64, 70], [17, 17], linewidth=3, color='black',
                       marker='>', markersize=10, markerfacecolor='black')
ax.add_line(arrow1)
ax.text(72, 17, 'Primary data flow', fontsize=10, va='center', weight='bold')

arrow2 = mlines.Line2D([64, 70], [14.5, 14.5], linewidth=2, color='black',
                       marker='>', markersize=8, markerfacecolor='black')
ax.add_line(arrow2)
ax.text(72, 14.5, 'Secondary flow', fontsize=10, va='center')

arrow3 = mlines.Line2D([64, 70], [12, 12], linewidth=3.5, color='#D32F2F',
                       marker='>', markersize=10, markerfacecolor='#D32F2F')
ax.add_line(arrow3)
ax.text(72, 12, 'Feedback/Update loop', fontsize=10, va='center', 
        weight='bold', color='#D32F2F')

# Main Title at bottom
ax.text(50, 3, 'System Architecture: Neural Learned Index with Online Drift Detection and Auto-Repair',
        fontsize=18, weight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=1', facecolor='white', 
                 edgecolor='#333333', linewidth=3))

plt.tight_layout()
plt.savefig('nli_system_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('nli_system_architecture.png', dpi=300, bbox_inches='tight')
print("✓ Architecture diagram saved successfully!")
plt.show()
