import sys; sys.path.insert(0, '.')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scripts.evaluate import expand_frontier_continuous
from models.config import FLAGS

CONFIG_ACCURACIES = {
    (0.25, 4): 89.69, (0.25, 8): 90.46, (0.25, 16): 90.57, (0.25, 32): 90.59,
    (0.5, 4): 90.19,  (0.5, 8): 91.83,  (0.5, 16): 91.79,  (0.5, 32): 91.83,
    (0.75, 4): 90.51, (0.75, 8): 92.12, (0.75, 16): 92.16, (0.75, 32): 92.18,
    (1.0, 4): 90.66,  (1.0, 8): 92.32,  (1.0, 16): 92.37,  (1.0, 32): 92.37,
}

frontier = [{'config': (w, b), 'latency_ms': 0, 'std_ms': 0, 'acc': CONFIG_ACCURACIES[(w, b)]} for w, b in FLAGS.deploy_configs]
expanded = expand_frontier_continuous(frontier, step=0.05)

print(f"{'No.':<5} {'Width Mult':<12} {'Bit Width':<10} {'Accuracy (%)':<14} {'Type'}")
print('-' * 55)
for i, e in enumerate(expanded):
    w, b = e['config']
    kind = 'discrete' if (w, b) in CONFIG_ACCURACIES else 'interpolated'
    print(f"{i+1:<5} {w:<12.2f} {b:<10} {e['acc']:<14.2f} {kind}")
print(f"\nTotal: {len(expanded)} configurations")

# --- Plot ---
bit_widths = sorted({e['config'][1] for e in expanded})
colors = {4: '#e74c3c', 8: '#2ecc71', 16: '#3498db', 32: '#9b59b6'}
markers = {4: 'o', 8: 's', 16: '^', 32: 'D'}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Supernetwork Configurations — Accuracy & Bit-Width vs Width Multiplier', fontsize=14, fontweight='bold')

# Plot 1: Accuracy vs Width Multiplier (one line per bit-width)
ax1 = axes[0]
for b in bit_widths:
    pts = sorted([(e['config'][0], e['acc']) for e in expanded if e['config'][1] == b])
    xs, ys = zip(*pts)
    disc_xs = [x for x, _ in pts if (x, b) in CONFIG_ACCURACIES]
    disc_ys = [CONFIG_ACCURACIES[(x, b)] for x in disc_xs]
    ax1.plot(xs, ys, color=colors[b], linewidth=1.5, linestyle='--', alpha=0.6)
    ax1.scatter(xs, ys, color=colors[b], marker=markers[b], s=25, alpha=0.5)
    ax1.scatter(disc_xs, disc_ys, color=colors[b], marker=markers[b], s=80, zorder=5, edgecolors='black', linewidths=0.8)

ax1.set_xlabel('Width Multiplier', fontsize=11)
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
ax1.set_title('Accuracy vs Width Multiplier', fontsize=12)
ax1.set_xticks([0.25, 0.5, 0.75, 1.0])
ax1.set_ylim(89, 93)
ax1.grid(True, alpha=0.3)
patches = [mpatches.Patch(color=colors[b], label=f'{b}-bit') for b in bit_widths]
ax1.legend(handles=patches, title='Bit Width', fontsize=9)

# Plot 2: Scatter — all 60+ configs, x=width, y=accuracy, color=bit-width, size=bit-width
ax2 = axes[1]
for b in bit_widths:
    pts = [(e['config'][0], e['acc'], (e['config'][0], b) in CONFIG_ACCURACIES) for e in expanded if e['config'][1] == b]
    interp = [(x, y) for x, y, d in pts if not d]
    disc   = [(x, y) for x, y, d in pts if d]
    if interp:
        ax2.scatter(*zip(*interp), color=colors[b], marker=markers[b], s=30, alpha=0.4, label=f'{b}-bit (interp)')
    if disc:
        ax2.scatter(*zip(*disc), color=colors[b], marker=markers[b], s=120, edgecolors='black', linewidths=0.8, zorder=5, label=f'{b}-bit (measured)')

ax2.set_xlabel('Width Multiplier', fontsize=11)
ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
ax2.set_title(f'All {len(expanded)} Configurations (large = measured, small = interpolated)', fontsize=11)
ax2.set_xticks([0.25, 0.5, 0.75, 1.0])
ax2.set_ylim(89, 93)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig('results/supernetwork_configs.png', dpi=150, bbox_inches='tight')
print("\nGraph saved to results/supernetwork_configs.png")
plt.show()
