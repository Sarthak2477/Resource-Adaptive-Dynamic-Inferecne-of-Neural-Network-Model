import sys; sys.path.insert(0, '.')
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
