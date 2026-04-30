#!/usr/bin/env python3
"""Count trajectories per module across all node downloads."""
import torch, os, glob, sys

traj_dir = sys.argv[1] if len(sys.argv) > 1 else 'trajectories_generate/output/multi_trajectories_v2'
modules = {}
for node_dir in sorted(glob.glob(os.path.join(traj_dir, '*'))):
    if not os.path.isdir(node_dir):
        continue
    for pt_file in sorted(glob.glob(os.path.join(node_dir, '*.pt'))):
        module = os.path.basename(pt_file).replace('.pt', '')
        node = os.path.basename(node_dir)
        try:
            data = torch.load(pt_file, weights_only=False)
            n_traj = len(data.get('trajectories', []))
        except Exception as e:
            print(f"  WARN: {pt_file}: {e}", file=sys.stderr)
            n_traj = -1  # corrupted
        if module not in modules:
            modules[module] = []
        modules[module].append((node, n_traj))

fmt = "{:<55} {:>6} {:>10}"
print(fmt.format("Module", "Copies", "Total"))
print("-" * 75)
total = 0
for mod in sorted(modules.keys()):
    entries = modules[mod]
    n_total = sum(t for _, t in entries)
    total += n_total
    print(fmt.format(mod, len(entries), n_total))
print("-" * 75)
print(fmt.format("TOTAL", len(modules), total))
