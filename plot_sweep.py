"""
Plot teacher sweep results from logs/dev/*.txt

Usage:
    python plot_sweep.py [glob_pattern]

Examples:
    python plot_sweep.py                     # all logs in logs/dev/
    python plot_sweep.py "logs/dev/abc*.txt" # specific logs
"""

import sys
import re
import glob
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log(filepath):
    """Extract config and val loss trajectory from a log file."""
    config = {}
    val_points = []  # (step, val_loss, train_time_ms)

    with open(filepath) as f:
        for line in f:
            # Parse teacher config line
            m = re.match(r'Teacher config: (\d+)L / (\d+)H / (\d+)D / (\d+)dim / (\d+) steps(?: / bs=(\d+))?', line)
            if m:
                config['layers'] = int(m.group(1))
                config['heads'] = int(m.group(2))
                config['head_dim'] = int(m.group(3))
                config['dim'] = int(m.group(4))
                config['steps'] = int(m.group(5))
                if m.group(6):
                    config['batch_size'] = int(m.group(6))

            # Parse batch size from hparams logging, phase lines, or embedded source
            m = re.match(r'batch_size[=:](\d+)', line)
            if m and 'batch_size' not in config:
                config['batch_size'] = int(m.group(1))
            # Parse from embedded source: BATCH_SIZE=N in env or sweep runner output
            m = re.search(r'BATCH_SIZE=(\d+)', line)
            if m and 'batch_size' not in config:
                config['batch_size'] = int(m.group(1))

            # Parse total params
            m = re.search(r'Total parameters: ([\d,]+)', line)
            if m:
                config['params'] = m.group(1)

            # Parse val loss lines: step:N/M val_loss:X.XXXX train_time:NNNms step_avg:NNms
            m = re.match(r'step:(\d+)/(\d+) val_loss:([\d.]+) train_time:(\d+)ms(?: step_avg:([\d.]+)ms)?', line)
            if m:
                step = int(m.group(1))
                total = int(m.group(2))
                val_loss = float(m.group(3))
                train_time_ms = int(m.group(4))
                val_points.append((step, val_loss, train_time_ms))
                config['steps'] = total
                if m.group(5):
                    config['step_avg_ms'] = float(m.group(5))

            # Parse best val loss
            m = re.match(r'Best val_loss: ([\d.]+)', line)
            if m:
                config['best_val_loss'] = float(m.group(1))

    # Compute best_val_loss from val_points if not explicitly logged
    if 'best_val_loss' not in config and val_points:
        config['best_val_loss'] = min(p[1] for p in val_points)

    return config, val_points


def make_label(filepath, config):
    """Create a short label from config."""
    steps = config.get('steps', '?')
    best = config.get('best_val_loss', '?')
    if isinstance(best, float):
        best = f"{best:.4f}"
    bs = config.get('batch_size')
    bs_str = f"bs={bs//1024}k" if bs else "bs=?"
    layers = config.get('layers', '?')
    return f"{layers}L {steps}steps {bs_str} (val={best})"


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "logs/dev/pretrain-teacher/*.txt"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for pattern: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} log files")

    # Parse all logs
    runs = []
    for f in files:
        config, val_points = parse_log(f)
        if val_points:
            runs.append((f, config, val_points))
            print(f"  {Path(f).name}: {config.get('steps', '?')} steps, "
                  f"best={config.get('best_val_loss', '?')}, "
                  f"{len(val_points)} val points")

    if not runs:
        print("No valid runs found")
        sys.exit(1)

    # Sort by best val loss for legend ordering
    runs.sort(key=lambda r: r[1].get('best_val_loss', 99))

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Teacher Pretraining Sweep ({len(runs)} runs)', fontsize=14, fontweight='bold')

    colors = plt.cm.tab10.colors

    for idx, (filepath, config, val_points) in enumerate(runs):
        color = colors[idx % len(colors)]
        label = make_label(filepath, config)
        steps = [p[0] for p in val_points]
        losses = [p[1] for p in val_points]
        times_s = [p[2] / 1000 for p in val_points]  # ms → seconds

        # Skip step 0 for cleaner plots (initial loss ~10.8 compresses the scale)
        s, l, t = steps[1:], losses[1:], times_s[1:]

        # (0,0) Val loss vs step
        axes[0, 0].plot(s, l, '-o', color=color, label=label, markersize=3, linewidth=1.5)

        # (0,1) Val loss vs wall time
        axes[0, 1].plot(t, l, '-o', color=color, label=label, markersize=3, linewidth=1.5)

        # (1,0) Val loss vs total tokens seen
        if config.get('steps'):
            # Infer batch size from timing: total_time / steps ≈ step_time
            # But we can also compute tokens from step * batch_size
            # Since we may not have batch_size logged, estimate from step avg
            last_time = val_points[-1][2]  # ms
            last_step = val_points[-1][0]
            step_avg_ms = last_time / max(last_step, 1)
            bs = config.get('batch_size', 131072)
            tokens = [p[0] * bs for p in val_points[1:]]
            axes[1, 0].plot(tokens, l, '-o', color=color, label=label, markersize=3, linewidth=1.5)

        # (1,1) Step avg time over training (efficiency)
        step_avgs = []
        for p in val_points[1:]:
            if p[0] > 0:
                step_avgs.append(p[2] / p[0])  # train_time_ms / step
        if step_avgs:
            axes[1, 1].plot(s, step_avgs, '-o', color=color, label=label, markersize=3, linewidth=1.5)

    # Format axes
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Val Loss')
    axes[0, 0].set_title('Val Loss vs Step')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc='upper right')

    axes[0, 1].set_xlabel('Wall Time (s)')
    axes[0, 1].set_ylabel('Val Loss')
    axes[0, 1].set_title('Val Loss vs Wall Time')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Tokens Seen (approx)')
    axes[1, 0].set_ylabel('Val Loss')
    axes[1, 0].set_title('Val Loss vs Tokens')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Avg Step Time (ms)')
    axes[1, 1].set_title('Step Efficiency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = 'sweep_teacher_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")


if __name__ == '__main__':
    main()
