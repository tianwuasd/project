"""Plot the saved, single-seed development diagnostics without new evaluation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
pilot = ROOT/'results/risk_pilot_001'
scales = json.loads((ROOT/'results/value_scale_001/experiment.json').read_text())['summary']
baseline = json.loads((pilot/'experiment.json').read_text())['validation_summary']['conditional_lp']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for variant, label, color in [('pattern', 'Pattern + risk context', '#176b91'),
                              ('risk_insertion', 'Risk-insertion features', '#b45a36')]:
    log = json.loads((pilot/f'{variant}_11.json').read_text())['log']
    axes[0].plot([r['episode'] for r in log], [r['validation_mean'] for r in log],
                 marker='o', label=label, color=color)
    values = scales[variant]['scale_means']
    axes[1].plot([float(v) for v in values], list(values.values()), marker='o', color=color, label=label)
for ax in axes:
    ax.axhline(baseline, color='#464c50', linestyle='--', linewidth=1.3, label='Conditional LP')
    ax.set_ylim(74.5, 84)
    ax.set_ylabel('Mean validation saving (proxy units/day)')
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_xlabel('Training episodes')
axes[0].set_title('More training did not improve the selected policy', fontsize=10)
axes[1].set_xlabel('Frozen continuation-value multiplier')
axes[1].set_title('Post hoc scaling did not close the baseline gap', fontsize=10)
axes[1].legend(loc='lower right', fontsize=8, frameon=False)
fig.suptitle('One-seed development diagnostics; no independent confirmation', fontsize=11)
fig.savefig(ROOT/'diagnostics_03.png', dpi=180)
plt.close(fig)
