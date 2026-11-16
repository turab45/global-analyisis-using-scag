"""Audit saved SENCoG results and plot influence against concept association."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(directory):
    output = directory / 'followup_audit'
    output.mkdir(exist_ok=True)
    selections = pd.read_csv(directory / 'ablation_selections_test.csv')
    summary = selections.groupby(['concept', 'budget', 'strategy'], as_index=False).agg(test_iou=('heldout_mean_concept_iou', 'mean'), kl=('kl', 'mean'), train_iou=('mean_semantic_score', 'mean'), strength=('mean_strength', 'mean'), activation=('mean_activation', 'mean'))
    assert np.isfinite(summary[['test_iou', 'kl']]).all().all()
    summary.to_csv(output / 'influence_specificity.csv', index=False)
    concepts = sorted(summary.concept.unique())
    budgets = sorted(summary.budget.unique())
    colors = {'functional': '#e58b39', 'joint': '#148f80', 'semantic': '#8c7adc', 'random': '#89929b'}
    fig, axes = plt.subplots(len(concepts), len(budgets), figsize=(15, 11), squeeze=False, layout='constrained')
    for i, concept in enumerate(concepts):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            for strategy, color in colors.items():
                row = summary[(summary.concept == concept) & (summary.budget == budget) & (summary.strategy == strategy)].iloc[0]
                ax.scatter(row.test_iou, row.kl, color=color, s=65, label=strategy)
                ax.annotate(strategy, (row.test_iou, row.kl), xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax.set(title=f'{concept} | {budget:.0%} channels', xlabel='Mean selected-channel test concept IoU', ylabel='Mean prediction KL')
            ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
            ax.margins(.25)
            ax.grid(alpha=.2)
    fig.suptitle('Observed influence versus concept association — ACDC ResNet18 final stage\nRandom points average control draws; IoU measures overlap, not exclusivity or causal faithfulness')
    for extension in ['png', 'pdf']:
        fig.savefig(output / f'influence_specificity.{extension}', dpi=180)
    plt.close(fig)
    coverage = []
    with np.load(directory / 'representations.npz') as archive:
        for split in ['train', 'val', 'test']:
            S = archive[f'S_{split}']
            for gamma in [.005, .01, .015, .02]:
                coverage.append({'split': split, 'gamma': gamma, 'labelled_channels': int((S.max(axis=1) >= gamma).sum()), 'total_channels': len(S), 'mean_max_iou': S.max(axis=1).mean()})
    pd.DataFrame(coverage).to_csv(output / 'coverage_sensitivity.csv', index=False)
    matched = selections.query("strategy == 'matched'").groupby(['concept', 'budget'], as_index=False).agg(valid_fraction=('match_valid', 'mean'), max_gap=('match_max_percentile_gap', 'max'), mean_gap=('match_max_percentile_gap', 'mean'), overlap=('match_overlap_fraction', 'mean'))
    matched.to_csv(output / 'matching_diagnostics.csv', index=False)
    lines = ['# Saved-result follow-up audit', '', 'These are observed results; no inference, retraining or threshold fitting was performed.', '', '## Influence versus concept association', '', 'IoU is concept overlap, not exclusivity to that concept. Random points average sampled controls. The plot is descriptive; it has no horizontal patient uncertainty intervals.', '']
    for concept in concepts:
        group = summary[(summary.concept == concept) & (summary.budget == max(budgets))].set_index('strategy')
        joint, functional, semantic = group.loc['joint'], group.loc['functional'], group.loc['semantic']
        lines.append(f'- {concept}, {max(budgets):.0%}: joint IoU={joint.test_iou:.5f}, KL={joint.kl:.5f}; functional IoU={functional.test_iou:.5f}, KL={functional.kl:.5f}; semantic IoU={semantic.test_iou:.5f}, KL={semantic.kl:.5f}.')
    lines += ['', '## Coverage sensitivity', '', '```text', pd.DataFrame(coverage).to_string(index=False), '```', '', '## Matching quality', '', 'Gap is the largest per-channel covariate percentile difference. The prespecified caliper is 0.10. Invalid matches do not support a controlled causal comparison; do not relax the caliper to manufacture a win.', '', '```text', matched.to_string(index=False), '```', '', 'Coverage differences alone do not identify preprocessing error or distribution shift. Inspect the existing train/val/test concept overlays and patient-level IoUs before choosing a cause. Further architectures and budget changes remain unrun.']
    (output / 'AUDIT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:10]))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layer_directory', type=Path)
    analyze(parser.parse_args().layer_directory)
