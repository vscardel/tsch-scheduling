"""The learning figure the reviewers asked for, both learners on one page.

Reviewer 3.5 asked for a convergence criterion and the reward curve averaged
over the runs, for both proposals. Reviewer 1.8 asked whether the table had
settled or was still in a transient. The answer is two panels per learner,
read from the JSON that learning_report.py writes with --out:

  left   the reward per decision, mean over runs with the spread across runs
  right  how often a decision changed the greedy action of some row, which is
         the policy still moving

Each learner is drawn against its own random control on the same seeds, so a
rise the control shows too is the environment and not learning. Dotted lines
mark the disturbances.

  python learning_panels.py --dynq dynq/relatorio.json \
      --qstatic qstatic/relatorio.json --disturbances 0.40 0.70 --out fig.pdf
"""
from __future__ import print_function

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import figures


PANELS = [
    ('reward', 'Reward per decision'),
    ('churn',  'Policy changes per decision'),
]


def arms_of(report):
    """(learned, control) labels, the control being the one named so."""
    labels = sorted(report)
    controle = [l for l in labels if 'aleatorio' in l]
    aprendido = [l for l in labels if l not in controle]
    if len(controle) != 1 or len(aprendido) != 1:
        raise ValueError('expected one learned arm and one control, got %r'
                         % labels)
    return aprendido[0], controle[0]


def draw(ax, report, chave, marcos):
    aprendido, controle = arms_of(report)
    colours, _ = figures.styles_for(2)
    for label, nome, colour in (
        (aprendido, 'learned policy', colours[0]),
        (controle,  'random control', colours[1]),
    ):
        curva = report[label]['curves'][chave]
        media = np.array(curva['mean'], dtype=float)
        x = (np.arange(len(media)) + 0.5) / len(media)
        ok = np.isfinite(media)
        ax.plot(x[ok], media[ok], color=colour, linewidth=2, label=nome)
        baixo = np.array(curva['low'], dtype=float)
        alto = np.array(curva['high'], dtype=float)
        ax.fill_between(x[ok], baixo[ok], alto[ok], color=colour, alpha=0.2,
                        linewidth=0)
    for m in marcos:
        ax.axvline(m, color='black', linestyle=':', linewidth=1)
    ax.set_xlim(0, 1)
    ax.grid(True, linewidth=0.5, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynq', required=True)
    parser.add_argument('--qstatic', required=True)
    parser.add_argument('--disturbances', type=float, nargs='*', default=[])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    figures.use_paper_style()
    reports = [('DynQ', json.load(open(args.dynq))),
               ('Q-static', json.load(open(args.qstatic)))]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex='col')
    for linha, (nome, report) in enumerate(reports):
        for coluna, (chave, ylabel) in enumerate(PANELS):
            ax = axes[linha][coluna]
            draw(ax, report, chave, args.disturbances)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(nome, loc='left', fontweight='bold', pad=6)
            if linha == 1:
                ax.set_xlabel('Fraction of the run')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=2.0, w_pad=2.0)
    fig.savefig(args.out)
    print('escrito em', args.out)


if __name__ == '__main__':
    main()
