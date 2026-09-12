"""How much of the score each factor of the 2^3 design accounts for.

Replaces figs/factor_contributions.pdf, whose numbers were written into
generate_plots.py by hand. Reads the JSON that factorial_anova.py writes and
draws one bar per effect, the share of the effects' sum of squares, with the
significant ones marked.

  python factorial_figure.py --anova anova.json --out contributions.pdf
"""
from __future__ import division
from __future__ import print_function

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anova', required=True)
    parser.add_argument('--metric', default='score')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    with open(args.anova) as f:
        dados = json.load(f)[args.metric]
    efeitos = dados['effects']

    figures.use_paper_style()
    cor = figures.PALETTE[0]
    rotulos = [e['effect'] for e in efeitos]
    alturas = [100 * e['share_of_effects'] for e in efeitos]

    fig, ax = plt.subplots()
    pos = np.arange(len(efeitos))
    barras = ax.bar(pos, alturas, color=cor, edgecolor='black', linewidth=1.2)
    for barra, efeito in zip(barras, efeitos):
        if efeito['p'] < 0.05:
            ax.text(barra.get_x() + barra.get_width() / 2,
                    barra.get_height() + 1, '*', ha='center', fontsize=12)
    ax.set_xticks(pos)
    ax.set_xticklabels(rotulos)
    ax.set_ylabel('Share of the explained variation (%)')
    ax.set_xlabel('Factor (T = traffic, Q = queue, C = charge)')
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    for e in efeitos:
        print('%-8s %5.1f%%  p = %.4f%s' % (
            e['effect'], 100 * e['share_of_effects'], e['p'],
            ' *' if e['p'] < 0.05 else ''))
    print('escrito em', args.out)


if __name__ == '__main__':
    main()
