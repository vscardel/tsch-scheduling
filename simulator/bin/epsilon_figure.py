"""The exploration schedule of each learner, drawn from a real run.

Section 6.5 shows epsilon over a run for one node. The two learners decay
epsilon the same way but use it differently: DynQ keeps drawing random
actions with probability epsilon down to its floor, Q-static switches to
exploitation once and for all when epsilon crosses its threshold. One node
of each, on the ASN axis, with the threshold marked.

  python epsilon_figure.py --dynq simData/dynq_aprendido_n50 \
      --qstatic simData/qstatic_aprendido_n50 --threshold 0.3 --out eps.pdf
"""
from __future__ import print_function

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import figures


def stats_files(inputfolder):
    """Every qlearning_stats.json below the folder, python 2 and 3 alike."""
    for raiz, _, arquivos in os.walk(inputfolder):
        for nome in arquivos:
            if nome == 'qlearning_stats.json':
                yield os.path.join(raiz, nome)


def trajectory(inputfolder, slotframe_length=101):
    """(fraction of the run, epsilon) for the mote with most decisions."""
    melhor = None
    for caminho in stats_files(inputfolder):
        with open(caminho) as f:
            d = json.load(f)
        if not d.get('DECISIONS') or not d.get('EPSILON'):
            continue            # a run from before the trace existed
        if melhor is None or len(d['DECISIONS']) > len(melhor['DECISIONS']):
            melhor = d
    if melhor is None:
        raise ValueError('no decision trace under %s' % inputfolder)
    asn_por_passo = {int(k): v['asn'] for k, v in melhor['DECISIONS'].items()}
    pontos = sorted(
        (asn_por_passo[int(k)], v)
        for k, v in melhor['EPSILON'].items() if int(k) in asn_por_passo
    )
    x = np.array([p[0] for p in pontos], dtype=float)
    y = np.array([p[1] for p in pontos], dtype=float)
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dynq', required=True)
    parser.add_argument('--qstatic', required=True)
    parser.add_argument('--threshold', type=float, required=True,
                        help='the Q-static exploitation threshold')
    parser.add_argument('--slotframes', type=int, default=15000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    figures.use_paper_style()
    colours, _ = figures.styles_for(2)
    total = float(args.slotframes * 101)
    fig, ax = plt.subplots()
    for nome, pasta, colour in (('DynQ', args.dynq, colours[0]),
                                ('Q-static', args.qstatic, colours[1])):
        x, y = trajectory(pasta)
        ax.plot(x / total, y, color=colour[:3], linewidth=1.8, label=nome)
        print('%-9s %d decisions, epsilon from %.3f to %.3f'
              % (nome, len(x), y[0], y[-1]))
    ax.axhline(args.threshold, color='gray', linestyle='--', linewidth=1.2)
    ax.text(0.99, args.threshold + 0.02,
            r'Q-static threshold $\epsilon_{th}$', ha='right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Fraction of the run')
    ax.set_ylabel(r'$\epsilon$')
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(args.out)
    print('escrito em', args.out)


if __name__ == '__main__':
    main()
