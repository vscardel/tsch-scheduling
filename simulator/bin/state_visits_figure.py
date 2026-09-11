"""How often the agent was in each state, for the reviewer who asked.

Reviewer 3.2 suspected the charge factor never changes, which would leave
DynQ with four effective states rather than eight. The visited-state counters
that state_visits.py records during the run answer that directly: one bar per
row of the Q-table, the share of all decisions that landed there, mean over
runs with its confidence interval.

The row number is the binary state read in the order the learner builds it,
which is what --order says. The default is the order the published code
produces (queue, charge, traffic).

  python state_visits_figure.py --inputfolder simData/dynq_aprendido_n50 \
      --out states.pdf
"""
from __future__ import print_function

import argparse
import collections
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


def per_run_shares(inputfolder, rows):
    """One list per row: the share of visits it got in each run."""
    por_run = collections.defaultdict(collections.Counter)
    for caminho in stats_files(inputfolder):
        run = [p for p in caminho.split(os.sep) if p.startswith('run_')]
        if not run:
            continue
        with open(caminho) as f:
            visitas = json.load(f).get('STATE_VISITS', {})
        for estado, n in visitas.items():
            por_run[run[-1]][int(estado)] += n
    if not por_run:
        raise ValueError('no qlearning_stats.json under %s' % inputfolder)
    shares = [[] for _ in range(rows)]
    for run, contagem in sorted(por_run.items()):
        total = float(sum(contagem.values()))
        for estado in range(rows):
            shares[estado].append(100.0 * contagem[estado] / total)
    return shares, len(por_run)


def label(estado, order):
    bits = [(estado >> (len(order) - 1 - i)) & 1 for i in range(len(order))]
    return '\n'.join('%s=%d' % (nome[0].upper(), b)
                     for nome, b in zip(order, bits))


def draw(shares, labels, path, confidence=0.95):
    """Eight rows of one table are categories, not series, so one colour."""
    figures.use_paper_style()
    colour = figures.PALETTE[0]
    medias = [np.mean(s) for s in shares]
    erros = []
    for serie, media in zip(shares, medias):
        baixo, _ = figures.confidence_interval(serie, confidence)
        erros.append(media - baixo if np.isfinite(baixo) else 0.0)
    fig, ax = plt.subplots()
    pos = np.arange(len(shares))
    ax.bar(pos, medias, yerr=erros, capsize=4, color=colour,
           edgecolor='black', linewidth=1.2)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Share of decisions (%)')
    ax.set_xlabel('State (Q = queue, C = charge, T = traffic)')
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', required=True)
    parser.add_argument('--order', nargs='+',
                        default=['queue', 'charge', 'traffic'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rows = 2 ** len(args.order)
    shares, runs = per_run_shares(args.inputfolder, rows)
    labels = [label(e, args.order) for e in range(rows)]
    for l, s in zip(labels, shares):
        print('%-18s %5.1f%%' % (l.replace('\n', ' '), sum(s) / len(s)))
    print('%d runs' % runs)
    draw(shares, labels, args.out)
    print('escrito em', args.out)


if __name__ == '__main__':
    main()
