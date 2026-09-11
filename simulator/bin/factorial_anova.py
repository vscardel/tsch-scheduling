"""The 2^3 factorial ANOVA, on the score and on each metric, from the .kpi files.

The published Table 2k came from compute_effects.py, which read a 'score'
list per cell from final_results.json and ran MSF in the empty cell, so every
main effect compared a state factor against a different scheduling function.
This keeps its arithmetic, the Yates contrasts of a 2^k design with equal
replication, and changes what goes in: the runs are read from the .kpi files
(every CPU, not only cpu0), the response is recomputed per run through
score_model so the constants are the published ones in one place, and the
empty cell is the stateless learner unless asked otherwise.

For each of the seven effects (three main, three two-way, one three-way):
the effect estimate and its sign, the sum of squares, its share of the
effects' sum of squares (what the published table reports as contribution),
its share of the total including error, and the p-value of the F test with
one and 8(n-1) degrees of freedom.

  python factorial_anova.py --inputfolder simData --motes 50 --latex table.tex
"""
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import json
import os

import numpy as np
from scipy import stats

import score_model
from compare_schedulers import METRICS, load_runs
from factorial_per_metric import FACTORS, EMPTY_CELL, EMPTY_CELL_MSF, cells, \
    cell_name, read_cell


SHORT = {'traffic': 'T', 'queue': 'Q', 'charge': 'C'}


def effects_of(factors):
    """Every non-empty subset of the factors, main effects first."""
    saida = []
    for k in range(1, len(factors) + 1):
        for subset in itertools.combinations(factors, k):
            saida.append(subset)
    return saida


def sign(effect, present):
    s = 1
    for f in effect:
        s *= 1 if f in present else -1
    return s


def anova(values_by_cell, vazia):
    """values_by_cell: {cell name: list of per-run responses}, equal lengths.

    Returns (rows, error) where rows is a list of dicts, one per effect, and
    error carries the residual sum of squares and its degrees of freedom.
    """
    n = {len(v) for v in values_by_cell.values()}
    if len(n) != 1:
        raise ValueError('unequal replication across cells: %r'
                         % {k: len(v) for k, v in values_by_cell.items()})
    n = n.pop()
    k = len(FACTORS)
    todas = [x for v in values_by_cell.values() for x in v]
    media = np.mean(todas)
    ss_total = float(sum((x - media) ** 2 for x in todas))

    rows = []
    for effect in effects_of(FACTORS):
        contrast = 0.0
        for nome, present in cells(vazia):
            contrast += sign(effect, present) * sum(values_by_cell[nome])
        estimate = contrast / (2 ** (k - 1) * n)
        ss = contrast ** 2 / (2 ** k * n)
        rows.append({'effect': effect, 'estimate': estimate, 'ss': ss})

    ss_effects = sum(r['ss'] for r in rows)
    ss_error = ss_total - ss_effects
    df_error = 2 ** k * (n - 1)
    ms_error = ss_error / df_error if df_error else float('nan')
    for r in rows:
        r['share_of_effects'] = r['ss'] / ss_effects if ss_effects else 0.0
        r['share_of_total'] = r['ss'] / ss_total if ss_total else 0.0
        f = r['ss'] / ms_error if ms_error else float('nan')
        r['F'] = f
        r['p'] = float(1 - stats.f.cdf(f, 1, df_error)) if df_error else float('nan')
    error = {'ss': ss_error, 'df': df_error,
             'share_of_total': ss_error / ss_total if ss_total else 0.0,
             'replicas': n}
    return rows, error


def label(effect):
    return '/'.join(SHORT[f] for f in effect)


def print_table(nome, direction, rows, error):
    print('')
    print('=== %s (%s is better; %d runs per cell) ===' % (nome, direction, error['replicas']))
    print('  %-8s %+10s %8s %8s %8s   %s' % ('effect', 'estimate', '%effects', '%total', 'p', ''))
    for r in rows:
        sig = '*' if r['p'] < 0.05 else ''
        print('  %-8s %+10.4f %8.1f %8.1f %8.4f %s' % (
            label(r['effect']), r['estimate'], 100 * r['share_of_effects'],
            100 * r['share_of_total'], r['p'], sig))
    print('  %-8s %10s %8s %8.1f' % ('error', '', '', 100 * error['share_of_total']))


def latex_rows(rows, direction):
    """Rows for the paper's Table 2k: sign, contribution, p-value.

    The sign column says whether the factor moves the response towards the
    better end, so it reads the same way for a lower-is-better score and a
    higher-is-better delivery ratio.
    """
    linhas = []
    for r in rows:
        melhora = (r['estimate'] < 0) if direction == 'lower' else (r['estimate'] > 0)
        sinal = '$-$' if r['estimate'] < 0 else '$+$'
        p = '$< 0.05$' if r['p'] < 0.05 else '$> 0.05$'
        prefix = '\\rowcolor[HTML]{c2c2c0} ' if (melhora and r['p'] < 0.05) else ''
        linhas.append('%s%s & %s & %.1f & %s \\\\' % (
            prefix, label(r['effect']), sinal, 100 * r['share_of_effects'], p))
    return '\n'.join(linhas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', required=True)
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--empty-cell', default=EMPTY_CELL,
                        help='folder of the cell with no state factor; pass %s '
                             'to reproduce the published arithmetic' % EMPTY_CELL_MSF)
    parser.add_argument('--out', default=None, help='JSON of every table')
    parser.add_argument('--latex', default=None,
                        help='write the score table rows in the paper format')
    args = parser.parse_args()

    vazia = args.empty_cell
    if vazia == EMPTY_CELL_MSF:
        print('AVISO: a celula vazia e o MSF, entao cada efeito principal')
        print('       compara um fator de estado contra outro escalonador.')

    runs_by_cell = {}
    for nome, _ in cells(vazia):
        runs = read_cell(args.inputfolder, nome, args.motes)
        if not runs:
            raise SystemExit('sem resultados para a celula %s' % nome)
        runs_by_cell[nome] = runs

    responses = [('score', score_model.run_score, 'lower')] + [
        (m, reader, direction) for m, reader, direction in METRICS
        if m != 'score'
    ]
    resumo = {}
    for nome, reader, direction in responses:
        values = {}
        for cell, runs in runs_by_cell.items():
            v = [reader(r) for r in runs.values()]
            v = [x for x in v if x is not None]
            values[cell] = v
        if any(not v for v in values.values()):
            continue
        try:
            rows, error = anova(values, vazia)
        except ValueError as e:
            print('%s: %s' % (nome, e))
            continue
        print_table(nome, direction, rows, error)
        resumo[nome] = {
            'direction': direction,
            'cell_means': {c: float(np.mean(v)) for c, v in values.items()},
            'effects': [dict(r, effect=label(r['effect'])) for r in rows],
            'error': error,
        }
        if nome == 'score' and args.latex:
            with open(args.latex, 'w') as f:
                f.write(latex_rows(rows, direction) + '\n')
            print('  linhas LaTeX em %s' % args.latex)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(resumo, f, indent=2)
        print('')
        print('escrito em %s' % args.out)


if __name__ == '__main__':
    main()
