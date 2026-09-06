"""Pairwise comparison of scheduling functions, with tests and effect sizes.

Two reviewers asked for the same thing: the comparative claims are not backed
by statistics, and with ten runs the power to separate DynQ from Q-static is
limited. This computes what they asked for, from the .kpi files a run leaves
behind.

The comparison is paired, and that is not a detail. Each run seeds the
simulator with exec_randomSeed + run_id, so run 3 of DynQ and run 3 of MSF see
the same topology and the same link qualities. Pairing on the run index removes
the variance between topologies, which is the largest source of spread in these
numbers, and buys far more power than the same number of unpaired runs would.

Reports, per metric and per pair of schedulers:

  - the median of the paired differences, and a bootstrap confidence interval
    for it, which is the interval a reader wants and the manuscript lacks
  - the Wilcoxon signed-rank p value, Holm-corrected across the pairs of a
    metric, since every pair is being tested against the same data
  - the matched-pairs rank-biserial correlation as the effect size, so a
    difference that is real but tiny reads as tiny

Usage:
    python compare_schedulers.py --inputfolder simData --motes 50 \\
        --schedulers Qlearning QlearningSBRC24 RLSF MSF EMSF
"""
from __future__ import division
from __future__ import print_function

import argparse
import glob
import itertools
import json
import math
import os
import random

import numpy as np
from scipy.stats import wilcoxon

from runExperiments import compute_run_lifetime


# metric name -> (how to read it from one run, whether more is better)
def _latency(run):
    return run['global-stats']['e2e-upstream-latency'][0]['mean']


def _pdr(run):
    return run['global-stats']['e2e-upstream-delivery'][0]['value']


def _join(run):
    return run['global-stats']['joining-time'][0]['mean'] / 100.0


def _lifetime_min(run):
    return run['global-stats']['network_lifetime'][0]['min']


def _lifetime_mean(run):
    return compute_run_lifetime(run)


def _sixp_per_packet(run):
    overhead = run['global-stats'].get('sixp-overhead')
    if not overhead:
        return None
    value = overhead[0].get('per_app_packet')
    return None if value == 'N/A' else value


def _sixp_transactions(run):
    overhead = run['global-stats'].get('sixp-overhead')
    return overhead[1]['total'] if overhead else None


METRICS = [
    ('latency',          _latency,          'lower'),
    ('pdr',              _pdr,              'higher'),
    ('join_time',        _join,             'lower'),
    ('lifetime_min',     _lifetime_min,     'higher'),
    ('lifetime_mean',    _lifetime_mean,    'higher'),
    ('sixp_per_packet',  _sixp_per_packet,  'lower'),
    ('sixp_transactions', _sixp_transactions, 'lower'),
]


def load_runs(folder):
    """Every .kpi in the folder, merged and keyed by run id.

    load_kpis elsewhere opens only output_cpu0.dat.kpi, which drops every run
    that landed on another core. This reads all of them.
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(folder, '*.kpi'))):
        with open(path) as f:
            for run_id, run in json.load(f).items():
                runs[int(run_id)] = run
    return runs


def paired_series(runs_a, runs_b, reader):
    """The metric for the runs both schedulers have, in run order."""
    shared = sorted(set(runs_a) & set(runs_b))
    a, b = [], []
    for run_id in shared:
        va, vb = reader(runs_a[run_id]), reader(runs_b[run_id])
        if va is None or vb is None:
            continue
        a.append(va)
        b.append(vb)
    return a, b


def bootstrap_ci(differences, confidence=0.95, resamples=10000, seed=1):
    """Percentile interval for the median difference."""
    if not differences:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    n = len(differences)
    medians = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        medians.append(np.median(sample))
    low = (1 - confidence) / 2 * 100
    return (np.percentile(medians, low), np.percentile(medians, 100 - low))


def rank_biserial(differences):
    """Matched-pairs rank-biserial correlation, on [-1, 1].

    The share of the signed rank mass that falls on one side. Zero means the
    two schedulers win as often and by as much as each other.
    """
    nonzero = [d for d in differences if d != 0]
    if not nonzero:
        return 0.0
    ranks = _ranks([abs(d) for d in nonzero])
    total = sum(ranks)
    positive = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    negative = total - positive
    return (positive - negative) / total


def _ranks(values):
    """Ranks with ties averaged, which is what the signed-rank test uses."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        media = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = media
        i = j + 1
    return ranks


def holm(pvalues):
    """Holm-Bonferroni, which controls the family-wise error without
    throwing away as much power as plain Bonferroni."""
    indexed = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    n = len(pvalues)
    adjusted = [0.0] * n
    running = 0.0
    for position, i in enumerate(indexed):
        value = (n - position) * pvalues[i]
        running = max(running, min(1.0, value))
        adjusted[i] = running
    return adjusted


def compare(runs_by_scheduler, metric_reader, direction):
    """Every pair of schedulers on one metric."""
    names = sorted(runs_by_scheduler)
    rows = []
    for a, b in itertools.combinations(names, 2):
        serie_a, serie_b = paired_series(
            runs_by_scheduler[a], runs_by_scheduler[b], metric_reader
        )
        if len(serie_a) < 3:
            continue
        differences = [x - y for x, y in zip(serie_a, serie_b)]
        try:
            _, p = wilcoxon(serie_a, serie_b)
        except ValueError:
            # every difference is zero, so there is nothing to test
            p = 1.0
        rows.append({
            'a': a, 'b': b, 'n': len(differences),
            'median_a': float(np.median(serie_a)),
            'median_b': float(np.median(serie_b)),
            'median_diff': float(np.median(differences)),
            'ci': bootstrap_ci(differences),
            'p': p,
            'effect': rank_biserial(differences),
            'better': _who_wins(np.median(differences), direction, a, b),
        })
    for row, adjusted in zip(rows, holm([r['p'] for r in rows])):
        row['p_holm'] = adjusted
    return rows


def _who_wins(median_diff, direction, a, b):
    if median_diff == 0:
        return 'tie'
    a_is_larger = median_diff > 0
    if direction == 'higher':
        return a if a_is_larger else b
    return b if a_is_larger else a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', default='simData')
    parser.add_argument('--motes', type=int, required=True)
    parser.add_argument('--schedulers', nargs='+', required=True)
    parser.add_argument('--prefix', default='',
                        help='folder name prefix, e.g. n50_ for n50_MSF')
    parser.add_argument('--out', default=None, help='write the report as JSON')
    args = parser.parse_args()

    runs_by_scheduler = {}
    for sf in args.schedulers:
        folder = os.path.join(
            args.inputfolder,
            '{0}{1}'.format(args.prefix, sf),
            'exec_numMotes_{0}'.format(args.motes)
        )
        runs = load_runs(folder)
        if not runs:
            print('no .kpi under {0}, skipping {1}'.format(folder, sf))
            continue
        runs_by_scheduler[sf] = runs
        print('{0}: {1} runs'.format(sf, len(runs)))

    if len(runs_by_scheduler) < 2:
        raise SystemExit('need at least two schedulers with results')

    report = {}
    for name, reader, direction in METRICS:
        rows = compare(runs_by_scheduler, reader, direction)
        if not rows:
            continue
        report[name] = rows
        print('')
        print('=== {0} ({1} is better) ==='.format(name, direction))
        print('%-22s %4s %11s %11s %13s %22s %9s %8s' % (
            'pair', 'n', 'median A', 'median B', 'median diff',
            '95% CI of diff', 'p (Holm)', 'effect'))
        for r in rows:
            marca = ' *' if r['p_holm'] < 0.05 else ''
            print('%-22s %4d %11.4f %11.4f %13.4f  [%9.4f,%9.4f] %9.4f %8.3f%s' % (
                '{0} vs {1}'.format(r['a'], r['b']), r['n'],
                r['median_a'], r['median_b'], r['median_diff'],
                r['ci'][0], r['ci'][1], r['p_holm'], r['effect'], marca))

    print('')
    print('* Holm-corrected p below 0.05. The effect is the matched-pairs')
    print('  rank-biserial correlation: near zero means the two win as often')
    print('  and by as much as each other, whatever the p value says.')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2, default=float)
        print('written to {0}'.format(args.out))


if __name__ == '__main__':
    main()
