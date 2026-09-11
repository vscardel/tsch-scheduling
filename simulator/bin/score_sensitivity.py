"""Does the ranking of the schedulers depend on the constants in the score?

Reviewer 4.4: the aggregate score has hand-picked thresholds and smoothness
constants, and the rankings depend on it, so show how they change when the
constants do. This recomputes the score of every run under a family of
variants of those constants and reports, for each variant, the ranking of the
schedulers by mean score, and how often the baseline ranking survives.

The variants move one constant at a time: each threshold to 0.5, 0.75, 1.25
and 1.5 times its value, each smoothness to half and double, each weight to
half and double. One at a time, because that is what a reader can interpret,
and because the question is whether any single constant carries the ranking.

  python score_sensitivity.py --inputfolder simData --motes 50 \
      --arms dynq qstatic rlsf msf emsf
"""
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os

import numpy as np

import score_model
from compare_schedulers import load_runs


THRESHOLD_FACTORS = (0.5, 0.75, 1.25, 1.5)
SMOOTHNESS_FACTORS = (0.5, 2.0)
WEIGHT_FACTORS = (0.5, 2.0)


def variants():
    """(name, weights, thresholds, smoothness), the baseline first."""
    W, T, K = score_model.WEIGHTS, score_model.THRESHOLDS, score_model.SMOOTHNESS
    yield 'baseline', dict(W), dict(T), dict(K)
    for metric in sorted(T):
        for f in THRESHOLD_FACTORS:
            t = dict(T); t[metric] = T[metric] * f
            yield 'threshold %s x%.2f' % (metric, f), dict(W), t, dict(K)
    for metric in sorted(K):
        for f in SMOOTHNESS_FACTORS:
            k = dict(K); k[metric] = K[metric] * f
            yield 'smoothness %s x%.1f' % (metric, f), dict(W), dict(T), k
    for metric in sorted(W):
        for f in WEIGHT_FACTORS:
            w = dict(W); w[metric] = W[metric] * f
            yield 'weight %s x%.1f' % (metric, f), w, dict(T), dict(K)


def mean_scores(runs_by_arm, weights, thresholds, smoothness):
    return {
        arm: np.mean([
            score_model.run_score(run, weights, thresholds, smoothness)
            for run in runs.values()
        ])
        for arm, runs in runs_by_arm.items()
    }


def ranking(scores):
    """Arms from best (lowest score) to worst."""
    return tuple(sorted(scores, key=scores.get))


def kendall_tau(a, b):
    """Rank correlation of two orderings of the same items, in [-1, 1]."""
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    concordant = discordant = 0
    for x, y in itertools.combinations(a, 2):
        s = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
        if s > 0:
            concordant += 1
        elif s < 0:
            discordant += 1
    pares = concordant + discordant
    return (concordant - discordant) / pares if pares else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', required=True,
                        help='folder holding <arm>_n<motes>/exec_numMotes_<motes>')
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--arms', nargs='+', required=True)
    args = parser.parse_args()

    runs_by_arm = {}
    for arm in args.arms:
        pasta = os.path.join(args.inputfolder, '%s_n%d' % (arm, args.motes),
                             'exec_numMotes_%d' % args.motes)
        runs_by_arm[arm] = load_runs(pasta)
        if not runs_by_arm[arm]:
            raise ValueError('no runs under %s' % pasta)

    resultados = []
    for nome, w, t, k in variants():
        scores = mean_scores(runs_by_arm, w, t, k)
        resultados.append((nome, scores, ranking(scores)))
    base = resultados[0][2]

    print('%-28s %-38s %6s  %s' % ('variant', 'ranking (best first)', 'tau',
                                    ' '.join('%8s' % a for a in base)))
    for nome, scores, ordem in resultados:
        print('%-28s %-38s %6.2f  %s' % (
            nome, ' > '.join(ordem), kendall_tau(base, ordem),
            ' '.join('%8.4f' % scores[a] for a in base)))

    outros = resultados[1:]
    identicos = sum(1 for _, _, o in outros if o == base)
    mesmo_topo = sum(1 for _, _, o in outros if o[0] == base[0])
    taus = [kendall_tau(base, o) for _, _, o in outros]
    print()
    print('baseline ranking: %s' % ' > '.join(base))
    print('%d variants besides the baseline' % len(outros))
    print('  full ranking preserved in %d (%.0f%%)'
          % (identicos, 100.0 * identicos / len(outros)))
    print('  best scheduler preserved in %d (%.0f%%)'
          % (mesmo_topo, 100.0 * mesmo_topo / len(outros)))
    print('  Kendall tau against the baseline: min %.2f, median %.2f'
          % (min(taus), float(np.median(taus))))
    print()
    print('pairs that ever swap:')
    for x, y in itertools.combinations(base, 2):
        trocas = [nome for nome, _, o in outros
                  if o.index(x) > o.index(y)]
        if trocas:
            print('  %s below %s in %d variants: %s'
                  % (x, y, len(trocas), '; '.join(trocas)))


if __name__ == '__main__':
    main()
