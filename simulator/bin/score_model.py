"""The aggregate score of one run, in one place.

The score drives the hyperparameter optimisation and the factorial analysis,
and it was written out three times: in runExperiments, which computes it while
running, and in the two analysis tools. Three copies of a formula a reviewer
has already objected to is three chances for them to drift apart.

It is a weighted sum of four measured quantities, each passed through a
sigmoid around a threshold, so a value on the wrong side of its threshold
costs its weight and a value on the right side costs nothing. Lower is better.
The constants are the published ones.
"""
from __future__ import division

import math

WEIGHTS = {
    'latency': 0.25, 'pdr': 0.25, 'lifetime': 0.25, 'join_time': 0.25
}
THRESHOLDS = {
    'latency': 1.5, 'pdr': 0.95, 'lifetime': 1.0, 'join_time': 1000.0
}
SMOOTHNESS = {
    'latency': 1.0, 'pdr': 0.2, 'lifetime': 1.5, 'join_time': 0.004
}
# whether the score punishes being above the threshold or below it
ABOVE_IS_BAD = {
    'latency': True, 'join_time': True, 'pdr': False, 'lifetime': False
}


def sigmoid(x, threshold, k):
    return 1.0 / (1.0 + math.exp(-k * (x - threshold)))


def run_lifetime(run):
    """Mean battery lifetime over the motes of one run, in years.

    A mote whose lifetime could not be estimated reports a string rather than
    a number, and counts as zero.
    """
    lifetimes = []
    for mote, mote_kpis in run.items():
        if mote == 'global-stats':
            continue
        lifetime = mote_kpis.get('lifetime_AA_years')
        if not isinstance(lifetime, (int, float)):
            lifetime = 0
        lifetimes.append(lifetime)
    if not lifetimes:
        return 0.0
    return sum(lifetimes) / float(len(lifetimes))


def run_metrics(run):
    """The four quantities the score is built from, for one run."""
    g = run['global-stats']
    return {
        'latency': g['e2e-upstream-latency'][0]['mean'],
        'pdr': g['e2e-upstream-delivery'][0]['value'],
        'join_time': g['joining-time'][0]['mean'] / 100.0,
        'lifetime': run_lifetime(run),
    }


def score(metrics, weights=None, thresholds=None, smoothness=None):
    weights = WEIGHTS if weights is None else weights
    thresholds = THRESHOLDS if thresholds is None else thresholds
    smoothness = SMOOTHNESS if smoothness is None else smoothness
    total = 0.0
    for nome, valor in metrics.items():
        acima = sigmoid(valor, thresholds[nome], smoothness[nome])
        total += weights[nome] * (acima if ABOVE_IS_BAD[nome] else 1.0 - acima)
    return total


def run_score(run, weights=None, thresholds=None, smoothness=None):
    return score(run_metrics(run), weights, thresholds, smoothness)
