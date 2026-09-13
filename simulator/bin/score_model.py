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
        # Not every key of a run is a mote. compare_schedulers attaches the
        # learning quantities under 'learning-stats', and counting that as a
        # mote with no lifetime put a zero into the mean: the average divided
        # by 51 instead of 50, which lowered the lifetime of every learner by
        # about 2% and raised its score. MSF and EMSF leave no trace, so they
        # kept theirs, and every comparison between a learner and one of them
        # was biased against the learner.
        #
        # A mote always carries lifetime_AA_years, as a number or, when it
        # could not be estimated, as a string that still counts as zero. An
        # entry without the key is not a mote.
        if not isinstance(mote_kpis, dict) or 'lifetime_AA_years' not in mote_kpis:
            continue
        lifetime = mote_kpis.get('lifetime_AA_years')
        if not isinstance(lifetime, (int, float)):
            lifetime = 0
        lifetimes.append(lifetime)
    if not lifetimes:
        return 0.0
    return sum(lifetimes) / float(len(lifetimes))


def as_number(value, divide_by=1.0):
    """The value as a float, or None when the simulator could not compute it.

    compute_kpis writes the string 'N/A' wherever a statistic has no sample:
    a run where nothing arrived has no latency and no delivery ratio, and on
    the Linear topology, where the far motes may never join, whole runs come
    out that way. Reading the string as a number scored a run on a value that
    does not exist, and in python 2 it did so silently, since a string
    compares greater than any float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value / divide_by


def run_metrics(run):
    """The four quantities the score is built from, for one run.

    A quantity the run does not have is None, and a run missing any of them
    cannot be scored.
    """
    g = run['global-stats']
    return {
        'latency': as_number(g['e2e-upstream-latency'][0]['mean']),
        'pdr': as_number(g['e2e-upstream-delivery'][0]['value']),
        'join_time': as_number(g['joining-time'][0]['mean'], 100.0),
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
    metrics = run_metrics(run)
    if any(v is None for v in metrics.values()):
        return None
    return score(metrics, weights, thresholds, smoothness)
