"""Locating the results of the run that just finished.

runSim.py runs compute_kpis.py and plot.py as post-simulation actions, with no
arguments, so both have to find the results on their own. This lives on its own
rather than in compute_kpis.py so that importing it does not drag in SimEngine.
"""
from __future__ import absolute_import

import os


def latest_subfolder(root='simData'):
    """The results directory with the most recent mtime.

    A run writes into simData/<log_directory>/exec_numMotes_N/, so this
    descends one level when it can.
    """
    candidates = [os.path.join(root, name) for name in os.listdir(root)]
    candidates = [path for path in candidates if os.path.isdir(path)]
    if not candidates:
        raise RuntimeError('no results directory found under {0}'.format(root))

    newest = max(candidates, key=os.path.getmtime)

    combinations = [os.path.join(newest, name) for name in os.listdir(newest)]
    combinations = [path for path in combinations if os.path.isdir(path)]
    if combinations:
        return max(combinations, key=os.path.getmtime)
    return newest
