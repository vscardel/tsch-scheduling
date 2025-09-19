
from __future__ import print_function

# =========================== imports =========================================

# standard
from builtins import range
import os
import argparse
import json
import time
import glob
from collections import OrderedDict
import numpy as np
from runExperiments import load_kpis, compute_score

from skopt import gp_minimize
from skopt.plots import plot_convergence


# third party
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    # parse options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inputfolder',
        help       = 'The simulation result folder.',
        default    = 'simData',
    )

    return parser.parse_args()

if __name__ == '__main__':

    options = parse_args()
    curr_output_folder_path = os.path.join(
        'simData',
        options.inputfolder,
        'exec_numMotes_{0}'.format(50)
    )
    kpis_loaded = False
    try:
        kpis = load_kpis(curr_output_folder_path, 50)
        kpis_loaded = True
    except Exception as e:
        ret = os.system('python2 compute_kpis.py --subfolder {0}'.format(curr_output_folder_path))
        time.sleep(5)
    if not kpis_loaded:
        kpis = load_kpis(curr_output_folder_path, 50)
    score = compute_score(kpis)
    print(score)

