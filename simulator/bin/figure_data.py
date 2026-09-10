# -*- coding: utf-8 -*-
"""The runs behind a figure, read the same way the statistics read them.

A figure and the number quoted beside it in the text have to come from the
same runs and the same definition of the metric, or the paper contradicts
itself. So both go through METRICS and through load_runs, which reads every
output_cpu*.dat.kpi rather than the first one.
"""
from __future__ import division
from __future__ import print_function

import os

from compare_schedulers import load_runs, per_run_metrics, METRICS

LEITOR = dict((nome, funcao) for nome, funcao, _ in METRICS)
DIRECAO = dict((nome, direcao) for nome, _, direcao in METRICS)

# Labels for the axes, in the units the paper uses. Written so that mathtext
# and LaTeX both render them.
ROTULOS = {
    'score'            : 'Score',
    'latency'          : 'Latency (s)',
    'pdr'              : 'PDR',
    'join_time'        : 'Join time (s)',
    'lifetime_min'     : 'Lifetime, worst mote (years)',
    'lifetime_mean'    : 'Lifetime, mean (years)',
    'sixp_per_packet'  : '6P transactions per packet',
    'sixp_transactions': '6P transactions',
    'mean_reward'      : 'Mean reward per decision',
    'final_policy_gap' : 'Final policy gap',
}

PRECISA_TRACO = ('mean_reward', 'final_policy_gap')


def folder_of(inputfolder, arm, motes):
    return os.path.join(inputfolder, arm, 'exec_numMotes_{0}'.format(motes))


def series(inputfolder, arm, motes, metric):
    """The metric for every run of one arm, in run order.

    Runs where the metric is missing are dropped rather than zeroed: the
    published reward plot filled absent episodes with zeros, which pulls a
    mean toward zero for a reason that has nothing to do with the network.
    """
    pasta = folder_of(inputfolder, arm, motes)
    runs = load_runs(pasta)
    if not runs:
        return []
    if metric in PRECISA_TRACO:
        for run_id, valores in per_run_metrics(pasta).items():
            if run_id in runs:
                runs[run_id]['learning-stats'] = valores
    leitor = LEITOR[metric]
    saida = []
    for run_id in sorted(runs):
        valor = leitor(runs[run_id])
        if valor is not None:
            saida.append(valor)
    return saida


def paired_series(inputfolder, arms, motes, metric):
    """The metric for the runs every arm has, so the boxes are comparable.

    Each arm seeds the simulator with exec_randomSeed plus the run id, so run
    3 of one arm and run 3 of another see the same topology. A box drawn on a
    different set of runs per arm would not be paired with the statistics.
    """
    por_arm = {}
    for arm in arms:
        pasta = folder_of(inputfolder, arm, motes)
        runs = load_runs(pasta)
        if metric in PRECISA_TRACO:
            for run_id, valores in per_run_metrics(pasta).items():
                if run_id in runs:
                    runs[run_id]['learning-stats'] = valores
        por_arm[arm] = runs

    comuns = None
    for runs in por_arm.values():
        ids = set(runs)
        comuns = ids if comuns is None else (comuns & ids)
    comuns = sorted(comuns or [])

    leitor = LEITOR[metric]
    saida = []
    for arm in arms:
        valores = [leitor(por_arm[arm][i]) for i in comuns]
        saida.append([v for v in valores if v is not None])
    return saida
