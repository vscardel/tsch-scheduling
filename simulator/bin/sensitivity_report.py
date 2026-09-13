"""What each setting did to the network, and whether it converged.

Two things matter and they are not the same thing. A configuration can score
better and never settle, or settle on a policy that scores worse; the
published DynQ does the first and the published Q-static does the second. A
table that reports only one of them answers half the question.

So every arm gets both: the paired difference against its own baseline, on the
same seeds, with the interval and the effect size the reviewers asked for, and
the three-part convergence verdict.

The statistics are the ones already in use. Nothing new is invented here:
paired_series, bootstrap_ci and rank_biserial come from compare_schedulers,
and the criterion comes from learning_report.

The score appears as one column among the metrics rather than as the verdict.
It aggregates four quantities with weights nobody defended, and the agent is
not optimising it, so it is a summary and not the finding.

Usage:
    python sensitivity_report.py --inputfolder resultados --motes 50 \\
        --manifest sensitivity_manifest.json
"""
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from compare_schedulers import (
    METRICS, bootstrap_ci, load_runs, paired_series, rank_biserial
)
from learning_report import load_runs as load_learning_runs
from learning_report import per_run_metrics, verdict


def folder_of(inputfolder, label, motes):
    return os.path.join(
        inputfolder, '{0}_n{1}'.format(label, motes),
        'exec_numMotes_{0}'.format(motes)
    )


def read_arm(inputfolder, label, motes):
    """The network runs of one arm, with its learning quantities attached."""
    pasta = folder_of(inputfolder, label, motes)
    runs = load_runs(pasta)
    if not runs:
        return None
    for run_id, valores in per_run_metrics(pasta).items():
        if run_id in runs:
            runs[run_id]['learning-stats'] = valores
    return runs


def compare_to_baseline(runs_arm, runs_base):
    """Every metric, paired run by run against the baseline."""
    linha = {}
    for nome, leitor, direcao in METRICS:
        a, b = paired_series(runs_arm, runs_base, leitor)
        if len(a) < 3:
            continue
        diferencas = [x - y for x, y in zip(a, b)]
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0
        linha[nome] = {
            'median': float(np.median(a)),
            'median_baseline': float(np.median(b)),
            'median_diff': float(np.median(diferencas)),
            'ci': bootstrap_ci(diferencas),
            'p': p,
            'effect': rank_biserial(diferencas),
            'direction': direcao,
            'better': _melhor(np.median(diferencas), direcao),
        }
    return linha


def _melhor(diferenca, direcao):
    if diferenca == 0:
        return 'igual'
    subiu = diferenca > 0
    if direcao == 'higher':
        return 'braco' if subiu else 'base'
    return 'base' if subiu else 'braco'


def convergence_of(inputfolder, label, motes, bins):
    pasta = folder_of(inputfolder, label, motes)
    runs = load_learning_runs(pasta)
    if not runs:
        return None
    return verdict(runs, bins)


def marca(v):
    """Three letters, one per part of the criterion, then the verdict."""
    if v is None:
        return '   sem traco'
    def letra(valor):
        return '-' if valor is None else ('s' if valor else 'n')
    return ' %s%s%s  %s' % (
        letra(v['reward_flat']['flat']),
        letra(v['table_still']['settled']),
        letra(v['policy_settled']['settled']),
        'CONVERGIU' if v['converged'] else '         ',
    )


DESTAQUE = ('score', 'latency', 'pdr', 'lifetime_mean', 'mean_reward')


def report(inputfolder, motes, manifesto, bins, destaque):
    """One table per factor, arms in the order the sweep defines them."""
    por_fator = {}
    for label, info in manifesto.items():
        if info['factor'] == 'baseline':
            continue
        por_fator.setdefault((info['learner'], info['factor']), []).append(
            (info['value'], label, info['baseline'])
        )

    bases = {}
    saida = {}
    for (learner, factor) in sorted(por_fator):
        entradas = sorted(por_fator[(learner, factor)])
        nome_base = entradas[0][2]
        if nome_base not in bases:
            bases[nome_base] = read_arm(inputfolder, nome_base, motes)
        runs_base = bases[nome_base]
        if runs_base is None:
            print('sem resultados para a linha de base {0}'.format(nome_base))
            continue

        setting = manifesto[entradas[0][1]]['setting']
        print('')
        print('=== {0}, {1} ({2}) ==='.format(learner, factor, setting))
        cabecalho = '%-10s' % 'valor'
        for metrica in destaque:
            cabecalho += ' %14s' % metrica
        cabecalho += '  %s  %s' % ('abc', 'veredito')
        print(cabecalho)

        v_base = convergence_of(inputfolder, nome_base, motes, bins)
        linha_base = compare_to_baseline(runs_base, runs_base)
        print(_linha('publicado', linha_base, destaque, v_base, base=True))

        for valor, label, _ in entradas:
            runs = read_arm(inputfolder, label, motes)
            if runs is None:
                print('%-10s  sem resultados' % ('%g' % valor))
                continue
            comparacao = compare_to_baseline(runs, runs_base)
            v = convergence_of(inputfolder, label, motes, bins)
            print(_linha('%g' % valor, comparacao, destaque, v))
            saida[label] = {
                'learner': learner, 'factor': factor, 'setting': setting,
                'value': valor, 'metrics': comparacao, 'convergence': v,
            }
    return saida


def _linha(rotulo, comparacao, destaque, v, base=False):
    texto = '%-10s' % rotulo
    for metrica in destaque:
        dados = comparacao.get(metrica)
        if not dados:
            texto += ' %14s' % '-'
            continue
        if base:
            texto += ' %14.4f' % dados['median']
        else:
            estrela = '*' if dados['p'] < 0.05 else ' '
            texto += ' %13.4f%s' % (dados['median'], estrela)
    texto += ' ' + marca(v)
    return texto


LEGENDA = """
As tres letras sao as tres partes do criterio, nesta ordem: (a) a recompensa
achatou, (b) a tabela parou de se mexer, (c) a politica assentou. s e sim, n e
nao, - e sem dados para decidir. O veredito exige as tres.

O asterisco marca p abaixo de 0.05 no Wilcoxon pareado contra a linha de base
publicada, uma metrica por vez e sem correcao entre metricas. O score e uma
coluna entre as outras, nao o veredito: ele agrega quatro quantidades com pesos
que ninguem defendeu, e o agente nao esta otimizando ele.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', required=True)
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--manifest', default='sensitivity_manifest.json')
    parser.add_argument('--bins', type=int, default=30)
    parser.add_argument('--metrics', nargs='+', default=list(DESTAQUE))
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifesto = json.load(f)

    saida = report(
        args.inputfolder, args.motes, manifesto, args.bins, args.metrics
    )
    print(LEGENDA)

    if args.out and saida:
        with open(args.out, 'w') as f:
            json.dump(saida, f, indent=2, default=float)
        print('escrito em {0}'.format(args.out))


if __name__ == '__main__':
    main()
