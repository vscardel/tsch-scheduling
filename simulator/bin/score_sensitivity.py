"""Does the ranking survive the score's own parameters?

A reviewer objects that the aggregate score drives both the hyperparameter
optimisation and the factorial analysis, yet its thresholds and smoothness
constants were hand-picked with no justification and no sensitivity analysis,
and asks to be shown how the rankings change with different thresholds and
weights.

This answers it from the .kpi files a run already produced. The score is a
function of four measured quantities, so changing its constants is arithmetic,
not another simulation. Nothing here needs the simulator.

Three sweeps, each reporting how often every scheduler comes first:

  thresholds  each threshold moved on its own, and all of them together
  weights     the four corners where one metric carries everything, plus a
              grid over the simplex
  smoothness  the k of each sigmoid, which decides how sharply the score
              punishes a value on the wrong side of its threshold

A ranking that only holds at the published constants is a ranking that was
chosen along with them.
"""
from __future__ import division
from __future__ import print_function

import argparse
import collections
import itertools
import json
import math
import os

from compare_schedulers import load_runs
from score_model import (
    ABOVE_IS_BAD, SMOOTHNESS as BASE_SMOOTHNESS, THRESHOLDS as BASE_THRESHOLDS,
    WEIGHTS as BASE_WEIGHTS, run_metrics, score, sigmoid
)


def mean_score(runs, weights, thresholds, smoothness):
    valores = [
        score(run_metrics(run), weights, thresholds, smoothness)
        for run in runs.values()
    ]
    return sum(valores) / float(len(valores)) if valores else float('nan')


def ranking(metrics_by_sf, weights, thresholds, smoothness):
    """Schedulers ordered best first. Lower score is better."""
    pontuados = [
        (mean_score(runs, weights, thresholds, smoothness), sf)
        for sf, runs in metrics_by_sf.items()
    ]
    pontuados.sort()
    return [sf for _, sf in pontuados]


def normalised(weights):
    total = float(sum(weights.values()))
    return dict((k, v / total) for k, v in weights.items())


def sweep_thresholds(runs_by_sf, factors):
    """Each threshold moved on its own, then all of them together."""
    saida = []
    for nome in sorted(BASE_THRESHOLDS):
        for factor in factors:
            limiares = dict(BASE_THRESHOLDS)
            limiares[nome] = BASE_THRESHOLDS[nome] * factor
            saida.append((
                '{0} x{1:.1f}'.format(nome, factor),
                ranking(runs_by_sf, BASE_WEIGHTS, limiares, BASE_SMOOTHNESS)
            ))
    for factor in factors:
        limiares = dict(
            (k, v * factor) for k, v in BASE_THRESHOLDS.items()
        )
        saida.append((
            'todos x{0:.1f}'.format(factor),
            ranking(runs_by_sf, BASE_WEIGHTS, limiares, BASE_SMOOTHNESS)
        ))
    return saida


def sweep_weights(runs_by_sf, steps):
    """The four corners, then a grid over the simplex."""
    saida = []
    for nome in sorted(BASE_WEIGHTS):
        pesos = dict((k, 0.0) for k in BASE_WEIGHTS)
        pesos[nome] = 1.0
        saida.append((
            'so {0}'.format(nome),
            ranking(runs_by_sf, pesos, BASE_THRESHOLDS, BASE_SMOOTHNESS)
        ))

    nomes = sorted(BASE_WEIGHTS)
    for combo in itertools.product(range(1, steps + 1), repeat=len(nomes)):
        pesos = normalised(dict(zip(nomes, [float(c) for c in combo])))
        rotulo = ' '.join(
            '{0}={1:.2f}'.format(n[:3], pesos[n]) for n in nomes
        )
        saida.append((
            rotulo,
            ranking(runs_by_sf, pesos, BASE_THRESHOLDS, BASE_SMOOTHNESS)
        ))
    return saida


def sweep_smoothness(runs_by_sf, factors):
    saida = []
    for nome in sorted(BASE_SMOOTHNESS):
        for factor in factors:
            suavidade = dict(BASE_SMOOTHNESS)
            suavidade[nome] = BASE_SMOOTHNESS[nome] * factor
            saida.append((
                '{0} k x{1:.1f}'.format(nome, factor),
                ranking(runs_by_sf, BASE_WEIGHTS, BASE_THRESHOLDS, suavidade)
            ))
    return saida


def report(titulo, resultados, esperado):
    primeiros = collections.Counter(r[0] for _, r in resultados)
    total = len(resultados)
    print('')
    print('=== {0} ({1} configuracoes) ==='.format(titulo, total))
    for sf, n in primeiros.most_common():
        print('  primeiro em %4d de %4d (%5.1f%%)  %s' % (
            n, total, 100.0 * n / total, sf))
    viradas = [rotulo for rotulo, r in resultados if r[0] != esperado]
    if viradas:
        print('  o vencedor muda em {0} delas, por exemplo: {1}'.format(
            len(viradas), ', '.join(viradas[:4])))
    else:
        print('  o vencedor nunca muda')
    return primeiros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', default='simData')
    parser.add_argument('--motes', type=int, required=True)
    parser.add_argument('--schedulers', nargs='+', required=True)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--steps', type=int, default=3,
                        help='grid resolution per weight')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    runs_by_sf = {}
    for sf in args.schedulers:
        folder = os.path.join(
            args.inputfolder, '{0}{1}'.format(args.prefix, sf),
            'exec_numMotes_{0}'.format(args.motes)
        )
        runs = load_runs(folder)
        if runs:
            runs_by_sf[sf] = runs
            print('{0}: {1} runs'.format(sf, len(runs)))

    if len(runs_by_sf) < 2:
        raise SystemExit('need at least two schedulers with results')

    base = ranking(runs_by_sf, BASE_WEIGHTS, BASE_THRESHOLDS, BASE_SMOOTHNESS)
    print('')
    print('ranking nos valores publicados: {0}'.format(' < '.join(base)))
    esperado = base[0]

    factors = [0.5, 0.75, 1.25, 1.5, 2.0]
    resultados = {
        'limiares': sweep_thresholds(runs_by_sf, factors),
        'pesos': sweep_weights(runs_by_sf, args.steps),
        'suavidade': sweep_smoothness(runs_by_sf, factors),
    }

    resumo = {'base': base}
    for titulo, dados in resultados.items():
        contagem = report(titulo, dados, esperado)
        resumo[titulo] = {
            'total': len(dados),
            'primeiro': dict(contagem),
        }

    todos = [r for dados in resultados.values() for _, r in dados]
    ganhos = collections.Counter(r[0] for r in todos)
    print('')
    print('=== tudo junto ({0} configuracoes) ==='.format(len(todos)))
    for sf, n in ganhos.most_common():
        print('  primeiro em %4d de %4d (%5.1f%%)  %s' % (
            n, len(todos), 100.0 * n / len(todos), sf))
    resumo['tudo'] = {'total': len(todos), 'primeiro': dict(ganhos)}

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(resumo, f, indent=2)
        print('')
        print('escrito em {0}'.format(args.out))


if __name__ == '__main__':
    main()
