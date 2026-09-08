"""How often the agent stood in each row of its Q-table.

A reviewer asks for the distribution of visited states, and specifically
whether the state space is being used or whether the agent spends the run in
a handful of rows. This reads the per-mote counters the simulator writes and
answers it over every mote of every run.

Three things come out, and the third is the one that decides whether the
Q-table did any work at all:

  the histogram        how the visits divide over the rows, with the share
                       the busiest row takes and the rows never reached
  the coverage         the entropy of that histogram over the entropy of a
                       flat one, so 1.0 is every row used equally and a value
                       near 0 means the agent effectively had one state
  the exploited share  how many decisions the Q-table actually chose, per
                       row. A row filled entirely by exploration was never
                       used as a policy, whatever its visit count says.

Usage:
    python state_histogram.py --inputfolder resultados/charge
"""
from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import json
import math
import os


def load_stats(folder):
    """Every per-mote statistics file under the folder."""
    padrao = os.path.join(folder, '**', 'qlearning_stats.json')
    try:
        caminhos = glob.glob(padrao, recursive=True)
    except TypeError:  # python 2 has no recursive glob
        caminhos = []
        for raiz, _, arquivos in os.walk(folder):
            if 'qlearning_stats.json' in arquivos:
                caminhos.append(os.path.join(raiz, 'qlearning_stats.json'))
    saida = []
    for caminho in sorted(caminhos):
        try:
            with open(caminho) as f:
                saida.append(json.load(f))
        except ValueError:
            # a run killed mid-write leaves a truncated file; one mote of
            # thousands is not worth failing the whole reading over
            print('ilegivel, ignorado: {0}'.format(caminho))
    return saida


def totals(stats):
    """Visits per row, and the explored/greedy split per row, over all motes."""
    visitas = collections.Counter()
    explorado = collections.Counter()
    ganancioso = collections.Counter()
    acoes = collections.defaultdict(collections.Counter)
    for mote in stats:
        for estado, n in mote.get('STATE_VISITS', {}).items():
            visitas[int(estado)] += n
        for estado, linha in mote.get('STATE_ACTION', {}).items():
            e = int(estado)
            for acao, n in linha.get('explored', {}).items():
                explorado[e] += n
                acoes[e][int(acao)] += n
            for acao, n in linha.get('greedy', {}).items():
                ganancioso[e] += n
                acoes[e][int(acao)] += n
    return visitas, explorado, ganancioso, acoes


FACTORS = ('traffic', 'queue', 'charge')


def rows_of(nome, visitas):
    """How many rows the Q-table of this configuration has.

    A one-factor configuration has two rows, not eight, and reporting rows two
    to seven as never visited would invent a state space it never had. The
    factorial names its folders after the factors it switched on, so the name
    says how many rows there are. Q-static always uses all three factors, and
    anything unrecognised falls back to the largest row actually seen.
    """
    if nome.startswith('qlearningSBRC24'):
        return 2 ** len(FACTORS)
    presentes = [f for f in FACTORS if f in nome.split('_')]
    if presentes:
        return 2 ** len(presentes)
    return (max(visitas) + 1) if visitas else 0


def coverage(visitas, num_rows):
    """Entropy of the visit distribution over the entropy of a flat one.

    One number for how much of the state space the agent actually inhabited.
    A single row gives 0, every row used equally gives 1. The denominator is
    the rows the table has and not the rows that were reached, otherwise an
    agent that only ever saw two of eight rows would score as if it had used
    everything it was given.
    """
    total = sum(visitas.values())
    if total == 0 or num_rows < 2:
        return 0.0
    h = 0.0
    for n in visitas.values():
        p = n / total
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(num_rows)


def report(nome, visitas, explorado, ganancioso, acoes, num_states):
    total = sum(visitas.values())
    print('')
    print('=== {0} ==='.format(nome))
    if not total:
        print('  nenhuma decisao registrada')
        return {}
    print('  decisoes registradas: {0}'.format(total))
    print('  %5s %10s %8s %10s %10s  %s' % (
        'linha', 'visitas', 'share', 'da tabela', 'ao acaso', 'acoes'))
    linhas = range(num_states) if num_states else sorted(visitas)
    for estado in linhas:
        n = visitas.get(estado, 0)
        g = ganancioso.get(estado, 0)
        e = explorado.get(estado, 0)
        decisoes = g + e or 1
        dist = acoes.get(estado, {})
        resumo = ' '.join(
            '%d:%.0f%%' % (a, 100.0 * c / sum(dist.values()))
            for a, c in sorted(dist.items())
        ) if dist else '-'
        print('  %5d %10d %7.1f%% %9.1f%% %9.1f%%  %s' % (
            estado, n, 100.0 * n / total,
            100.0 * g / decisoes, 100.0 * e / decisoes, resumo))
    nunca = [s for s in linhas if visitas.get(s, 0) == 0]
    maior = max(visitas.values()) / total
    exploradas = sum(explorado.values())
    decisoes = exploradas + sum(ganancioso.values())
    print('')
    print('  linha mais visitada: %.1f%% das decisoes' % (100.0 * maior))
    print('  linhas nunca visitadas: {0}'.format(
        ', '.join(str(s) for s in nunca) if nunca else 'nenhuma'))
    print('  cobertura do espaco de estados: %.3f' % coverage(visitas, len(linhas)))
    if decisoes:
        print('  decisoes escolhidas pela tabela: %.1f%%' % (
            100.0 * (decisoes - exploradas) / decisoes))
    return {
        'decisions': total,
        'visits': dict(visitas),
        'rows': len(linhas),
        'coverage': coverage(visitas, len(linhas)),
        'largest_share': maior,
        'never_visited': nunca,
        'greedy_share': (
            (decisoes - exploradas) / decisoes if decisoes else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', nargs='+', required=True,
                        help='one folder per configuration to report')
    parser.add_argument('--states', type=int, default=0,
                        help='rows the Q-table has; 0 reads it off the folder name')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    resumo = {}
    for folder in args.inputfolder:
        stats = load_stats(folder)
        if not stats:
            print('sem qlearning_stats.json em {0}'.format(folder))
            continue
        nome = os.path.basename(os.path.normpath(folder))
        visitas, explorado, ganancioso, acoes = totals(stats)
        num_states = args.states or rows_of(nome, visitas)
        resumo[nome] = report(
            '{0} ({1} motes-run, {2} linhas)'.format(nome, len(stats), num_states),
            visitas, explorado, ganancioso, acoes, num_states
        )

    if args.out and resumo:
        with open(args.out, 'w') as f:
            json.dump(resumo, f, indent=2)
        print('')
        print('escrito em {0}'.format(args.out))


if __name__ == '__main__':
    main()
