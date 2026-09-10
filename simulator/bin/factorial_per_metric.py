"""The 2^k factorial, one metric at a time instead of one aggregate score.

The factorial's response variable was the aggregate score, and a reviewer
objected that the score's constants were hand picked. Two of the four terms
also turn out to do no work: PDR sits far below its threshold in every arm, so
its sigmoid is saturated and it contributes the same quarter to everybody,
while lifetime sits right on its threshold, where the sigmoid is steepest and
amplifies differences of a few weeks.

Reading the factorial per metric avoids both problems. It says what each state
factor does to latency, to delivery, to join time, to lifetime and to control
overhead, separately, and lets a trade-off be seen instead of averaged away.

Main effects only. With one cell per combination and thirty runs in each, the
effect of a factor is the mean over the cells that have it minus the mean over
the cells that do not.
"""
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import json
import os

from compare_schedulers import METRICS, load_runs

FACTORS = ['traffic', 'queue', 'charge']


# The empty cell of the design is the learner with no state factors, a
# one-row table. The published factorial ran MSF there, under the name
# baseline, so every main effect compared a state factor against another
# scheduling function: the latency effect of the traffic factor read as
# -0.805 s with that cell in the arithmetic and -0.005 s without it. So the
# stateless learner is the default, and the contaminated arithmetic has to be
# asked for by name.
EMPTY_CELL = 'sem_estado'
EMPTY_CELL_MSF = 'baseline'


def cell_name(factors, empty=EMPTY_CELL):
    return '_'.join(factors) if factors else empty


def cells(empty=EMPTY_CELL):
    """The eight combinations, each as (name, the factors it switched on)."""
    saida = []
    for combination in itertools.product([0, 1], repeat=len(FACTORS)):
        presentes = [FACTORS[i] for i, on in enumerate(combination) if on]
        saida.append((cell_name(presentes, empty), presentes))
    return saida


def read_cell(folder, nome, motes):
    caminho = os.path.join(folder, nome, 'exec_numMotes_{0}'.format(motes))
    return load_runs(caminho)


def cell_means(runs, reader):
    valores = [reader(r) for r in runs.values()]
    valores = [v for v in valores if v is not None]
    return sum(valores) / float(len(valores)) if valores else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', required=True)
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--out', default=None)
    parser.add_argument(
        '--empty-cell', default=EMPTY_CELL,
        help=(
            'folder of the cell where no state factor is on. The default is '
            'the stateless learner. Pass {0} to reproduce the published '
            'arithmetic, which ran MSF there and so mixed every main effect '
            'with a different scheduling function.'.format(EMPTY_CELL_MSF)
        )
    )
    args = parser.parse_args()

    vazia = args.empty_cell
    if vazia == EMPTY_CELL_MSF:
        print('AVISO: a celula vazia e o MSF, entao cada efeito principal')
        print('       compara um fator de estado contra outro escalonador.')

    medias = {}
    for nome, _ in cells(vazia):
        runs = read_cell(args.inputfolder, nome, args.motes)
        if runs:
            medias[nome] = runs
    ausentes = [n for n, _ in cells(vazia) if n not in medias]
    if ausentes:
        print('sem resultados para: {0}'.format(', '.join(ausentes)))

    resumo = {}
    for metrica, reader, direcao in METRICS:
        por_celula = {}
        for nome, runs in medias.items():
            valor = cell_means(runs, reader)
            if valor is not None:
                por_celula[nome] = valor
        if len(por_celula) < len(list(cells(vazia))):
            continue

        print('')
        print('=== {0} ({1} e melhor) ==='.format(metrica, direcao))
        for nome, valor in sorted(por_celula.items(), key=lambda kv: kv[1]):
            print('  %-24s %12.4f' % (nome, valor))

        efeitos = {}
        for fator in FACTORS:
            com = [por_celula[cell_name(f, vazia)]
                   for n, f in cells(vazia) if fator in f]
            sem = [por_celula[cell_name(f, vazia)]
                   for n, f in cells(vazia) if fator not in f]
            efeitos[fator] = sum(com) / len(com) - sum(sem) / len(sem)
        print('  efeitos principais:')
        for fator, efeito in sorted(efeitos.items(), key=lambda kv: -abs(kv[1])):
            ajuda = (efeito < 0) if direcao == 'lower' else (efeito > 0)
            print('    %-8s %+12.4f  %s' % (
                fator, efeito, 'ajuda' if ajuda else 'atrapalha'))
        resumo[metrica] = {
            'direction': direcao, 'cells': por_celula, 'effects': efeitos
        }

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(resumo, f, indent=2)
        print('')
        print('escrito em {0}'.format(args.out))


if __name__ == '__main__':
    main()
