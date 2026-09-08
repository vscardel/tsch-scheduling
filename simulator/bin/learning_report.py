"""Whether the agent learned anything, and by what criterion.

Twelve reviewer comments turn on one question. The manuscript answers it with
Figure 6 and one sentence: a cumulative reward curve, averaged over the motes
of a single run, with no variance band, no control, no convergence criterion,
and only for DynQ. Reviewer 3 asks for exactly that figure to be redone with
the average taken over runs and for both learners, and to be given a
convergence criterion. Reviewer 1 asks whether the table has converged at all
or is still in a transient, and points out that independent learners in a
shared network make the environment non-stationary, so convergence is not
guaranteed by the usual argument.

This reads the per-decision trace the simulator now writes and reports:

  reward curve      averaged over runs, with a bootstrap band, on an ASN axis
                    so two learners that decide on different triggers can be
                    drawn together. Gaps are carried forward from the last
                    value, not filled with zero, which is what makes a
                    cumulative curve sag when a mote stops deciding
  temporal difference   in absolute value. If it decays there is convergence;
                    if it settles on a plateau, that plateau is the evidence
                    of non-stationarity rather than an assertion about it
  policy churn      how many rows changed their preferred action per decision.
                    Reaching zero and staying there is the convergence
                    criterion, and it is legible: from episode X on, no
                    visited row changed its mind again
  the final table   the separation within each row, and how many visited rows
                    end with a strictly preferred action. A row whose columns
                    are equal is a row where the policy is arbitrary, which is
                    the practical consequence of a reward that does not depend
                    on the action

The unit of the statistics is the run, not the mote. Motes inside one run
share a topology and a random seed and are not independent; runs are.

Usage:
    python learning_report.py --inputfolder resultados/dynq_aprendido \\
        --control resultados/dynq_aleatorio --plot figuras/
"""
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import random


# ------------------------------------------------------------------ reading

def load_runs(folder):
    """The per-mote statistics under a folder, grouped by run."""
    runs = collections.defaultdict(list)
    for raiz, _, arquivos in os.walk(folder):
        if 'qlearning_stats.json' not in arquivos:
            continue
        caminho = os.path.join(raiz, 'qlearning_stats.json')
        try:
            with open(caminho) as f:
                runs[run_id_of(caminho)].append(json.load(f))
        except ValueError:
            # a run killed mid-write leaves a truncated file, and one mote of
            # thousands is not worth failing the whole reading over
            print('ilegivel, ignorado: {0}'.format(caminho))
    return dict(runs)


def run_id_of(caminho):
    for parte in caminho.split(os.sep):
        if parte.startswith('run_'):
            return parte
    return caminho


def decisions_of(mote):
    """The mote's decisions in the order they were taken."""
    trace = mote.get('DECISIONS', {})
    return [trace[k] for k in sorted(trace, key=int)]


def span_of(runs):
    """The last ASN any decision was taken at, over every run."""
    maior = 0
    for motes in runs.values():
        for mote in motes:
            for registro in decisions_of(mote):
                maior = max(maior, registro['asn'])
    return maior


# ------------------------------------------------------------------- curves

def forward_fill(serie):
    """Carry the last value over a gap, rather than calling the gap zero.

    A mote that took no decision in a window did not earn a reward of zero
    there; it earned nothing there and still holds what it had. Filling with
    zero is what makes an averaged cumulative curve dip, and the existing
    plotting script does exactly that.
    """
    saida = list(serie)
    ultimo = None
    for i, valor in enumerate(saida):
        if valor is None:
            saida[i] = ultimo
        else:
            ultimo = valor
    return saida


def mote_series(mote, span, bins):
    """One mote's decisions binned onto the common ASN axis."""
    soma = collections.defaultdict(float)
    soma_td = collections.defaultdict(float)
    soma_trocas = collections.defaultdict(float)
    contagem = collections.defaultdict(int)
    ultimo_acumulado = {}

    acumulado = 0.0
    for registro in decisions_of(mote):
        i = bin_of(registro['asn'], span, bins)
        soma[i] += registro['reward']
        soma_td[i] += abs(registro['td_error'])
        soma_trocas[i] += registro['policy_changes']
        contagem[i] += 1
        acumulado += registro['reward']
        ultimo_acumulado[i] = acumulado

    recompensa, td, trocas, cumulativa = [], [], [], []
    for i in range(bins):
        n = contagem[i]
        recompensa.append(soma[i] / n if n else None)
        td.append(soma_td[i] / n if n else None)
        trocas.append(soma_trocas[i] / n if n else None)
        cumulativa.append(ultimo_acumulado.get(i))
    return {
        'reward'    : forward_fill(recompensa),
        'td_error'  : forward_fill(td),
        'churn'     : forward_fill(trocas),
        'cumulative': forward_fill(cumulativa),
    }


def bin_of(asn, span, bins):
    if span <= 0:
        return 0
    return min(int(bins * asn / float(span)), bins - 1)


def run_series(motes, span, bins):
    """A run's curve is the mean over its motes, ignoring motes not yet born."""
    por_mote = [mote_series(m, span, bins) for m in motes]
    saida = {}
    for chave in ('reward', 'td_error', 'churn', 'cumulative'):
        linha = []
        for i in range(bins):
            valores = [s[chave][i] for s in por_mote if s[chave][i] is not None]
            linha.append(sum(valores) / len(valores) if valores else None)
        saida[chave] = linha
    return saida


def band(por_run, resamples=2000, seed=1):
    """Mean over runs at each bin, with a percentile bootstrap band."""
    rng = random.Random(seed)
    media, baixo, alto = [], [], []
    bins = len(por_run[0]) if por_run else 0
    if not bins:
        return media, baixo, alto
    for i in range(bins):
        valores = [serie[i] for serie in por_run if serie[i] is not None]
        if not valores:
            media.append(None)
            baixo.append(None)
            alto.append(None)
            continue
        media.append(sum(valores) / len(valores))
        n = len(valores)
        medias = sorted(
            sum(valores[rng.randrange(n)] for _ in range(n)) / n
            for _ in range(resamples)
        )
        baixo.append(medias[int(0.025 * resamples)])
        alto.append(medias[int(0.975 * resamples)])
    return media, baixo, alto


# ------------------------------------------------------------- the Q-table

def table_summary(runs):
    """What the tables look like when the run ends.

    Only rows the agent actually stood in count. A row it never reached holds
    the zeros it was created with, and counting those as 'no preference' would
    describe the initialisation rather than the learning.
    """
    separacoes = []
    com_preferencia = 0
    linhas = 0
    for motes in runs.values():
        for mote in motes:
            visitadas = set(mote.get('STATE_VISITS', {}))
            for linha, valores in mote.get('Q_TABLE', {}).items():
                if linha not in visitadas or len(valores) < 2:
                    continue
                ordenado = sorted(valores, reverse=True)
                linhas += 1
                separacoes.append(ordenado[0] - ordenado[-1])
                if ordenado[0] > ordenado[1]:
                    com_preferencia += 1
    if not linhas:
        return {'rows': 0, 'mean_separation': 0.0, 'share_with_preference': 0.0}
    return {
        'rows': linhas,
        'mean_separation': sum(separacoes) / len(separacoes),
        'share_with_preference': com_preferencia / linhas,
    }


def run_number(run):
    """The integer the paired statistics key on, out of a run_N folder name."""
    try:
        return int(run.split('_')[-1])
    except ValueError:
        return run


def per_run_metrics(folder):
    """One number per run, for the paired statistics in compare_schedulers.

    mean_reward is the mean over motes of a mote's mean per-decision reward.
    final_policy_gap is the mean separation inside the visited rows of the
    table the run ended with: a table whose columns come out level has learned
    nothing worth acting on, whatever its reward curve did.

    Both are in the units of that learner's own reward, so they are for
    comparing a learner with its own random control and not with the other
    learner.
    """
    saida = {}
    for run, motes in load_runs(folder).items():
        recompensas = []
        for mote in motes:
            decisoes = decisions_of(mote)
            if decisoes:
                recompensas.append(
                    sum(d['reward'] for d in decisoes) / len(decisoes)
                )
        if not recompensas:
            continue
        saida[run_number(run)] = {
            'mean_reward': sum(recompensas) / len(recompensas),
            'final_policy_gap': table_summary(
                {run: motes}
            )['mean_separation'],
        }
    return saida


def convergence(churn, bins):
    """The bin from which no visited row ever changed its mind again.

    This is the convergence criterion reviewer 3 asked for, in the terms a
    reader can check: from this point on the policy stopped moving. None means
    it never stopped.
    """
    ultimo = None
    for i, valor in enumerate(churn):
        if valor:                       # None is no data, not a change
            ultimo = i
    if ultimo is None:
        return 0
    if ultimo >= bins - 1:
        return None
    return ultimo + 1


# ------------------------------------------------------------------ report

def summarise(nome, runs, bins):
    span = span_of(runs)
    series = dict(
        (run, run_series(motes, span, bins))
        for run, motes in sorted(runs.items())
    )
    curvas = {}
    for chave in ('reward', 'td_error', 'churn', 'cumulative'):
        por_run = [series[run][chave] for run in sorted(series)]
        media, baixo, alto = band(por_run)
        curvas[chave] = {'mean': media, 'low': baixo, 'high': alto}

    decisoes = sum(
        len(decisions_of(m)) for motes in runs.values() for m in motes
    )
    parada = convergence(curvas['churn']['mean'], bins)

    resumo = {
        'runs'      : len(runs),
        'motes'     : sum(len(m) for m in runs.values()),
        'decisions' : decisoes,
        'span_asn'  : span,
        'curves'    : curvas,
        'table'     : table_summary(runs),
        'mean_reward'      : media_de(curvas['reward']['mean']),
        'reward_last_third': media_de(ultimo_terco(curvas['reward']['mean'])),
        'td_first_third'   : media_de(primeiro_terco(curvas['td_error']['mean'])),
        'td_last_third'    : media_de(ultimo_terco(curvas['td_error']['mean'])),
        'converged_at_bin' : parada,
        'greedy_share'     : greedy_share(runs),
    }
    imprimir(nome, resumo, bins)
    return resumo


def media_de(serie):
    valores = [v for v in serie if v is not None]
    return sum(valores) / len(valores) if valores else 0.0


def primeiro_terco(serie):
    return serie[:max(1, len(serie) // 3)]


def ultimo_terco(serie):
    return serie[-max(1, len(serie) // 3):]


def greedy_share(runs):
    """How many decisions the table chose rather than a draw.

    It means different things in the two learners, which is why it is reported
    and not compared. DynQ tosses a coin on every decision, so its share
    describes a policy. Q-static switches phase once when epsilon crosses a
    threshold, so its share describes a schedule.
    """
    ganancioso = explorado = 0
    for motes in runs.values():
        for mote in motes:
            for linha in mote.get('STATE_ACTION', {}).values():
                ganancioso += sum(linha.get('greedy', {}).values())
                explorado += sum(linha.get('explored', {}).values())
    total = ganancioso + explorado
    return ganancioso / total if total else 0.0


def imprimir(nome, r, bins):
    print('')
    print('=== {0} ==='.format(nome))
    print('  {0} runs, {1} motes-run, {2} decisoes, ate ASN {3}'.format(
        r['runs'], r['motes'], r['decisions'], r['span_asn']))
    print('  recompensa media por decisao: %.4f' % r['mean_reward'])
    print('    no ultimo terco da rodada:  %.4f' % r['reward_last_third'])
    print('  |erro TD| no primeiro terco:  %.4f' % r['td_first_third'])
    print('    no ultimo terco:            %.4f' % r['td_last_third'])
    if r['td_first_third']:
        queda = 100.0 * (1 - r['td_last_third'] / r['td_first_third'])
        print('    queda: %.1f%%' % queda)
    parada = r['converged_at_bin']
    if parada is None:
        print('  politica: ainda trocava de acao no ultimo bin, nao estabilizou')
    else:
        print('  politica: parou de trocar de acao a partir do bin %d de %d'
              % (parada, bins))
    t = r['table']
    print('  tabela final: %d linhas visitadas, separacao media %.4f' % (
        t['rows'], t['mean_separation']))
    print('    com uma acao estritamente preferida: %.1f%%' % (
        100.0 * t['share_with_preference']))
    print('  decisoes escolhidas pela tabela: %.1f%%' % (
        100.0 * r['greedy_share']))


CAVEAT = """
Uma ressalva sobre a ultima linha. "Escolhida pela tabela" nao quer dizer a
mesma coisa nos dois metodos. No DynQ e uma moeda por decisao, entao a fracao
descreve uma politica. No Q-static e um interruptor de fase que vira uma vez,
quando epsilon cruza o limiar, entao a fracao descreve um cronograma. Os dois
numeros nao se comparam entre si; cada um se compara com o seu proprio
controle aleatorio.
"""


# ------------------------------------------------------------------- plots

def plot(destino, curvas_por_nome, bins):
    """The figure reviewer 3 asked for: averaged over runs, with a band."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib ausente, sem figuras')
        return

    titulos = [
        ('reward', 'recompensa media por decisao'),
        ('cumulative', 'recompensa acumulada'),
        ('td_error', '|erro TD|'),
        ('churn', 'linhas que trocaram de acao, por decisao'),
    ]
    if not os.path.isdir(destino):
        os.makedirs(destino)

    for chave, titulo in titulos:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for nome, resumo in sorted(curvas_por_nome.items()):
            curva = resumo['curves'][chave]
            x = [i / float(bins) for i in range(bins)]
            pares = [
                (xi, m, lo, hi) for xi, m, lo, hi in
                zip(x, curva['mean'], curva['low'], curva['high'])
                if m is not None
            ]
            if not pares:
                continue
            xs = [p[0] for p in pares]
            ax.plot(xs, [p[1] for p in pares], label=nome)
            ax.fill_between(xs, [p[2] for p in pares], [p[3] for p in pares],
                            alpha=0.2)
        ax.set_xlabel('fracao da rodada (eixo de ASN)')
        ax.set_ylabel(titulo)
        ax.legend(fontsize='small')
        fig.tight_layout()
        caminho = os.path.join(destino, 'learning_{0}.png'.format(chave))
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
        print('escrito em {0}'.format(caminho))


# -------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', nargs='+', required=True,
                        help='one folder per arm, the learned arms')
    parser.add_argument('--control', nargs='*', default=[],
                        help='the matching random-policy arms')
    parser.add_argument('--bins', type=int, default=40,
                        help='windows the ASN axis is divided into')
    parser.add_argument('--plot', default=None, help='where to write figures')
    parser.add_argument('--out', default=None, help='write the report as JSON')
    args = parser.parse_args()

    resumos = {}
    for folder in list(args.inputfolder) + list(args.control):
        runs = load_runs(folder)
        nome = os.path.basename(os.path.normpath(folder))
        if not runs:
            print('sem qlearning_stats.json em {0}'.format(folder))
            continue
        resumos[nome] = summarise(nome, runs, args.bins)

    if not resumos:
        raise SystemExit('nada para relatar')

    print(CAVEAT)

    if args.plot:
        plot(args.plot, resumos, args.bins)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(resumos, f, indent=2)
        print('escrito em {0}'.format(args.out))


if __name__ == '__main__':
    main()
