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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from SimEngine import network_trace


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



# ------------------------------------------------------- the convergence test

"""Three things have to hold at once, and each is reported on its own.

Watkins and Dayan guarantee convergence only when the learning rate shrinks
and every state-action pair is visited without end. Neither learner meets
both, so the asymptotic guarantee is not available and is not claimed. What is
used instead is the practical criterion: the reward curve has levelled off,
the table has stopped moving, and the policy has stopped changing its mind.

Failing one and passing two is a diagnosis rather than a verdict, which is why
the three are never collapsed into a single number.
"""

FLAT_CONFIDENCE = 0.95
STEP_TOLERANCE = 0.01     # of the reward scale
SETTLED_SHARE = 0.95      # of the visited rows


def percentil(valores, p):
    if not valores:
        return 0.0
    ordenado = sorted(valores)
    return ordenado[min(len(ordenado) - 1, int(p * len(ordenado)))]


def decode_policy(codigo, rows, base):
    """The greedy action of each row, back out of the integer."""
    saida = []
    for _ in range(rows):
        saida.append(codigo % base)
        codigo //= base
    return saida


def mote_settled(mote, span, bins):
    """Per window, the share of visited rows already at their final action.

    Rows the mote never stood in are left out. A row it never reached holds
    the action its tie-break gave it and has not settled on anything.
    """
    tabela = mote.get('Q_TABLE', {})
    decisoes = decisions_of(mote)
    if not tabela or not decisoes or 'policy_code' not in decisoes[-1]:
        return [None] * bins
    ordem = sorted(tabela, key=lambda r: int(r))
    base = max(len(valores) for valores in tabela.values())
    visitadas = set(mote.get('STATE_VISITS', {}))
    # the visit counter is keyed by string, and a table read back from
    # JSON is too, but one still in memory need not be
    indices = [
        i for i, linha in enumerate(ordem) if str(linha) in visitadas
    ]
    if not indices or base < 2:
        return [None] * bins

    final = decode_policy(decisoes[-1]['policy_code'], len(ordem), base)
    ultimo = {}
    for registro in decisoes:
        ultimo[bin_of(registro['asn'], span, bins)] = registro['policy_code']

    serie = []
    for i in range(bins):
        codigo = ultimo.get(i)
        if codigo is None:
            serie.append(None)
            continue
        politica = decode_policy(codigo, len(ordem), base)
        iguais = sum(1 for j in indices if politica[j] == final[j])
        serie.append(iguais / len(indices))
    return forward_fill(serie)


def settled_curve(runs, span, bins):
    """Mean over motes, then over runs, of the share of settled rows."""
    por_run = []
    for _, motes in sorted(runs.items()):
        por_mote = [mote_settled(m, span, bins) for m in motes]
        linha = []
        for i in range(bins):
            valores = [s[i] for s in por_mote if s[i] is not None]
            linha.append(sum(valores) / len(valores) if valores else None)
        por_run.append(linha)
    media, _, _ = band(por_run)
    return media


def settled_from(curva, bins):
    """The first window from which the policy stays settled to the end.

    The stretch has to cover at least the final third, which is the same
    stretch the other two parts of the criterion are read over. Without that
    floor the last window always counts as settled, since it is the window the
    final policy is taken from, and every run would report convergence in its
    own last moment.
    """
    inicio = None
    for i in range(bins - 1, -1, -1):
        valor = curva[i] if i < len(curva) else None
        if valor is None:
            continue
        if valor >= SETTLED_SHARE:
            inicio = i
        else:
            break
    if inicio is None or inicio > 2 * bins // 3:
        return None
    return inicio


def flatness(runs, span, bins):
    """Whether the reward curve has stopped moving, and where it stopped.

    The last third against the middle third, bootstrapped over runs. Flat when
    the interval contains zero. The level is reported beside it, because a
    curve that has levelled off below zero has not converged on anything worth
    having.
    """
    diferencas = []
    niveis = []
    for _, motes in sorted(runs.items()):
        serie = run_series(motes, span, bins)['reward']
        fim = media_de(ultimo_terco(serie))
        meio = media_de(serie[len(serie) // 3: 2 * len(serie) // 3])
        if fim is None or meio is None:
            continue
        diferencas.append(fim - meio)
        niveis.append(fim)
    if len(diferencas) < 3:
        return {'flat': None, 'change': None, 'ci': (None, None), 'level': None}

    rng = random.Random(1)
    n = len(diferencas)
    medias = sorted(
        sum(diferencas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(2000)
    )
    baixo = medias[int(0.025 * 2000)]
    alto = medias[int(0.975 * 2000)]
    return {
        'flat'  : baixo <= 0 <= alto,
        'change': sum(diferencas) / n,
        'ci'    : (baixo, alto),
        'level' : sum(niveis) / len(niveis),
    }


def step_sizes(runs, span, bins):
    """How big the updates still are at the end, against the reward scale.

    The size of an update is recorded rather than derived from the configured
    learning rate, so this reads the same whether the rate is constant or on a
    schedule. The scale is the spread of the reward rather than its mean,
    which stays defined when the mean passes through zero.
    """
    recompensas = []
    finais = []
    corte = 2 * bins // 3
    for motes in runs.values():
        for mote in motes:
            for registro in decisions_of(mote):
                recompensas.append(registro['reward'])
                if 'delta_q' not in registro:
                    continue
                if bin_of(registro['asn'], span, bins) >= corte:
                    finais.append(abs(registro['delta_q']))
    if not recompensas or not finais:
        return {'settled': None, 'step': None, 'scale': None, 'tolerance': None}

    escala = percentil(recompensas, 0.95) - percentil(recompensas, 0.05)
    passo = percentil(finais, 0.95)
    tolerancia = STEP_TOLERANCE * escala
    if escala <= 0:
        # a reward that never varies gives nothing to be a hundredth of, and
        # nothing to learn either. Undecidable rather than failed.
        return {'settled': None, 'step': passo, 'scale': escala,
                'tolerance': tolerancia}
    return {
        'settled'  : passo < tolerancia,
        'step'     : passo,
        'scale'    : escala,
        'tolerance': tolerancia,
    }


def verdict(runs, bins):
    """The three parts, and whether all of them hold."""
    span = span_of(runs)
    curva = settled_curve(runs, span, bins)
    inicio = settled_from(curva, bins)
    partes = {
        'reward_flat' : flatness(runs, span, bins),
        'table_still' : step_sizes(runs, span, bins),
        'policy_settled': {
            # a trace with no policy_code at all is undecidable, not failed:
            # the runs made before the field existed are still readable
            'settled': None if not [v for v in curva if v is not None]
                       else inicio is not None,
            'from_bin': inicio,
            'from_fraction': (inicio / bins) if inicio is not None else None,
            'final_share': curva[-1] if curva else None,
        },
        'curve': curva,
    }
    partes['converged'] = bool(
        partes['reward_flat']['flat']
        and partes['table_still']['settled']
        and partes['policy_settled']['settled']
    )
    return partes


def imprimir_veredito(v, bins):
    a = v['reward_flat']
    b = v['table_still']
    c = v['policy_settled']
    print('  criterio de convergencia:')
    if a['flat'] is None:
        print('    (a) recompensa achatou:   sem rodadas suficientes')
    else:
        print('    (a) recompensa achatou:   %s   variacao %+.4f '
              '[%+.4f,%+.4f], nivel %.4f' % (
                  'sim' if a['flat'] else 'NAO', a['change'],
                  a['ci'][0], a['ci'][1], a['level']))
    if b['settled'] is None:
        print('    (b) tabela parou:         sem dados de |dQ|')
    else:
        print('    (b) tabela parou:         %s   |dQ| p95 %.5f contra '
              'tolerancia %.5f (1%% de %.3f)' % (
                  'sim' if b['settled'] else 'NAO', b['step'],
                  b['tolerance'], b['scale']))
    if c['settled'] is None:
        print('    (c) politica assentou:    sem policy_code no traco')
    elif c['settled']:
        print('    (c) politica assentou:    sim   a partir de %.0f%% da rodada'
              % (100.0 * c['from_fraction']))
    elif (c['final_share'] or 0) >= SETTLED_SHARE:
        # reached the mark but not early enough to have held for the final
        # third, which is the stretch the other two parts are read over
        print('    (c) politica assentou:    NAO  chegou a %.1f%% das linhas, '
              'mas tarde demais para ter durado o terco final'
              % (100.0 * c['final_share']))
    else:
        print('    (c) politica assentou:    NAO  chegou a %.1f%% das linhas'
              % (100.0 * (c['final_share'] or 0)))
    print('    veredito: %s' % ('CONVERGIU' if v['converged'] else 'nao convergiu'))



# ---------------------------------------------------- tracking a disturbance

"""What the agent did when the network changed under it.

In a non-stationary environment the question is not whether the table settled
but whether the agent recovers, so these read the network time series the
engine now writes and report, per disturbance, how far delivery fell and how
long it took to come back.

Both are comparative. A disturbance that permanently changes what the network
can deliver moves the achievable ratio, and then no agent returns to the old
level however well it adapts, so a recovery time means something only against
another arm on the same disturbance and the same seeds.
"""


def load_network_traces(folder):
    """The per-run network time series, keyed the way the runs are."""
    saida = {}
    for raiz, _, arquivos in os.walk(folder):
        if 'network_trace.json' not in arquivos:
            continue
        caminho = os.path.join(raiz, 'network_trace.json')
        try:
            with open(caminho) as f:
                saida[run_id_of(caminho)] = json.load(f)
        except ValueError:
            print('ilegivel, ignorado: {0}'.format(caminho))
    return saida


def tracking(traces, fractions):
    """Per disturbance, the fall and the return, over the runs.

    The run length comes from the trace itself rather than from a config, so
    the report needs to be told only where the disturbances were.
    """
    saida = []
    for fracao in fractions:
        quedas, voltas, nunca = [], [], 0
        for trace in traces.values():
            if not trace.get('asn'):
                continue
            fim = max(trace['asn'])
            asn = int(fim * fracao)
            serie = network_trace.delivery_series(trace)
            queda = network_trace.drop(serie, asn)
            volta = network_trace.recovery(serie, asn)
            if queda is not None:
                quedas.append(queda)
            if volta is None:
                nunca += 1
            else:
                voltas.append(volta)
        saida.append({
            'fraction'      : fracao,
            'runs'          : len(traces),
            'drop'          : media_de(quedas),
            'recovery_asn'  : media_de(voltas),
            'recovered'     : len(voltas),
            'never_recovered': nunca,
        })
    return saida


def imprimir_rastreamento(linhas, slotframe_length=101):
    if not linhas:
        return
    print('  rastreamento das perturbacoes:')
    for linha in linhas:
        volta = linha['recovery_asn']
        print('    em %.0f%% da rodada: queda %s, voltou em %d de %d rodadas'
              % (
                  100 * linha['fraction'],
                  '%.1f%%' % (100 * linha['drop'])
                  if linha['drop'] is not None else 'sem dados',
                  linha['recovered'], linha['runs'],
              ))
        if volta is not None:
            print('       tempo medio de volta: %.0f slotframes'
                  % (volta / float(slotframe_length)))
        if linha['never_recovered']:
            print('       nao voltou em %d rodadas' % linha['never_recovered'])


# ------------------------------------------------------------------ report

def summarise(nome, runs, bins, fractions=(), traces=None):
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
        'verdict'          : verdict(runs, bins),
        'tracking'         : tracking(traces or {}, fractions),
    }
    imprimir(nome, resumo, bins)
    return resumo


def media_de(serie):
    """None, not zero, when no run reached this stretch of the run.

    Early windows are often empty: a mote decides nothing before it has
    joined. Reporting that as a mean of zero would put a reward of zero and a
    settled temporal difference where there is simply no measurement.
    """
    valores = [v for v in serie if v is not None]
    return sum(valores) / len(valores) if valores else None


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


def numero(valor):
    return 'sem dados' if valor is None else '%.4f' % valor


def imprimir(nome, r, bins):
    print('')
    print('=== {0} ==='.format(nome))
    print('  {0} runs, {1} motes-run, {2} decisoes, ate ASN {3}'.format(
        r['runs'], r['motes'], r['decisions'], r['span_asn']))
    print('  recompensa media por decisao: %s' % numero(r['mean_reward']))
    print('    no ultimo terco da rodada:  %s' % numero(r['reward_last_third']))
    print('  |erro TD| no primeiro terco:  %s' % numero(r['td_first_third']))
    print('    no ultimo terco:            %s' % numero(r['td_last_third']))
    if r['td_first_third'] and r['td_last_third'] is not None:
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
    imprimir_veredito(r['verdict'], bins)
    imprimir_rastreamento(r.get('tracking') or [])


CAVEAT = """
Uma ressalva sobre a ultima linha. "Escolhida pela tabela" nao quer dizer a
mesma coisa nos dois metodos. No DynQ e uma moeda por decisao, entao a fracao
descreve uma politica. No Q-static e um interruptor de fase que vira uma vez,
quando epsilon cruza o limiar, entao a fracao descreve um cronograma. Os dois
numeros nao se comparam entre si; cada um se compara com o seu proprio
controle aleatorio.
"""


# ------------------------------------------------------------------- plots

def plot(destino, curvas_por_nome, bins, fractions=()):
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
        for fracao in fractions:
            ax.axvline(fracao, color='0.4', linestyle=':', linewidth=1)
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
    parser.add_argument('--disturbances', type=float, nargs='*', default=[],
                        help='where the disturbances were, as fractions of '
                             'the run; the report is told rather than reading '
                             'a config')
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
        resumos[nome] = summarise(
            nome, runs, args.bins, args.disturbances, load_network_traces(folder)
        )

    if not resumos:
        raise SystemExit('nada para relatar')

    print(CAVEAT)

    if args.plot:
        plot(args.plot, resumos, args.bins, args.disturbances)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(resumos, f, indent=2)
        print('escrito em {0}'.format(args.out))


if __name__ == '__main__':
    main()
