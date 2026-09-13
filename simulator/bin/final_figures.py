"""Every figure of the results section, from one results folder.

For a folder holding <arm>_n<motes>/exec_numMotes_<motes> for each scheduler:

  boxplots      one per metric, the schedulers side by side, paired by seed
  reward curve  the paper's Figure 6: mean cumulative reward per episode over
                the DynQ motes, now over 15000 slotframes
  epsilon       one node of each learner, the Q-static threshold marked

All through figures.py, so palette, style and PDF output match.

  python final_figures.py --inputfolder /dados --motes 50 \
      --arms dynq qstatic rlsf msf emsf --out /figuras
"""
from __future__ import division
from __future__ import print_function

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import figures
import figure_data
from epsilon_figure import trajectory


NOMES = {
    'dynq': 'DynQ', 'qstatic': 'Q-static', 'rlsf': 'RL-SF',
    'msf': 'MSF', 'emsf': 'EMSF', 'dynq_sem_remocao': 'DynQ, plain removal',
}
METRICAS = ['latency', 'pdr', 'join_time', 'lifetime_mean', 'score',
            'sixp_per_packet']


def nome(arm):
    return NOMES.get(arm, arm)


def boxplots(inputfolder, arms, motes, out):
    pastas = ['{0}_n{1}'.format(a, motes) for a in arms]
    for metrica in METRICAS:
        dados = figure_data.paired_series(inputfolder, pastas, motes, metrica)
        if isinstance(dados, dict):
            dados = [dados[p] for p in pastas]
        if not all(dados):
            print('%s: sem dados para algum braco, pulando' % metrica)
            continue
        caminho = os.path.join(out, '%s_boxplot.pdf' % metrica)
        figures.boxplots(dados, [nome(a) for a in arms],
                         figure_data.ROTULOS[metrica], caminho)
        print('escrito em', caminho)


def cumulative_reward_curve(pasta, zero_fill=False, min_share=0.5):
    """Mean cumulative reward per episode over the motes of every run.

    zero_fill reproduces the published arithmetic, which counted a mote that
    had stopped deciding as a zero. Without it, an episode is averaged over
    the motes that reached it, and the curve stops where fewer than min_share
    of the motes are still deciding, so its tail is not a headcount.
    """
    series = []
    for caminho in glob.glob(os.path.join(pasta, 'run_*', '*', 'qlearning_stats.json')):
        with open(caminho) as f:
            d = json.load(f)
        acumulado = d.get('CUMULATIVE_REWARD')
        if acumulado is None:
            continue
        if not acumulado and not zero_fill:
            # a mote that never decided has nothing to average; the published
            # arithmetic keeps it and counts it as zero in every episode
            continue
        series.append({int(k): v for k, v in acumulado.items()})
    if not series:
        raise ValueError('no CUMULATIVE_REWARD under %s' % pasta)
    ultimo = max(max(s) for s in series if s)
    episodios = range(1, ultimo + 1)
    media, presentes = [], []
    for e in episodios:
        valores = [s[e] for s in series if e in s]
        presentes.append(len(valores))
        if zero_fill:
            # the published arithmetic: a mote that has not reached this
            # episode, or never decided at all, counts as a zero
            media.append(sum(valores) / float(len(series)))
        else:
            media.append(np.mean(valores) if valores else float('nan'))
    if not zero_fill:
        corte = next((i for i, n in enumerate(presentes)
                      if n < min_share * len(series)), len(episodios))
        episodios, media = list(episodios)[:corte], media[:corte]
    return list(episodios), media, len(series)


def reward_figure(inputfolder, arm, motes, out, zero_fill):
    pasta = figure_data.folder_of(inputfolder, '{0}_n{1}'.format(arm, motes), motes)
    x, y, motes_lidos = cumulative_reward_curve(pasta, zero_fill)
    caminho = os.path.join(out, 'reward_cumulative_mean.pdf')
    figures.curves(x, [y], ['%s, mean over %d motes' % (nome(arm), motes_lidos)],
                   'Cumulative reward', caminho, xlabel='Episode')
    print('escrito em', caminho, '(%d episodios)' % len(x))


def epsilon(inputfolder, motes, out, threshold, slotframes):
    figures.use_paper_style()
    colours, _ = figures.styles_for(2)
    total = float(slotframes * 101)
    fig, ax = plt.subplots()
    for arm, colour in (('dynq', colours[0]), ('qstatic', colours[1])):
        pasta = figure_data.folder_of(inputfolder, '{0}_n{1}'.format(arm, motes), motes)
        x, y = trajectory(pasta)
        ax.plot(x / total, y, color=colour[:3], linewidth=1.8, label=nome(arm))
    ax.axhline(threshold, color='gray', linestyle='--', linewidth=1.2)
    ax.text(0.99, threshold + 0.02, r'Q-static threshold $\epsilon_{th}$',
            ha='right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Fraction of the run')
    ax.set_ylabel(r'$\epsilon$')
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    caminho = os.path.join(out, 'epsilon-decay.pdf')
    fig.savefig(caminho)
    plt.close(fig)
    print('escrito em', caminho)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputfolder', required=True)
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reward-arm', default='dynq')
    parser.add_argument('--zero-fill', action='store_true',
                        help="the published arithmetic for the reward curve")
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--slotframes', type=int, default=15000)
    parser.add_argument('--skip-epsilon', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    boxplots(args.inputfolder, args.arms, args.motes, args.out)
    if args.reward_arm in args.arms:
        reward_figure(args.inputfolder, args.reward_arm, args.motes, args.out,
                      args.zero_fill)
    if not args.skip_epsilon and 'dynq' in args.arms and 'qstatic' in args.arms:
        try:
            epsilon(args.inputfolder, args.motes, args.out, args.threshold,
                    args.slotframes)
        except ValueError as e:
            print('sem figura de epsilon: %s' % e)


if __name__ == '__main__':
    main()
