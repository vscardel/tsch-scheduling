# -*- coding: utf-8 -*-
"""One place where a figure of this paper decides how it looks.

The plots of generate_plots.py look right, and every number in them was read
the wrong way: its loader opens output_cpu0.dat.kpi alone, so with ten CPUs it
drew nine tenths of the runs away and the boxes came out narrower than the
data. The same file also fixes six series and their labels in the body of the
drawing function, which is why a comparison of any other set of schedulers
cannot reuse it.

So the drawing lives here, the reading lives in compare_schedulers, and a
caller passes data in. Three consequences worth having:

  - a figure and a statistic in the text are computed from the same runs and
    the same metric definition, because both come from METRICS
  - a figure of 100 or 200 motes is the same call with another folder
  - the number of series follows the data, and labels are the caller's

Works on Python 2 and 3, and with or without a LaTeX installed: the paper's
figures want usetex, and neither this container nor this machine has latex, so
mathtext stands in and the notation is written so that both render it.
"""
from __future__ import division
from __future__ import print_function

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# The palette and the hatches of the published figures, kept so that a new
# figure sits beside an old one without looking like another paper. Alpha is
# in the colour rather than on the artist, which is how the original did it.
PALETTE = [
    (33 / 255., 133 / 255., 197 / 255., 0.7),   # blue
    (151 / 255., 102 / 255., 255 / 255., 0.7),  # purple
    (255 / 255., 102 / 255., 178 / 255., 0.7),  # pink
    (224 / 255., 202 / 255., 60 / 255., 0.7),   # yellow
    (243 / 255., 66 / 255., 19 / 255., 0.7),    # red orange
    (38 / 255., 166 / 255., 91 / 255., 0.7),    # green
]
HATCHES = ['//', '..', 'xx', 'oo', '\\\\', '++']

STYLE = 'presentation.mplstyle'
SIZES = {
    'axes.titlesize' : 18,
    'axes.labelsize' : 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
}


def latex_available():
    """Whether text.usetex can be switched on without the run dying."""
    for caminho in os.environ.get('PATH', '').split(os.pathsep):
        if caminho and os.path.exists(os.path.join(caminho, 'latex')):
            return True
    return False


def use_paper_style():
    """The style of the published figures, and LaTeX only if there is one.

    Returns whether usetex ended up on, since a caller writing an axis label
    has to know which of the two syntaxes it can use.
    """
    if os.path.exists(STYLE):
        plt.style.use(STYLE)
    plt.rcParams.update(SIZES)
    com_latex = latex_available()
    plt.rcParams['text.usetex'] = com_latex
    return com_latex


def styles_for(n):
    """A colour and a hatch per series, and never two the same.

    Two series that look alike in a paper figure is a defect, not a
    degradation, so this refuses rather than cycling. Past six series the
    answer is to split the figure, not to invent a seventh hatch.
    """
    if n < 1:
        raise ValueError('a figure with no series')
    if n > len(PALETTE):
        raise ValueError(
            '{0} series and {1} distinct styles: split the figure by '
            'scenario or by network size instead of cycling'.format(
                n, len(PALETTE)
            )
        )
    return PALETTE[:n], HATCHES[:n]


def _check(data, labels):
    if len(data) != len(labels):
        raise ValueError(
            '{0} series and {1} labels: the labels of a figure are the '
            "caller's, and a mismatch is how the wrong name ends up under "
            'the wrong box'.format(len(data), len(labels))
        )
    vazias = [l for serie, l in zip(data, labels) if len(serie) == 0]
    if vazias:
        raise ValueError('no data for: {0}'.format(', '.join(vazias)))


def _finish(fig, ax, ylabel, path, handles=None, legend_loc='best'):
    if ylabel:
        ax.set_ylabel(ylabel)
    if handles:
        ax.legend(handles=handles, loc=legend_loc, frameon=True)
    fig.tight_layout()
    pasta = os.path.dirname(path)
    if pasta and not os.path.isdir(pasta):
        os.makedirs(pasta)
    fig.savefig(path, format='pdf', dpi=300)
    plt.close(fig)
    return path


def boxplots(data, labels, ylabel, path, marcar_media=True):
    """One box per series, in the order given.

    data is a list of lists of per run values, so the box shows the spread
    across runs, which is what the paired statistics are computed on.
    """
    _check(data, labels)
    use_paper_style()
    colours, hatches = styles_for(len(data))

    fig, ax = plt.subplots()
    box = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        showmeans=marcar_media,
        meanline=marcar_media,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=1.5, color='black'),
        meanprops=dict(linewidth=1.5, color='red'),
    )
    for patch, colour, hatch in zip(box['boxes'], colours, hatches):
        patch.set_facecolor(colour)
        patch.set_edgecolor('black')
        patch.set_hatch(hatch)
    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    for cap in box['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)
    for flier in box['fliers']:
        flier.set(marker='o', color='gray', alpha=0.5, markersize=5)

    handles = [
        mpatches.Patch(facecolor=c, edgecolor='black', hatch=h, label=l)
        for c, h, l in zip(colours, hatches, labels)
    ]
    return _finish(fig, ax, ylabel, path, handles)


def confidence_interval(values, confidence=0.95):
    """Normal interval for the mean, as the published bar plots used."""
    from scipy.stats import norm
    n = len(values)
    if n < 2:
        return (float('nan'), float('nan'))
    media = np.mean(values)
    erro = norm.ppf(1 - (1 - confidence) / 2) * np.std(values, ddof=1) / np.sqrt(n)
    return (media - erro, media + erro)


def bars(data, labels, ylabel, path, confidence=0.95):
    """The mean of each series with its confidence interval."""
    _check(data, labels)
    use_paper_style()
    colours, hatches = styles_for(len(data))

    medias = [np.mean(s) for s in data]
    baixos = []
    for serie, media in zip(data, medias):
        baixo, alto = confidence_interval(serie, confidence)
        baixos.append(media - baixo if np.isfinite(baixo) else 0.0)

    fig, ax = plt.subplots()
    posicoes = np.arange(len(data))
    barras = ax.bar(
        posicoes, medias, yerr=baixos, capsize=4,
        edgecolor='black', linewidth=1.2,
    )
    for barra, colour, hatch in zip(barras, colours, hatches):
        barra.set_facecolor(colour)
        barra.set_hatch(hatch)
    ax.set_xticks(posicoes)
    ax.set_xticklabels(labels)

    handles = [
        mpatches.Patch(facecolor=c, edgecolor='black', hatch=h, label=l)
        for c, h, l in zip(colours, hatches, labels)
    ]
    return _finish(fig, ax, ylabel, path, handles)


def curves(x, series, labels, ylabel, path, xlabel=None, bands=None,
           marcos=None):
    """One line per series over a shared x, with an optional band each.

    bands is a list of (low, high) pairs, one per series, for the spread
    across runs. marcos are vertical lines, which is where a disturbance goes.
    """
    _check(series, labels)
    use_paper_style()
    colours, _ = styles_for(len(series))

    fig, ax = plt.subplots()
    for serie, colour, label in zip(series, colours, labels):
        ax.plot(x, serie, color=colour[:3], linewidth=1.8, label=label)
    if bands:
        for (baixo, alto), colour in zip(bands, colours):
            ax.fill_between(x, baixo, alto, color=colour[:3], alpha=0.18,
                            linewidth=0)
    for marco in (marcos or []):
        ax.axvline(marco, color='gray', linestyle='--', linewidth=1.2)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(loc='best', frameon=True)
    return _finish(fig, ax, ylabel, path)
