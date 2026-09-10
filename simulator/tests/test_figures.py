"""One place decides how a figure looks, and it refuses the ways they go wrong.

The published plotting code had two defects of a kind a reader cannot see in
the output: it read one CPU's runs out of ten, and it fixed six series and
their labels inside the drawing function. A figure drawn on a quarter of the
data, or with the wrong name under a box, looks exactly as convincing as a
correct one, so these are held by tests rather than by care.
"""
import json
import os

import pytest

import figures
import figure_data


@pytest.fixture
def destino(tmpdir):
    return str(tmpdir.join('fig.pdf'))


# ---------------------------------------------------- refusing the mistakes

def test_a_label_per_series_or_nothing_is_drawn(destino):
    with pytest.raises(ValueError) as erro:
        figures.boxplots([[1, 2], [3, 4], [5, 6]], ['a', 'b'], 'y', destino)
    assert 'labels' in str(erro.value)
    assert not os.path.exists(destino)


def test_more_series_than_distinct_styles_is_refused(destino):
    dados = [[1, 2]] * (len(figures.PALETTE) + 1)
    rotulos = ['s%d' % i for i in range(len(dados))]
    with pytest.raises(ValueError) as erro:
        figures.boxplots(dados, rotulos, 'y', destino)
    assert 'split the figure' in str(erro.value)


def test_an_empty_series_is_refused_by_name(destino):
    with pytest.raises(ValueError) as erro:
        figures.boxplots([[1, 2], []], ['cheia', 'vazia'], 'y', destino)
    assert 'vazia' in str(erro.value)


def test_every_series_gets_its_own_colour_and_hatch():
    for n in range(1, len(figures.PALETTE) + 1):
        cores, hachuras = figures.styles_for(n)
        assert len(cores) == n and len(hachuras) == n
        assert len(set(cores)) == n
        assert len(set(hachuras)) == n


# ------------------------------------------------------- drawing something

def test_a_boxplot_is_written_as_a_pdf(destino):
    figures.boxplots([[1, 2, 3], [2, 3, 4]], ['a', 'b'], 'y', destino)
    assert os.path.getsize(destino) > 1000
    with open(destino, 'rb') as f:
        assert f.read(4) == b'%PDF'


def test_bars_and_curves_write_too(tmpdir):
    b = str(tmpdir.join('b.pdf'))
    figures.bars([[1, 2, 3], [2, 3, 4]], ['a', 'b'], 'y', b)
    assert os.path.getsize(b) > 1000
    c = str(tmpdir.join('c.pdf'))
    figures.curves([0, 1, 2], [[1, 2, 3], [3, 2, 1]], ['a', 'b'], 'y', c,
                   marcos=[1])
    assert os.path.getsize(c) > 1000


def test_a_missing_directory_is_created(tmpdir):
    destino = str(tmpdir.join('sub').join('dir').join('f.pdf'))
    figures.boxplots([[1, 2]], ['a'], 'y', destino)
    assert os.path.exists(destino)


def test_usetex_is_only_switched_on_when_latex_exists():
    import matplotlib.pyplot as plt
    figures.use_paper_style()
    assert plt.rcParams['text.usetex'] == figures.latex_available()


# ----------------------------------------------- reading the runs behind it

def _escreve_kpi(pasta, nome, runs):
    if not os.path.isdir(pasta):
        os.makedirs(pasta)
    with open(os.path.join(pasta, nome), 'w') as f:
        json.dump(runs, f)


def _run(pdr):
    return {
        'global-stats': {
            'e2e-upstream-delivery': [{'name': 'E2E Upstream Delivery Ratio',
                                       'value': pdr}],
        }
    }


def test_every_cpu_file_reaches_the_figure(tmpdir):
    """The defect that made every published box too narrow."""
    pasta = str(tmpdir.join('arm').join('exec_numMotes_50'))
    _escreve_kpi(pasta, 'output_cpu0.dat.kpi', {'0': _run(0.5), '1': _run(0.6)})
    _escreve_kpi(pasta, 'output_cpu1.dat.kpi', {'2': _run(0.7), '3': _run(0.8)})
    valores = figure_data.series(str(tmpdir), 'arm', 50, 'pdr')
    assert sorted(valores) == [0.5, 0.6, 0.7, 0.8]


def test_a_paired_figure_uses_only_the_runs_every_arm_has(tmpdir):
    """A box per arm drawn on a different set of runs is not paired with the
    statistics quoted beside it."""
    a = str(tmpdir.join('a').join('exec_numMotes_50'))
    b = str(tmpdir.join('b').join('exec_numMotes_50'))
    _escreve_kpi(a, 'output_cpu0.dat.kpi',
                 {'0': _run(0.5), '1': _run(0.6), '2': _run(0.7)})
    _escreve_kpi(b, 'output_cpu0.dat.kpi', {'0': _run(0.1), '1': _run(0.2)})
    series = figure_data.paired_series(str(tmpdir), ['a', 'b'], 50, 'pdr')
    assert [len(s) for s in series] == [2, 2]
    assert sorted(series[0]) == [0.5, 0.6]


def test_an_arm_with_no_results_reads_as_empty(tmpdir):
    assert figure_data.series(str(tmpdir), 'nao_existe', 50, 'pdr') == []


def test_every_metric_of_the_statistics_has_an_axis_label():
    """A figure whose axis has no unit is a figure nobody can read."""
    from compare_schedulers import METRICS
    for nome, _, _ in METRICS:
        assert nome in figure_data.ROTULOS, nome
        assert figure_data.ROTULOS[nome].strip()
