"""The score has to see every run, not one of them.

load_kpis opened output_cpu0.dat.kpi and nothing else. runSim gives each core
its own output file, so with ten cores it read one run and discarded nine. The
score computed from that drove the Bayesian optimisation and the factorial
analysis, and every experiment had to be pinned to a single core to be
trustworthy, which is also the slowest way to run one.
"""
from __future__ import absolute_import

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import runExperiments as rx


def escreve(pasta, nome, conteudo):
    caminho = os.path.join(str(pasta), nome)
    with open(caminho, 'w') as f:
        json.dump(conteudo, f)
    return caminho


def test_le_todos_os_nucleos(tmpdir):
    escreve(tmpdir, 'output_cpu0.dat.kpi', {'0': {'global-stats': {'x': 1}}})
    escreve(tmpdir, 'output_cpu1.dat.kpi', {'1': {'global-stats': {'x': 2}}})
    escreve(tmpdir, 'output_cpu2.dat.kpi', {'2': {'global-stats': {'x': 3}}})

    kpis = rx.load_kpis(str(tmpdir), 50)

    assert sorted(kpis.keys()) == ['0', '1', '2']


def test_um_nucleo_so_continua_funcionando(tmpdir):
    escreve(tmpdir, 'output_cpu0.dat.kpi',
            {'0': {'global-stats': {}}, '1': {'global-stats': {}}})
    kpis = rx.load_kpis(str(tmpdir), 50)
    assert sorted(kpis.keys()) == ['0', '1']


def test_pasta_vazia_devolve_none(tmpdir):
    """compute_score is guarded by "if kpis", so nothing must be truthy-empty."""
    assert rx.load_kpis(str(tmpdir), 50) is None


def test_um_arquivo_corrompido_nao_leva_os_outros(tmpdir):
    escreve(tmpdir, 'output_cpu0.dat.kpi', {'0': {'global-stats': {}}})
    with open(os.path.join(str(tmpdir), 'output_cpu1.dat.kpi'), 'w') as f:
        f.write('{"1": {')
    escreve(tmpdir, 'output_cpu2.dat.kpi', {'2': {'global-stats': {}}})

    kpis = rx.load_kpis(str(tmpdir), 50)

    assert sorted(kpis.keys()) == ['0', '2']


def test_nao_confunde_dat_com_kpi(tmpdir):
    """The raw .dat sits beside the .kpi and is not JSON."""
    with open(os.path.join(str(tmpdir), 'output_cpu0.dat'), 'w') as f:
        f.write('nao sou json\n')
    escreve(tmpdir, 'output_cpu0.dat.kpi', {'0': {'global-stats': {}}})

    assert sorted(rx.load_kpis(str(tmpdir), 50).keys()) == ['0']
