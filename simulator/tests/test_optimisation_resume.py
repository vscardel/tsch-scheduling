"""The optimisation has to survive being killed.

Two configurations never finished. Each attempt met the same accumulated risk
of the run stopping dead, and starting over reset the progress but not the
risk, so more attempts did not help. Recording every finished evaluation turns
a lost run into a lost evaluation.
"""
from __future__ import absolute_import

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import runExperiments as rx


@pytest.fixture
def em_pasta_limpa(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    return tmpdir


def test_sem_registro_comeca_do_zero(em_pasta_limpa):
    assert rx.load_evaluations('seja_o_que_for') == []


def test_uma_avaliacao_e_relida(em_pasta_limpa):
    rx.record_evaluation('exp', [0.1, 0.2], 0.5)
    assert rx.load_evaluations('exp') == [{'x': [0.1, 0.2], 'y': 0.5}]


def test_avaliacoes_se_acumulam_na_ordem(em_pasta_limpa):
    rx.record_evaluation('exp', [0.1], 0.9)
    rx.record_evaluation('exp', [0.2], 0.4)
    rx.record_evaluation('exp', [0.3], 0.7)
    registro = rx.load_evaluations('exp')
    assert [e['y'] for e in registro] == [0.9, 0.4, 0.7]


def test_configuracoes_nao_se_misturam(em_pasta_limpa):
    rx.record_evaluation('a', [0.1], 0.5)
    rx.record_evaluation('b', [0.9], 0.1)
    assert rx.load_evaluations('a') == [{'x': [0.1], 'y': 0.5}]
    assert rx.load_evaluations('b') == [{'x': [0.9], 'y': 0.1}]


def test_arquivo_corrompido_nao_derruba_a_rodada(em_pasta_limpa):
    """A run killed mid-write leaves half a file; that must not be fatal."""
    with open(rx.evaluations_path('exp'), 'w') as f:
        f.write('[{"x": [0.1], "y":')
    assert rx.load_evaluations('exp') == []


def test_valores_viram_float_serializavel(em_pasta_limpa):
    """skopt hands back numpy scalars, which json refuses."""
    import numpy as np
    rx.record_evaluation('exp', [np.float64(0.25)], np.float64(0.75))
    with open(rx.evaluations_path('exp')) as f:
        conteudo = json.load(f)
    assert conteudo == [{'x': [0.25], 'y': 0.75}]
