"""The removal rule, on and off.

A reviewer calls the utilisation-aware removal a hard-coded heuristic doing the
agent's job, and says it undermines the claim of an adaptive scheduler. The only
honest answer is to run DynQ both ways and report the difference, so the rule
has to be switchable without touching anything else.

What is not part of the rule, and stays on in both arms, is the floor of one
cell. MSF has the same floor, and without it a mote can remove its way into
silence.
"""
from __future__ import absolute_import

import pytest

from SimEngine.Mote import MoteDefines as d


class Celula(object):
    def __init__(self, slot, num_tx, num_tx_ack, options=None):
        self.slot_offset = slot
        self.channel_offset = 0
        self.num_tx = num_tx
        self.num_tx_ack = num_tx_ack
        self.num_rx = 0
        self.options = options or [d.CELLOPTION_TX]


def agente_com(sim_engine, monkeypatch, ligado, cells):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes'         : 4,
            'sf_class'              : 'Qlearning',
            'factorial_combinations': ['traffic', 'queue', 'charge'],
        }
    )
    mote = engine.motes[1]
    mote.sf.start()
    mote.sf.SMART_CELL_REMOVAL = ligado
    monkeypatch.setattr(mote.rpl, 'getPreferredParent', lambda: 'pai')
    monkeypatch.setattr(mote.tsch, 'get_cells', lambda vizinho, handle: cells)
    return mote.sf


def test_ligado_deixa_passar_so_a_celula_pouco_usada(sim_engine, monkeypatch):
    cells = [
        Celula(1, num_tx=10, num_tx_ack=10),   # 100%, fica
        Celula(2, num_tx=10, num_tx_ack=3),    # 30%, pode ir
    ]
    sf = agente_com(sim_engine, monkeypatch, True, cells)
    saida = sf._get_unused_cells([d.CELLOPTION_TX])
    assert [c['slotOffset'] for c in saida] == [2]


def test_desligado_oferece_todas(sim_engine, monkeypatch):
    """The stock behaviour picks among the allocated cells, used or not."""
    cells = [
        Celula(1, num_tx=10, num_tx_ack=10),
        Celula(2, num_tx=10, num_tx_ack=3),
        Celula(3, num_tx=0, num_tx_ack=0),
    ]
    sf = agente_com(sim_engine, monkeypatch, False, cells)
    saida = sf._get_unused_cells([d.CELLOPTION_TX])
    assert sorted(c['slotOffset'] for c in saida) == [1, 2, 3]


def test_desligado_nao_e_sempre_a_mesma_ordem(sim_engine, monkeypatch):
    """Random, not first-in-list, otherwise the ablation is its own rule."""
    cells = [Celula(i, num_tx=10, num_tx_ack=10) for i in range(8)]
    sf = agente_com(sim_engine, monkeypatch, False, cells)
    primeiras = set(
        sf._get_unused_cells([d.CELLOPTION_TX])[0]['slotOffset']
        for _ in range(60)
    )
    assert len(primeiras) > 1


def test_os_dois_lados_respeitam_a_opcao_da_celula(sim_engine, monkeypatch):
    cells = [
        Celula(1, num_tx=10, num_tx_ack=3),
        Celula(2, num_tx=10, num_tx_ack=3, options=[d.CELLOPTION_RX]),
    ]
    sf = agente_com(sim_engine, monkeypatch, True, cells)
    for ligado in (True, False):
        sf.SMART_CELL_REMOVAL = ligado
        saida = sf._get_unused_cells([d.CELLOPTION_TX])
        assert [c['slotOffset'] for c in saida] == [1]


def test_a_regra_vem_ligada_por_padrao(sim_engine):
    """The published method has it on; the ablation is the exception."""
    engine = sim_engine(diff_config={'exec_numMotes': 4, 'sf_class': 'Qlearning'})
    assert engine.motes[1].sf.SMART_CELL_REMOVAL is True
