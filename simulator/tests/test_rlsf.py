"""RL-SF, the baseline from Pratama and Chung, ICEIEC 2022.

Two reviewers asked for a comparison against an existing reinforcement learning
scheduling function, and both named this one. These tests pin the parts a
reviewer could check against the paper: the state is the queue occupancy, the
action is an absolute cell count rather than a change to one, the reward is
Eq. 3, and the Q update is the standard one rather than the double subtraction
the paper's Eq. 1 and Eq. 2 literally spell out.

They also pin the two things that keep the comparison honest: MSF's threshold
rule is off, so RL-SF's cell count comes from the agent and nowhere else, and
the agent cannot ask for zero cells.
"""
from __future__ import absolute_import

import random

import pytest

from SimEngine.Mote import MoteDefines as d


@pytest.fixture
def agente(sim_engine):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : 'RLSF',
        }
    )
    mote = engine.motes[1]
    mote.sf.start()
    return mote


# === o estado ============================================================

def test_estado_e_a_fila(agente, monkeypatch):
    """The state is the number of queued messages, nothing else."""
    sf = agente.sf
    for fila in (0, 1, 5):
        monkeypatch.setattr(agente.tsch, 'txQueue', [{}] * fila)
        assert sf._observe_state() == fila


def test_estado_satura_na_ultima_linha(agente, monkeypatch):
    """A queue longer than the table saturates instead of indexing past the end."""
    sf = agente.sf
    monkeypatch.setattr(agente.tsch, 'txQueue', [{}] * 400)
    estado = sf._observe_state()
    assert estado == sf.NUM_QUEUE_STATES - 1
    assert sf.q_table[estado]          # the row exists


# === a acao ==============================================================

def test_acao_e_contagem_absoluta_nao_variacao(agente, monkeypatch):
    """Action k means "hold k+1 cells", so 6P negotiates the gap to today's count."""
    sf = agente.sf
    pedidos = []
    monkeypatch.setattr(sf, '_num_negotiated_tx_cells', lambda vizinho: 2)
    monkeypatch.setattr(sf, '_request_adding_cells',
                        lambda **kw: pedidos.append(('add', kw['num_tx_cells'])))
    monkeypatch.setattr(sf, '_request_deleting_cells',
                        lambda **kw: pedidos.append(('del', kw['num_cells'])))
    sf.retry_count['pai'] = -1

    sf._apply_action('pai', state=0, action=4)      # quer 5, tem 2
    assert pedidos == [('add', 3)]


def test_acao_menor_que_o_atual_remove(agente, monkeypatch):
    sf = agente.sf
    pedidos = []
    monkeypatch.setattr(sf, '_num_negotiated_tx_cells', lambda vizinho: 6)
    monkeypatch.setattr(sf, '_request_adding_cells',
                        lambda **kw: pedidos.append(('add', kw['num_tx_cells'])))
    monkeypatch.setattr(sf, '_request_deleting_cells',
                        lambda **kw: pedidos.append(('del', kw['num_cells'])))
    sf.retry_count['pai'] = -1

    sf._apply_action('pai', state=0, action=1)      # quer 2, tem 6
    assert pedidos == [('del', 4)]


def test_acao_igual_ao_atual_nao_negocia(agente, monkeypatch):
    """Choosing the count it already holds must not spend a 6P transaction."""
    sf = agente.sf
    pedidos = []
    monkeypatch.setattr(sf, '_num_negotiated_tx_cells', lambda vizinho: 3)
    monkeypatch.setattr(sf, '_request_adding_cells',
                        lambda **kw: pedidos.append('add'))
    monkeypatch.setattr(sf, '_request_deleting_cells',
                        lambda **kw: pedidos.append('del'))
    sf.retry_count['pai'] = -1

    sf._apply_action('pai', state=0, action=2)      # quer 3, tem 3
    assert pedidos == []


def test_nao_negocia_com_transacao_em_voo(agente, monkeypatch):
    """6P allows one transaction per peer, so a busy neighbour is left alone."""
    sf = agente.sf
    pedidos = []
    monkeypatch.setattr(sf, '_num_negotiated_tx_cells', lambda vizinho: 1)
    monkeypatch.setattr(sf, '_request_adding_cells',
                        lambda **kw: pedidos.append('add'))
    sf.retry_count['pai'] = 0                       # ocupado

    sf._apply_action('pai', state=0, action=7)
    assert pedidos == []


def test_agente_nunca_pede_zero_celulas(agente):
    """The action space starts at one cell, so RL-SF cannot empty its schedule."""
    sf = agente.sf
    alvos = [acao + 1 for acao in range(sf.MAX_CELLS)]
    assert min(alvos) == 1
    assert max(alvos) == sf.MAX_CELLS


# === a recompensa, Eq. 3 a Eq. 6 =========================================

def test_normalizacao_leva_razao_para_menos_um_e_um(agente):
    sf = agente.sf
    assert sf._normalize(0.0) == -1.0
    assert sf._normalize(0.5) == 0.0
    assert sf._normalize(1.0) == 1.0


def test_recompensa_segue_a_equacao_3(agente, monkeypatch):
    """Half full queue, a quarter of the cells idle, no drop."""
    sf = agente.sf
    monkeypatch.setattr(agente.tsch, 'txQueue', [{}] * 5)
    monkeypatch.setattr(agente.tsch, 'txQueueSize', 10)
    sf.cells_elapsed = 4
    sf.cells_used = 3
    sf.dropped_a_packet = False

    # QU = 0.5 -> N(1-0.5) = 0.0 ; UC = 0.25 -> N(1-0.25) = 0.5 ; PD = 0
    assert sf._compute_reward() == pytest.approx(0.5)


def test_queda_de_pacote_penaliza(agente, monkeypatch):
    sf = agente.sf
    monkeypatch.setattr(agente.tsch, 'txQueue', [{}] * 5)
    monkeypatch.setattr(agente.tsch, 'txQueueSize', 10)
    sf.cells_elapsed = 4
    sf.cells_used = 3

    sf.dropped_a_packet = False
    sem_queda = sf._compute_reward()
    sf.dropped_a_packet = True
    com_queda = sf._compute_reward()

    assert com_queda == pytest.approx(sem_queda - sf.THETA_DROP)


def test_sem_celula_decorrida_nao_conta_desperdicio(agente):
    """A slotframe in which no cell came round reports no waste, not total waste."""
    sf = agente.sf
    sf.cells_elapsed = 0
    sf.cells_used = 0
    assert sf._unused_cell_ratio() == 0.0


def test_fila_infinita_nao_estoura(agente, monkeypatch):
    sf = agente.sf
    monkeypatch.setattr(agente.tsch, 'txQueue', [{}] * 3)
    monkeypatch.setattr(agente.tsch, 'txQueueSize', float('inf'))
    assert sf._queue_utilization() == 0.0


def test_overflow_marca_a_queda(agente):
    """The PD term is fed by the TX queue overflow indication."""
    sf = agente.sf
    assert sf.dropped_a_packet is False
    sf.indication_queue_full()
    assert sf.dropped_a_packet is True


# === o aprendizado =======================================================

def test_atualizacao_e_a_padrao_nao_a_do_artigo(agente):
    """Q <- (1-a)Q + a(r + B maxQ').

    Eq. 1 and Eq. 2 of the paper, read literally, subtract the old value twice
    and would keep the table from converging. The standard update is used.
    """
    sf = agente.sf
    sf.ALFA, sf.BETA = 0.5, 0.9
    sf.q_table[0][0] = 2.0
    sf.q_table[1] = [0.0] * sf.MAX_CELLS
    sf.q_table[1][3] = 10.0

    sf._update_q_table(state=0, action=0, reward=1.0, next_state=1)

    # (1-0.5)*2.0 + 0.5*(1.0 + 0.9*10.0) = 1.0 + 5.0
    assert sf.q_table[0][0] == pytest.approx(6.0)


def test_epsilon_decai_e_para_no_piso(agente):
    sf = agente.sf
    sf.epsilon = sf.EPSILON_INIT
    anterior = sf.epsilon
    for _ in range(20):
        sf.epsilon = max(sf.epsilon * sf.EPSILON_DECAY, sf.EPSILON_END)
        assert sf.epsilon <= anterior
        anterior = sf.epsilon

    sf.epsilon = sf.EPSILON_END
    sf.epsilon = max(sf.epsilon * sf.EPSILON_DECAY, sf.EPSILON_END)
    assert sf.epsilon == sf.EPSILON_END


def test_explora_com_epsilon_alto_e_explota_com_baixo(agente, monkeypatch):
    sf = agente.sf
    sf.q_table[0] = [0.0] * sf.MAX_CELLS
    sf.q_table[0][5] = 1.0

    sf.epsilon = 0.0
    assert sf._choose_action(0) == 5

    sf.epsilon = 1.0
    monkeypatch.setattr(random, 'randrange', lambda n: 2)
    assert sf._choose_action(0) == 2


def test_empate_em_linha_nova_nao_escolhe_sempre_a_mesma(agente):
    """A fresh row is all zeros; always taking the first column would freeze it."""
    sf = agente.sf
    sf.epsilon = 0.0
    escolhidas = set(sf._choose_action(0) for _ in range(200))
    assert len(escolhidas) > 1


# === a comparacao fica honesta ===========================================

def test_regra_de_limiar_do_msf_esta_desligada(agente, monkeypatch):
    """MSF's own rule must not add cells behind the agent's back."""
    sf = agente.sf
    pedidos = []
    monkeypatch.setattr(sf, '_request_adding_cells',
                        lambda **kw: pedidos.append('add'))
    monkeypatch.setattr(sf, '_request_deleting_cells',
                        lambda **kw: pedidos.append('del'))
    sf.retry_count['pai'] = -1
    sf.tx_cell_utilization = 1.0                    # bem acima do limiar do MSF

    sf._adapt_to_traffic('pai', sf.TX_CELL_OPT)
    assert pedidos == []


def test_nao_herda_a_remocao_inteligente_do_dynq(agente):
    """The utilisation-aware removal belongs to DynQ, not to this baseline."""
    sf = agente.sf
    assert not hasattr(sf, '_is_unused_cell')
    assert not hasattr(sf, '_get_unused_cells')


def test_conta_celulas_do_pai_para_a_recompensa(agente, monkeypatch):
    """Only cells towards the preferred parent feed the unused-cell term."""
    sf = agente.sf

    class Celula(object):
        def __init__(self, mac, options):
            self.mac_addr = mac
            self.options = options

    monkeypatch.setattr(agente.rpl, 'getPreferredParent', lambda: 'pai')
    monkeypatch.setattr(agente.tsch, 'txQueue', [])

    sf.indication_tx_cell_elapsed(Celula('pai', [d.CELLOPTION_TX]), True)
    sf.indication_tx_cell_elapsed(Celula('pai', [d.CELLOPTION_TX]), None)
    sf.indication_tx_cell_elapsed(Celula('outro', [d.CELLOPTION_TX]), True)

    assert sf.cells_elapsed == 2
    assert sf.cells_used == 1
