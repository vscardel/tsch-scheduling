"""The three things that stopped the agent from being able to learn.

One: exploration was a switch, not a chance. Epsilon was compared to a fixed
threshold, so a mote drew at random until the crossing and then never drew
again. Half the motes never reached the crossing, so their Q-table was built
and never read.

Two: the number of cells to move comes from the state, following the
manuscript, and it reaches zero. In the all-zero state inserting inserts
nothing and is the same as idling; in the all-one state removing removes
nothing. That is 39% of decisions in which two of the three columns of a row
cannot be told apart, and numpy's argmax breaks the tie the same way every time.

Three: the functions that discretise also append to the buffers behind the
moving average. A decision asked them four times, so the window covered two and
a half decisions instead of ten, and the row credited with the reward could
differ from the one the action was sized from.
"""
from __future__ import absolute_import

import random

import pytest

from SimEngine.Mote import MoteDefines as d


@pytest.fixture
def agent(sim_engine):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes'         : 4,
            'sf_class'              : 'Qlearning',
            'factorial_combinations': ['traffic', 'queue', 'charge'],
        }
    )
    mote = engine.motes[1]
    mote.sf.start()
    return mote


def decide(agent, monkeypatch, epslon, sorteio):
    """Run one decision with epsilon and the coin fixed, and report the action."""
    sf = agent.sf
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: 'parent')
    monkeypatch.setattr(agent, 'clear_to_send_EBs_DATA', lambda: True)
    monkeypatch.setattr(sf, 'compute_next_state', lambda factors: {})
    monkeypatch.setattr(sf, 'discretize_variables', lambda state: {
        'traffic': 0, 'queue': 0, 'charge': 0
    })
    monkeypatch.setattr(sf, 'map_discrete_state_to_number', lambda ds: 0)
    monkeypatch.setattr(random, 'random', lambda: sorteio)

    escolhida = []
    monkeypatch.setattr(sf, 'sixp_interface_add',
                        lambda **kw: escolhida.append(('add', kw['num_cells'])))
    monkeypatch.setattr(sf, 'sixp_interface_delete',
                        lambda **kw: escolhida.append(('del', kw['num_cells'])))

    # a tabela prefere remover, para distinguir da escolha aleatoria
    sf.Q_table[0] = [0.0, 1.0, 0.0]
    sf.EPISODE = 0
    sf.MIN_EPSLON = epslon
    sf.MAX_EPSLON = epslon        # epsilon fica fixo no valor pedido

    sf.adapt_to_traffic([d.CELLOPTION_TX], None, 'insertion')
    return sf.last_action, escolhida


def test_a_low_epsilon_makes_the_agent_read_its_table(agent, monkeypatch):
    # moeda alta, epsilon baixo: nao explora, usa a tabela
    acao, _ = decide(agent, monkeypatch, epslon=0.1, sorteio=0.9)

    assert acao == 1          # a tabela prefere remover


def test_a_high_epsilon_makes_the_agent_explore(agent, monkeypatch):
    # moeda baixa, epsilon alto: explora
    monkeypatch.setattr(random, 'choice', lambda opcoes: 2)
    acao, _ = decide(agent, monkeypatch, epslon=0.9, sorteio=0.1)

    assert acao == 2          # veio do sorteio, nao da tabela


def test_the_table_is_read_even_while_epsilon_is_high(agent, monkeypatch):
    # o ponto do epsilon-greedy: com epsilon 0.9 ainda ha 10% de uso da tabela.
    # Sob o limiar antigo esta decisao seria sorteada com certeza.
    acao, _ = decide(agent, monkeypatch, epslon=0.9, sorteio=0.95)

    assert acao == 1


def test_inserting_always_moves_at_least_one_cell(agent, monkeypatch):
    # estado 000: pela equacao do artigo N_insert seria zero
    monkeypatch.setattr(random, 'choice', lambda opcoes: 0)
    _, feito = decide(agent, monkeypatch, epslon=0.9, sorteio=0.1)

    assert feito == [('add', 1)]


def test_removing_always_moves_at_least_one_cell(agent, monkeypatch):
    sf = agent.sf
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: 'parent')
    monkeypatch.setattr(agent, 'clear_to_send_EBs_DATA', lambda: True)
    monkeypatch.setattr(sf, 'compute_next_state', lambda factors: {})
    # estado 111: pela equacao N_remove seria zero
    monkeypatch.setattr(sf, 'discretize_variables', lambda state: {
        'traffic': 1, 'queue': 1, 'charge': 1
    })
    monkeypatch.setattr(sf, 'map_discrete_state_to_number', lambda ds: 7)
    monkeypatch.setattr(random, 'random', lambda: 0.1)
    monkeypatch.setattr(random, 'choice', lambda opcoes: 1)

    feito = []
    monkeypatch.setattr(sf, 'sixp_interface_delete',
                        lambda **kw: feito.append(kw['num_cells']))
    sf.EPISODE = 0
    sf.MIN_EPSLON = sf.MAX_EPSLON = 0.9

    sf.adapt_to_traffic([d.CELLOPTION_TX], None, 'insertion')

    assert feito == [1]


def test_a_decision_discretises_once(agent, monkeypatch):
    sf = agent.sf
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: 'parent')
    monkeypatch.setattr(agent, 'clear_to_send_EBs_DATA', lambda: True)
    monkeypatch.setattr(sf, 'compute_next_state', lambda factors: {})
    monkeypatch.setattr(sf, 'sixp_interface_add', lambda **kw: None)
    monkeypatch.setattr(sf, 'sixp_interface_delete', lambda **kw: None)

    chamadas = []
    real = sf.discretize_variables
    monkeypatch.setattr(sf, 'discretize_variables',
                        lambda state: (chamadas.append(1),
                                       {'traffic': 0, 'queue': 0, 'charge': 0})[1])

    sf.EPISODE = 0
    sf.adapt_to_traffic([d.CELLOPTION_TX], None, 'insertion')
    primeira = len(chamadas)

    del chamadas[:]
    sf.adapt_to_traffic([d.CELLOPTION_TX], None, 'insertion')

    # a segunda decisao passa pelo compute_q_table, que antes discretizava
    # mais duas vezes. Uma leitura por decisao, sempre.
    assert primeira == 1
    assert len(chamadas) == 1


def test_the_moving_average_window_advances_once_per_decision(agent, monkeypatch):
    sf = agent.sf
    monkeypatch.setattr(agent.rpl, 'getPreferredParent', lambda: 'parent')
    monkeypatch.setattr(agent, 'clear_to_send_EBs_DATA', lambda: True)
    monkeypatch.setattr(sf, 'sixp_interface_add', lambda **kw: None)
    monkeypatch.setattr(sf, 'sixp_interface_delete', lambda **kw: None)

    sf.array_rxs_acks = []
    sf.EPISODE = 0

    for _ in range(3):
        sf.adapt_to_traffic([d.CELLOPTION_TX], None, 'insertion')

    # tres decisoes, tres observacoes. Eram doze.
    assert len(sf.array_rxs_acks) == 3
