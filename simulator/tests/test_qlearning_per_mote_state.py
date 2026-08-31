"""Each mote must run its own Q-learning agent.

The scheduling functions used to declare their mutable state as class
attributes, which made every mote in the network read and write one shared
Q-table and one shared set of moving-average windows.
"""
from __future__ import absolute_import

import pytest

# state that must not be shared between motes
MUTABLE_STATE = [
    'Q_table',
    'array_rxs_acks',
    'array_queue_ratio',
    'array_energy_consumed',
]


@pytest.fixture(params=['Qlearning', 'QlearningSBRC24'])
def sf_class(request):
    return request.param


def test_state_is_per_mote(sim_engine, sf_class):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : sf_class,
        }
    )

    for attr in MUTABLE_STATE:
        objects = [id(getattr(mote.sf, attr)) for mote in engine.motes]
        assert len(set(objects)) == len(engine.motes), (
            '{0}.{1} is shared by {2} motes'.format(
                sf_class, attr, len(engine.motes)
            )
        )


def test_state_lives_on_the_instance(sim_engine, sf_class):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : sf_class,
        }
    )

    for mote in engine.motes:
        for attr in MUTABLE_STATE:
            assert attr in mote.sf.__dict__, (
                '{0}.{1} is a class attribute'.format(sf_class, attr)
            )


def test_writes_do_not_leak_between_motes(sim_engine, sf_class):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : sf_class,
        }
    )

    first, second = engine.motes[0], engine.motes[1]

    first.sf.array_rxs_acks.append(1)
    assert second.sf.array_rxs_acks == []

    first.sf.Q_table[0] = ['written by the first mote']
    assert second.sf.Q_table.get(0) != ['written by the first mote']
