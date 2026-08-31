"""The Q-table must have a row for every state the agent can reach.

Each state variable is discretised to 0 or 1, so N factors give 2**N states.
STATE_SIZE used to come from the configuration independently of the factor
list, and a run configured with fewer rows than states crashed with a KeyError
the first time a mote reached a state that had no row.
"""
from __future__ import absolute_import

import pytest

FACTOR_SETS = [
    ['queue'],
    ['traffic', 'queue'],
    ['traffic', 'queue', 'charge'],
]


@pytest.fixture(params=FACTOR_SETS)
def factors(request):
    return request.param


def test_table_covers_every_state(sim_engine, factors):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes'         : 4,
            'sf_class'              : 'Qlearning',
            'factorial_combinations': factors,
            # deliberately wrong, and deliberately ignored
            'STATE_SIZE'            : 2,
        }
    )

    expected = 2 ** len(factors)

    for mote in engine.motes:
        if mote.dagRoot:
            # the root does not run an agent
            continue
        mote.sf.start()   # the table is built here, not in __init__
        assert mote.sf.STATE_SIZE == expected
        assert sorted(mote.sf.Q_table.keys()) == list(range(expected))


def test_every_reachable_state_has_a_row(sim_engine, factors):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes'         : 4,
            'sf_class'              : 'Qlearning',
            'factorial_combinations': factors,
        }
    )

    mote = engine.motes[1]
    mote.sf.start()

    # the state number is the binary digits of the discretised factors, so the
    # largest one is all ones
    highest = 2 ** len(factors) - 1
    assert highest in mote.sf.Q_table
