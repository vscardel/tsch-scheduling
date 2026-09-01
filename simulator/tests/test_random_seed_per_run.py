"""An integer seed has to make the batch reproducible without collapsing it.

exec_randomSeed defaulted to 'random', so a published configuration did not say
what actually ran. Handing an integer to every run of a batch is not the answer
either: every run would start from the same seed and the ten runs behind an
average would be ten copies of one run. The integer is the seed of the batch and
each run adds its own run id to it.
"""
from __future__ import absolute_import

import pytest

BATCH_SEED = 1000


@pytest.mark.parametrize('run_id', [0, 1, 7])
def test_a_run_uses_the_batch_seed_plus_its_run_id(sim_engine, run_id):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes'   : 4,
            'exec_randomSeed' : BATCH_SEED,
        },
        run_id = run_id
    )

    assert engine.random_seed == BATCH_SEED + run_id


def test_without_a_run_id_the_batch_seed_is_used_as_is(sim_engine):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes'   : 4,
            'exec_randomSeed' : BATCH_SEED,
        }
    )

    assert engine.random_seed == BATCH_SEED
