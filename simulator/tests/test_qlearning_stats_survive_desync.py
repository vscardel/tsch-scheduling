"""The recorded Q-learning history must survive a desynchronisation.

sf.stop() runs when a mote loses synchronisation, part way through a run,
not at the end of it. It used to wipe QLEARNING_STATS, so any mote that
desynchronised reached the end of the simulation with an empty history, and
that history is what SimEngine.save_qlearning_stats writes out.
"""
from __future__ import absolute_import


def _one_mote(sim_engine):
    engine = sim_engine(
        diff_config = {
            'exec_numMotes': 4,
            'sf_class'     : 'Qlearning',
        }
    )
    mote = engine.motes[1]
    # stop() deletes the scheduling function's slotframe, which start() creates
    mote.sf.start()
    return mote


def test_history_survives_stop(sim_engine):
    mote = _one_mote(sim_engine)

    mote.sf.QLEARNING_STATS['CUMULATIVE_REWARD'][1] = 0.5
    mote.sf.QLEARNING_STATS['EPSILON'][1] = 0.9

    mote.sf.stop()

    assert mote.sf.QLEARNING_STATS['CUMULATIVE_REWARD'][1] == 0.5
    assert mote.sf.QLEARNING_STATS['EPSILON'][1] == 0.9


def test_stop_still_restarts_exploration(sim_engine):
    mote = _one_mote(sim_engine)

    mote.sf.EPISODE = 40
    mote.sf.stop()

    # epsilon decays from the episode counter, so resetting it puts the agent
    # back into exploration after it rejoins
    assert mote.sf.EPISODE == 0


def test_recorded_step_does_not_rewind(sim_engine):
    mote = _one_mote(sim_engine)

    mote.sf.RECORDED_STEP = 12
    mote.sf.stop()

    # if this rewound, entries recorded after the desync would overwrite the
    # ones recorded before it
    assert mote.sf.RECORDED_STEP == 12
