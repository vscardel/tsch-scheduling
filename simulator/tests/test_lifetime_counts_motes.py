"""Only a mote counts as a mote in the lifetime mean.

Both lifetime functions walked every key of a run and treated each one as a
mote, skipping only global-stats. compare_schedulers attaches the learning
quantities under 'learning-stats' before reading the metrics, so that key
became a mote with no lifetime and counted as zero: the mean divided by 51
instead of 50.

The consequence was not symmetric. A learner leaves a trace and got the extra
zero; MSF and EMSF leave none and did not. So every comparison between a
learner and one of them was biased against the learner, on lifetime and on
the score that is built from it.
"""
import pytest

from score_model import run_lifetime, run_score
from runExperiments import compute_run_lifetime


def _mote(anos):
    return {'lifetime_AA_years': anos}


def _run(vidas, extras=None):
    run = dict(('%d' % i, _mote(v)) for i, v in enumerate(vidas))
    run['global-stats'] = {
        'e2e-upstream-latency': [{'mean': 1.0}],
        'e2e-upstream-delivery': [{'value': 0.7}],
        'joining-time': [{'mean': 100.0}],
    }
    run.update(extras or {})
    return run


@pytest.mark.parametrize('vida', [run_lifetime, compute_run_lifetime])
def test_the_learning_trace_is_not_a_mote(vida):
    limpa = _run([2.0, 4.0])
    com_traco = _run([2.0, 4.0],
                     {'learning-stats': {'mean_reward': 2.6,
                                         'final_policy_gap': 1.8}})
    assert vida(limpa) == 3.0
    assert vida(com_traco) == 3.0


@pytest.mark.parametrize('vida', [run_lifetime, compute_run_lifetime])
def test_a_mote_that_could_not_be_estimated_still_counts_as_zero(vida):
    """The behaviour the docstring promises, and the reason the guard cannot
    simply drop entries whose lifetime is not a number."""
    run = _run([3.0])
    run['1'] = {'lifetime_AA_years': 'n/a'}
    assert vida(run) == 1.5


@pytest.mark.parametrize('vida', [run_lifetime, compute_run_lifetime])
def test_a_run_of_no_motes_is_zero_and_does_not_divide_by_zero(vida):
    run = _run([])
    assert vida(run) == 0.0


def test_the_two_copies_agree():
    """They are the same function in two files, and the fix has to land in
    both or the score and the lifetime metric disagree."""
    run = _run([1.0, 2.0, 3.0],
               {'learning-stats': {'mean_reward': 1.0}})
    assert run_lifetime(run) == compute_run_lifetime(run)


def test_the_score_does_not_move_when_the_trace_is_attached():
    """This is what made the same configuration read 0.5591 in one report and
    0.5615 in another."""
    limpa = _run([2.0, 4.0])
    com_traco = _run([2.0, 4.0], {'learning-stats': {'mean_reward': 2.6}})
    assert run_score(limpa) == run_score(com_traco)
