"""The final comparison's arms differ only in what they are meant to differ in.

Six arms over three network sizes is a lot of configuration to get wrong
quietly, and the run costs a night. These check the things that would make the
comparison unfair rather than merely broken.
"""
import json
import math

import pytest

from runComparison import ARMS, build_config, square_side


@pytest.fixture
def base():
    with open('config.json') as f:
        return json.load(f)


def arm(label):
    return [a for a in ARMS if a[0] == label][0]


def test_density_is_held_constant_as_the_network_grows():
    """Otherwise a scalability claim measures size and crowding at once."""
    densities = [n / square_side(n) ** 2 for n in (50, 100, 200)]
    assert max(densities) - min(densities) < 1e-9


def test_rlsf_runs_on_its_own_papers_hyperparameters(base):
    """Victor's call: the baseline is fairer, and easier to defend, untuned."""
    settings = build_config(base, arm('rlsf'), 50, 30, 10, 3750)
    regular = settings['settings']['regular']
    assert regular['sf_class'] == 'RLSF'
    assert not [k for k in regular if k.startswith('RLSF_')]


def test_the_ablation_differs_from_dynq_only_in_the_removal_rule(base):
    """If anything else differs, the ablation attributes the wrong cause.

    The full arm leaves SMART_CELL_REMOVAL out of the config, and Qlearning
    reads a missing one as on, so the two arms differ in that key alone.
    """
    completo = build_config(base, arm('dynq'), 50, 30, 10, 3750)
    ablacao = build_config(base, arm('dynq_sem_remocao'), 50, 30, 10, 3750)
    a = completo['settings']['regular']
    b = ablacao['settings']['regular']
    diferencas = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    assert diferencas == ['SMART_CELL_REMOVAL']
    assert b['SMART_CELL_REMOVAL'] is False


def test_removal_is_on_unless_a_config_turns_it_off():
    """The default the ablation is measured against lives in the code, so a
    config that never mentions the key still runs the full method."""
    from SimEngine.Mote.scheduling_functions.Qlearning import (
        SchedulingFunctionQlearning
    )
    import inspect
    fonte = inspect.getsource(SchedulingFunctionQlearning.__init__)
    assert "'SMART_CELL_REMOVAL', True" in fonte


def test_every_arm_writes_somewhere_of_its_own(base):
    nomes = [
        build_config(base, a, n, 30, 10, 3750)['log_directory_name']
        for a in ARMS for n in (50, 100, 200)
    ]
    assert len(set(nomes)) == len(nomes) == 18


def test_the_learners_are_given_the_hyperparameters_that_were_tuned(base):
    dynq = build_config(base, arm('dynq'), 50, 30, 10, 3750)
    esperado = json.load(open('traffic_queue_charge_parameters.json'))
    for chave, valor in esperado.items():
        assert dynq['settings']['regular'][chave] == valor


def test_building_one_arm_does_not_leak_into_the_next(base):
    """The arms share a base config, and a view instead of a copy would make
    the last arm's overrides land on every arm after it."""
    build_config(base, arm('dynq_sem_remocao'), 50, 30, 10, 3750)
    msf = build_config(base, arm('msf'), 50, 30, 10, 3750)
    assert msf['settings']['regular'].get('SMART_CELL_REMOVAL') is not False
    assert msf['settings']['regular']['sf_class'] == 'MSF'


def test_all_six_arms_are_present():
    assert [a[0] for a in ARMS] == [
        'dynq', 'dynq_sem_remocao', 'qstatic', 'rlsf', 'msf', 'emsf'
    ]
