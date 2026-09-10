"""The final comparison's arms differ only in what they are meant to differ in.

Six arms over three network sizes is a lot of configuration to get wrong
quietly, and the run costs a night. These check the things that would make the
comparison unfair rather than merely broken.
"""
import json
import math

import pytest

from runComparison import ALL_ARMS, ARMS, build_config, square_side


@pytest.fixture
def base():
    with open('config.json') as f:
        return json.load(f)


def arm(label):
    return [a for a in ALL_ARMS if a[0] == label][0]


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


# ------------------------------------------------- the arms that measure learning


@pytest.mark.parametrize('learner', ['dynq', 'qstatic'])
def test_the_control_differs_from_its_arm_in_one_key(base, learner):
    """The check that has caught this twice already.

    Two arms came out byte identical in this revision because a hyperparameter
    was applied by position and never reached the simulator. A difference of
    exactly zero on every metric means plumbing, not a tie, so the difference
    between an arm and its control is pinned to the single key it is supposed
    to be.
    """
    aprendido = build_config(base, arm(learner + '_aprendido'), 50, 10, 10, 3750)
    aleatorio = build_config(base, arm(learner + '_aleatorio'), 50, 10, 10, 3750)
    a = aprendido['settings']['regular']
    b = aleatorio['settings']['regular']
    diferencas = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    assert diferencas == ['LEARNED_POLICY']
    assert a['LEARNED_POLICY'] is True
    assert b['LEARNED_POLICY'] is False


@pytest.mark.parametrize('label', [
    'dynq_aprendido', 'dynq_aleatorio', 'qstatic_aprendido', 'qstatic_aleatorio'
])
def test_both_learners_keep_the_removal_rule_in_every_learning_arm(base, label):
    """The rule is what the previous comparison turned on, so it is held
    constant here rather than varied alongside the thing being measured."""
    settings = build_config(base, arm(label), 50, 10, 10, 3750)
    regular = settings['settings']['regular']
    assert regular.get('SMART_CELL_REMOVAL') is not False
    assert regular.get('QSTATIC_SMART_CELL_REMOVAL') is not False


def test_the_learned_arm_is_the_scheduler_it_is_named_after(base):
    """dynq_aprendido has to be exactly dynq, or the gain over the control is
    measured on a method nobody else ran. Learning is on by default, so the
    two configurations come out identical and the control is the departure."""
    dynq = build_config(base, arm('dynq'), 50, 10, 10, 3750)
    aprendido = build_config(base, arm('dynq_aprendido'), 50, 10, 10, 3750)
    a = dynq['settings']['regular']
    b = aprendido['settings']['regular']
    assert [k for k in set(a) | set(b) if a.get(k) != b.get(k)] == []


def test_learning_arms_are_asked_for_by_name(base):
    """A plain run of the comparison is the six arms it always was, so nobody
    runs eight hours of controls by forgetting a flag."""
    assert 'dynq_aprendido' not in [a[0] for a in ARMS]


# ------------------------------------ the core values the sweep chose

ANCORA_ESCOLHIDA = {
    'dynq'   : {'ALFA': 0.1, 'BETA': 0.95, 'MIN_EPSLON': 0.15},
    'qstatic': {'BETA': 0.7},
}


def test_the_anchor_reaches_a_learner_by_its_scheduling_function():
    """Not by label, so the controls get the same core values.

    dynq_aleatorio is the same agent as dynq_aprendido with the table
    switched off. If the anchor were matched by label the control would run
    the published parameters and the pair would differ in four things
    instead of one.
    """
    from runComparison import anchor_for
    assert anchor_for('Qlearning', ANCORA_ESCOLHIDA) \
        == ANCORA_ESCOLHIDA['dynq']
    assert anchor_for('QlearningSBRC24', ANCORA_ESCOLHIDA) \
        == ANCORA_ESCOLHIDA['qstatic']


def test_a_scheduler_that_does_not_learn_gets_no_anchor():
    from runComparison import anchor_for
    for sf in ('MSF', 'EMSF', 'RLSF'):
        assert anchor_for(sf, ANCORA_ESCOLHIDA) == {}


def test_a_learning_pair_differs_only_in_consulting_the_table(base):
    """The whole design of the learning test rests on this."""
    from runComparison import LEARNING_ARMS, build_config
    por_nome = dict((a[0], a) for a in LEARNING_ARMS)
    for aprendido, aleatorio in [('dynq_aprendido', 'dynq_aleatorio'),
                                 ('qstatic_aprendido', 'qstatic_aleatorio')]:
        a = build_config(base, por_nome[aprendido], 50, 10, 10, 15000,
                         ANCORA_ESCOLHIDA, True)['settings']['regular']
        b = build_config(base, por_nome[aleatorio], 50, 10, 10, 15000,
                         ANCORA_ESCOLHIDA, True)['settings']['regular']
        diferencas = [
            k for k in set(a) | set(b) if a.get(k) != b.get(k)
        ]
        assert diferencas == ['LEARNED_POLICY'], (aprendido, diferencas)


def test_the_anchor_carries_into_the_learning_arms(base):
    from runComparison import LEARNING_ARMS, build_config
    for arm in LEARNING_ARMS:
        regular = build_config(base, arm, 50, 10, 10, 15000,
                               ANCORA_ESCOLHIDA, True)['settings']['regular']
        esperado = (ANCORA_ESCOLHIDA['dynq'] if arm[1] == 'Qlearning'
                    else ANCORA_ESCOLHIDA['qstatic'])
        for chave, valor in esperado.items():
            assert regular[chave] == valor, (arm[0], chave)


def test_an_arms_own_override_beats_the_anchor(base):
    """The anchor is a starting point, not a lock."""
    from runComparison import build_config
    arm = ('teste', 'Qlearning', 'traffic_queue_charge', {'ALFA': 0.42})
    regular = build_config(base, arm, 50, 10, 10, 15000,
                           ANCORA_ESCOLHIDA, True)['settings']['regular']
    assert regular['ALFA'] == 0.42
    assert regular['BETA'] == 0.95


def test_disturbances_are_all_or_nothing(base):
    from runComparison import ALL_ARMS, build_config, DISTURBANCES
    for arm in ALL_ARMS:
        com = build_config(base, arm, 50, 10, 10, 15000, None, True)
        sem = build_config(base, arm, 50, 10, 10, 15000, None, False)
        assert com['settings']['regular']['disturbances'] == DISTURBANCES
        assert sem['settings']['regular']['disturbances'] == []
