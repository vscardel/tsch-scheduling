"""The sweep's arms differ only in the one setting each is meant to differ in.

Forty-four arms is a lot of configuration to get wrong quietly, and the run
costs a night. A difference of exactly zero on every metric between an arm and
its baseline has meant, twice in this revision, a parameter that never reached
the simulator rather than a setting that made no difference. These hold each
arm to its one departure.
"""
import json

import pytest

from runSensitivity import (
    LEARNERS, arm_name, baseline_settings, build_config, manifest, plan
)


@pytest.fixture
def base():
    with open('config.json') as f:
        return json.load(f)


@pytest.fixture
def arms(base):
    return plan(base, ['dynq', 'qstatic'])


def regular(base, arms, label, motes=50):
    arm = [a for a in arms if a[0] == label][0]
    return build_config(base, arm, motes, 10, 10, 3750)['settings']['regular']


# ------------------------------------------------------- one key at a time

def test_every_arm_differs_from_its_baseline_in_one_setting(base, arms):
    for label, learner, setting, valor in arms:
        if setting is None:
            continue
        linha_base = regular(base, arms, '{0}_base'.format(learner))
        braco = regular(base, arms, label)
        diferencas = [
            k for k in set(linha_base) | set(braco)
            if linha_base.get(k) != braco.get(k)
        ]
        assert diferencas == [setting], (label, diferencas)
        assert braco[setting] == valor


def test_no_arm_repeats_the_published_value_under_another_name(base, arms):
    """An arm compared against itself reports a difference of exactly zero on
    every metric, which is what a missing parameter looks like."""
    for label, learner, setting, valor in arms:
        if setting is None:
            continue
        publicado = baseline_settings(base, learner).get(setting, 0)
        assert valor != publicado, label


def test_each_learner_gets_exactly_one_baseline(base, arms):
    bases = [a[0] for a in arms if a[2] is None]
    assert sorted(bases) == ['dynq_base', 'qstatic_base']


# --------------------------------------------- each learner's own knobs only

def test_the_energy_weight_is_swept_only_where_the_reward_has_weights(base):
    """Equation 11 has no weights, so a row for it would mean nothing."""
    dynq = [f[1] for f in LEARNERS['dynq']['factors']]
    qstatic = [f[1] for f in LEARNERS['qstatic']['factors']]
    assert 'W_ENERGY' in dynq
    assert 'W_ENERGY' not in qstatic


def test_the_phase_threshold_is_swept_only_where_there_is_a_phase(base):
    """DynQ tosses a coin per decision and has no threshold to cross."""
    dynq = [f[1] for f in LEARNERS['dynq']['factors']]
    qstatic = [f[1] for f in LEARNERS['qstatic']['factors']]
    assert 'EPSLON_THRESHOLD' in qstatic
    assert 'EPSLON_THRESHOLD' not in dynq


def test_both_learners_are_swept_on_what_they_share(base):
    for parametro in ('ALFA', 'BETA', 'SLOTFRAME_INTERVAL_SIZE',
                      'ALFA_DECAY_TAU'):
        for learner in ('dynq', 'qstatic'):
            nomes = [f[1] for f in LEARNERS[learner]['factors']]
            assert parametro in nomes, (learner, parametro)


# ------------------------------------------------------- the method holds

def test_removal_stays_on_in_every_arm(base, arms):
    """It is what the previous comparison turned on, so it is held constant
    rather than varied alongside what is being measured."""
    for label, _, _, _ in arms:
        linha = regular(base, arms, label)
        assert linha.get('SMART_CELL_REMOVAL') is not False
        assert linha.get('QSTATIC_SMART_CELL_REMOVAL') is not False


def test_the_learners_run_their_own_scheduling_function(base, arms):
    assert regular(base, arms, 'dynq_base')['sf_class'] == 'Qlearning'
    assert regular(base, arms, 'qstatic_base')['sf_class'] == 'QlearningSBRC24'


def test_dynq_keeps_its_three_state_factors(base, arms):
    linha = regular(base, arms, 'dynq_alfa_0p2')
    assert linha['factorial_combinations'] == ['traffic', 'queue', 'charge']
    assert linha['STATE_SIZE'] == 8


# ---------------------------------------------------------- the manifest

def test_the_manifest_says_what_each_arm_changed(base, arms):
    m = manifest(arms)
    assert m['dynq_base']['factor'] == 'baseline'
    entrada = m['qstatic_limiar_0p6']
    assert entrada['learner'] == 'qstatic'
    assert entrada['factor'] == 'limiar'
    assert entrada['setting'] == 'EPSLON_THRESHOLD'
    assert entrada['value'] == 0.6
    assert entrada['baseline'] == 'qstatic_base'


def test_every_arm_names_a_baseline_that_exists(base, arms):
    m = manifest(arms)
    nomes = set(m)
    for entrada in m.values():
        assert entrada['baseline'] in nomes


def test_a_float_survives_becoming_a_folder_name():
    assert arm_name('dynq', 'alfa', 0.05) == 'dynq_alfa_0p05'
    assert arm_name('dynq', 'beta', 0) == 'dynq_beta_0'
    assert '.' not in arm_name('dynq', 'w_energy', 0.01)


# ------------------------------------------------------- the built-in check

def test_a_threshold_of_zero_never_exploits(base, arms):
    """That arm has to reproduce the random control already measured, which
    makes it a check on the plumbing rather than a new question."""
    linha = regular(base, arms, 'qstatic_limiar_0')
    assert linha['EPSLON_THRESHOLD'] == 0.0
    assert linha['LEARNED_POLICY'] is True     # the table is still consulted,
    # it just never gets the chance, because epsilon is never below zero


def test_the_decay_grid_matches_how_often_a_cell_is_updated(base):
    """Measured on the runs of 2026-09-08: the median cell is updated five
    times at the published length. A tau of 50 would trim the rate by a tenth
    and a tau of 200 would do nothing at all."""
    for learner in ('dynq', 'qstatic'):
        grade = [
            f[2] for f in LEARNERS[learner]['factors']
            if f[1] == 'ALFA_DECAY_TAU'
        ][0]
        assert max(grade) <= 10
