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
    CORE_GRIDS, CORE_LEFT_ALONE, CORE_REFERENCES, DISTURBANCES, LEARNERS,
    arm_name, baseline_settings, build_config, factors_of, load_anchor,
    manifest, plan
)
from runExperiments import search_space


@pytest.fixture
def base():
    with open('config.json') as f:
        return json.load(f)


@pytest.fixture(params=['core', 'tarefa', 'legado'])
def grupo(request):
    return request.param


@pytest.fixture
def arms(base, grupo):
    return plan(base, ['dynq', 'qstatic'], grupo)


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

def test_the_energy_weight_is_swept_only_where_the_reward_has_weights():
    """Equation 11 has no weights, so a row for it would mean nothing."""
    dynq = [f[1] for f in factors_of('dynq', 'tarefa')]
    qstatic = [f[1] for f in factors_of('qstatic', 'tarefa')]
    assert 'W_ENERGY' in dynq
    assert 'W_ENERGY' not in qstatic


def test_the_window_is_swept_only_where_it_does_something():
    """Q-static discretises the raw value against a fixed threshold and
    throws the moving average away, which is what makes it the static one.
    Its four window arms in the sweep of 2026-09-08 came out byte identical
    to the baseline."""
    dynq = [f[1] for f in factors_of('dynq', 'tarefa')]
    qstatic = [f[1] for f in factors_of('qstatic', 'tarefa')]
    assert 'SLOTFRAME_INTERVAL_SIZE' in dynq
    assert 'SLOTFRAME_INTERVAL_SIZE' not in qstatic


def test_the_phase_threshold_is_swept_only_where_there_is_a_phase():
    """DynQ tosses a coin per decision and has no threshold to cross."""
    dynq = [f[1] for f in factors_of('dynq', 'core')]
    qstatic = [f[1] for f in factors_of('qstatic', 'core')]
    assert 'EPSLON_THRESHOLD' in qstatic
    assert 'EPSLON_THRESHOLD' not in dynq


def test_both_learners_are_swept_on_the_core_they_share():
    for parametro in ('ALFA', 'BETA', 'MIN_EPSLON'):
        for learner in ('dynq', 'qstatic'):
            nomes = [f[1] for f in factors_of(learner, 'core')]
            assert parametro in nomes, (learner, parametro)


# --------------------------------------------- the core grid and the ranges

def test_every_core_grid_point_lies_inside_its_range():
    """The ranges are what the citations defend, so a grid point outside one
    would be a value with nothing behind it."""
    for learner in ('dynq', 'qstatic'):
        faixas = dict(search_space(LEARNERS[learner]['sf_class']))
        for _, setting, pontos in factors_of(learner, 'core'):
            baixo, alto = faixas[setting]
            for ponto in pontos:
                assert baixo <= ponto <= alto, (setting, ponto)


def test_the_core_grid_reaches_both_ends_of_each_range():
    """Otherwise the sweep would report an optimum inside a box it never
    tested the edges of."""
    faixas = dict(search_space('QlearningSBRC24'))
    for _, setting, pontos in CORE_GRIDS:
        baixo, alto = faixas[setting]
        assert min(pontos) == baixo, setting
        assert max(pontos) == alto, setting


def test_the_grid_contains_what_the_neighbouring_papers_use():
    pontos = dict((setting, p) for _, setting, p in CORE_GRIDS)
    assert 0.01 in pontos['ALFA']       # Pratama, Chung and Fawwaz 2024
    assert 0.1 in pontos['ALFA']        # Pratama and Chung 2022
    assert 0.95 in pontos['BETA']       # both of them
    assert 0.1 in pontos['MIN_EPSLON']  # both of them


def test_the_epsilon_decay_is_deliberately_left_alone():
    """It governs how fast epsilon reaches the floor rather than where the
    floor is, so it is held rather than swept. Pinned so the omission stays a
    decision instead of becoming an oversight."""
    assert 'EPSLON_DECAY_RATE' in CORE_LEFT_ALONE
    for learner in ('dynq', 'qstatic'):
        nomes = [f[1] for f in factors_of(learner, 'core')]
        assert 'EPSLON_DECAY_RATE' not in nomes


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


def test_dynq_keeps_its_three_state_factors(base):
    arms = plan(base, ['dynq'], 'core')
    linha = regular(base, arms, 'dynq_alfa_0p1')
    assert linha['factorial_combinations'] == ['traffic', 'queue', 'charge']
    assert linha['STATE_SIZE'] == 8


# ---------------------------------------------------------- the manifest

def test_the_manifest_says_what_each_arm_changed(base):
    arms = plan(base, ['dynq', 'qstatic'], 'core')
    m = manifest(arms, 'core')
    assert m['dynq_base']['factor'] == 'baseline'
    entrada = m['qstatic_limiar_0p6']
    assert entrada['learner'] == 'qstatic'
    assert entrada['factor'] == 'limiar'
    assert entrada['group'] == 'core'
    assert entrada['setting'] == 'EPSLON_THRESHOLD'
    assert entrada['value'] == 0.6
    assert entrada['baseline'] == 'qstatic_base'


def test_every_arm_names_a_baseline_that_exists(base, arms, grupo):
    m = manifest(arms, grupo)
    nomes = set(m)
    for entrada in m.values():
        assert entrada['baseline'] in nomes


def test_a_float_survives_becoming_a_folder_name():
    assert arm_name('dynq', 'alfa', 0.05) == 'dynq_alfa_0p05'
    assert arm_name('dynq', 'beta', 0) == 'dynq_beta_0'
    assert '.' not in arm_name('dynq', 'w_energy', 0.01)


# ------------------------------------------------------- the built-in check

def test_a_threshold_of_zero_never_exploits(base):
    """In the legacy sweep that arm reproduced the random control already
    measured, which made it a check on the plumbing rather than a question."""
    arms = plan(base, ['qstatic'], 'legado')
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
            f[2] for f in factors_of(learner, 'legado')
            if f[1] == 'ALFA_DECAY_TAU'
        ][0]
        assert max(grade) <= 10


# ------------------------------------------------- the anchor and its box

ANCORA = {'dynq': {'ALFA': 0.1, 'MIN_EPSLON': 0.1}, 'qstatic': {'BETA': 0.95}}


@pytest.fixture
def arms_ancorados(base, grupo):
    return plan(base, ['dynq', 'qstatic'], grupo, None, ANCORA)


def ancorado(base, arms, label, motes=50):
    arm = [a for a in arms if a[0] == label][0]
    return build_config(
        base, arm, motes, 10, 10, 15000, ANCORA, True
    )['settings']['regular']


def test_the_anchor_moves_the_baseline_and_nothing_else(base):
    """Every departure of an anchored arm is still its own one setting."""
    arms = plan(base, ['dynq', 'qstatic'], 'core', None, ANCORA)
    for label, learner, setting, valor in arms:
        if setting is None:
            continue
        linha_base = ancorado(base, arms, '{0}_base'.format(learner))
        braco = ancorado(base, arms, label)
        diferencas = [
            k for k in set(linha_base) | set(braco)
            if linha_base.get(k) != braco.get(k)
        ]
        assert diferencas == [setting], (label, diferencas)
        assert braco[setting] == valor


def test_the_anchored_baseline_differs_from_the_published_one_by_the_anchor(base):
    for learner, mudancas in ANCORA.items():
        publicado = baseline_settings(base, learner)
        com_ancora = baseline_settings(base, learner, ANCORA)
        diferentes = set(
            k for k in set(publicado) | set(com_ancora)
            if publicado.get(k) != com_ancora.get(k)
        )
        assert diferentes == set(mudancas), (learner, diferentes)
        for chave, valor in mudancas.items():
            assert com_ancora[chave] == valor


def test_no_anchored_arm_repeats_the_anchor_under_another_name(base, arms_ancorados):
    for label, learner, setting, valor in arms_ancorados:
        if setting is None:
            continue
        # a setting with no entry defaults in the learner, as plan reads it
        publicado = baseline_settings(base, learner, ANCORA).get(setting, 0)
        assert valor != publicado, label


def test_every_anchor_value_lies_inside_its_range(base):
    """The point the sweep departs from is one the literature admits.

    That is the whole reason for the anchor: a coordinate sweep reports
    sensitivity around one point, and the point has to be defensible.
    """
    for learner, mudancas in ANCORA.items():
        faixas = dict(search_space(LEARNERS[learner]['sf_class']))
        for chave, valor in mudancas.items():
            baixo, alto = faixas[chave]
            assert baixo <= valor <= alto, (learner, chave, valor)


def test_the_published_values_outside_the_box_are_run_as_references(base):
    """Each one gets an arm on the curve of its own factor.

    Reporting that the box costs nothing, or that it costs something, needs
    the out of range value measured in the same scenario as the rest.
    """
    arms = plan(base, ['dynq', 'qstatic'], 'core', None, ANCORA)
    for learner, referencias in CORE_REFERENCES.items():
        faixas = dict(search_space(LEARNERS[learner]['sf_class']))
        for setting, valor in referencias:
            baixo, alto = faixas[setting]
            assert not baixo <= valor <= alto, (learner, setting, valor)
            iguais = [
                a for a in arms
                if a[1] == learner and a[2] == setting and a[3] == valor
            ]
            assert len(iguais) == 1, (learner, setting, valor, iguais)


def test_the_references_only_appear_in_the_core_sweep(base):
    for grupo in ('tarefa', 'legado'):
        arms = plan(base, ['dynq', 'qstatic'], grupo, None, ANCORA)
        for learner, referencias in CORE_REFERENCES.items():
            for setting, valor in referencias:
                assert not [
                    a for a in arms if a[2] == setting and a[3] == valor
                ], (grupo, learner, setting)


def test_the_reference_arm_sits_on_the_curve_of_its_factor(base):
    """The manifest groups it with the values it is there to be read against."""
    arms = plan(base, ['dynq', 'qstatic'], 'core', None, ANCORA)
    manifesto = manifest(arms, 'core', ANCORA, True)
    assert manifesto[arm_name('dynq', 'alfa', 0.7863156043331933)]['factor'] \
        == 'alfa'
    assert manifesto[arm_name('qstatic', 'beta', 0.05)]['factor'] == 'beta'


def test_no_anchor_file_means_the_published_configuration(base):
    assert load_anchor(None) == {}
    assert baseline_settings(base, 'dynq', {}) \
        == baseline_settings(base, 'dynq')


# --------------------------------------------------------- the disturbances

def test_every_arm_of_a_disturbed_sweep_carries_the_same_disturbances(base):
    """Including the baseline, or the pairing compares two scenarios."""
    arms = plan(base, ['dynq', 'qstatic'], 'core', None, ANCORA)
    for arm in arms:
        regular = build_config(
            base, arm, 50, 10, 10, 15000, ANCORA, True
        )['settings']['regular']
        assert regular['disturbances'] == DISTURBANCES, arm[0]


def test_an_undisturbed_sweep_declares_no_disturbance(base, arms_ancorados):
    for arm in arms_ancorados:
        regular = build_config(
            base, arm, 50, 10, 10, 15000, ANCORA, False
        )['settings']['regular']
        assert regular['disturbances'] == []


def test_the_manifest_records_the_scenario_and_the_anchor(base):
    arms = plan(base, ['dynq'], 'core', None, ANCORA)
    manifesto = manifest(arms, 'core', ANCORA, True)
    for label, info in manifesto.items():
        assert info['disturbed'] is True, label
        assert info['anchor'] == ANCORA['dynq'], label
    limpo = manifest(arms, 'core', None, False)
    for label, info in limpo.items():
        assert info['disturbed'] is False
        assert info['anchor'] == {}
