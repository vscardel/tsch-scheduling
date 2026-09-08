"""A hyperparameter in a parameters file must reach the simulator.

The factorial applied parameters positionally, in the order of the search
space of whatever -sf the run was given. It runs with -sf Qlearning, which has
four hyperparameters, and Q-static has five, so the Q-static cell lost
EPSLON_THRESHOLD without a word: it ran on config.json's 0.58 rather than the
0.30 the optimisation chose. Three control arms built by varying exactly that
parameter came out byte-identical, which is how it was found.
"""
import json

import pytest

import runExperiments
from runExperiments import configure_settings


class _Args(object):
    combinations = [50]
    num_cpus = 10
    num_runs = 30
    sched_function = 'Qlearning'
    conn_class = 'Random'
    num_slots = 3750
    factor_combinations = None
    output_folder = 'x'
    sync_required = False


@pytest.fixture(autouse=True)
def _args(monkeypatch):
    monkeypatch.setattr(runExperiments, 'args', _Args(), raising=False)


@pytest.fixture
def settings():
    with open('config.json') as f:
        return json.load(f)


def test_a_mapping_applies_every_key_it_holds(settings):
    """Including one the running scheduling function's search space omits."""
    runExperiments.parameters_position[:] = [
        'ALFA', 'BETA', 'EPSLON_DECAY_RATE', 'MIN_EPSLON'
    ]
    saida = configure_settings(settings, {
        'ALFA': 0.5, 'EPSLON_THRESHOLD': 0.3
    })
    regular = saida['settings']['regular']
    assert regular['ALFA'] == 0.5
    assert regular['EPSLON_THRESHOLD'] == 0.3


def test_a_positional_list_still_works(settings):
    """The optimiser hands over positions, not names."""
    runExperiments.parameters_position[:] = ['ALFA', 'BETA']
    saida = configure_settings(settings, [0.7, 0.8])
    regular = saida['settings']['regular']
    assert regular['ALFA'] == 0.7
    assert regular['BETA'] == 0.8


def test_no_parameters_leaves_the_config_alone(settings):
    antes = dict(settings['settings']['regular'])
    saida = configure_settings(settings, None)
    depois = saida['settings']['regular']
    for chave in ['ALFA', 'BETA', 'EPSLON_THRESHOLD']:
        assert depois[chave] == antes[chave]
