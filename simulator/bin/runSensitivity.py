"""One factor at a time, around each learner's published configuration.

Two reviewers asked for this and nothing in the manuscript answers either.
Reviewer 1 calls the energy weight of 0.01 negligible and asks what the reward
does without it, and says the moving average window of 10 is used without
justification. That window is not a detail: it is what the dynamic learner
discretises with, so it is the thing that separates it from the static one.

The design is the conventional one on purpose. Every arm is the published
configuration with a single setting changed, on the same seeds, so each arm
pairs run by run against its own baseline and the difference is attributable
to that one setting.

Three factors go beyond what was asked, because the convergence question needs
them: the learning rate, tuned to 0.79 in DynQ and therefore nearly
memoryless; the knobs that govern how much the agent explores and how far
ahead it looks; and a decaying learning rate, which is the only setting here
that can make the convergence criterion say yes, since every constant rate
fails the Watkins condition alike.

Each learner is swept on the knobs it actually has. The energy weight is not
in the static learner, whose reward is Equation 11 and has no weights, and the
phase threshold is not in the dynamic one, which tosses a coin per decision.
Sweeping a knob a method does not have would produce a row that means nothing.

Usage:
    python runSensitivity.py --learners dynq qstatic
    python runSensitivity.py --factors alfa,janela --dry-run
"""
from __future__ import print_function

import argparse
import json
import os


FACTORS_STATE = ['traffic', 'queue', 'charge']

# The values of tau are small on purpose. Measured on the runs of 2026-09-08,
# the median cell of the table is updated five times at the published length,
# so a tau of 50 would trim the rate by a tenth and a tau of 200 would do
# nothing at all.
DECAY_GRID = [1, 3, 10]

LEARNERS = {
    'dynq': {
        'sf_class'  : 'Qlearning',
        'parameters': 'traffic_queue_charge',
        'factors'   : [
            ('w_energy', 'W_ENERGY',                [0.0, 0.01, 0.1, 5.0]),
            ('janela',   'SLOTFRAME_INTERVAL_SIZE', [1, 3, 20, 50]),
            ('alfa',     'ALFA',                    [0.05, 0.2, 0.5, 0.95]),
            ('epsilon',  'MIN_EPSLON',              [0.0, 0.05, 0.4, 0.8]),
            ('beta',     'BETA',                    [0.0, 0.2, 0.7, 0.9]),
            ('decaimento', 'ALFA_DECAY_TAU',        DECAY_GRID),
        ],
    },
    'qstatic': {
        'sf_class'  : 'QlearningSBRC24',
        'parameters': 'qlearningSBRC24',
        'factors'   : [
            ('janela',   'SLOTFRAME_INTERVAL_SIZE', [1, 3, 20, 50]),
            ('alfa',     'ALFA',                    [0.01, 0.2, 0.5, 0.9]),
            ('beta',     'BETA',                    [0.0, 0.3, 0.6, 0.9]),
            ('limiar',   'EPSLON_THRESHOLD',        [0.0, 0.1, 0.6, 1.0]),
            ('decaimento', 'ALFA_DECAY_TAU',        DECAY_GRID),
        ],
    },
}


def load_parameters(nome):
    with open('./{0}_parameters.json'.format(nome)) as f:
        return json.load(f)


def baseline_settings(base, learner):
    """The published configuration of one learner, as its arms depart from it."""
    regular = json.loads(json.dumps(base))['settings']['regular']
    spec = LEARNERS[learner]
    regular.update(load_parameters(spec['parameters']))
    regular['sf_class'] = spec['sf_class']
    if spec['sf_class'] == 'Qlearning':
        regular['factorial_combinations'] = FACTORS_STATE
        regular['STATE_SIZE'] = 2 ** len(FACTORS_STATE)
    return regular


def arm_name(learner, factor, valor):
    """A folder name that survives being a float."""
    texto = ('%g' % valor).replace('.', 'p').replace('-', 'm')
    return '{0}_{1}_{2}'.format(learner, factor, texto)


def plan(base, learners, factors=None):
    """Every arm to run: one baseline per learner, then one per swept value.

    A value equal to the published one is not run again under another name. It
    is the baseline, and an arm compared against itself would report a
    difference of exactly zero on every metric, which in this project has
    twice meant a parameter that never arrived.
    """
    arms = []
    for learner in learners:
        regular = baseline_settings(base, learner)
        arms.append(('{0}_base'.format(learner), learner, None, None))
        escolhidos = [
            f for f in LEARNERS[learner]['factors']
            if not factors or f[0] in factors
        ]
        for factor, setting, valores in escolhidos:
            publicado = regular.get(setting, 0)
            for valor in valores:
                if valor == publicado:
                    continue
                arms.append(
                    (arm_name(learner, factor, valor), learner, setting, valor)
                )
    if factors:
        conhecidos = set()
        for spec in LEARNERS.values():
            conhecidos.update(f[0] for f in spec['factors'])
        faltando = set(factors) - conhecidos
        if faltando:
            raise ValueError('no such factor: {0}'.format(', '.join(faltando)))
    return arms


def build_config(base, arm, num_motes, num_runs, num_cpus, slotframes):
    label, learner, setting, valor = arm
    settings = json.loads(json.dumps(base))
    regular = baseline_settings(base, learner)
    if setting is not None:
        regular[setting] = valor
    regular['exec_numSlotframesPerRun'] = slotframes
    settings['settings']['regular'] = regular
    settings['settings']['combination']['exec_numMotes'] = [num_motes]
    settings['execution'] = {'numRuns': num_runs, 'numCPUs': num_cpus}
    settings['log_directory_name'] = '{0}_n{1}'.format(label, num_motes)
    return settings


def manifest(arms):
    """What each arm changed, so the report can group the sweeps by factor."""
    saida = {}
    for label, learner, setting, valor in arms:
        if setting is None:
            saida[label] = {'learner': learner, 'factor': 'baseline',
                            'setting': None, 'value': None,
                            'baseline': label}
            continue
        factor = [
            f[0] for f in LEARNERS[learner]['factors'] if f[1] == setting
        ][0]
        saida[label] = {'learner': learner, 'factor': factor,
                        'setting': setting, 'value': valor,
                        'baseline': '{0}_base'.format(learner)}
    return saida


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--cpus', type=int, default=10)
    parser.add_argument('--slotframes', type=int, default=3750)
    parser.add_argument('--learners', nargs='+', default=['dynq', 'qstatic'])
    parser.add_argument('--factors', help='comma separated labels, default all')
    parser.add_argument('--arms', help='comma separated arm names to run')
    parser.add_argument('--dry-run', action='store_true',
                        help='write the configs and the manifest, run nothing')
    args = parser.parse_args()

    pedidos = (
        [f.strip() for f in args.factors.split(',')] if args.factors else None
    )
    with open('config.json') as f:
        base = json.load(f)

    arms = plan(base, args.learners, pedidos)
    if args.arms:
        querido = [a.strip() for a in args.arms.split(',')]
        arms = [a for a in arms if a[0] in querido]

    with open('sensitivity_manifest.json', 'w') as f:
        json.dump(manifest(arms), f, indent=2)
    print('{0} bracos, manifesto em sensitivity_manifest.json'.format(len(arms)))

    for arm in arms:
        settings = build_config(
            base, arm, args.motes, args.runs, args.cpus, args.slotframes
        )
        nome = settings['log_directory_name']
        config_name = 'config_{0}.json'.format(nome)
        with open(config_name, 'w') as f:
            json.dump(settings, f, indent=4)
        if args.dry_run:
            mudou = '{0}={1}'.format(arm[2], arm[3]) if arm[2] else 'publicado'
            print('  {0:28s} {1}'.format(nome, mudou))
            continue

        print('=== {0} ==='.format(nome))
        os.system('python2 runSim.py --config {0}'.format(config_name))
        pasta = os.path.join(
            'simData', nome, 'exec_numMotes_{0}'.format(args.motes)
        )
        os.system('python2 compute_kpis.py --subfolder {0}'.format(pasta))
        print('=== {0} pronto ==='.format(nome))


if __name__ == '__main__':
    main()
