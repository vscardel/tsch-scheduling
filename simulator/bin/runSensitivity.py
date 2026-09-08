"""One factor at a time, around the published configuration.

Two reviewers asked for this and nothing in the manuscript answers either.
Reviewer 1 calls the energy weight of 0.01 negligible and asks what the reward
would do without it; reviewer 1 also notes the moving average window of 10 is
used without justification, and that window is what separates the dynamic
learner from the static one, so its value is not a detail.

The design is deliberately the conventional one. Every arm is the published
configuration with a single setting changed, run on the same seeds, so each
arm pairs run by run against the baseline and the difference is attributable
to the one setting. A grid over several factors at once would be cheaper per
point and would answer a different question.

Two factors are swept beyond what was asked, because the convergence
discussion needs them: the learning rate, tuned to 0.79 and therefore nearly
memoryless, and the pair that governs how much the agent explores and how far
ahead it looks.

A caveat that belongs in the manuscript rather than in a footnote: sweeping
constant learning rates says which value scores best. It says nothing about
convergence, because every constant value fails the second Watkins condition
alike. That question needs a decaying schedule, which is a different run.

Usage:
    python runSensitivity.py --runs 10 --motes 50
    python runSensitivity.py --factors alfa,janela --dry-run
"""
from __future__ import print_function

import argparse
import json
import os


BASELINE_PARAMETERS = 'traffic_queue_charge'
FACTORS_STATE = ['traffic', 'queue', 'charge']

# label -> (setting, values). The published value is added to every grid from
# the parameters file, so one point of each sweep is the baseline itself and
# is run once rather than once per factor.
FACTORS = [
    ('w_energy', 'W_ENERGY',                [0.0, 0.01, 0.1, 0.5, 2.0, 5.0]),
    ('janela',   'SLOTFRAME_INTERVAL_SIZE', [1, 3, 5, 20, 50]),
    ('alfa',     'ALFA',                    [0.05, 0.1, 0.3, 0.5, 0.95]),
    ('epsilon',  'MIN_EPSLON',              [0.0, 0.05, 0.4, 0.8]),
    ('beta',     'BETA',                    [0.0, 0.2, 0.7, 0.9]),
]


def load_parameters(nome):
    with open('./{0}_parameters.json'.format(nome)) as f:
        return json.load(f)


def baseline_settings(base):
    """The published configuration, as the arms depart from it."""
    regular = json.loads(json.dumps(base))['settings']['regular']
    regular.update(load_parameters(BASELINE_PARAMETERS))
    regular['sf_class'] = 'Qlearning'
    regular['factorial_combinations'] = FACTORS_STATE
    regular['STATE_SIZE'] = 2 ** len(FACTORS_STATE)
    return regular


def arm_name(factor, valor):
    """A folder name that survives being a float."""
    texto = ('%g' % valor).replace('.', 'p').replace('-', 'm')
    return 'sens_{0}_{1}'.format(factor, texto)


def plan(base, factors=None):
    """Every arm to run: the baseline once, then one per swept value.

    A value equal to the published one is not run again under another name;
    it is the baseline, and comparing an arm against itself would report a
    difference of exactly zero and look like a result.
    """
    regular = baseline_settings(base)
    escolhidos = [f for f in FACTORS if not factors or f[0] in factors]
    if factors:
        faltando = set(factors) - set(f[0] for f in escolhidos)
        if faltando:
            raise ValueError('no such factor: {0}'.format(', '.join(faltando)))

    arms = [('baseline', None, None, regular[
        'W_ENERGY' if False else 'ALFA'] and None)] if False else []
    arms.append(('baseline', None, None))
    for factor, setting, valores in escolhidos:
        publicado = regular[setting]
        for valor in valores:
            if valor == publicado:
                continue
            arms.append((arm_name(factor, valor), setting, valor))
    return arms


def build_config(base, arm, num_motes, num_runs, num_cpus, slotframes):
    label, setting, valor = arm
    settings = json.loads(json.dumps(base))
    regular = baseline_settings(base)
    if setting is not None:
        regular[setting] = valor
    regular['exec_numSlotframesPerRun'] = slotframes
    settings['settings']['regular'] = regular
    settings['settings']['combination']['exec_numMotes'] = [num_motes]
    settings['execution'] = {'numRuns': num_runs, 'numCPUs': num_cpus}
    settings['log_directory_name'] = '{0}_n{1}'.format(label, num_motes)
    return settings


def manifest(arms):
    """What each arm changed, for the report to group the sweeps by factor."""
    saida = {}
    for label, setting, valor in arms:
        if setting is None:
            saida[label] = {'factor': 'baseline', 'setting': None, 'value': None}
            continue
        factor = [f[0] for f in FACTORS if f[1] == setting][0]
        saida[label] = {'factor': factor, 'setting': setting, 'value': valor}
    return saida


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motes', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--cpus', type=int, default=10)
    parser.add_argument('--slotframes', type=int, default=3750)
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

    arms = plan(base, pedidos)
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
            print('  {0}: {1}'.format(nome, arm[1] and '{0}={1}'.format(
                arm[1], arm[2]) or 'publicado'))
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
