"""The final comparison: every scheduling function, at three network sizes.

Two reviewers said the comparative claims are not backed by statistics and
that ten runs cannot separate DynQ from Q-static, and a third asked how the
method behaves as the network grows. This is the experiment that answers all
three. It runs every scheduler over the same seeds at each size, so the
comparison is paired run by run, and compare_schedulers.py does the testing.

The arms:

  dynq              the best cell of the factorial, all three state factors
  dynq_sem_remocao  the same, with the utilisation-aware cell removal off.
                    DynQ's advantage could come from its removal rule rather
                    than from anything it learned, and this separates them.
  qstatic           the fixed-threshold learner
  rlsf              the RL-SF baseline, on the hyperparameters of Pratama and
                    Chung rather than ones tuned here, which is the fairer
                    comparison and the easier one to defend
  msf               RFC 9033
  emsf              the enhanced variant

Density is held constant as the network grows: motes are placed uniformly in
a square, so the side scales with the square root of the count. Keeping the
side fixed instead would make the 200-mote network four times denser than the
50-mote one, and a scalability claim would then be measuring two things at
once. The choice belongs in the manuscript.
"""
from __future__ import print_function

import argparse
import json
import math
import os


BASE_MOTES = 50
BASE_SIDE = 2.0


def load_parameters(nome):
    caminho = './{0}_parameters.json'.format(nome)
    with open(caminho, 'r') as f:
        return json.load(f)


ARMS = [
    ('dynq',             'Qlearning',       'traffic_queue_charge', {}),
    ('dynq_sem_remocao', 'Qlearning',       'traffic_queue_charge',
     {'SMART_CELL_REMOVAL': False}),
    ('qstatic',          'QlearningSBRC24', 'qlearningSBRC24',      {}),
    # no RLSF_ overrides, so RLSF.py falls back to its published defaults
    ('rlsf',             'RLSF',            None,                   {}),
    ('msf',              'MSF',             None,                   {}),
    ('emsf',             'EMSF',            None,                   {}),
]

FACTORS = ['traffic', 'queue', 'charge']


def square_side(num_motes):
    """The side that keeps motes per unit area at the 50-mote value."""
    return BASE_SIDE * math.sqrt(num_motes / float(BASE_MOTES))


def build_config(base, arm, num_motes, num_runs, num_cpus, slotframes):
    label, sf_class, parameters_name, overrides = arm
    settings = json.loads(json.dumps(base))  # a copy, not a view
    regular = settings['settings']['regular']

    if parameters_name:
        regular.update(load_parameters(parameters_name))
    regular.update(overrides)

    regular['sf_class'] = sf_class
    regular['exec_numSlotframesPerRun'] = slotframes
    regular['conn_random_square_side'] = square_side(num_motes)
    if sf_class == 'Qlearning':
        regular['factorial_combinations'] = FACTORS
        regular['STATE_SIZE'] = 2 ** len(FACTORS)

    settings['settings']['combination']['exec_numMotes'] = [num_motes]
    settings['execution'] = {'numRuns': num_runs, 'numCPUs': num_cpus}
    settings['log_directory_name'] = '{0}_n{1}'.format(label, num_motes)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motes', type=int, nargs='+', default=[50, 100, 200])
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--cpus', type=int, default=10)
    parser.add_argument('--slotframes', type=int, default=3750)
    parser.add_argument('--arms', help='comma separated labels, default all')
    args = parser.parse_args()

    pedidos = (
        [a.strip() for a in args.arms.split(',')] if args.arms else None
    )
    arms = [a for a in ARMS if not pedidos or a[0] in pedidos]
    if pedidos:
        faltando = set(pedidos) - set(a[0] for a in arms)
        if faltando:
            raise ValueError('no such arm: {0}'.format(', '.join(faltando)))

    with open('config.json', 'r') as f:
        base = json.load(f)

    for num_motes in args.motes:
        for arm in arms:
            settings = build_config(
                base, arm, num_motes, args.runs, args.cpus, args.slotframes
            )
            nome = settings['log_directory_name']
            config_name = 'config_{0}.json'.format(nome)
            with open(config_name, 'w') as f:
                json.dump(settings, f, indent=4)

            print('=== {0}: {1} motes, lado {2:.2f}, {3} runs ==='.format(
                nome, num_motes,
                settings['settings']['regular']['conn_random_square_side'],
                args.runs))
            os.system('python2 runSim.py --config {0}'.format(config_name))

            pasta = os.path.join(
                'simData', nome, 'exec_numMotes_{0}'.format(num_motes)
            )
            os.system('python2 compute_kpis.py --subfolder {0}'.format(pasta))
            print('=== {0} pronto ==='.format(nome))


if __name__ == '__main__':
    main()
