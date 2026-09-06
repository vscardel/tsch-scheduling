import argparse
import json
import os
import shutil
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import time

from skopt import gp_minimize
from skopt.plots import plot_convergence

MAX_FUNCTION_VALUE = 1
ALL_SCORES = []

def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif isinstance(obj, np.integer): 
        return int(obj)
    elif isinstance(obj, np.floating): 
        return float(obj)
    else:
        return obj

# What the Bayesian search varies, and over what range. Q-static still decides
# whether to explore by comparing epsilon against EPSLON_THRESHOLD, so the
# threshold is a real parameter for it. DynQ is epsilon-greedy: epsilon is the
# chance of exploring and there is no threshold to cross, so searching it there
# would spend evaluations on a dimension that changes nothing.
SEARCH_SPACE = {
    'Qlearning': [
        ("ALFA",              (0.1, 0.9)),
        ("BETA",              (0.1, 0.9)),
        ("EPSLON_DECAY_RATE", (0.01, 0.09)),
        ("MIN_EPSLON",        (0.05, 0.1)),
    ],
    'QlearningSBRC24': [
        ("ALFA",              (0.1, 0.9)),
        ("BETA",              (0.1, 0.9)),
        ("EPSLON_DECAY_RATE", (0.01, 0.09)),
        ("MIN_EPSLON",        (0.05, 0.1)),
        ("EPSLON_THRESHOLD",  (0.5, 0.7)),
    ],
    # RL-SF gets the same budget and the same number of dimensions as DynQ, so
    # neither method is the only one that was tuned. Its three reward weights
    # are left out for the same reason DynQ's are: searching the reward changes
    # what the agent is being asked to do, not how well it does it.
    'RLSF': [
        ("RLSF_ALFA",          (0.1, 0.9)),
        ("RLSF_BETA",          (0.1, 0.9)),
        ("RLSF_EPSILON_DECAY", (0.99, 0.9999)),
        ("RLSF_EPSILON_END",   (0.05, 0.2)),
    ],
}

def search_space(sched_function):
    """The names and ranges the search varies for a scheduling function."""
    return SEARCH_SPACE.get(sched_function, SEARCH_SPACE['QlearningSBRC24'])


DEFAULT_NUM_EVALUATIONS   = 40
DEFAULT_NUM_RANDOM_STARTS = 10


def optimisation_budget(num_evaluations, num_random_starts):
    """How many evaluations the search gets, and how many of those are random.

    skopt samples at random for its first n_random_starts evaluations and only
    then fits the surrogate. If n_calls equals that number every evaluation is
    a draw, and the search is a random search wearing the name of Bayesian
    optimisation. Returns the pair, or raises if the budget buys no guided
    evaluation at all.
    """
    n_random_starts = num_random_starts or DEFAULT_NUM_RANDOM_STARTS
    n_calls = num_evaluations or DEFAULT_NUM_EVALUATIONS
    if n_calls <= n_random_starts:
        raise ValueError(
            'n_calls ({0}) must exceed n_random_starts ({1}), otherwise no '
            'evaluation is guided by the model and this is a random '
            'search.'.format(n_calls, n_random_starts)
        )
    return n_calls, n_random_starts

parameters_position = []

metrics_vector_position = [
   "latency",
   "join_time",
   "network_lifetime",
   "packet_delivery_ratio",
]

#hotspot scoring function

kpis_weights = {
    'latency': 0.25,
    'pdr': 0.25,
    'lifetime': 0.25,
    'join_time': 0.25
}
kpis_tresholds = {
    'latency': 1.5, #s
    'pdr': 0.95, #%|
    'lifetime': 1, #y
    'join_time': 1000,#s
}

metrics = {
    'latencies': [],
    'pdrs': [],
    'join_times': [],
    'lifetimes': []
}

def smooth_threshold_above(x, T, k=1):
    return 1 / (1 + np.exp(-k * (x - T)))

def smooth_threshold_below(x, T, k=1):
    return (1 - smooth_threshold_above(x, T, k))
########################

def remove_results_folder(subfolder):
    shutil.rmtree(subfolder)
    time.sleep(5)

def load_config():
    settings = None
    with open('config.json', 'r') as f:
        json_string = f.read() 
        settings = json.loads(json_string)
    return settings

def load_optimal_parameters(factor_combination):
    paramaters = None
    parameters_list = [None] * len(parameters_position)
    with open('./{0}_parameters.json'.format(factor_combination), 'r') as f:
        parameters = json.load(f)
        # for compatibility
        for position, name in enumerate(parameters_position):
            parameters_list[position] = parameters[name]
    return parameters_list

def configure_settings(settings, parameters):
    settings['settings']['combination']['exec_numMotes'] = args.combinations
    settings['execution']['numCPUs'] = args.num_cpus
    settings['execution']['numRuns'] = args.num_runs
    settings['settings']['regular']['sf_class'] = args.sched_function
    settings['settings']['regular']['conn_class'] = args.conn_class
    settings['settings']['regular']['exec_numSlotframesPerRun'] = args.num_slots
    factor_combinations = args.factor_combinations
    if not factor_combinations:
        settings['settings']['regular']['factorial_combinations'] = ['traffic', 'queue', 'charge']
    else:
        settings['settings']['regular']['factorial_combinations'] = factor_combinations.split(',')
    settings['log_directory_name']= args.output_folder
    settings['get_sync_node_info'] = args.sync_required

    # Configure simulator with the parameters
    if parameters:
        for position, parameter_name in enumerate(parameters_position):
            parameter_value = parameters[position]
            settings['settings']['regular'][parameter_name] = parameter_value
    return settings


def save_curr_run_config(config_name, settings):
    with open(config_name, 'w') as f:
        json.dump(settings, f, indent=4)

def load_kpis(folder_path, num_motes):
    kpis = None
    for tentativa in range(3):
        try:
            with open(
                os.path.join(
                    folder_path,
                    'output_cpu0.dat.kpi'.format(num_motes)
                )
            , 'r') as f:
                json_string = f.read()
                kpis = json.loads(json_string)
        except Exception as e:
            print(e)
            print("Something went wrong reading KPIs on try {0}".format(tentativa))
    return kpis

def compute_run_lifetime(run_kpis):
    """Mean battery lifetime over the motes of one run, in years.

    A mote whose lifetime could not be estimated reports a string rather than a
    number, and counts as zero.
    """
    lifetimes = []
    for mote, mote_kpis in run_kpis.items():
        if mote == 'global-stats':
            continue
        lifetime = mote_kpis.get('lifetime_AA_years')
        if not isinstance(lifetime, (int, float)):
            lifetime = 0
        lifetimes.append(lifetime)
    if not lifetimes:
        return 0.0
    return sum(lifetimes) / float(len(lifetimes))


def compute_average_lifetime(kpis):
    """Mean battery lifetime over every run."""
    per_run = [compute_run_lifetime(kpis[run]) for run in kpis]
    if not per_run:
        return 0.0
    return sum(per_run) / float(len(per_run))
        

def compute_score(kpis):
    scores = []
    for run in kpis:
        try:
            #seconds
            latency = kpis[run]['global-stats']['e2e-upstream-latency'][0]['mean']
            #convert join_time to seconds
            join_time = kpis[run]['global-stats']["joining-time"][0]['mean'] / 100
            #years
            # this run's lifetime, not the average over every run. Taking the
            # average here gave all the runs the same value, so a quarter of the
            # score was a constant and could not tell one run from another.
            network_lifetime = compute_run_lifetime(kpis[run])
            packet_delivery_ratio = kpis[run]['global-stats']['e2e-upstream-delivery'][0]['value']

            metrics_vector = [
                (latency, 'latency'), 
                (join_time, 'join_time'), 
                (network_lifetime, 'lifetime'), 
                (packet_delivery_ratio, 'pdr')
            ]
            score = 0.0
            for metric in metrics_vector:
                metric_value, metric_name = metric[0],metric[1]
                if metric_name == 'latency':
                    score += kpis_weights[metric_name] * smooth_threshold_above(metric_value, kpis_tresholds[metric_name], k = 1)
                if metric_name == 'join_time':
                    score += kpis_weights[metric_name] * smooth_threshold_above(metric_value, kpis_tresholds[metric_name],k=0.004)
                elif metric_name =='pdr':
                    score += kpis_weights[metric_name] * smooth_threshold_below(metric_value, kpis_tresholds[metric_name], k = 0.2)
                elif metric_name == 'lifetime':
                    score += kpis_weights[metric_name] * smooth_threshold_below(metric_value, kpis_tresholds[metric_name], k = 1.5)
            scores.append(score)
        except Exception as e:
            print(e)
            print('Failed to calculate score. Returning MAX VALUE.')
            scores.append(1)
            return None
    return scores

# to be called by the gp_minimize function
def efficience_function(parameters): 
    global args, ALL_SCORES, metrics

    config_name = 'config_{0}.json'.format(args.output_folder)
    settings = load_config()
    settings = configure_settings(settings, parameters)
    save_curr_run_config(config_name, settings)
    settings = convert_types(settings)

    # Run simulator
    os.system('python2 runSim.py --config {0}'.format(config_name))
    curr_output_folder_path = os.path.join(
        'simData',
        args.output_folder,
        'exec_numMotes_{0}'.format(args.combinations[0])
    )
    os.system('python2 compute_kpis.py --subfolder {0}'.format(curr_output_folder_path))
    time.sleep(3)

    # Get results
    num_motes = settings['settings']['combination']['exec_numMotes'][0]
    kpis = load_kpis(curr_output_folder_path, num_motes)
                
    if kpis:
        scores = compute_score(kpis)
        mean_scores = sum(scores)/float(len(scores))
        remove_results_folder(curr_output_folder_path)
        if mean_scores:
            ALL_SCORES.append(mean_scores)
            return mean_scores
        return MAX_FUNCTION_VALUE
    return MAX_FUNCTION_VALUE

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-nc','--num_cpus', type=int, help='Number of cpu cores', required=True)
    parser.add_argument('-nr','--num_runs', type=int, help='Number of times each combination of motes will be run', required=True)
    parser.add_argument('-cb','--combinations',type=int, nargs='+', help='Combination of number of Motes', required=True)
    parser.add_argument('-sf','--sched_function', help='Scheduling function alias', required=True)
    parser.add_argument('-app','--application', help='Application Type', required=True)
    parser.add_argument('-of','--output_folder', help='Output folder name', required=True)
    parser.add_argument('-ne','--num_evaluations', type=int, help='Number of evaluations of efficiency function', required=False)
    parser.add_argument('-af','--aquisition_function', type=str, help='The aquisition function used in gp_minimize', required=False)
    parser.add_argument('-sr','--sync_required', type=bool, help='if sync info is obtained in simulation', required=False)
    parser.add_argument('-nrs','--num_random_starts', type=int, help='num random starts of gp.minimize', required=False)
    parser.add_argument('-rs','--random_state', type=int, default=1, help='seed of the search itself, so the optimisation reproduces', required=False)
    parser.add_argument('-cc','--conn_class', type=str, help='connectivity_matrix', required=True)
    parser.add_argument('-nslots','--num_slots', type=int, help='number of slotframes (time) of simulation', required=True)
    parser.add_argument('-is_min','--experiment_type', type=str, help='determines the time of experiment (minimization or 2^k)', required=True)

    parser.add_argument(
        '-fc', '--factor_combinations',
        help='List of factor combinations',
        type=str,
        required=False
    )


    args = parser.parse_args()

    # which parameters this scheduling function has, and in what order. Both the
    # minimization and the 2^k path read the order from here.
    espaco = search_space(args.sched_function)
    parameters_position[:] = [nome for nome, _ in espaco]

    if args.experiment_type == 'minimization':
        print('searching {0} parameters: {1}'.format(
            len(parameters_position), ', '.join(parameters_position)))

        try:
            n_calls, n_random_starts = optimisation_budget(
                args.num_evaluations, args.num_random_starts
            )
        except ValueError as erro:
            raise SystemExit(str(erro))
        print('{0} evaluations, {1} of them random, {2} guided'.format(
            n_calls, n_random_starts, n_calls - n_random_starts))

        res = gp_minimize(efficience_function,
                        [faixa for _, faixa in espaco],
                        n_calls          = n_calls,
                        n_random_starts  = n_random_starts,
                        acq_func         = args.aquisition_function or 'gp_hedge',
                        random_state     = args.random_state,
                    )

        time.sleep(30)

        print('Optimal set of parameters\n')
        parameters_to_save = {}
        for i,paramater in enumerate(parameters_position):
            current_value = res.x[i]
            print('{0}: {1}\n'.format(paramater, current_value))
            parameters_to_save[paramater] = current_value

        with open('./{0}_parameters.json'.format(args.output_folder), 'w') as f:
            json.dump(parameters_to_save, f)

        print('Best value: {0}'.format(min(res.func_vals)))

        ys = ALL_SCORES
        xs = [i+1 for i in range(len(ALL_SCORES))]  # Evaluation numbers (x-axis)

        my_plot = plt.plot(xs, ys ,color='red', linewidth=2)
        plt.scatter(xs, ys, color='red', s=50, edgecolors='black', zorder=3)
        plt.title("Optimization Convergence")
        plt.xlabel("Number of Evaluations")
        plt.ylabel("Minimum Objective Function Value")
        plt.xticks(range(1, len(ALL_SCORES)+1, 1))  
        random_num = random.randint(1, 50)
        plt.savefig("all_values_convergence{0}.png".format(random_num))

        ax = plot_convergence(res)
        ax.set_title("Optimization Convergence")
        ax.set_xlabel("Number of Evaluations")
        ax.set_ylabel("Minimum Objective Function Value")
        plt.savefig("convergence_plot{0}.png".format(random_num))

    else:
        print('lets do the 2^k factorial experiment')   

        # build all possibilities of factors
        factors = ['traffic', 'queue', 'charge']
        all_combinations = []
        combinations = list(itertools.product([0, 1], repeat=3))
        for combination in combinations:
            current_combination = []
            for i,index in enumerate(combination):
                if index:
                    current_combination.append(factors[i])
            all_combinations.append(current_combination)

        all_combinations.sort(key=len)
        all_combinations.reverse()
        all_combinations.insert(0,['qlearningSBRC24'])
        # run each combination
        for factor_combination in all_combinations:

            # empty combination
            if not factor_combination:
                output_folder = 'baseline'
                parameters_list = []
            else:
                output_folder = '_'.join(factor_combination)
                parameters_list = load_optimal_parameters('_'.join(factor_combination))

            config_name = 'config_{0}.json'.format(output_folder)
            settings = load_config()
            settings = configure_settings(settings, parameters_list)
            settings['log_directory_name'] = output_folder
            settings['settings']['regular']['factorial_combinations'] = factor_combination
            settings['settings']['regular']['STATE_SIZE'] = 2**(len(factor_combination))
            
            #baseline runs MSF
            if not factor_combination:
                settings['settings']['regular']['sf_class'] = 'MSF'
            elif factor_combination == ['qlearningSBRC24']:
                settings['settings']['regular']['sf_class'] = 'QlearningSBRC24'

            save_curr_run_config(config_name, settings)
            settings = convert_types(settings)   

            os.system('python2 runSim.py --config {0}'.format(config_name))

            curr_output_folder_path = os.path.join(
                'simData',
                output_folder,
                'exec_numMotes_{0}'.format(args.combinations[0])
            )

            os.system('python2 compute_kpis.py --subfolder {0}'.format(curr_output_folder_path))
            os.system('python2 plot.py --inputfolder {0}'.format(curr_output_folder_path))

            import time 
            time.sleep(10)

            kpis = load_kpis(curr_output_folder_path, args.combinations[0])
            scores = compute_score(kpis)
            final_results = {
                'score': scores, 
                # to be computed
                'comulative reward': None
            }
            with open(os.path.join(curr_output_folder_path, 'final_results.json'), 'w') as f:
                json.dump(final_results, f, indent=4)
            time.sleep(2)
