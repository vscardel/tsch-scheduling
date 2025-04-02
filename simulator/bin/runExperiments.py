import argparse
import json
import os
import shutil
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
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

parameters_position = [
   "ALFA",
   "BETA",
   "EPSLON_DECAY_RATE",
   "MIN_EPSLON",
   "EPSLON_THRESHOLD"
]

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
    'latency': 0.54,
    'pdr': 0.98,
    'lifetime': 0.49,
    'join_time': 880,
}

metrics = {
    'latencies': [],
    'pdrs': [],
    'join_times': [],
    'lifetimes': []
}

def smooth_threshold_above(x, T, k=10):
    return 1 / (1 + np.exp(-k * (x - T)))

def smooth_threshold_below(x, T, k=10):
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

def compute_score(kpis):
    scores = []
    for run in kpis:
        #seconds
        latency = kpis[run]['global-stats']['e2e-upstream-latency'][0]['mean']
        #convert join_time to seconds
        join_time = kpis[run]['global-stats']["joining-time"][0]['mean'] / 100
        #years
        network_lifetime = kpis[run]['global-stats']['network_lifetime'][0]['min']
        
        packet_delivery_ratio = kpis[run]['global-stats']['e2e-upstream-delivery'][0]['value']

        metrics['latencies'].append(latency)
        metrics['pdrs'].append(packet_delivery_ratio)
        metrics['lifetimes'].append(network_lifetime)
        metrics['join_times'].append(join_time)

        try:
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
                    score += kpis_weights[metric_name] * smooth_threshold_above(metric_value, kpis_tresholds[metric_name])
                if metric_name == 'join_time':
                    score += kpis_weights[metric_name] * smooth_threshold_above(metric_value, kpis_tresholds[metric_name],k=0.01)
                elif metric_name in ['pdr', 'lifetime']:
                    score += kpis_weights[metric_name] * smooth_threshold_below(metric_value, kpis_tresholds[metric_name])
            scores.append(score)
        except Exception as e:
            print(e)
            print('Failed to calculate score. Returning MAX VALUE.')
            return None
    return scores

# to be called by the gp_minimize function
def efficience_function(parameters):
    import ipdb;
    ipdb.set_trace()    
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

if args.experiment_type == 'minimization':
    res = gp_minimize(efficience_function,  # The function to minimize
                    [(0.1, 0.9),  # ALFA
                    (0.1, 0.9),  # BETA
                    #    (5,10),     # SLOTFRAME_INTERVAL_SIZE
                    (0.001, 0.005), # EPSLON_DECAY_RATE
                    (0.05, 0.1),  # MIN_EPSLON
                    #    (50, 100),   # MAX_TX_CELLS_PASSED
                    #    (50, 100),   # MAX_RX_CELLS_PASSED
                    (0.5, 0.7)],   # EPSLON_THRESHOLD   
                    n_calls=25
                )

    time.sleep(10)

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

        #baseline runs MSF
        if not factor_combination:
            settings['settings']['regular']['sf_class'] = 'MSF'

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
