import argparse
import json
import os
import shutil
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plot
import numpy as np

MAX_FUNCTION_VALUE = 100000

max_values = {
    'latency': -1,
    'join_time': -1,
    'network_lifetime': -1,
    'packet_delivery_ratio': -1
}

min_values = {
    'latency': 100000,
    'join_time': 100000,
    'network_lifetime': 100000,
    'packet_delivery_ratio': 100000
}

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

parameters_position = {
   "ALFA": 0,
   "BETA": 1,
   "SLOTFRAME_INTERVAL_SIZE": 2,
   "EPSLON_DECAY_RATE": 3,
   "MIN_EPSLON": 4,
   "MAX_TX_CELLS_PASSED": 5,
   "MAX_RX_CELLS_PASSED": 6,
   "DISCRETIZE_ENERGY_PARAMETER": 7,
   "LAMBDA": 8
}

def remove_results_folder(args):
    shutil.rmtree(        
            os.path.join(
            'simData', 
            args.output_folder
        )
    )

def update_min_max(metric_name, value):
    global max_values, min_values
    max_values[metric_name] = max(max_values[metric_name], value)
    min_values[metric_name] = min(min_values[metric_name], value)

def normalize(metric_name, value):
    max_val = max_values[metric_name]
    min_val = min_values[metric_name]
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

def efficience_function(parameters):
    global args

    settings = None

    with open('config.json','r') as f:
        json_string = f.read()
        settings = json.loads(json_string)

    settings['settings']['combination']['exec_numMotes'] = args.combinations
    settings['execution']['numCPUs'] = args.num_cpus
    settings['execution']['numRuns'] = args.num_runs
    settings['settings']['regular']['sf_class'] = args.sched_function
    settings['log_directory_name']= args.output_folder

    # Configure simulator with the parameters
    for parameter_name, position in parameters_position.items():
        parameter_value = parameters[position]
        settings['settings']['regular'][parameter_name] = parameter_value

    settings = convert_types(settings)

    with open('config.json', 'w') as f:
        f.write(json.dumps(settings, indent=4))

    # Run simulator
    os.system('python2 runSim.py')

    # Get results
    num_motes = settings['settings']['combination']['exec_numMotes'][0]

    kpis = None
    for tentativa in range(3):
        try:
            with open(
                os.path.join(
                    'simData', 
                    args.output_folder, 
                    'exec_numMotes_{0}.dat.kpi'.format(num_motes)
                )
            , 'r') as f:
                json_string = f.read()
                kpis = json.loads(json_string)
                break
        except Exception as e:
            print(e)
            print("Something went wrong reading KPIs on try {0}".format(tentativa))
                
    if kpis:

        latency = kpis['0']['global-stats']['e2e-upstream-latency'][0]['mean']
        join_time = kpis['0']['global-stats']["joining-time"][0]['mean']
        network_lifetime = kpis['0']['global-stats']['network_lifetime'][0]['min']
        packet_delivery_ratio = kpis['0']['global-stats']['e2e-upstream-delivery'][0]['value']

        update_min_max('latency', latency)
        update_min_max('join_time', join_time)
        update_min_max('network_lifetime', network_lifetime)
        update_min_max('packet_delivery_ratio', packet_delivery_ratio)

        latency = normalize('latency', latency)
        join_time = normalize('join_time', join_time)
        network_lifetime = normalize('network_lifetime', network_lifetime)
        packet_delivery_ratio = normalize('packet_delivery_ratio', packet_delivery_ratio)


        try:
            score = latency - join_time + network_lifetime - packet_delivery_ratio
        except Exception as e:
            print('Failed to calculate score. Returning infinity.')
            return MAX_FUNCTION_VALUE
        
        # Remove results folder
        remove_results_folder(args)

        return score
    else:
        return MAX_FUNCTION_VALUE

parser = argparse.ArgumentParser()

parser.add_argument('-nc','--num_cpus', type=int, help='Number of cpu cores', required=True)
parser.add_argument('-nr','--num_runs', type=int, help='Number of times each combination of motes will be run', required=True)
parser.add_argument('-cb','--combinations',type=int, nargs='+', help='Combination of number of Motes', required=True)
parser.add_argument('-sf','--sched_function', help='Scheduling function alias', required=True)
parser.add_argument('-app','--application', help='Application Type', required=True)
parser.add_argument('-of','--output_folder', help='Output folder name', required=True)
parser.add_argument('-ne','--num_evaluations', type=int, help='Number of evaluations of efficiency function', required=True)
parser.add_argument('-af','--aquisition_function', type=str, help='The aquisition function used in gp_minimize', required=True)


args = parser.parse_args()

res = gp_minimize(efficience_function,  # The function to minimize
                  [(0.1, 0.9),  # ALFA
                   (0.1, 0.9),  # BETA
                   (1,10),     # SLOTFRAME_INTERVAL_SIZE
                   (0.1, 0.3), # EPSLON_DECAY_RATE
                   (0.05, 0.1),  # MIN_EPSLON
                   (1, 100),   # MAX_TX_CELLS_PASSED
                   (1, 100),   # MAX_RX_CELLS_PASSED
                   (0.5, 0.9),  # DISCRETIZE_ENERGY_PARAMETER
                   (1, 10)],    # LAMBDA    
                  acq_func="gp_hedge",        # The acquisition function
                  n_calls=args.num_evaluations,  # The number of evaluations of f
                  n_random_starts=5,   # The number of random initialization points
                  noise=0.1**2,        # The noise level (optional)
                  random_state=1234)   # The random seed


print('Optimal set of parameters:')
print(res.x)

print('Best value:')
print(min(res.func_vals))

ax = plot_convergence(res)
ax.set_title("Optimization Convergence")
ax.set_xlabel("Number of Evaluations")
ax.set_ylabel("Minimum Objective Function Value")

plot.savefig("convergence_plot_{0}.png".format(args.aquisition_function))
