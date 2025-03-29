import os
import json

#traffic, queue, charge
# a, b, c

def compute_dataframe_effects(factors_score):
    for factor_name, score in factors_score.items():
        if factor_name != 'baseline':
            function_name = f'{factor_name}_effect'
            result = globals()[function_name](factors_score)
            print(f'{factor_name}: {result}')


def traffic_effect(factors_score):
    return (
        (factors_score['traffic']
        - factors_score['baseline']
        + factors_score['traffic_queue']
        - factors_score['queue']
        + factors_score['queue_charge']
        - factors_score['charge']
        + factors_score['traffic_queue_charge']
        - factors_score['queue_charge'] )/8
    )

def queue_effect(factors_score):
    return (
        (factors_score['queue']
        + factors_score['traffic_queue']
        + factors_score['queue_charge']
        + factors_score['traffic_queue_charge']
        - factors_score['baseline']
        - factors_score['traffic']
        - factors_score['charge']
        - factors_score['traffic_charge']) /8
    )

def charge_effect(factors_score):
    return (
        (factors_score['charge']
        + factors_score['traffic_charge']
        + factors_score['queue_charge']
        + factors_score['traffic_queue_charge']
        - factors_score['baseline']
        - factors_score['traffic']
        - factors_score['queue']
        - factors_score['traffic_queue']) /8
    )

def traffic_queue_effect(factors_score):
    return (
        (factors_score['traffic_queue']
        - factors_score['traffic']
        - factors_score['queue']
        + factors_score['baseline']
        + factors_score['traffic_queue_charge']
        - factors_score['queue_charge']
        - factors_score['traffic_charge']
        + factors_score['charge']) /8
    )

def traffic_charge_effect(factors_score):
    return (
        (factors_score['baseline']
        - factors_score['traffic']
        + factors_score['queue']
        - factors_score['traffic_queue']
        - factors_score['charge']
        + factors_score['traffic_charge']
        - factors_score['queue_charge']
        + factors_score['traffic_queue_charge']) /8
    )

def queue_charge_effect(factors_score):
     return (
        (factors_score['baseline']
        + factors_score['traffic']
        - factors_score['queue']
        - factors_score['traffic_queue']
        - factors_score['charge']
        - factors_score['traffic_charge']
        + factors_score['queue_charge']
        + factors_score['traffic_queue_charge']) /8
    )   

def traffic_queue_charge_effect(factors_score):
     return (
        (factors_score['traffic_queue_charge']
        - factors_score['queue_charge']
        - factors_score['traffic_charge']
        + factors_score['charge']
        - factors_score['traffic_queue']
        + factors_score['queue']
        + factors_score['traffic']
        - factors_score['baseline']) /8
    )   

def read_scores():
    experiment_dirs = os.listdir('simData')
    factors_scores = {}
    for dir in experiment_dirs:
        curr_path = os.path.join('./simData', dir, 'exec_numMotes_50','final_results.json')
        try:
            with open(curr_path, 'r') as f:
                results = json.load(f)
                curr_score = results['score']
                factors_scores[dir] = curr_score
        except: pass

    return factors_scores


if __name__ == '__main__':
    factors_score = read_scores()
    compute_dataframe_effects(factors_score)