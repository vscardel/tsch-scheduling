import os
import json
from pprint import pprint
import pandas as pd
import scipy.stats as stats



#traffic, queue, charge
# a, b, c

input_anova = {}
all_scores = []


def compute_sst(all_scores):
    mean_score = sum(all_scores)/len(all_scores)
    sst_total = 0
    for score in all_scores:
        sst_total += (score - mean_score)**2
    return sst_total
    

# how much each combination contribute to the result
def compute_percent_contribution():
    ss_of_squares = 0.0
    for combination,value in input_anova.items():
        ss_of_squares += value['sum_of_squares']
    contributions = {}
    for combination,value in input_anova.items():
        percent_contribution = value['sum_of_squares'] / ss_of_squares
        contributions[combination] = percent_contribution
    for k,v in contributions.items():
        input_anova[k]['percent_contribution'] = v 

def compute_dataframe_effects(factors_score):
    for factor_name, score in factors_score.items():
        if factor_name != 'baseline':
            function_name = f'{factor_name}_effect'
            globals()[function_name](factors_score)
            compute_percent_contribution()
        
    df = pd.DataFrame.from_dict(input_anova, orient='index')
    return df

def traffic_effect(factors_score):
    contrast = \
        (factors_score['traffic']
        - factors_score['baseline']
        + factors_score['traffic_queue']
        - factors_score['queue']
        + factors_score['queue_charge']
        - factors_score['charge']
        + factors_score['traffic_queue_charge']
        - factors_score['queue_charge'] )
    effect = contrast / (4*input_anova['traffic']['num_replicas'])
    sum_of_squares = (contrast**2) / (8*input_anova['traffic']['num_replicas'])
    input_anova['traffic']['effect_estimate'] = effect
    input_anova['traffic']['sum_of_squares'] = sum_of_squares

def queue_effect(factors_score):
    contrast = \
        (factors_score['queue']
        + factors_score['traffic_queue']
        + factors_score['queue_charge']
        + factors_score['traffic_queue_charge']
        - factors_score['baseline']
        - factors_score['traffic']
        - factors_score['charge']
        - factors_score['traffic_charge']) 
    
    effect = contrast / (4*input_anova['queue']['num_replicas'])
    sum_of_squares = (contrast**2) / (8*input_anova['queue']['num_replicas'])
    input_anova['queue']['effect_estimate'] = effect
    input_anova['queue']['sum_of_squares'] = sum_of_squares

def charge_effect(factors_score):
    contrast = \
        (factors_score['charge']
        + factors_score['traffic_charge']
        + factors_score['queue_charge']
        + factors_score['traffic_queue_charge']
        - factors_score['baseline']
        - factors_score['traffic']
        - factors_score['queue']
        - factors_score['traffic_queue']) 
    
    effect = contrast / (4*input_anova['charge']['num_replicas'])
    sum_of_squares = (contrast**2) / (8*input_anova['charge']['num_replicas'])
    input_anova['charge']['effect_estimate'] = effect
    input_anova['charge']['sum_of_squares'] = sum_of_squares
    

def traffic_queue_effect(factors_score):
    contrast = \
        (factors_score['traffic_queue']
        - factors_score['traffic']
        - factors_score['queue']
        + factors_score['baseline']
        + factors_score['traffic_queue_charge']
        - factors_score['queue_charge']
        - factors_score['traffic_charge']
        + factors_score['charge']) 
    
    effect = contrast / (4*input_anova['traffic_queue']['num_replicas'])
    sum_of_squares = (contrast**2) / (8*input_anova['traffic_queue']['num_replicas'])
    input_anova['traffic_queue']['effect_estimate'] = effect
    input_anova['traffic_queue']['sum_of_squares'] = sum_of_squares

def traffic_charge_effect(factors_score):
    contrast = \
        (factors_score['baseline']
        - factors_score['traffic']
        + factors_score['queue']
        - factors_score['traffic_queue']
        - factors_score['charge']
        + factors_score['traffic_charge']
        - factors_score['queue_charge']
        + factors_score['traffic_queue_charge']) 
    
    effect = contrast / (4*input_anova['traffic_charge']['num_replicas'])
    sum_of_squares = (contrast**2) / (8*input_anova['traffic_charge']['num_replicas'])
    input_anova['traffic_charge']['effect_estimate'] = effect
    input_anova['traffic_charge']['sum_of_squares'] = sum_of_squares  

def queue_charge_effect(factors_score):
    contrast = \
        (factors_score['baseline']
        + factors_score['traffic']
        - factors_score['queue']
        - factors_score['traffic_queue']
        - factors_score['charge']
        - factors_score['traffic_charge']
        + factors_score['queue_charge']
        + factors_score['traffic_queue_charge']) 
    
    effect = contrast / (4*input_anova['queue_charge']['num_replicas'])
    sum_of_squares = (contrast**2) / (8*input_anova['queue_charge']['num_replicas'])
    input_anova['queue_charge']['effect_estimate'] = effect
    input_anova['queue_charge']['sum_of_squares'] = sum_of_squares  
    

def traffic_queue_charge_effect(factors_score):
    contrast = \
        (factors_score['traffic_queue_charge']
        - factors_score['queue_charge']
        - factors_score['traffic_charge']
        + factors_score['charge']
        - factors_score['traffic_queue']
        + factors_score['queue']
        + factors_score['traffic']
        - factors_score['baseline']) 
    
    effect = contrast / (4*input_anova['traffic_queue_charge']['num_replicas'])
    sum_of_squares = (contrast**2) / (8*input_anova['traffic_queue_charge']['num_replicas'])
    input_anova['traffic_queue_charge']['effect_estimate'] = effect
    input_anova['traffic_queue_charge']['sum_of_squares'] = sum_of_squares  
    

def read_scores():
    global total_sse
    experiment_dirs = os.listdir('simData')
    factors_scores = {}
    for dir in experiment_dirs:
        curr_path = os.path.join('./simData', dir, 'exec_numMotes_50','final_results.json')
        try:
            with open(curr_path, 'r') as f:
                results = json.load(f)
                for score in results['score']:
                    all_scores.append(score)
                curr_score = sum(results['score'])
                factors_scores[dir] = curr_score
                if dir != 'baseline':
                    input_anova[dir] = {
                        'num_replicas': len(results['score']),
                        'total_score': curr_score,
                        'effect_estimate': 0.0,
                        'sum_of_squares': 0.0,
                        'percent_contribution': 0.0
                    }
        except: pass    
    return factors_scores


def perform_anova(input_anova_df):
    # Número total de observações (8 combinações × 3 réplicas = 24)
    N = len(all_scores)
    
    # Graus de liberdade
    df_total = N - 1
    df_effects = 7  # 3 principais + 3 interações de 2 fatores + 1 interação de 3 fatores
    df_error = df_total - df_effects  # GL do erro = Total GL - GL dos efeitos
    
    SS_total = compute_sst(all_scores)
    
    SS_effects = input_anova_df[input_anova_df.index != 'baseline']['sum_of_squares'].sum()
    
    SS_error = SS_total - SS_effects
    
    MS_error = SS_error / df_error
    
    # Estatísticas F e p-valores
    input_anova_df['F_value'] = input_anova_df['sum_of_squares'] / MS_error
    input_anova_df['p_value'] = 1 - stats.f.cdf(input_anova_df['F_value'], 1, df_error)
    
    return input_anova_df

factors_score = read_scores()
input_anova_df = compute_dataframe_effects(factors_score)
anova_df = perform_anova(input_anova_df)
final_df = anova_df.drop('num_replicas', axis=1)
pprint(final_df)
final_df.to_excel('anova.xlsx', index=False) 