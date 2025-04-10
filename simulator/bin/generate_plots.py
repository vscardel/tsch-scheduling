import json
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.stats import norm


def generate_box_plots(data_lists, metric_label):
    plt.style.use("presentation.mplstyle")
    plt.rcParams['text.usetex'] = True

    colors = [
        (33/255, 133/255, 197/255, 0.7),  # Azul
        (224/255, 202/255, 60/255, 0.7),  # Amarelo
        (243/255, 66/255, 19/255, 0.7),   # Laranja avermelhado
        (38/255, 166/255, 91/255, 0.7)
    ]
    labels = ['Q-poisson', 'Q-static', 'MSF', 'EMSF']

    fig, ax = plt.subplots()
    box = ax.boxplot(
        data_lists,
        patch_artist=True,
        labels=labels,
        showmeans=True,
        meanline=True,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=1.5, color='black'),
        meanprops=dict(linewidth=1.5, color='red')
    )

    # Pintar as caixas com as cores
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    for cap in box['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)
    for flier in box['fliers']:
        flier.set(marker='o', color='gray', alpha=0.5, markersize=5)

    if metric_label == 'Latencies':
        plt.ylabel('Latencies (s)')
    elif metric_label == 'Lifetimes':
        plt.ylabel('Lifetimes (Years)')
    elif metric_label == 'PDRS':
        plt.ylabel('PDRS (\\%)')
    elif metric_label == 'Scores':
        plt.ylabel('')
    elif metric_label == 'Join Times':
        plt.ylabel('Join Times (s)')

    plt.tight_layout()
    os.makedirs('./images/boxplots', exist_ok=True)
    plt.savefig(f'./images/boxplots/{metric_label.lower().replace(" ", "_")}_boxplot.pdf', format='pdf', dpi=300)
    plt.clf()


def generate_bar_plots(metrics_data, metric_label):
    plt.style.use("presentation.mplstyle")
    plt.rcParams['text.usetex'] = True
    barWidth = 0.1
    colors = [
        (33/255, 133/255, 197/255, 0.7),  # Azul
        (224/255, 202/255, 60/255, 0.7),  # Amarelo
        (243/255, 66/255, 19/255, 0.7),   # Laranja avermelhado
        (38/255, 166/255, 91/255, 0.7)    # verde
    ]
    labels = ['Q-poisson', 'Q-static', 'MSF', 'EMSF']
    
    base_positions = np.arange(1)

    for i in range(len(metrics_data)):
        mean_val = metrics_data[i][0]
        conf_interval = metrics_data[i][1]
        try:
            conf_height = conf_interval[1] - conf_interval[0]
        except:
            conf_height = conf_interval
        bar_position = base_positions + i * barWidth

        plt.bar(
            bar_position,
            mean_val,
            width=barWidth,
            color=colors[i],
            edgecolor='black',
            yerr=conf_height,
            capsize=7,
            label=labels[i]
        )

    plt.xticks(
        [r + barWidth for r in base_positions],
        [metric_label]
    )
    if metric_label == 'Latencies':
        plt.ylabel('Latencies (s)')
    elif metric_label == 'Lifetimes':
        plt.ylabel('Lifetimes (Years)')
    elif metric_label == 'PDRS':
        plt.ylabel('PDRS (\\%)')
    elif metric_label == 'Scores':
        plt.ylabel('')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./images/barplots/{metric_label.lower().replace(" ", "_")}.pdf', format='pdf', dpi=300)
    plt.clf()


def compute_confidence_interval(values, confidence=0.95):
    n = len(values)
    sample_mean = np.mean(values)
    sample_sd = np.std(values, ddof=1)  # amostral
    z_score = norm.ppf(1 - (1 - confidence) / 2)  # z para 95% ≈ 1.96
    margin_error = z_score * (sample_sd / np.sqrt(n))
    return (sample_mean - margin_error, sample_mean + margin_error)

def load_kpis(folder_path, num_motes):
    kpis = None
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
    return kpis


def generate_folder_path(method_name):
    output_folder_path = os.path.join(
        'simData',
        method_name,
        'exec_numMotes_50'
    )
    return output_folder_path

def get_metric_list(metric_name, kpis):
    metrics = []
    for run in kpis:
        metric = 0.0
        if metric_name == 'latency':
            metric = kpis[run]['global-stats']['e2e-upstream-latency'][0]['mean']
        elif metric_name == 'join_time':
            metric = kpis[run]['global-stats']["joining-time"][0]['mean'] / 100
        elif metric_name == 'lifetime':
            metric = kpis[run]['global-stats']['network_lifetime'][0]['min']
        elif metric_name == 'pdr':
            metric = kpis[run]['global-stats']['e2e-upstream-delivery'][0]['value']
        metrics.append(metric)
    return metrics

def load_scores(folder_path):
    results = None
    with open(os.path.join(folder_path, 'final_results.json')) as f:
        results = json.load(f)['score']
    return results

if __name__ == '__main__':
    os.makedirs('./images/barplots', exist_ok=True)
    # load the kpis
    kpis_traffic_queue_charge = load_kpis(
        generate_folder_path('traffic_queue_charge'),
        50
    )

    kpis_qlearningSBRC24 = load_kpis(
        generate_folder_path('qlearningSBRC24'),
        50
    )

    kpis_msf = load_kpis(
        generate_folder_path('baseline'),
        50
    )

    kpis_emsf = load_kpis(
        generate_folder_path('EMSF'),
        50
    )

    latencies_tfq = get_metric_list('latency', kpis_traffic_queue_charge)
    joins_time_tfq = get_metric_list('join_time', kpis_traffic_queue_charge)
    lifetimes_tfq = get_metric_list('lifetime', kpis_traffic_queue_charge)
    pdrs_tfq = get_metric_list('pdr', kpis_traffic_queue_charge)
    final_scores_tfq = load_scores(generate_folder_path('traffic_queue_charge'))

    latencies_qsbrc24= get_metric_list('latency', kpis_qlearningSBRC24)
    joins_time_qsbrc24 = get_metric_list('join_time', kpis_qlearningSBRC24)
    lifetimes_qsbrc24 = get_metric_list('lifetime', kpis_qlearningSBRC24)
    pdrs_qsbrc24 = get_metric_list('pdr', kpis_qlearningSBRC24)
    final_scores_qsbrc24 = load_scores(generate_folder_path('qlearningSBRC24'))

    latencies_msf = get_metric_list('latency', kpis_msf)
    joins_time_msf = get_metric_list('join_time', kpis_msf)
    lifetimes_msf = get_metric_list('lifetime', kpis_msf)
    pdrs_msf = get_metric_list('pdr', kpis_msf)
    final_scores_msf = load_scores(generate_folder_path('baseline'))

    latencies_emsf = get_metric_list('latency', kpis_emsf)
    joins_time_emsf = get_metric_list('join_time', kpis_emsf)
    lifetimes_emsf = get_metric_list('lifetime', kpis_emsf)
    pdrs_emsf = get_metric_list('pdr', kpis_emsf)
    final_scores_emsf = load_scores(generate_folder_path('traffic_queue_charge'))

    latencies_data = [
        (np.mean(latencies_tfq), compute_confidence_interval(latencies_tfq)),
        (np.mean(latencies_qsbrc24), compute_confidence_interval(latencies_qsbrc24)),
        (np.mean(latencies_msf), compute_confidence_interval(latencies_msf)),
         (np.mean(latencies_emsf), compute_confidence_interval(latencies_emsf)),
    ]


    join_time_data = [
        (np.mean(joins_time_tfq), compute_confidence_interval(joins_time_tfq)),
        (np.mean(joins_time_qsbrc24), compute_confidence_interval(joins_time_qsbrc24)),
        (np.mean(joins_time_msf), compute_confidence_interval(joins_time_msf)),
         (np.mean(joins_time_emsf), compute_confidence_interval(joins_time_emsf)),
    ]

    lifetime_data = [
        (np.mean(lifetimes_tfq), compute_confidence_interval(lifetimes_tfq)),
        (np.mean(lifetimes_qsbrc24), compute_confidence_interval(lifetimes_qsbrc24)),
        (np.mean(lifetimes_msf), compute_confidence_interval(lifetimes_msf)),
        (np.mean(lifetimes_emsf), compute_confidence_interval(lifetimes_emsf)),
    ]

    pdr_data = [
        (np.mean(pdrs_tfq), compute_confidence_interval(pdrs_tfq)),
        (np.mean(pdrs_qsbrc24), compute_confidence_interval(pdrs_qsbrc24)),
        (np.mean(pdrs_msf), compute_confidence_interval(pdrs_msf)),
        (np.mean(pdrs_emsf), compute_confidence_interval(pdrs_emsf)),

    ]

    score_data = [
        (np.mean(final_scores_tfq), compute_confidence_interval(final_scores_tfq)),
        (np.mean(final_scores_qsbrc24), compute_confidence_interval(final_scores_qsbrc24)),
        (np.mean(final_scores_msf), compute_confidence_interval(final_scores_msf)),
        (np.mean(final_scores_emsf), compute_confidence_interval(final_scores_emsf)),
    ]

    generate_bar_plots(latencies_data, 'Latencies')
    generate_bar_plots(join_time_data, 'Join Times')
    generate_bar_plots(lifetime_data, 'Lifetimes')
    generate_bar_plots(pdr_data, 'PDRS')
    generate_bar_plots(score_data, 'Scores')

    generate_box_plots([latencies_tfq, latencies_qsbrc24, latencies_msf, latencies_emsf], 'Latencies')
    generate_box_plots([joins_time_tfq, joins_time_qsbrc24, joins_time_msf, joins_time_emsf], 'Join Times')
    generate_box_plots([lifetimes_tfq, lifetimes_qsbrc24, lifetimes_msf, lifetimes_emsf], 'Lifetimes')
    generate_box_plots([pdrs_tfq, pdrs_qsbrc24, pdrs_msf, pdrs_emsf], 'PDRS')
    generate_box_plots([final_scores_tfq, final_scores_qsbrc24, final_scores_msf, final_scores_emsf], 'Scores')


