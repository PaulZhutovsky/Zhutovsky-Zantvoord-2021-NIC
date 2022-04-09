import os
import os.path as osp
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_all_perf(result_folder):
    return pd.read_csv(osp.join(result_folder, 'all_networks_average_performance.csv'))


def load_perf_files(result_folder):
    return sorted(glob(osp.join(result_folder, '*_clf_performance.csv')))


def get_component_ids(performance_files):
    return np.char.rpartition(np.char.partition(performance_files, '_clf')[:, 0], os.sep)[:, -1]


def load_permutation_distribution(permutation_file):
    return pd.read_csv(permutation_file, index_col=0, header=[0, 1])


def get_metrics_dict():
    return {'AUC': 'test_AUC', 'ACC': 'test_ACC', 'SENS': 'test_SENS', 'SPEC': 'test_SPEC', 'PPV': 'test_PPV',
            'NPV': 'test_NPV'}


def get_perf_statistics():
    return ['mean', 'SD', 'SE', 'median', 'p', 'p_FWE']


def calculate_p_values(comp_id, neutral_perm, perm_dist_all, neutral_max_value):
    # uncorrected p-value:
    perm_distribution = np.concatenate((perm_dist_all.loc[comp_id].to_numpy(), [neutral_perm]))
    p_value = np.mean(perm_distribution >= neutral_perm)

    # corrected p-value:
    max_distribution = np.concatenate((perm_dist_all.max().to_numpy(), [neutral_max_value]))
    p_FWE = np.mean(max_distribution >= neutral_perm)

    return p_value, p_FWE


def run(result_folder, permutation_file=None):
    # all performance is needed in case FWE correction via the maximum statistic is performed
    df_all_perf = load_all_perf(result_folder)
    perf_files = load_perf_files(result_folder)
    component_ids = get_component_ids(perf_files)
    metrics_dict = get_metrics_dict()
    metrics = metrics_dict.keys()
    perf_stats = get_perf_statistics()

    df_perm = None
    if permutation_file:
        df_perm = load_permutation_distribution(permutation_file)

    df_results = pd.DataFrame(index=component_ids, columns=pd.MultiIndex.from_product((metrics, perf_stats)),
                              dtype='float64')

    for i_comp, perf_file in enumerate(tqdm(perf_files)):
        comp_id = component_ids[i_comp]
        df_perf_comp = pd.read_csv(perf_file)

        for metric in metrics:

            df_results.loc[comp_id, (metric, 'mean')] = df_perf_comp[metrics_dict[metric]].mean()
            df_results.loc[comp_id, (metric, 'SD')] = df_perf_comp[metrics_dict[metric]].std()
            df_results.loc[comp_id, (metric, 'SE')] = (df_results.loc[comp_id, (metric, 'SD')] /
                                                       np.sqrt(df_perf_comp.shape[0]))
            df_results.loc[comp_id, (metric, 'median')] = df_perf_comp[metrics_dict[metric]].median()

            # are we calculating p-values?
            if df_perm is not None:
                p_val, p_FWE = calculate_p_values(comp_id,
                                                  df_results.loc[comp_id, (metric, 'mean')],
                                                  df_perm[metrics_dict[metric]],
                                                  df_all_perf[metrics_dict[metric]].max())

                df_results.loc[comp_id, (metric, 'p')] = p_val
                df_results.loc[comp_id, (metric, 'p_FWE')] = p_FWE

    df_results.sort_values(by=[('AUC', 'mean')], axis='index', ascending=False, inplace=True)
    df_results.to_csv(osp.join(result_folder, 'all_results.csv'))
