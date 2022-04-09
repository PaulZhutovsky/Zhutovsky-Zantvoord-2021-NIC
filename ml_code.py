import os
import os.path as osp
from time import time
import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import make_scorer, recall_score, precision_score

import nibabel as nib
from tqdm import tqdm
import warnings
from sklearn import set_config
# to speed up scikit-learn computation I switch off data checks
set_config(assume_finite=True)
warnings.filterwarnings("ignore")


def ensure_folder(folder_path):
    if not osp.exists(folder_path):
        os.makedirs(folder_path)


def load_nifti(nifti_file, mask=None, dtype=np.float):
    data = nib.load(nifti_file).get_fdata().astype(dtype)

    if mask is not None:
        data = data[mask]
    return data


def load_signal_comps(spatial_template, signal_labels, mask_file):
    mask = load_nifti(mask_file, dtype=bool)
    # 4th dimension are subjects
    n_subj = load_nifti(spatial_template.format(signal_labels[0])).shape[-1]
    n_comp = signal_labels.size
    n_voxel = mask.sum()
    print(f'Loading data... n_comp: {n_comp}; n_subj: {n_subj}; n_voxel: {n_voxel}')
    all_signal_components = np.zeros((n_comp, n_subj, n_voxel))

    for i_signal, signal_label in enumerate(tqdm(signal_labels)):
        # tranpose because we go from n_voxel x n_subj to n_subj x n_voxel
        all_signal_components[i_signal] = load_nifti(spatial_template.format(signal_label), mask=mask, dtype=np.float).T

    return all_signal_components


def load_between_network_features(corr_file, pcorr_file):
    print('Loading between network features: correlation + partial correlation')
    X = np.array([pd.read_csv(corr_file, header=None).to_numpy(),
                  pd.read_csv(pcorr_file, header=None).to_numpy()])
    labels = np.array(['corr', 'pcorr'])
    return X, labels


def get_subjids(final_file_path):
    print('Get subject-ids of all included patients')
    final_file = np.loadtxt(final_file_path, dtype=str)
    # extract the subj_id from the full path of the preprocessed functional files
    subj_ids = np.char.rpartition(np.char.rpartition(final_file, '_task')[:, 0], 'func/')[:, -1]
    print('#Patients: {}'.format(subj_ids.size))
    return subj_ids


def get_response_status(info_file_path, subj_ids, response_status='responder_30perc'):
    print('Get responder-status: {}'.format(response_status))
    df_info = pd.read_csv(info_file_path, sep='\t', index_col='participant_id')
    response_included = df_info.loc[subj_ids, response_status]
    print(response_included.value_counts(dropna=False))
    return response_included.to_numpy()


def get_signal_ids(signal_file):
    print('Get signal components')
    # necessary to use engine='python' because the columns have an annoying empty space after the "," and that's the
    # only way to load it conveniently
    # df_comp_info = pd.read_csv(signal_file, sep=', ', engine='python', index_col='IDs')
    df_comp_info = pd.read_csv(signal_file, index_col='IDs')
    # FSLeyes counts components starting from 1 (1 to 70) but I saved them in Dual Regression style: starting from 0
    # (0 to 69)
    signal_comp = df_comp_info.loc[~df_comp_info.REJECT].index.to_numpy() - 1
    print('#Signal components: {}/{}'.format(signal_comp.size, df_comp_info.shape[0]))
    # adding the number to the name used for saving: "ic{:04d}"
    return np.char.add('ic', np.char.zfill(signal_comp.astype(str), 4))


def get_ml_pipeline(random_state):
    return make_pipeline(MinMaxScaler(feature_range=(-1, 1)), LinearSVC(random_state=random_state, max_iter=2000,
                                                                        class_weight='balanced'))


def specificity(y, y_pred, **kwargs):
    # recall == sensitivity
    # recall for other class (0) == specificity
    return recall_score(y_true=y, y_pred=y_pred, pos_label=0)


def negative_predictive_value(y, y_pred, **kwargs):
    # precision == positive predictive value
    # precision for other class (0) == negative predictive value
    return precision_score(y_true=y, y_pred=y_pred, pos_label=0)


def get_metrics():
    return ['test_AUC', 'test_ACC', 'test_SENS', 'test_SPEC', 'test_PPV', 'test_NPV']


def ml_run(all_signal_comp, between_network_features, y, signal_component_labels, between_network_labels, result_folder,
           n_splits=5, n_repeats=50, random_state=42, permutation=False, n_jobs=15):
    metrics = get_metrics()
    labels = np.concatenate((signal_component_labels, between_network_labels))
    df_metrics_all = pd.DataFrame(index=labels, columns=metrics, dtype='float64')

    # if we are running a permutation test we already have a tqdm bar for that and don't want to print an additional one
    # for the individual component classifications
    if permutation:
        iterator = labels
    else:
        iterator = tqdm(labels)

    for i_comp, label in enumerate(iterator):
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        ml_pipeline = get_ml_pipeline(random_state=random_state + 1)

        if label in signal_component_labels:
            data = all_signal_comp[i_comp]
        else:
            id_between = i_comp % signal_component_labels.size
            data = between_network_features[id_between]

        spec = make_scorer(specificity)
        npv = make_scorer(negative_predictive_value)
        scores = cross_validate(ml_pipeline, data, y, scoring={'AUC': 'roc_auc',
                                                               'ACC': 'balanced_accuracy',
                                                               'SENS': 'recall',
                                                               'SPEC': spec,
                                                               'PPV': 'precision',
                                                               'NPV': npv},
                                cv=cv, n_jobs=n_jobs, error_score='raise')
        df_scores = pd.DataFrame(scores)
        df_metrics = df_scores[metrics]

        if not permutation:
            df_metrics.to_csv(osp.join(result_folder, '{}_clf_performance.csv'.format(label)), index=False,
                              float_format='%g')

        df_metrics_all.loc[label] = df_metrics.mean(axis=0)

    if permutation:
        return df_metrics_all
    else:
        df_metrics_all.to_csv(osp.join(result_folder, 'all_networks_average_performance.csv'), float_format='%g')
        print(df_metrics_all.max())
        print(df_metrics_all.loc[df_metrics_all["test_AUC"].idxmax()])


def permutation_test(all_signal_comp, between_network_features, y, signal_component_labels, between_network_labels,
                     result_folder, n_splits=5, n_repeats=50, random_state=42, n_perm=2000, n_jobs=15):
    n_subj = y.size
    perm_index = ['perm_{:04d}'.format(i) for i in range(n_perm)]
    metrics = get_metrics()
    labels = np.concatenate((signal_component_labels, between_network_labels))
    df_perm_metrics_all = pd.DataFrame(index=labels, dtype='float64',
                                       columns=pd.MultiIndex.from_product((metrics, perm_index)))
    random_dist = np.random.RandomState(seed=random_state)
    # synchronized permutations: same permutation for each of the components (n_perm times)
    for i_perm in tqdm(range(n_perm)):
        id_permute = random_dist.permutation(n_subj)
        df_perm_metrics = ml_run(all_signal_comp, between_network_features, y[id_permute], signal_component_labels,
                                 between_network_labels, result_folder, n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state, permutation=True, n_jobs=n_jobs)

        for metric in metrics:
            df_perm_metrics_all.loc[:, (metric, perm_index[i_perm])] = df_perm_metrics[metric]

    df_perm_metrics_all.to_csv(osp.join(result_folder, 'all_networks_permutation_average_performance.csv'),
                               float_format='%g')


def run(final_file_path, info_file_path, spatial_comp_template, mask_file, signal_file, corr_file, pcorr_file,
        result_folder, n_splits=5, n_repeats=50, response_status='responder_30perc', n_perm=2000, random_seed=None,
        n_jobs=15):
    ensure_folder(result_folder)

    if random_seed is None:
        random_seed = int(time())

    print('Random seed: {}'.format(random_seed))
    np.save(osp.join(result_folder, 'random_seed.npy'), np.array([random_seed]))

    subj_ids = get_subjids(final_file_path)
    y = get_response_status(info_file_path, subj_ids, response_status=response_status)
    signal_comp_labels = get_signal_ids(signal_file)
    all_signal_comp = load_signal_comps(spatial_comp_template, signal_comp_labels, mask_file)
    between_network_features, between_network_labels = load_between_network_features(corr_file, pcorr_file)

    print('Running ML-pipeline....')
    ml_run(all_signal_comp, between_network_features, y, signal_comp_labels, between_network_labels, result_folder,
           n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed, permutation=False, n_jobs=n_jobs)

    if n_perm > 1:
        print('Running Permutation tests....')
        permutation_test(all_signal_comp, between_network_features, y, signal_comp_labels, between_network_labels,
                         result_folder, n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed + 2,
                         n_perm=n_perm, n_jobs=n_jobs)
