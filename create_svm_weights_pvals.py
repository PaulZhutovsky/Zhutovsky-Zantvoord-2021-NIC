import os.path as osp
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from run_ml_pipeline import (final_file_path, info_file_path, response_status, signal_file, mask_file, result_folder,
                             spatial_comp_template, corr_mat_file_path, pcorr_mat_file_path, n_splits, n_repeats)
from ml_code import (get_subjids, get_response_status, get_signal_ids, load_signal_comps, ensure_folder,
                     load_between_network_features, get_ml_pipeline)
from calc_linearsvm_pvals_gaonkar import calculate_p_values


def load_result_file(result_folder):
    return pd.read_csv(osp.join(result_folder, 'all_results.csv'), index_col=0, header=[0, 1])


def load_random_seed(result_folder):
    return np.load(osp.join(result_folder, 'random_seed.npy'))[0]


def load_data(final_file_path, info_file_path, response_status, signal_file, spatial_comp_template, mask_file,
              corr_file, pcorr_file):
    subj_ids = get_subjids(final_file_path)
    y = get_response_status(info_file_path, subj_ids, response_status=response_status)
    signal_comp_labels = get_signal_ids(signal_file)
    all_signal_comp = load_signal_comps(spatial_comp_template, signal_comp_labels, mask_file)
    between_network_features, between_network_labels = load_between_network_features(corr_file, pcorr_file)
    return y, all_signal_comp, signal_comp_labels, between_network_features, between_network_labels


def store_pvals(p_values, sign_of_stat, mask, affine, save_folder, label_clf):
    p_values_brain_log = np.zeros_like(mask, dtype=np.float)
    p_values_brain_log_signed = np.zeros_like(p_values_brain_log)
    p_values_brain_bonferroni = np.zeros_like(p_values_brain_log, dtype=np.int)
    p_values_brain_fdr = np.zeros_like(p_values_brain_log, dtype=np.int)
    p_values_brain = np.zeros_like(p_values_brain_log)

    # transfer the p-values np.log10(p) makes them all negative the smallest p-values being the maximally negative ones
    # -np.log10(p) makes the smallest p-values to have the highest value (visualization purposes)
    p_values_brain[mask] = p_values
    p_values_brain_log[mask] = -np.log10(p_values)
    p_values_brain_log_signed[mask] = (-np.log10(p_values)) * sign_of_stat
    p_values_brain_bonferroni[mask] = (p_values < (0.05/p_values.size)).astype(np.int)
    p_values_brain_fdr[mask] = multipletests(p_values, alpha=0.05, method='fdr_tsbh')[0].astype(int)

    print('# surviving Bonferroni-correction: {}'.format(p_values_brain_bonferroni.sum()))
    print('# surviving FDR-correction: {}'.format(p_values_brain_fdr.sum()))
    img_p_log = nib.Nifti1Image(p_values_brain_log, affine=affine)
    img_p_log_signed = nib.Nifti1Image(p_values_brain_log_signed, affine=affine)
    img_p = nib.Nifti1Image(p_values_brain, affine=affine)
    img_p_bonferroni = nib.Nifti1Image(p_values_brain_bonferroni, affine=affine)
    img_p_fdr = nib.Nifti1Image(p_values_brain_fdr, affine=affine)

    nib.save(img_p_log_signed, osp.join(save_folder, f'{label_clf}_pvals_minuslog10_signed.nii.gz'))
    nib.save(img_p_log, osp.join(save_folder, f'{label_clf}_pvals_minuslog10.nii.gz'))
    nib.save(img_p, osp.join(save_folder, f'{label_clf}_pvals.nii.gz'))
    nib.save(img_p_bonferroni, osp.join(save_folder, f'{label_clf}_pvals_bonferroni.nii.gz'))
    nib.save(img_p_fdr, osp.join(save_folder, f'{label_clf}_pvals_fdr.nii.gz'))


def calc_and_store_pvals(X, y, weights_mean, weights_whole, mask_file, label_classification, save_folder):

    p_mean, signs_mean = calculate_p_values(X, y, weights_mean)
    p_whole, signs_whole = calculate_p_values(X, y, weights_whole)
    mask_img = nib.load(mask_file)
    affine = mask_img.affine
    mask = mask_img.get_fdata().astype(bool)

    print('mean weight:')
    store_pvals(p_mean, signs_mean, mask, affine, save_folder, label_classification + '_meanW')
    print('total-sample weight')
    store_pvals(p_whole, signs_whole, mask, affine, save_folder, label_classification + '_totalW')


def run_ml(X, y, expected_accuracy, n_repeats, n_splits, random_state):
    n_iter = n_splits * n_repeats
    cv = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=random_state)
    # the + 1 is like in the original ml_code to make the call identical
    ml_pipeline = get_ml_pipeline(random_state + 1)
    ml_pipeline.fit(X, y)
    weights_whole = ml_pipeline.named_steps['linearsvc'].coef_.squeeze()
    acc_all = np.zeros(n_iter)
    weights_all = np.zeros((n_iter, X.shape[1]))

    for i_cv, (train_id, test_id) in enumerate(tqdm(cv.split(X, y))):
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]

        ml_pipeline_cv = clone(ml_pipeline)
        ml_pipeline_cv.fit(X_train, y_train)
        y_pred = ml_pipeline_cv.predict(X_test)

        acc_all[i_cv] = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
        weights_all[i_cv] = ml_pipeline_cv.named_steps['linearsvc'].coef_.squeeze()

    assert np.allclose(acc_all.mean(), expected_accuracy), 'Missmatch in performance?!'

    return weights_all, weights_whole


def run():
    weights_folder = osp.join(result_folder, 'svm_weights_pvals')
    ensure_folder(weights_folder)

    (y, all_within_networks, all_within_labels,
     all_between_networks, all_between_labels) = load_data(final_file_path, info_file_path, response_status,
                                                           signal_file, spatial_comp_template, mask_file,
                                                           corr_mat_file_path, pcorr_mat_file_path)

    all_labels = np.concatenate((all_within_labels, all_between_labels))
    all_labels_ids = np.concatenate((np.arange(all_within_labels.size), np.arange(all_between_labels.size)))

    random_seed_experiment = load_random_seed(result_folder)

    df_results = load_result_file(result_folder)
    # it's actually just one...
    # ids_significant = df_results.index[df_results[('ACC', 'p_FWE')] < 0.05]
    # for the supplementary Figure I take all components which are p_unc < 0.05
    ids_significant = df_results.index[df_results[('ACC', 'p')] < 0.05]

    for id_significant in ids_significant:
        print(id_significant)

        expected_accuracy = df_results.loc[id_significant, ('ACC', 'mean')]
        id_data = all_labels_ids[all_labels == id_significant]

        if id_significant in all_within_labels:
            X = all_within_networks[id_data].squeeze()
        elif id_significant in all_between_labels:
            X = all_between_networks[id_data].squeeze()
        else:
            raise RuntimeError(f'{id_significant} not in {all_within_labels} or {all_between_labels}')
        weights_all, weights_whole = run_ml(X, y, expected_accuracy, n_repeats, n_splits, random_seed_experiment)

        weights_mean = weights_all.mean(axis=0)
        r = np.corrcoef(weights_mean, weights_whole)[0, 1]
        print(f'Correlation between average weight and weight across whole sample: {r:.4}')

        # We need to scale X to make it correspond to how it was used to calculate the weights
        ml_pipeline = get_ml_pipeline(random_seed_experiment + 1)
        scaler = ml_pipeline.named_steps['minmaxscaler']
        X_scl = scaler.fit_transform(X)

        calc_and_store_pvals(X_scl, y, weights_mean, weights_whole, mask_file, id_significant, weights_folder)


if __name__ == '__main__':
    run()
