import os.path as osp
from ml_code import run as run_ml

DIM = 70
project_folder = '/data/shared/ptsdchild'
analysis_folder = osp.join(project_folder, 'analysis', 'resting_state')
melodic_folder = osp.join(analysis_folder, f'group_ica_HC_dim{DIM}.ica', 'meta_melodic')
gigica_folder = osp.join(analysis_folder, f'patients_ptsdchild_maps_dim{DIM}.gigica')
group_folder = osp.join(gigica_folder, 'group_comparison', 'between_networks')
corr_mat_file_path = osp.join(group_folder, 'unique_correlation_matrix.csv')
pcorr_mat_file_path = osp.join(group_folder, 'unique_partial_correlation_matrix.csv')
result_folder = osp.join(gigica_folder, 'ml_analysis')

final_file_path = osp.join(project_folder, 'code', 'resting_state', 'fmriprep_preprocessing',
                           'final_patient_data_dr.txt')
info_file_path = osp.join(project_folder, 'bids_data', 'participants.tsv')
response_status = 'responder_30perc'
spatial_comp_template = osp.join(gigica_folder, 'gigica_spatialmap_{}.nii')
mask_file = osp.join(gigica_folder, 'mask.nii')
signal_file = osp.join(melodic_folder, 'signal_networks_fsleyes_v1_1_additional_QC.csv')
# 50 x 5-CV
n_splits = 5
n_repeats = 50
n_perm = 2000
n_jobs = 22
# None indicates that it will be seeded with time
# Final seed which was generated and used: 1588353789
random_seed = None


def run(final_file_path, info_file_path, spatial_comp_template, mask_file, signal_file, corr_mat_file_path,
        pcorr_mat_file_path, result_folder, n_splits, n_repeats, response_status, n_perm, random_seed, n_jobs):
    run_ml(final_file_path, info_file_path, spatial_comp_template, mask_file, signal_file, corr_mat_file_path,
        pcorr_mat_file_path, result_folder, n_splits=n_splits, n_repeats=n_repeats, response_status=response_status,
        n_perm=n_perm, random_seed=random_seed, n_jobs=n_jobs)

if __name__ == '__main__':
    run(final_file_path, info_file_path, spatial_comp_template, mask_file, signal_file, corr_mat_file_path,
        pcorr_mat_file_path, result_folder, n_splits, n_repeats, response_status, n_perm, random_seed, n_jobs)
