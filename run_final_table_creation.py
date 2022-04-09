import os.path as osp
from create_final_table import run


DIM = 70
project_folder = '/data/shared/ptsdchild'
analysis_folder = osp.join(project_folder, 'analysis', 'resting_state')
gigica_folder = osp.join(analysis_folder, f'patients_ptsdchild_maps_dim{DIM}.gigica')
result_folder = osp.join(gigica_folder, 'ml_analysis')
permutation_file = None
if osp.exists(osp.join(result_folder, 'all_networks_permutation_average_performance.csv')):
    permutation_file = osp.join(result_folder, 'all_networks_permutation_average_performance.csv')

run(result_folder, permutation_file)
