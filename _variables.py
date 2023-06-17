from pathlib import Path

tasks = ['agnews',
         'beans',
        'breastcancer', 'ionosphere']
_tasks_paper_names = {
    'agnews': 'AG News',
    'beans': 'Dry Bean',
    'breastcancer': 'Breastcancer',
    'ionosphere': 'Ionosphere'
}

explanation_abbreviations = ['vg', 'sg', 'ig', 'ks', 'li']
_explanation_abbreviations_paper = {e: e.upper() for e in explanation_abbreviations}

base_dir = './'

data_path = 'data/TrainOnly'  # see --directory given in run_*.sh

results_prefix = 'rashomon'

metric_names = [
    'feature disagreement',
    'sign disagreement',
    'euclid',
    'euclid abs',
]

_map_metric_to_color = {
    'feature disagreement': 'COLORBLUE',
    'sign disagreement': 'COLORGREEN',
    'euclid': 'COLORORANGE',
    'euclid abs': 'COLORRED'
}

_metric_paper_names = {
    'feature disagreement': "Feature Disagreement",
    'sign disagreement': "Sign Disagreement",
    'euclid': "Euclidean",
    'euclid abs': "Euclidean-abs"
}

_eval_hyper_param_expl_n_samples = {
    'sg': [25, 50, 75, 100, 125],
    'ig': [25, 50, 75, 100, 125],
    'ks': [25, 50, 75, 100, 125],
    'li': [25, 50, 75, 100, 125],
}

dataset_agreement_k = {
    'agnews': 11,
    'beans': 4,
    'breastcancer': 8,
    'ionosphere': 8
}


def __create_dir(dir):
    dir.mkdir(exist_ok=True, parents=True)


def get_data_dir(task):
    dir = Path(base_dir, data_path, task)
    __create_dir(dir)
    return dir


def get_result_dir(task):
    dir = Path(base_dir, 'results', results_prefix, task)
    __create_dir(dir)
    return dir


def get_plot_dir(task):
    dir = Path(base_dir, 'plots', results_prefix, task)
    __create_dir(dir)
    return dir