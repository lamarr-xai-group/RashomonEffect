import torch
from pathlib import Path

import _variables
import datasets
from util import load_idxs_from_multiple_models, load_meta_data, get_all_model_seeds_sorted, load_accuracies, dump_pkl
from attribution import integrated_gradients, smooth_grad, kernelshap, lime

from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

import os
os.environ['PYTHONWARNINGS']='ignore::UserWarning'
import warnings


def save_selected_seeds(seeds, dir, fname):
    pth = Path(dir, fname)
    seeds = '\n'.join([str(seed) for seed in seeds])
    with open(pth, 'w') as f:
        f.writelines(seeds)


if __name__ == '__main__':

    with warnings.catch_warnings():
        _base_dir = _variables.base_dir
        tasks = _variables.tasks
        model_idxs = [-1]
        _expl_n_samples = _variables._eval_hyper_param_expl_n_samples
        n_samplings = 10  # how often to compute an explanation for each sample at each hyperparameter-value

        for task in tqdm(tasks, desc=f"tasks", position=0, leave=True):
            data_dir = _variables.get_data_dir(task)
            model_dir = Path(data_dir, 'models')
            results_dir = _variables.get_result_dir(task)
            results_dir.mkdir(parents=True, exist_ok=True)
            model_seeds = get_all_model_seeds_sorted(data_dir)
            print(f"found {len(model_seeds)} models")
            _accuracies = load_accuracies(data_dir)
            # ---------------------------------------------------------- Removing models with less accuracy
            _acc_final = {k: v[-1] for k, v in _accuracies.items()}
            cutoff = max(_acc_final.values()) - 0.05
            selected_seeds = [s for s in model_seeds if _acc_final[s] >= cutoff]
            save_selected_seeds(selected_seeds, results_dir, f"{task}_selected_models.csv")
            print(f"cutoff at {cutoff:.5f}")
            print(f"{len(selected_seeds)} remaining")

            # ----------------------------------------------------------

            metadata = load_meta_data(data_dir, task)
            _models = load_idxs_from_multiple_models(data_dir, task, selected_seeds, idxs=model_idxs, return_fns=True)

            data, targets = metadata['X'], metadata['Y']

            if task not in datasets.nlp_tasks:
                data_val_range = torch.abs(torch.max(data) - torch.min(data))
            else:
                data_val_range = 1.  # ~variance of BiLSTM.embedding

            kernelshap_mask = torch.arange(0, data.shape[1])

            _expls = [
                ('sg', lambda x, y, args: smooth_grad(**args, std=data_val_range, data=x, targets=y),
                 ),
                ('ig', lambda x, y, args: integrated_gradients(**args, data=x, targets=y,
                                                               return_convergence_delta=False),
                 ),
                ('ks', lambda x, y, args: kernelshap(**args, data=x, targets=y, masks=kernelshap_mask),
                 ),
                ('li', lambda x, y, args: lime(**args, data=x, targets=y),
                 ),
            ]
            assert len(_expl_n_samples) == len(_expls)
            assert all([_e[0] in _variables.explanation_abbreviations for _e in _expls])

            results_by_model = []
            for (model, inference_fn, preprocess_fn) in tqdm(_models, desc="models", position=1, leave=True):
                results_dict = {}
                for (expl_str, expl_fn) in tqdm(_expls, desc="epxls", position=2, leave=True):
                    results_dict[expl_str] = []

                    if expl_str == 'ig':  # integral is deterministic
                        n_samplings = 1
                    else:
                        n_samplings = 10

                    if task in datasets.nlp_tasks:
                        if expl_str == 'sg':
                            n_jobs = 5  # cpu fully utilized @ n_samples=1000, ram @ 45gb
                        else:  # lstm models parallelize automatically, utilizing cpu fully
                            n_jobs = 3
                    else:
                        n_jobs = 10

                    expl_args = {
                        'model': model
                    }
                    if expl_str == 'sg':
                        expl_args['random_state'] = int(torch.rand(1).item()*100000000)

                    if task in datasets.nlp_tasks and expl_str in ['ig', 'sg']:
                        expl_args['inference_fn'] = inference_fn
                        expl_args['pre_process_fn'] = preprocess_fn

                    with parallel_backend(backend='loky', n_jobs=n_jobs):
                        for n_samples in tqdm(_expl_n_samples[expl_str], desc="sampling", position=3, leave=True):
                            expl_args['n_samples'] = n_samples
                            _results_sampling = Parallel(verbose=False)(delayed(expl_fn)
                                                  (data, targets, expl_args)
                                                                        for _n in range(n_samplings)
                                                  )
                            results_dict[expl_str].append(_results_sampling)
                results_by_model.append(results_dict)
            dump_pkl(results_by_model, Path(results_dir, f"{task}_stability_new.pkl"))
            del results_by_model
                # results_all_tasks[task].append(results_by_model)