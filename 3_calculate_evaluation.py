import os

from util import *
from util import _load_meta_data_by_pid, _get_outputs
from datasets import nlp_tasks

from eval import _functional_distance, _explanation_distance, \
     _top_k_eucl

from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    np.random.seed(42)

    n_jobs = os.cpu_count()-1

    import _variables
    tasks = _variables.tasks
    _all_accuracies = []
    _all_cm_frob_normalized = []
    _all_jsd = []


    for task in tqdm(tasks, position=0, desc="tasks"):


        print(f'starting eval on {task} dataset')
        data_dir = _variables.get_data_dir(task)
        print(f'loading from {data_dir}')
        result_dir = _variables.get_result_dir(task)
        print(f'result dir: {result_dir}')

        losses, _accuracies = load_losses(data_dir), load_accuracies(data_dir)

        selected_seed_fname = Path(result_dir, f"{task}_selected_models.csv")
        with open(selected_seed_fname, 'r') as f:
            model_seeds = [str(int(seed)) for seed in f.readlines()]
        print(f"found selected seeds\ncontinuing with {len(model_seeds)} models\n\n")

        _accuracies = [_accuracies[seed] for seed in model_seeds]
        dump_pkl(_accuracies, Path(result_dir, 'accuracies.pkl'))
        _all_accuracies.append(_accuracies)

        # meta_data = load_meta_data(data_dir, just_one=False)
        meta_data = _load_meta_data_by_pid(data_dir)
        _data = [d['X'] for d in meta_data.values()]
        _targets = [d['Y'] for d in meta_data.values()]
        eq = []
        for i in range(len(_data)):
            for j in range(i, len(_data)):
                eq.append(_data[i] == _data[j])
        assert torch.all(torch.stack(eq))  # assert that all metadata in dir uses same testset 'X' <-> we don't have issue with randomzation

        expl_abbrv = _variables.explanation_abbreviations
        expls = {}

        print("loading models")
        _models = load_idxs_from_multiple_models(data_dir, task, model_seeds, idxs=[-1], return_fns=True)


        for abb in expl_abbrv:
            expls[abb] = load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs=[-1], explanation_method=abb)[0]
            if abb == 'ig':
                expls[abb] = [e for e, _ in expls[abb]]  # because ig is (attrs, delta)
            expls[abb] = [e.cpu().detach().numpy() for e in expls[abb]] # ([explanations], epoch_batch) -> keep only explanations because we only look at last batch

            if task in nlp_tasks and len(expls[abb][0].shape) > 2:
                expls[abb] = [np.sum(e, -1) for e in expls[abb]]

        # --------------------------------------------------------------------------------------------------------------

        from disagreement_metrics import _top_k_feature_agreement, _top_k_sign_agreement
        clip_zero = lambda a: np.clip(a, a_min=0., a_max=None)

        _k = _variables.dataset_agreement_k[task]

        metrics = [
            ('feature disagreement',
             lambda e1, e2: 1. - _explanation_distance(e1, e2, metric=_top_k_feature_agreement, k=_k)
             ),
            ('sign disagreement',
             lambda e1, e2: 1. - _explanation_distance(e1, e2, metric=_top_k_sign_agreement, k=_k)
             ),
            ('euclid',
             lambda e1, e2: _explanation_distance(e1, e2, metric=_top_k_eucl, k=1.)
             ),
            ('euclid abs',
             lambda e1, e2: _explanation_distance(np.abs(e1), np.abs(e2), metric=_top_k_eucl, k=1.)
             ),
        ]

        for (metric_name, _), _m_name in zip(metrics, _variables.metric_names):
            assert metric_name == _m_name


        _model_outputs = [_get_outputs(model=model, data=_data[0], device='cpu',
                                          inference_fn=inference_fn if task not in nlp_tasks else lambda x: inference_fn(preprocess_fn(x))
                                       ).detach()
                    for model, inference_fn, preprocess_fn in _models]

        _model_predictions = [torch.argmax(o, dim=1) for o in _model_outputs]

        _model_prediction_masks = torch.zeros(len(_models), len(_models), _data[0].shape[0]).to(bool)
        for i in range(len(_models)):
            for j in range(i+1, len(_models)):
                _model_prediction_masks[i, j] = (_targets[0] == _model_predictions[i]) & \
                                                (_targets[0] == _model_predictions[j])

        _expls_stacked = np.vstack([expls[abb] for abb in expl_abbrv])  # shape=(n_methods*n_models, n_samples, dim_data)
        _predictions_stacked = torch.vstack([torch.tensor(np.array([x.numpy() for x in _model_predictions])) for _ in range(len(expl_abbrv))])
        dists = []

        get_mask = lambda p1, p2, t: (p1==t) & (p2==t)

        _jsd = [_functional_distance(_model_outputs[i].numpy(),_model_outputs[j].numpy())
                                    for i in range(len(_model_predictions))
                                        for j in range(i + 1, len(_model_predictions))
                                   ]
        dump_pkl(_jsd, Path(result_dir, 'jsd.pkl'))

        Y = _targets[0].detach().numpy()
        _cms = [confusion_matrix(Y, y.numpy()) for y in _model_predictions]
        _normalization = np.sqrt(2 * Y.shape[0]**2)
        cm_frob_normalized = lambda x: np.linalg.norm(x) / _normalization
        _cms_frobenius = [cm_frob_normalized(_cms[i] - _cms[j])
                                            for i in range(len(_cms))
                                                for j in range(i + 1, len(_cms))
                                             ]
        dump_pkl(_cms_frobenius, Path(result_dir, 'confmatdists.pkl'))

        with parallel_backend(backend='loky', n_jobs=n_jobs):
            # pre_compute masks because they will not differ for epxlanations and would be computed mulitple
            # times in Parallel() clause below
            for (_, metric) in tqdm(metrics, position=1, desc="distances"):
                # compute upper triangle of full distance matrix
                _dists = Parallel(verbose=1)(delayed(metric)
                           (_expls_stacked[i, get_mask(_predictions_stacked[i], _predictions_stacked[j], _targets[0])],
                            _expls_stacked[j, get_mask(_predictions_stacked[i], _predictions_stacked[j], _targets[0])])
                           for i in range(_expls_stacked.shape[0])
                           for j in range(i + 1, _expls_stacked.shape[0])
                           )
                dists.append(_dists)

        dump_pkl(dists, Path(result_dir, f'expl_dists.pkl'))





