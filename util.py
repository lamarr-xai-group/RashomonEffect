import torch
import logging
import numpy as np
import pickle as pkl

from pathlib import Path

from models import BiLSTMClassif, make_fmnist_small, make_ff

from datasets import TorchRandomSeed, nlp_tasks, cv_tasks, _get_dim_classes

from torch.utils.data import DataLoader, TensorDataset
# ---

# ---

def _set_random_state(random_state=None):
    if random_state is None:
        npr = np.random.RandomState(1234)
    elif isinstance(random_state, int):
        npr = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        npr = random_state
    else:
        raise Exception('Unknown input type of random state:', type(random_state))
    np.random.set_state(npr)


def _sample_new_model(task, modelparams, seed) -> (callable, callable, callable):
    with TorchRandomSeed(seed):
        if task in nlp_tasks:
            vocab_size, embed_dim, hid_size, nr_classes = modelparams
            model = BiLSTMClassif(nr_classes=nr_classes, embed_dim=embed_dim,
                                  hid_size=hid_size, vocab_size=vocab_size)
            return model, model._forward_embedded_softmax, model.embed_sequences

        elif task in cv_tasks:
            _, n_classes = modelparams
            model = make_fmnist_small(n_classes)
            return model, model.predict_batch_softmax, None

        else:
            model = make_ff(modelparams)
            return model, model.predict_batch_softmax, None
            # return model, None, None


# ---


def _get_n_digits(num: int):
    """
    Count how many decimals a number as
    :param num: natural number
    :return: number of decimals
    """
    n = 0
    while num != 0:
        num = num // 10
        n += 1
    return n


def _check_improved(test_accs, losses, acc_window=5, loss_window=10):
    """Determines whether the model has improved wrt accuracy on test set and significant decrease in loss"""

    # if most recent accuracy is not 1.75 as good as baseline -> False
    baseline_acc = sum(test_accs[:acc_window])/acc_window
    baseline_acc = min(1.75*baseline_acc, 1.)  #
    recent = sum(test_accs[-acc_window:])/acc_window
    logging.debug(f'_check_improved: baseline acc {baseline_acc} vs recent acc {recent}')
    if baseline_acc > recent:
        return False
    # if most recent loss still more than half of initial loss -> False
    baseline_loss = 0.5 * sum(losses[:loss_window])/loss_window
    recent_loss = sum(losses[-loss_window:])/loss_window
    logging.debug(f'_check_improved: baseline loss {baseline_loss} vs recent loss {recent_loss}')
    if baseline_loss <= recent_loss:
        return False
    # so if accuracy on test-set has improved and loss has decreased we say the model has improved!
    return True


def _training_finished(test_accs, losses, thresh_acc=0.01, thresh_loss=0.01, last_n=3, win_size=9):
    """
    _check_improved says the model has improved,
    _training_finished checks whether we still want to go on,
        eg loss keeps decreasing, test_acc increases again ...

    we do not consider the case where last_n or win_size is larger than length of test_accs or losses
    but since _training_finished only gets called afer _check_improved is True, this is probably fine

    """

    loss_recent = sum(losses[-last_n:])/last_n
    loss_win = sum(losses[-win_size:])/win_size
    loss_thresh = loss_win * thresh_loss
    loss_keeps_changing = abs(loss_win - loss_recent) > loss_thresh

    acc_recent = sum(test_accs[-last_n:])/last_n
    acc_win = sum(test_accs[-win_size:])/win_size
    acc_thresh = acc_win*thresh_acc
    test_acc_changing = abs(acc_win - acc_recent) > acc_thresh
    logging.debug(f'acc mark:{acc_win}, recent:{acc_recent}, thresh:{acc_thresh}')
    logging.debug(f'loss mark:{loss_win}, recent:{loss_recent}, thresh:{loss_thresh}')
    return not (loss_keeps_changing or test_acc_changing)


# ---
def create_checkpoint(path, model, optimizer=None):
    if optimizer is None:
        torch.save({
            'model_state_dict': model.state_dict(),
        }, path)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)


def restore_checkpoint(path, model, optimizer=None, train=True):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(path, map_location=torch.device('cuda'))
    for k, v in ckpt['model_state_dict'].items():
        if not v.is_contiguous():
            ckpt['model_state_dict'][k] = v.contiguous()
    model.load_state_dict(ckpt['model_state_dict'])
    if train:
        model.train()
    try:
        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except KeyError as ke:
        print(str(ke))
        print("RESTORE_CHECKPOINT: Optimizer given but no Optimizer found in state dict.")

# ---


def _get_filenames(pth: Path):
    return [
        str(f) for f in pth.iterdir() if f.is_file()
    ]


def _get_batchnum(fname):
    batch = fname.split('_')[3].split('-')[1]
    batch = batch.split('.')[0].split('_')[0]  # first split suffices for models
    return batch


def _get_epoch(fname):
    return fname.split('_')[3].split('-')[0]


def _filter_by_data_seed(fnames, seed):
    return sorted([fname for fname in fnames if seed == fname.split('_')[2]])


def _filter_by_model_seed(fnames, seed):
    return sorted([fname for fname in fnames if seed == fname.split('_')[1]])


def _filter_by_explanation_method(fnames, xai_method):
    return sorted([fname for fname in fnames if xai_method == fname.split('_')[-1].split('.')[0]])


def get_all_model_seeds_sorted(data_dir):
    fnames = _get_filenames(Path(data_dir, 'models'))
    seeds = []
    # task_modelseed_dataseed_epoch-batch
    for fname in fnames:
        seed = fname.split('_')[1]
        seeds.append(seed)
    seeds = np.unique(seeds)
    seeds = sorted(seeds)
    return seeds


def _sort_fnames_by_epoch_batch(fnames):
    return sorted(fnames, key=lambda x: (int(_get_epoch(x)), int(_get_batchnum(x))))


# def _loss_idx_to_eval_idx(idxs, eval_freq=1):
#     # needed if eval was not done after each batch
#     if eval_freq == 1:
#         return idxs
#     else:
#         return [np.floor(idx/eval_freq, dtype=np.int) for idx in idxs]
#
#     pass


def load_losses(data_dir):  # have option to only load a certain modelID?
    fnames = _get_filenames(Path(data_dir, 'losses'))
    losses = {}
    # model seed becomes key
    for fname in fnames:
        modelSeed = fname.split('_')[1]
        with open(fname, 'rb') as file:
            _losses_model = pkl.load(file)
            losses[modelSeed] = _losses_model
    return losses


def load_accuracies(data_dir):  # have option to only load a certain modelID?
    fnames = _get_filenames(Path(data_dir, 'accuracies'))
    accuracies = {}
    # model seed becomes key
    for fname in fnames:
        modelSeed = fname.split('_')[1]
        with open(fname, 'rb') as file:
            accs = pkl.load(file)
            accuracies[modelSeed] = accs
    return accuracies


def load_outputs_sorted(data_dir, model_seed):
    outputs = []
    fnames = _get_filenames(Path(data_dir, 'outputs'))
    fnames = _filter_by_model_seed(fnames, model_seed)

    for fname in fnames:
        with open(fname, 'rb') as f:
            out = pkl.load(f)
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg, ...}.pkl
            out['batch'] = _get_batchnum(fname)
            out['epoch'] = _get_epoch(fname)
            outputs.append(out)
    outputs = sorted(outputs, key= lambda o: (int(o['epoch']), int(o['batch'])))
    return outputs


def load_idxs_from_model(data_dir, task, model_seed, idxs: list[int], return_fns=False):
    # get all modelparams =
    meta_data = load_meta_data(data_dir, just_one=True)
    modelparams = meta_data['modelparams']
    del meta_data
    dim, n_classes = _get_dim_classes(task)
    modelparams = [dim] + modelparams + [n_classes]
    fnames_models = [str(f) for f in Path(data_dir, 'models').iterdir()]
    fnames_models = _filter_by_model_seed(fnames_models, model_seed)
    fnames_models = _sort_fnames_by_epoch_batch(fnames_models)

    models = []
    for idx in idxs:
        fname = fnames_models[idx]
        model, inference_fn, preprocess_fn = _sample_new_model(task, modelparams, seed=0)
        restore_checkpoint(fname, model)
        models.append(
            (model, inference_fn, preprocess_fn) if return_fns else model
        )

    return models

def load_idxs_from_multiple_models(data_dir, task, model_seeds: list, idxs: list[int], return_fns=False):
    '''
    :return: list(model(idxs)) if len(idxs)>1 else list(models)
    '''
    meta_data = load_meta_data(data_dir, just_one=True)
    modelparams = meta_data['modelparams']
    del meta_data
    dim, n_classes = _get_dim_classes(task)
    modelparams = [dim] + modelparams + [n_classes]

    fnames_models = [str(f) for f in Path(data_dir, 'models').iterdir()]
    models = []
    for seed in model_seeds:
        fnames_models_filtered = _filter_by_model_seed(fnames_models, seed)
        fnames_models_filtered = _sort_fnames_by_epoch_batch(fnames_models_filtered)
        if len(idxs) > 1:
            models.append([])
        for idx in idxs:
            fname = fnames_models_filtered[idx]
            # this won't work for nlp models
            model, inference_fn, preprocess_fn = _sample_new_model(task, modelparams, seed=0)
            restore_checkpoint(fname, model)
            if len(idxs) > 1:
                models[-1].append(
            (model, inference_fn, preprocess_fn) if return_fns else model
        )
            else:
                models.append(
            (model, inference_fn, preprocess_fn) if return_fns else model
        )

    return models

def load_outputs_by_idxs(data_dir, model_seed, idxs: list[int]):
    # load the idx's
    fnames = _get_filenames(Path(data_dir, 'outputs'))
    fnames = _filter_by_model_seed(fnames, model_seed)
    fnames = _sort_fnames_by_epoch_batch(fnames)
    outputs = []
    for idx in idxs:
        fname = fnames[idx]
        with open(fname, 'rb') as f:
            out = pkl.load(f)
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
            out['batch'] = _get_batchnum(fname)
            out['epoch'] = _get_epoch(fname)
            outputs.append(out)
    return outputs

def load_idxs_from_multiple_outputs(data_dir, model_seeds, idxs: list[int]):
    '''
    :return: list(model(idxs)) if len(idxs)>1 else list(models)
    '''
    fnames = _get_filenames(Path(data_dir, 'outputs'))
    outputs = []

    for seed in model_seeds:
        fnames_outputs_filtered = _filter_by_model_seed(fnames, seed)
        fnames_outputs_filtered = _sort_fnames_by_epoch_batch(fnames_outputs_filtered)
        if len(idxs) > 1:
            outputs.append([])
        for idx in idxs:
            fname = fnames_outputs_filtered[idx]
            with open(fname, 'rb') as f:
                out = pkl.load(f)
                # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
                out['batch'] = _get_batchnum(fname)
                out['epoch'] = _get_epoch(fname)
                if len(idxs) > 1:
                    outputs[-1].append(out)
                else:
                    outputs.append(out)
    return outputs


def load_explanations_by_idxs(data_dir, model_seed, idxs, explanation_method='sg'):
    fnames = _get_filenames(Path(data_dir, 'explanations'))
    fnames = _filter_by_model_seed(fnames, model_seed)
    fnames = _filter_by_explanation_method(fnames, explanation_method)
    fnames = _sort_fnames_by_epoch_batch(fnames)
    explanations = []
    epoch_batch = []
    for idx in idxs:
        fname = fnames[idx]
        with open(fname, 'rb') as f:
            explanations.append(pkl.load(f))
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
            epoch_batch.append((_get_epoch(fname), _get_batchnum(fname)))

    return explanations, epoch_batch

def load_idxs_from_multiple_explanations(data_dir, model_seeds, idxs: list[int], explanation_method='ig', train=False):

    if train:
        fnames = _get_filenames(Path(data_dir, 'explanationsTrain'))
    else:
        fnames = _get_filenames(Path(data_dir, 'explanations'))
    explanations = []
    epoch_batch = []

    for seed in model_seeds:
        fnames_seed = _filter_by_model_seed(fnames, seed)
        fnames_expl = _filter_by_explanation_method(fnames_seed, explanation_method)
        fnames_expl = _sort_fnames_by_epoch_batch(fnames_expl)

        if len(idxs) > 1:
            explanations.append([])
            epoch_batch.append([])
        for idx in idxs:
            fname = fnames_expl[idx]
            with open(fname, 'rb') as f:
                if len(idxs) > 1:
                    explanations[-1].append(pkl.load(f))
                    epoch_batch[-1].append((_get_epoch(fname), _get_batchnum(fname)))
                else:
                    explanations.append(pkl.load(f))
                    epoch_batch.append(
                        (_get_epoch(fname), _get_batchnum(fname))
                    )

    return explanations, epoch_batch


def load_explanations(data_dir, model_seed, explanation_method='sg'):
    explanations = []
    epoch_batch = []
    fnames = _get_filenames(Path(data_dir, 'explanations'))
    fnames = _filter_by_model_seed(fnames, model_seed)
    fnames = _filter_by_explanation_method(fnames, explanation_method)
    fnames = _sort_fnames_by_epoch_batch(fnames)

    for fname in fnames:
        with open(fname, 'rb') as f:
            explanations.append(pkl.load(f))
            # TASK_MODELID_epoch-batchnum_{ig, sg, vg}.pkl
            epoch_batch.append((_get_epoch(fname), _get_batchnum(fname)))

    return explanations, epoch_batch


def load_meta_data(data_dir, just_one=False):
    fnames = _get_filenames(Path(data_dir))
    meta_data = {}
    for fname in fnames:
        if 'meta_data' in fname:
            #print(f'loading {fname}')
            dataseed = fname.split('_')[-1].split('.')[0]
            with open(fname, 'rb') as md:
                data = pkl.load(md)
                if just_one:
                    return data
                else:
                    meta_data[dataseed] = data

    return meta_data


def _load_meta_data_by_pid(data_dir):
    fnames = _get_filenames(Path(data_dir))
    meta_data = {}
    for fname in fnames:
        if 'meta_data' in fname:
            #print(f'loading {fname}')
            pid = fname.split('_')[2]
            with open(fname, 'rb') as md:
                data = pkl.load(md)
                meta_data[pid] = data
    return meta_data


def dump_pkl(array, fname):
    if type(fname) == Path:
        fname = str(fname)
    with open(fname, 'wb') as f:
        pkl.dump(array, f)


def load_pkl(fname):
    if type(fname) == Path:
        fname = str(fname)
    with open(fname, 'rb') as f:
        data = pkl.load(f)
    return data

# def _load_pkls(fnames):
#     pkls = []
#     for fname in fnames:
#         with open(fname, 'rb') as f:
#             pkls.append(pkl.load(f))
#     return pkls


def _get_outputs(inference_fn, data, model, device, batch_size=256):

    _data = DataLoader(TensorDataset(data), shuffle=False, batch_size=batch_size)

    try:
        _y_out = []
        for x in _data:
            _y = inference_fn(x[0].to(device))
            _y_out.append(_y.cpu())
        return torch.vstack(_y_out)
    except RuntimeError as re:
        if "CUDA out of memory" in str(re):
            model.to('cpu')
            outputs = _get_outputs(inference_fn, data, model, 'cpu')
            model.to('cuda')
            return outputs
        else:
            raise re


def _get_targets(inference_fn, data, model, device):
    return torch.argmax(_get_outputs(inference_fn, data, model, device), dim=1)
