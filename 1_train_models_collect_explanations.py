import argparse
import logging
import pickle as pkl

from torch import nn
from time import time
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix

import _variables
from datasets import *
from attribution import integrated_gradients, smooth_grad, vanilla_grad, kernelshap, lime
from util import create_checkpoint, _get_n_digits, _check_improved, _training_finished, _sample_new_model
from datasets import _get_dataset_callable, nlp_tasks


def save_eval(base_dir, expl_dir, outputs_dir,
              _dict_metrics, _dict_explanations, prefix):
    """
    saves explanations and metrics in expl_dir and outputs_dir
    :param base_dir:
    :param expl_dir:
    :param model_dir:
    :param _dict_metrics: dictionary of misc metrics to be saved
    :param _dict_explanations: dictionary of explanations to be saved
    :param prefix: applied to filenames of both explanations and metrics
    """
    # task_ModelSeed_DataSeed_epoch_batch
    if len(_dict_metrics) > 0:
        with open(f"{outputs_dir}{prefix}_out.pkl", 'wb') as f:
            pkl.dump(_dict_metrics, f)

    if len(_dict_explanations) > 0:
        for _exp_name, _expls in _dict_explanations.items():
            with open(f"{expl_dir}{prefix}_{_exp_name}.pkl", 'wb') as f:
                pkl.dump(_expls, f)


def eval_model(model, X, Y, inference_fn, expl_funs: list[tuple[str, callable]] = [], preprocess_fn=None):
    '''
    calculate
    - model-outputs
    - explanations from callables
    :param model: callable
    :param X: test-data
    :param Y: labels for X
    :param expl_funs: list of tuple(explanation name: string, explanation(data,target): callable), eg ('ig', integrated_gradients)
    :return:
    '''

    logging.debug("EVAL MODEL")
    start = time()

    model.eval()
    device = next(model.parameters()).device

    # if preprocess_fn is not None:
    #     _X = preprocess_fn(X.to(device))
    # else:
    #     _X = X
    X = X.to(device)
    _metrics_dict = eval_performance(model, X, Y)
    _expl_dict = calc_explanations(X, Y, expl_funs)

    model.train()  # tracking gradients not possible lstm otherwise; we have to zero grads then before exit?

    end = time()

    logging.debug(f"EVAL TOOK {end-start} seconds")

    return _metrics_dict, _expl_dict


def eval_performance(model, X, Y):
    """
    compute accuracy, output distribution and confusion matrix
    :param model: the model, used only to determine on what device to run evaluation on
    :param X: input data
    :param Y: target labels
    :return: dict with keyes 'accuracy', 'output_distribution', 'confusion_matrix' containing results
    """

    with torch.no_grad():
        _y_distr = model(X)
        _y = _y_distr.argmax(-1).to('cpu')
        accuracy = (_y == Y).to(torch.float32).mean().item()

    _confusion_matrix = confusion_matrix(Y, _y)
    _metrics_dict = {
        'accuracy': accuracy,
        'output_distribution': _y_distr,
        'confusion_matrix': _confusion_matrix,
    }

    return _metrics_dict


def calc_explanations(X, Y, expl_funs: list[tuple[str, callable]] = []):
    """
    call functions in expl_funs with args=(X,Y)
    :param X: input data
    :param Y: target labels
    :param expl_funs: list of tuple(explanation name: string, explanation(data,target): callable), eg ('ig', integrated_gradients)
    :return: dict[explanation name] containing explanations
    """

    _expl_dict = {}
    for _name_str, _expl_fun in expl_funs:
        _expl_dict[_name_str] = _expl_fun(X, Y)

    return _expl_dict


def train_step(model, optim, loss_fn, X, Y, device='cpu'):
    """
    Performs standard trainstep in pytorch
    :param model: model to be trained
    :param optim: optimizer holding models parameters
    :param loss_fn: loss function with input (model prediction, target labels)
    :param X: one batch of training data
    :param Y: target labels
    :param device: devidce where to run on
    :return: loss value as python type
    """
    X = X.to(device)
    Y = Y.to(device)
    out = model(X)
    loss = loss_fn(out, Y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    optim.zero_grad()

    return loss.item()


def set_logger_level(l):
    l = l.upper()
    if l in logging._nameToLevel.keys():
        logging.getLogger().setLevel(logging._nameToLevel[l])
    else:
        print(f'loglevel {l} not recognized')


def make_parser():
    """
    Defines options for the script and default values
    :return: parser object
    """
    def int_list(input: str) -> list[int]:
        # parse string of list "[1, 2, 3]" -> [1, 2, 3]; [1,] is an invalid input
        input = input.replace('[', '').replace(']', '').replace(' ', '')
        input = input.split(',')
        if len(input) == 0:
            return []
        else:
            return [int(i) for i in input]

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', default=-1, type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch-sizes', type=int_list, default=[64, 128])
    parser.add_argument('--modelparams', type=int_list, default=[], help="parameters beside input/output size")
    parser.add_argument('--directory', default='./data/')
    parser.add_argument('--data-seed', default=42, type=int)
    parser.add_argument('--num-runs', default=50, type=int)
    parser.add_argument('--max-epochs', default=20, type=int)
    parser.add_argument('--eval-freq', default=1, type=int, help="frequency in batches how often model is evaluated")
    parser.add_argument('--save-freq', default=1, type=int, help="frequency in batches how often model is saved")
    parser.add_argument('--model-seed', default=42, type=int)
    parser.add_argument('--loglevel', default='ERROR', type=str)
    parser.add_argument('--training-length', default=0, type=int, help="if > 0, is the exact number of batches one "
                                                                       "model will be trained for")
    parser.add_argument('--track-explanations', default=1, type=int, help="whether or not to compute exlpanations"
                                                                              "during training; regardless, explanations"
                                                                              "will always be computed before and after"
                                                                              "training")

    return parser


if __name__ == '__main__':

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except NameError or ModuleNotFoundError:
        pass

    # can cause RuntimeError to be thrown if one of the operations is used where no deterministic impl is available, see:
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    # torch.use_deterministic_algorithms(True)
    # the CNNs use some functions where no deterministic alternative is available

    # parse args
    args = make_parser().parse_args()
    set_logger_level(args.loglevel)
    # get args
    n_models = args.num_runs
    task = args.dataset.lower(); assert callable(_get_dataset_callable(task))
    device = 'cpu' if torch.cuda.device_count()-1 > args.gpu_id < 0  else f"cuda:{args.gpu_id}"
    logging.debug(f'device: {device}')
    modelparams = args.modelparams
    batch_sizes = args.batch_sizes
    training_length = args.training_length

    # make/ check folders
    base_dir = args.directory+f'/{task}/'; logging.debug(f'base dir: {base_dir}')
    model_dir = base_dir+'models/'; Path(model_dir).mkdir(parents=True, exist_ok=True)
    expl_dir = base_dir+'explanations/'; Path(expl_dir).mkdir(exist_ok=True)
    outputs_dir = base_dir+'outputs/'; Path(outputs_dir).mkdir(exist_ok=True)
    losses_dir = base_dir+'losses/'; Path(losses_dir).mkdir(exist_ok=True)
    acc_dir = base_dir+'accuracies/'; Path(acc_dir).mkdir(exist_ok=True)

    # generate seeds for models
    with TorchRandomSeed(args.model_seed):
        seeds = torch.randint(high=420000, size=(n_models,), dtype=torch.int)
        seeds = [s.item() for s in seeds]

    # get dataloaders and input/ output dims for task
    train_loader, test_loader, input_size, n_classes = \
        _get_dataset_callable(task)(random_state=args.data_seed, batch_sizes=batch_sizes)

    # Save test data and some basic statistics
    Y_train = []
    n_batches = 0

    for _, y in train_loader:
        Y_train.append(y)
        n_batches += 1
        if training_length > 0 and n_batches >= training_length:
            break

    # used in epoch-batch prefix in filenames to avoid ordering 1,10,11,..., 2, 20 ...
    n_digits_batch = _get_n_digits(n_batches)
    n_digits_epoch = _get_n_digits(args.max_epochs)

    Y_train = torch.cat(Y_train)
    Y_train_distr = torch.unique(Y_train, return_counts=True)[1].to(torch.float)
    Y_train_distr /= torch.sum(Y_train_distr).item()
    print(Y_train_distr)


    X_test, Y_test = next(iter(test_loader))
    Y_test_distr = torch.unique(Y_test, return_counts=True)[1].to(torch.float)
    Y_test_distr /= torch.sum(Y_test_distr); logging.debug(Y_test_distr)
    print(Y_test_distr)
    assert len(torch.unique(Y_test)) == n_classes

    pid = os.getpid()
    metadata_fname = f'/meta_data_pid{pid}_{task}_{args.data_seed}.pkl'
    with open(base_dir+metadata_fname, 'wb') as f:
        meta_data = {
            'dataset': task,
            'X': X_test,
            'Y': Y_test,
            'Y_train_distr': Y_train_distr,
            'Y_test_distr': Y_test_distr,
            }
        meta_data.update(vars(args))
        pkl.dump(meta_data, f)
        logging.debug(f'saved meta_data: {[k for k in meta_data.keys()]}')
    # prepare model setup
    modelparams = [input_size] + modelparams + [n_classes]
    logging.debug(f'model params: {modelparams}')

    # prepare arguments for explanations (SG)
    expl_kwargs = {}
    if task not in nlp_tasks:
        X_train = torch.cat([x for x, _ in train_loader])
        X_train_val_range = torch.abs(torch.max(X_train) - torch.min(X_train))
    else:
        X_train_val_range = 2.  # ~ diameter of space occupied by embedding vectors


    # n_batches = 1234, max_epochs=100, e=2, b=64 -> "0100-0064"
    def epoch_batch_prefix(e, b):
        epochs = ''.join((n_digits_epoch-_get_n_digits(e))*['0']+[str(e)]) if e != 0 else ''.join(n_digits_epoch*['0'])
        batches = ''.join((n_digits_batch-_get_n_digits(b))*['0']+[str(b)]) if b != 0 else ''.join(n_digits_batch*['0'])
        return f'{epochs}-{batches}'

    kernelshap_mask = torch.arange(0, X_test.shape[1]).unsqueeze(0)

    for s, seed in enumerate(seeds): # run everything; data remains the same
        print(f'STARTING SEED {s}/{n_models}')
        logging.debug(f'running with seed {seed}')
        # only in case of nlp will inference_fn and preprocessing be given, the latter to embed sequence for IG/SG
        model, inference_fn, _preprocess_for_explanations = _sample_new_model(task, modelparams, seed)
        model.to(device)
        model_id = f'{seed}_{args.data_seed}'
        model_id = f'{task}_{model_id}'
        if inference_fn is None:
            inference_fn = model

        _grad_expls_args = {
            'model': model,
            'inference_fn': inference_fn,
            'pre_process_fn': _preprocess_for_explanations
        }


        # setup check to stop training
        performance_improved = False
        training_finished = False
        test_acc, losses = [], []

        expl_funs = [
            ('ig', lambda x, y: integrated_gradients(**_grad_expls_args, data=x, targets=y,
                                                     n_samples=200, return_convergence_delta=True)),
            ('sg', lambda x, y: smooth_grad(**_grad_expls_args, std=X_train_val_range, data=x, targets=y, n_samples=100)),
            ('vg', lambda x, y: vanilla_grad(**_grad_expls_args, data=x, targets=y)),
            ('ks', lambda x, y: kernelshap(model=model, data=x, targets=y, masks=kernelshap_mask,
                                           n_samples=300)),  # diff kernelshap and lime: lime uses different weighting
            ('li', lambda x, y: lime(model=model, data=x, targets=y, n_samples=200))
        ]

        assert sorted([e[0] for e in expl_funs]) == sorted(_variables.explanation_abbreviations)

        expl_funs_during_training = []
        if args.track_explanations:
            expl_funs_during_training = expl_funs

        optim = torch.optim.Adam(model.parameters())
        loss_function = nn.CrossEntropyLoss()

        # eval + save before training
        _dict_metrics, _dict_expls = \
            eval_model(model=model, X=X_test, Y=Y_test, inference_fn=inference_fn,
                       expl_funs=expl_funs, preprocess_fn=_preprocess_for_explanations)
        test_acc.append(_dict_metrics['accuracy'])
        prefix = model_id + '_' + epoch_batch_prefix(0, 0)  # task_ModelSeed_DataSeed_epoch-batch

        logging.debug(f'saving with prefix {prefix}')
        save_eval(
            base_dir, expl_dir, outputs_dir,
            _dict_metrics, _dict_expls,
            prefix=prefix
        )
        fname = model_dir + model_id + '_' + epoch_batch_prefix(0, 0) + '.ckpt'
        create_checkpoint(fname,
                          model, optimizer=None)
        logging.debug(f'checkpoint created {fname}')
        # keep count on how many batches we trained; esp. relevant if args.training_length > 0
        n_batches_trained_on = 0

        with TorchRandomSeed(args.data_seed):
            for epoch in range(args.max_epochs):
                print(f'    STARTING EPOCH {epoch}/{args.max_epochs}')
                for i, (x, y) in tqdm(enumerate(train_loader)):

                    # do one training step, log loss
                    loss = train_step(model, optim, loss_function, x, y, device=device)
                    losses.append(loss)
                    n_batches_trained_on += 1

                    if i % args.eval_freq == 0:
                        # _X_test = X_test if not callable(_preprocess_for_explanations) else _preprocess_for_explanations(X_test)
                        _dict_metrics, _dict_expls = \
                            eval_model(model=model, X=X_test, Y=Y_test, inference_fn=inference_fn,
                                       expl_funs=expl_funs_during_training,
                                       preprocess_fn=_preprocess_for_explanations)
                        test_acc.append(_dict_metrics['accuracy'])
                        prefix = model_id + '_' + epoch_batch_prefix(epoch, i+1)  # task_ModelSeed_DataSeed_epoch-batch
                        logging.debug(f'saving with prefix {prefix}')
                        save_eval(
                            base_dir, expl_dir, outputs_dir,
                            _dict_metrics, _dict_expls,
                            prefix=prefix
                        )

                    if args.save_freq > 0 and i % args.save_freq == 0:
                        fname = model_dir+model_id+'_'+epoch_batch_prefix(epoch, i+1)+'.ckpt'
                        create_checkpoint(fname,
                                          model, optimizer=None)
                        logging.debug(f'checkpoint created {fname}')

                    if i % 100 == 0:
                        print(f'    FINISHED: {epoch}.{i}')

                    if 0 < training_length: # if parameter training_length is set
                        print(training_length, n_batches_trained_on)
                        if training_length <= n_batches_trained_on:  # and we have trained for enough batches, break
                            break
                    else:
                        if i+epoch*i < 10*args.eval_freq:  # don't do anything before we have not evaluated model at least 10x
                            continue
                        if not performance_improved: # if performance has not improved until previous batch, continue training anyway
                            performance_improved = _check_improved(test_acc, losses)
                            continue
                        # performance had improved since last batch, check if we're done
                        training_finished = _training_finished(test_acc, losses)
                        if training_finished:  # break out of batch loop
                            break

                # break out of epoch loop
                if training_finished or (0 < training_length and training_length <= n_batches_trained_on):
                    print(f'        FINISHED MODEL #{s}@{epoch}{i} after {n_batches_trained_on} batches total')
                    break
                print(f'        last test accuracies: \n\t\t\t{test_acc[-10:]}')

        if args.save_freq < 0: # meaning we have not saved a single model throught training; nlp models are too large..
            create_checkpoint(model_dir+model_id+'_'+epoch_batch_prefix(epoch, i)+'.ckpt',
                                  model, optimizer=None)

        _dict_metrics, _dict_expls = \
                        eval_model(model=model, X=X_test, Y=Y_test, inference_fn=inference_fn,
                                   expl_funs=expl_funs, preprocess_fn=_preprocess_for_explanations)
        prefix = model_id + '_' + epoch_batch_prefix(args.max_epochs, len(train_loader)+2)  # task_ModelSeed_DataSeed_epoch-batch
        logging.debug(f'saving with prefix {prefix}')
        save_eval(
            base_dir, expl_dir, outputs_dir,
            _dict_metrics, _dict_expls,
            prefix=prefix
        )
        # task_ModelSeed_DataSeed_loss/testacc
        fname_loss = losses_dir+model_id+'_loss.pkl'
        fname_acc = acc_dir+model_id+'_testacc.pkl'
        logging.debug(f'saving loss, acc in \n\t{fname_loss}\n\t{fname_acc}')
        with open(fname_loss, 'wb') as f:
            pkl.dump(losses, f)
        with open(fname_acc, 'wb') as f:
            pkl.dump(test_acc, f)

        del losses, test_acc, model, optim
