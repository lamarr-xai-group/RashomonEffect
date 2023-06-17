import os

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from time import time
from pathlib import Path

from scipy.special import softmax
from scipy.sparse import triu
from scipy.spatial.distance import jensenshannon as JSD, cdist, hamming, euclidean, cosine

from util import *

"""
For multiple datasets, one model has been trained with a different random initialization multiple times.
We have collected:
    - Sequences of Models as they evolve throughout training 
    - Loss on each training step
    - Model outputs and explanations for a fixed set of datapoints,

Evaluation for:
    0. Plot losses and test accuracies over training
    1. Functional similarity -> primarily model output distributions, confusion matrices
        1.1 How does the Model-function change during training?
        1.2 How similar are models at the end of their training? (however we define 'end')
    2. Explanation similarity -> explanations for the fixed subset
        2.1 How much do explanations change over the course of the training?
        2.2 How different are explanations between the final models?
        2.3 Do explanations change when model performance is not changing?
    3. Feature based evaluation:
        3.1 How large is the difference of attribution scores for one feature during training? Variance? Moving Average?
        3.2 Is there a subset of features that becomes important early on & remains important?
        3.3 How large is the overlap of 'important subsets' between different models?

30.05: 
    Problem: too many models, too many training steps; loss/test-acc curves are not the same for all models, 
             hence need to pick relative points in time to compare
    Compare models wrt. functional sim and explanation sim at certain points in training:
        - Beginning
        - When they started learning but haven't plateaued yet
        - when they plateau
        - at the end

"""


def _match_models_accuracy(accs: dict[list[float]], thresholds=[0.25, 0.5, 0.75, 1.]) -> dict[list[float]]:
    # given: list of testing accuracy scores during training for multiple models; lists of floats may vary in length
    # return: for each model an index where its performance was (closest to one) in _thresholds
    # threshold/ criteria: mean of best performance of each model
    _maxs = [max(acc) for acc in accs.values()]
    _mean = sum(_maxs)/len(_maxs)
    print(f'mean peak performance: {_mean}')
    _thresholds = [_mean*t for t in thresholds]
    matching_points = {}

    for k, acc in accs.items():
        mp = []
        for t in _thresholds:
            # look what index in time series is closest to threshold; in theory this could violate chronological order
            # __we exclude first and last element of acc__
            idx = np.argmin(np.abs(np.array(acc[1:-1]) - t))+1
            mp.append(idx)
        matching_points[k] = mp

    return matching_points


def important_subsets(epxlanation_sequence):  # (T, N, *data_shape)
    pass

# --- (DIS-) SIMILARITIES


def _functional_distance(out1, out2, reduction=np.mean, shape=None, eps=1e-10):
    if shape is not None:
        out1 = out1.reshape(shape)
        out2 = out2.reshape(shape)
    # JSD uses scipy.special.rel_entr as the kullback leibler divergence
    # if y in rel_entr(x, y) is 0, inf is returned; since JSD computes dkl both ways,
    # we need to check in both arrays
    if np.any(out1 == 0.):
        out1 = out1 + eps
    if np.any(out2 == 0.):
        out2 = out2 + eps
    # Jensen-Shannon-Metric
    d = JSD(out1.T, out2.T)  # , base=2.) DEFAULT
    # filter NaNs that appear for some reason
    nan_idxs = np.where([np.isnan(x) for x in d])[0]
    if len(nan_idxs) > 0:
        for idx in nan_idxs:
            print(out1[idx], out2[idx])
            assert np.all(np.isclose((out1[idx] - out2[idx]) ** 2, 0, atol=1e-5))  # ugh
            d[idx] = 0.
    d = d**2
    return reduction(d)


def _symmetric_weighted_hamming(x1, x2):
    # Matthias: Provlem ist, wenn die Sortierung um 1 Feature verschoben ist, ist die aehnlichkeit == 0
    # xi.shape = wi.shape = (N, -1)
    w1 = np.abs(np.sort(x1))  # elements in x1 can be negative, what to do about them? abs? don't use weights at all??
    w2 = np.abs(np.sort(x2))
    s1 = np.argsort(x1)
    s2 = np.argsort(x2)
    l = s1.shape[1]
    d = np.sum((s1 != s2)*(w1 + w2), axis=1)/l
    # d2 = np.sum((s1 != s2) * w2, axis=1)/l
    # d1 = np.array([hamming(a, b, w) for a, b, w in zip(s1, s2, w1)])
    # d2 = np.array([hamming(a, b, w) for a, b, w in zip(s2, s1, w2)])
    # return d1+d2
    return d

def _sets_windowed(a, b, window_size=3):
    n_datapoints, n_features = a.shape
    similarities = np.zeros((n_datapoints))
    pair_similarities = []

    for datapoint_index in range(len(a)):
        for w_start in range(0, n_features-window_size, 1):
            w_stop = w_start + window_size

            set_a = set(a[datapoint_index, w_start:w_stop])
            set_b = set(b[datapoint_index, w_start:w_stop])

            # Because distance measure: 1-similarity
            pair_similarities.append(1 - (len(set_a.intersection(set_b)) / window_size))

        similarities[datapoint_index] = np.mean(pair_similarities)
    return similarities


def _rankings_windowed(x1, x2, window_size=3):

    assert x1.shape[0] == x2.shape[0]
    n_datapoints = x1.shape[0]
    n_features = x1.shape[1]

    # Create "rankings"
    r1 = np.argsort(x1)
    r2 = np.argsort(x2)

    similarities = _sets_windowed(r1, r2, window_size=window_size)

    return similarities


def _top_k_indices(x1, x2, cutoff):
    r1 = np.argsort(x1)[:, -cutoff:]
    r2 = np.argsort(x2)[:, -cutoff:]

    return r1, r2

def _neg_k_indices(x1, x2, cutoff):
    r1 = np.argsort(x1)[:, :cutoff]
    r2 = np.argsort(x2)[:, :cutoff]

    return r1, r2

def _worst_k_indices(x1, x2, cutoff):
    r1 = np.argsort(np.abs(x1))[:, :cutoff]
    r2 = np.argsort(np.abs(x2))[:, :cutoff]

    return r1, r2


def _k_rankings(x1, x2, k=0.25, select='top'):
    assert x1.shape[0] == x2.shape[0]
    n_features = x1.shape[1]
    n_datapoints = x1.shape[0]

    cutoff = int(n_features * k)

    if select == 'top':
        r1, r2 = _top_k_indices(x1, x2, cutoff)
    elif select == 'worst':
        r1, r2 = _worst_k_indices(x1, x2, cutoff)
    elif select == 'negative':
        r1, r2 = _neg_k_indices(x1, x2, cutoff)
    else:
        raise RuntimeError('Unknown selection strategy', select)

    similarities = _sets_windowed(r1, r2, window_size=3)

    return similarities


def _spearman_foot_rule(x1, x2, select='top'):
    max_dist = np.floor( 0.5 * x1.shape[1]**2)
    r1 = np.argsort(np.argsort(x1))
    r2 = np.argsort(np.argsort(x2))
    d = np.sum(np.abs(r1 - r2), axis=1)
    d = d / max_dist
    return d

# TODO: These two methods are just here to pass them to the multithreading code below easily. 
# Can this be done better? I dont know how to set `select` parameter otherwise
def _top_k_rankings(x1, x2, k=0.25):
    return _k_rankings(x1, x2, k=k, select='top')

def _worst_k_rankings(x1, x2, k=0.5):
    return _k_rankings(x1, x2, k=k, select='worst')

def _neg_k_rankings(x1, x2, k=0.25):
    return _k_rankings(x1, x2, k=k, select='negative')


def _k_euclid(x1, x2, k=0.25, select='top'):

    assert x1.shape[0] == x2.shape[0]
    n_features = x1.shape[1]
    cutoff = int(n_features * k)

    if select == 'top':
        x1args = np.argsort(x1)[:, -cutoff:]
        x1 = x1[:, x1args]
        x2args = np.argsort(x2)[:, -cutoff:]
        x2 = x2[:, x2args]
    elif select == 'worst':
        x1 = np.sort(np.abs(x1))[:, :cutoff]
        x2 = np.sort(np.abs(x2))[:, :cutoff]
    elif select == 'negative':
        x1 = np.sort(x1)[:, :cutoff]
        x2 = np.sort(x2)[:, :cutoff]
    else:
        raise RuntimeError('Unknown selection strategy', select)

    return _explanation_distance(x1, x2)

def _top_k_eucl(x1, x2, k=0.25):
    return _k_euclid(x1, x2, k=k, select='top')

def _worst_k_eucl(x1, x2, k=0.5):
    return _k_euclid(x1, x2, k=k, select='worst')

def _neg_k_eucl(x1, x2, k=0.25):
    return _k_euclid(x1, x2, k=k, select='negative')


def _precision_k(k, x1, x2):
    return np.array([len(np.setdiff1d(x1[row_idx, :k], x2[row_idx, :k], assume_unique=True))/k for row_idx in range(x1.shape[0])])

# x1, x2 -> np.argsort(x_i)
def mean_precision(x1, x2, upto=0.33):
    if upto is None:
        upto = x1.shape[1]
    elif 0 < upto < 1.:
        l = x1.shape[1]
        upto = int(np.ceil(upto * l))
    precisions = np.zeros((x1.shape[0]))
    for k in range(1, upto):
        precisions += _precision_k(k, x1, x2)
    return precisions / (upto-1)


'''
from time import time
w1 = np.abs(np.sort(x1))
s1 = np.argsort(x1)
s2 = np.argsort(x2)
st1 = time()
d = (np.sum(s1 != s2, axis=1) * w1)/l
st2 = time()
d1 = np.array([hamming(a, b, w) for a, b, w in zip(s1, s2, w1)])
f = time()
print(f'{st2-st1} vs {f-st2}')
'''


def _elementwise_eucl(a, b):
    return np.linalg.norm(a-b, axis=1)

def _elementwise_cosine(a, b):
    _a = np.linalg.norm(a, axis=1).astype(np.double) + 1e-20
    _b = np.linalg.norm(b, axis=1).astype(np.double) + 1e-20
    _ab = _a*_b

    _similarity = 0.5 * (np.sum(a*b, axis=1)/_ab + 1.)  # move [-1, 1] to [0, 1]
    return _similarity

def _explanation_distance(expl1, expl2, reduction=np.mean, metric=_elementwise_eucl, shape=None, **kwargs):
    '''
`
    :param expl1:
    :param expl2:
    :param reduction:
    :param metric: input 2 2-D arrays, expected to return triu(D, k=1).data
    :return:
    '''
    d = metric(expl1, expl2, **kwargs)
    return reduction(d)

# ------------------JSD



def tock(start):
    end = time()
    print(f'time: {end - start:.4f}')


def plot_fmnist_expls(images, labels, explanations, model_id):
    def _denormalize_mnist(imgs):
        return (imgs * 0.5) + 0.5
    start = 100
    for i in range(start, len(images)):
        f, axs = plt.subplots(1, len(explanations)+1)
        img = _denormalize_mnist(images[i]).squeeze().detach().numpy()
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('input')
        for j, (_name, _expls, eb) in enumerate(explanations, 1):
                _expls, eb = _expls[::-1], eb[::-1]
                axs[j].imshow(_expls[0][i].squeeze().detach().numpy(), cmap='gray')
                axs[j].set_title(f'{_name}')
        plt.show()
        if i > 10+start:
            break


def filter_models_performance_percentile(seeds: list, accuracies:dict, threshold=.9):
    # keep models that reach threshold * best accuracy at some point during training
    seeds_filtered = []
    max_acc = 0

    for v in accuracies.values():
        mv = max(v)
        if mv > max_acc:
            max_acc = mv

    for seed in seeds:
        acc = accuracies[seed]
        if max(acc) >= threshold * max_acc:
            seeds_filtered.append(seed)

    return seeds_filtered


def explanations_classwise(explanations: list[np.ndarray], class_masks: list[np.ndarray]) -> list[np.ndarray]:
    # sort explanations classwise for easy box plot
    # explanations modelwise -> {modelid: explanations}
    # need {class : explanations}

    class_explanations = []
    for i, mask in enumerate(class_masks):

        class_explanations.append([])
        for model_expls in explanations:
            expls = model_expls[mask]
            class_explanations[-1].append(expls)

        class_explanations[-1] = np.vstack(class_explanations[-1])

    return class_explanations

def explanation_classwise_ranked(explanations: dict[str: np.ndarray], class_masks: list[np.ndarray]):
    # get sorted explanations and additionally do argsort for ranking
    expls = explanations_classwise(explanations, class_masks)
    expls_rk = []
    for exp in expls:
        ranked_exp = np.argsort(exp, axis=1)
        expls_rk.append(ranked_exp)
    return expls_rk


def _historic_differences_explanations(data_dir, model_seed, explanation_method='sg', metric=_explanation_distance):
    expls, _ = load_explanations(data_dir, model_seed, explanation_method=explanation_method)
    dists = [metric(e1, e2) for e1, e2 in zip(expls[:-1], expls[1:])]
    return dists


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=float, default=0.)
    return parser

if __name__ == '__main__':

    # TODO normalize euclid explanation distances with sqrt(0.5 * n_dims)

    from tqdm import tqdm
    from joblib import Parallel, delayed, parallel_backend
    from plotting import get_cmap

    n_jobs = os.cpu_count() - 2
    args = make_parser().parse_args()
    print(args)
    tasks = ['classification']#['beans', 'waveform', 'breastcancer', 'ionosphere', 'classification' ]  # , 'fmnist']
    base_dir = './'
    is_softmax = True
    assert type(args.filter) is float and 0 <= args.filter <= 1.
    filter_models = args.filter > 0.
    filter_thresholds = {t: args.filter for t in tasks}
    print(f'FILTER : {filter_models}  SOFTMAX: {is_softmax}')
    for task in tqdm(tasks):
        if filter_models:
            filter_threshold = filter_thresholds[task]
            results_prefix = f'filtered{int(1000 * filter_threshold)}'
            print(f'filtering models. threshold: {filter_threshold}')
        else:
            results_prefix = 'unfiltered'

        print(f'starting eval on {task} dataset')
        # data_dir = Path(base_dir, 'data', task)
        data_dir = Path(base_dir, 'data', 'softmax' if is_softmax else '', task)
        print(f'loading from {data_dir}')
        result_dir = Path(base_dir, 'results', 'softmax' if is_softmax else '', results_prefix, task); result_dir.mkdir(exist_ok=True, parents=True)
        print(f'result dir: {result_dir}')
        plot_dir = Path(base_dir, 'plots', 'softmax' if is_softmax else '', results_prefix, task); plot_dir.mkdir(exist_ok=True, parents=True)

        # box plot over output similarity
        # meta_data = load_meta_data(data_dir)
        # get all model names
        model_seeds = get_all_model_seeds_sorted(data_dir)
        meta_data = load_meta_data(data_dir, just_one=True)


# LINE PLOT LOSSES / ACCURACIES
        losses, accuracies = load_losses(data_dir), load_accuracies(data_dir)
        cmap = get_cmap(len(losses), name='hsv')  # cmap before filtering
        # Filter model seeds
        if filter_models:
            model_seeds = filter_models_performance_percentile(model_seeds, accuracies, threshold=filter_threshold)
            losses = {k: losses[k] for k in model_seeds}
            accuracies = {k: accuracies[k] for k in model_seeds}

        # line_plot_dict_of_lists(losses, f'{task} losses', fname=str(Path(plot_dir, 'losses.pdf')), cmap=cmap)
        # line_plot_dict_of_lists(accuracies, f'{task} accs', fname=str(Path(plot_dir, 'accuracies.pdf')), cmap=cmap)

# BOX PLOTS OVER DISTANCES: OUTPUTS (JSD), CONFUSION MATRICES (FROB) AND EXPLANATIONS (EUCL, RANKING) (SG, IG)
        # get points of time in training for each model that we want to compare amongst
        print('get matching points')
        thresholds = [0.25, 0.5, 0.75, 1.]
        matching_points = _match_models_accuracy(accuracies, thresholds=thresholds)
        for k, v in matching_points.items():  # add start and end point of model training
            end = len(accuracies[k])-1
            matching_points[k] = [0]+v+[end]

        # load all collected model outputs
        # outputs = { s: load_outputs_sorted(data_dir, s) for s in model_seeds}
        output_distributions = {}
        confusion_matrices = {}
        expl_sg = {}
        expl_ig = {}
        print('load data from results')
        with parallel_backend(backend='loky', n_jobs=n_jobs):
            outs = Parallel()(delayed(load_outputs_by_idxs)(data_dir, seed, matching_points[seed]) for seed in model_seeds)
            eig = Parallel()(delayed(load_explanations_by_idxs)(data_dir, seed, idxs=matching_points[seed], explanation_method='ig') for seed in model_seeds)
        # esg = [e for e, _ in esg]
        eig = [e for e, _ in eig]
        print("done loading")
        for i, seed in enumerate(model_seeds):
            output_distributions[seed] = [o['output_distribution'].cpu().detach().numpy() for o in outs[i]]
            confusion_matrices[seed] = [o['confusion_matrix'] for o in outs[i]]
            # expl_sg[seed] = [e.cpu().detach().numpy() for e in esg[i]]
            expl_ig[seed] = [e.cpu().detach().numpy() for e in eig[i]]

        # 'transpose' {model: outputs} to [[model outputs @ matching point 1], [model outputs @ mp2 ..],..]
        # matched outputs: {seed : outputs[mp1, m2 ... ]}
        outputs_at_threshold = []
        confusion_matrices_at_threshold = []
        # expls_at_threshold_sg = []
        expls_at_threshold_ig = []
        # Argsorted explanations
        argsorted_expl = []
        argsorted_expl_rev = []
        argsorted_worst = []
        inputs = meta_data['X'].detach().cpu().numpy()
        for i in range(len(thresholds) + 2):
            outputs_at_threshold.append([])
            confusion_matrices_at_threshold.append([])
            argsorted_expl.append([]); argsorted_expl_rev.append([]); argsorted_worst.append([])
            # expls_at_threshold_sg.append([])
            expls_at_threshold_ig.append([])
            for k in output_distributions.keys():
                outputs_at_threshold[-1].append(output_distributions[k][i])
                confusion_matrices_at_threshold[-1].append(confusion_matrices[k][i])
                # expls_at_threshold_sg[-1].append(expl_sg[k][i])
                grads = expl_ig[k][i]
                local_explanations = grads*inputs
                expls_at_threshold_ig[-1].append(local_explanations)
                argsorted_expl[-1].append(np.argsort(local_explanations))
                argsorted_expl_rev[-1].append(np.argsort(-local_explanations))
                argsorted_worst[-1].append(np.argsort(np.abs(local_explanations)))

        # compute what datapoints are classified correctly by all models
        labels = meta_data['Y'].cpu().detach().numpy()
        # predictions = np.array([np.argmax(outs, axis=1) for outs in outputs_at_threshold[-1]])
        # overlap = np.sum(labels == predictions, axis=0)
        # mask = overlap >= 0.5*len(model_seeds)
        # _distr_samples_left = np.bincount(labels[mask])
        # print(f"\n\n{task}\nkeeping {np.sum(mask)} of {labels.shape[0]} samples.")
        # print(f"class distribution in sample:\n{_distr_samples_left}\n\n")
        # check what datapoints remain/ whether there are still 'enough' samples per class

        classes = np.unique(labels)
        class_masks = [labels == c for c in classes]
                                                                     # [-1] -> explanations after last batch of training
        # _feature_attribution_mean_vars_sg = explanations_classwise(expls_at_threshold_sg[-1], class_masks)
        # dump_pkl(_feature_attribution_mean_vars_sg, Path(result_dir, 'feature_attribution_mean_vars_sg.pkl'))

        _feature_attribution_mean_vars_ig = explanations_classwise(expls_at_threshold_ig[-1], class_masks)
        dump_pkl(_feature_attribution_mean_vars_ig, Path(result_dir, 'feature_attribution_mean_vars_ig.pkl'))

        # _feature_attribution_ranked_sg = explanation_classwise_ranked(expls_at_threshold_sg[-1], class_masks)
        # dump_pkl(_feature_attribution_ranked_sg, Path(result_dir, 'feature_attribution_ranked_sg.pkl'))

        _feature_attribution_ranked_ig = explanation_classwise_ranked(expls_at_threshold_ig[-1], class_masks)
        dump_pkl(_feature_attribution_ranked_ig, Path(result_dir, 'feature_attribution_ranked_ig.pkl'))



        # for all outputs per point of interest, compute upper half of distance matrix
        if not is_softmax:
            sm = lambda x: softmax(x, axis=-1)
            sm_outs = [sm(o) for o in outputs_at_threshold]  # outputs come from linear layer, needs softmax
        else:
            sm_outs = outputs_at_threshold
        dists_outputs = []
        dists_confmats = []
        dists_explanations_ig = []
        dists_explanations_rankings_ig = []
        dists_explanations_top_k_rankings_ig = []
        dists_explanations_worst_k_rankings_ig = []
        dists_explanations_neg_k_ranking_ig = []
        dists_preck = []
        dists_preck_rev = []
        dists_preck_worst = []
        expls_at_points = []
        print('comp dists')
        with parallel_backend(backend='loky', n_jobs=n_jobs):
            for cms, outs, expls_ig, as_expls, as_expls_rev, as_expls_worst in tqdm(zip(confusion_matrices_at_threshold, sm_outs, expls_at_threshold_ig, argsorted_expl, argsorted_expl_rev, argsorted_worst), total=len(sm_outs)):

                dists_outputs.append(
                    Parallel()(delayed(_functional_distance)
                                   (outs[i], outs[j])
                                   for i in range(len(outs))
                                   for j in range(i + 1, len(outs))
                                   )
                               )
                dists_confmats.append(
                    Parallel()(delayed(np.linalg.norm)
                                        (cms[i] - cms[j])
                                        for i in range(len(cms))
                                        for j in range(i + 1, len(cms))
                                        )
                                      )
                # EXPL EUCL --------------------------------------------------------------------------------------------
                dists_explanations_ig.append(
                    Parallel()(delayed(_explanation_distance)
                                     (expls_ig[i].reshape(expls_ig[0].shape[0], -1), expls_ig[j].reshape(expls_ig[0].shape[0], -1))
                                     for i in range(len(expls_ig))
                                     for j in range(i + 1, len(expls_ig))
                                     )
                                  )
                # EXPL RANK --------------------------------------------------------------------------------------------
                dists_explanations_rankings_ig.append(
                    Parallel()(delayed(_explanation_distance)
                                     (expls_ig[i].reshape(expls_ig[0].shape[0], -1), expls_ig[j].reshape(expls_ig[0].shape[0], -1), metric=_rankings_windowed)
                                     for i in range(len(expls_ig))
                                     for j in range(i + 1, len(expls_ig))
                                     )
                                  )
                # EXPL TOP/ WORST K RANK -------------------------------------------------------------------------------
                dists_explanations_top_k_rankings_ig.append(
                    Parallel()(delayed(_explanation_distance)
                                     (expls_ig[i].reshape(expls_ig[0].shape[0], -1), expls_ig[j].reshape(expls_ig[0].shape[0], -1), metric=_top_k_rankings)
                                     for i in range(len(expls_ig))
                                     for j in range(i+1, len(expls_ig))
                                     )
                                  )
                dists_explanations_worst_k_rankings_ig.append(
                    Parallel()(delayed(_explanation_distance)
                                     (expls_ig[i].reshape(expls_ig[0].shape[0], -1), expls_ig[j].reshape(expls_ig[0].shape[0], -1), metric=_worst_k_rankings)
                                     for i in range(len(expls_ig))
                                     for j in range(i+1, len(expls_ig))
                                     )
                                  )


                # # EXPL NEG K RANK -------------------------------------------------------------------------------

                dists_explanations_neg_k_ranking_ig.append(
                    Parallel()(delayed(_explanation_distance)
                                     (expls_ig[i].reshape(expls_ig[0].shape[0], -1), expls_ig[j].reshape(expls_ig[0].shape[0], -1), metric=_neg_k_rankings)
                                     for i in range(len(expls_ig))
                                     for j in range(i+1, len(expls_ig))
                                     )
                                  )

                # EXPL PREC K -------------------------------------------------------------------------------
                dists_preck.append(
                    Parallel()(delayed(_explanation_distance)
                                     (as_expls[i].reshape(as_expls[0].shape[0], -1), as_expls[j].reshape(as_expls[0].shape[0], -1), metric=mean_precision)
                                     for i in range(len(as_expls))
                                     for j in range(i+1, len(as_expls))
                                     )
                                  )

                dists_preck_rev.append(
                    Parallel()(delayed(_explanation_distance)
                                     (as_expls_rev[i].reshape(as_expls_rev[0].shape[0], -1), as_expls_rev[j].reshape(as_expls_rev[0].shape[0], -1), metric=mean_precision)
                                     for i in range(len(as_expls_rev))
                                     for j in range(i+1, len(as_expls_rev))
                                     )
                                  )
                dists_preck_worst.append(
                    Parallel()(delayed(_explanation_distance)
                               (as_expls_worst[i].reshape(as_expls_worst[0].shape[0], -1),
                                as_expls_worst[j].reshape(as_expls_worst[0].shape[0], -1), metric=mean_precision)
                               for i in range(len(as_expls_worst))
                               for j in range(i + 1, len(as_expls_worst))
                               )
                )
                expls_at_points.append(
                    np.vstack(expls_ig)
                )


        # --- save results
        meta = {'matching_points': matching_points, 'thresholds': thresholds}
        dump_pkl(losses, Path(result_dir, 'losses.pkl')); dump_pkl(accuracies, Path(result_dir, 'accuracies.pkl'))

        dists_outputs = np.array(dists_outputs); dump_pkl(dists_outputs, Path(result_dir, 'dists_outputs.pkl'))
        dists_confmats = np.array(dists_confmats); dump_pkl(dists_confmats, Path(result_dir, 'dists_confmats.pkl'))

        dists_explanations_ig = np.array(dists_explanations_ig); dump_pkl(dists_explanations_ig, Path(result_dir, 'dists_explanations_ig.pkl'))
        dists_explanations_rankings_ig = np.array(dists_explanations_rankings_ig); dump_pkl(dists_explanations_rankings_ig, Path(result_dir, 'dists_explanations_rankings_ig.pkl'))

        dists_explanations_top_k_rankings_ig = np.array(dists_explanations_top_k_rankings_ig); dump_pkl(dists_explanations_top_k_rankings_ig, Path(result_dir, 'dists_explanations_top_k_rankings_ig.pkl'))

        dists_explanations_worst_k_rankings_ig = np.array(dists_explanations_worst_k_rankings_ig); dump_pkl(dists_explanations_worst_k_rankings_ig, Path(result_dir, 'dists_explanations_worst_k_rankings_ig.pkl'))

        dists_explanations_neg_k_ranking_ig = np.array(dists_explanations_neg_k_ranking_ig); dump_pkl(dists_explanations_neg_k_ranking_ig, Path(result_dir, 'dists_explanations_neg_k_ranking_ig.pkl'))

        dists_preck = np.array(dists_preck); dump_pkl(dists_preck, Path(result_dir, 'dists_preck.pkl'))
        dists_preck_rev = np.array(dists_preck_rev); dump_pkl(dists_preck_rev, Path(result_dir, 'dists_preck_rev.pkl'))
        dists_preck_worst = np.array(dists_preck_worst); dump_pkl(dists_preck_worst, Path(result_dir, 'dists_preck_worst.pkl'))
        expls_ig_at_points = np.array(expls_at_points); dump_pkl(expls_ig_at_points, Path(result_dir, 'expls_ig_at_points.pkl'))
