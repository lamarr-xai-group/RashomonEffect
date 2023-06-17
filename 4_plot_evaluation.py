import matplotlib as mpl
import matplotlib.pyplot as plt
plt.interactive(True)
from tqdm import tqdm

from scipy.spatial.distance import squareform

from joblib import Parallel, delayed, parallel_backend

import _variables
from util import *
from plotting import \
    plot_standard_distance_matrix, plot_sailplot, plot_dist_hists,\
    plot_accuracy_curves, plot_jsd, _plot_110_boxplots

from scipy.stats import kendalltau


def results_by_model_to_results_by_expl(models):
    keys = [k for k in models[0].keys()]
    explanations = {k: [] for k in keys}
    for m in models:
        for k in m.keys():
            explanations[k].append(m[k])
    return explanations


def agg_expl_dist(sampled_expls):
    # [sampled[ expls for multiple points
    dists = []
    for i in range(len(sampled_expls)):
        a = sampled_expls[i]
        for j in range(i+1, len(sampled_expls)):
            b = sampled_expls[j]
            d = torch.sqrt(torch.sum((a-b)**2, dim=1))
            dists.append(d)
    dists = torch.stack(dists)
    # (take mean of mean of distances, take mean of std of distances)
    res = (torch.mean(torch.mean(dists, 0)), torch.mean(torch.std(dists, 0)))
    return res


def agg_ig_diff(ex1, ex2):
    d = torch.sqrt(torch.sum((ex1 - ex2) ** 2, dim=1))
    mu, sig = torch.mean(d), torch.std(d)
    return mu, sig


def aggregate_modelwise(models):
    n_sampling = len(models[0])
    results = []
    for i in range(n_sampling):
        mus, sigs = [], []
        for m in models:
            mus.append(m[i][0])
            sigs.append(m[i][1])
        _mu, _sig = np.mean(mus), np.mean(sigs)
        results.append((_mu, _sig))
    return results


def eval_stability(explanations: dict):
    # eval by computing the average euclidean distance between all samples for a single datapoint
    # take average and std over all models
    agg_results = {}
    results_model_wise = {}
    explanations = results_by_model_to_results_by_expl(explanations)

    for k, expls in explanations.items():
        # expls = [model [param [n_sampling [reference set x data dim]]]]
        agg_results[k] = None
        _modelwise = []
        for m in expls:
            _sampling = []
            if k != 'ig':
                for sampling in m:
                    (mu, sig) = agg_expl_dist(sampling)
                    _sampling.append((mu, sig))
            else:
                for ex1, ex2 in zip(m[:-1], m[1:]):
                    (mu, sig) = agg_ig_diff(ex1[0], ex2[0])
                    _sampling.append((mu, sig))
            _modelwise.append(_sampling)
        _agg_all = aggregate_modelwise(_modelwise)
        agg_results[k] = _agg_all
        results_model_wise[k] = _modelwise
    return agg_results, results_model_wise


def calc_model_param_choice(sampling_var_by_model, expl_params, factor=2):
    results = {}
    for k, expl in sampling_var_by_model.items():
        params = expl_params[k]
        if k == 'ig':
            idxs = []
            mus = []
            for m in expl:
                mus.append([m[j][0].item() for j in range(len(m))]) # for all models the means of differences of all samples for the j params
            for mu in mus:
                idx = None
                for i, (m1, m2) in enumerate(zip(mu[:-1], mu[1:])):
                    if m1 < factor*m2:
                        idx = i
                if idx is None: idx = len(params)-1
                idxs.append(idx)
        else:
            mus = []
            for m in expl:
                mus.append([m[j][0].item() for j in range(len(m))]) # for all models the means of differences of all samples for the j params
            idxs = []
            for mu in mus:

                idx = None
                for i, (m1, m2, m3) in enumerate(zip(mu[:-1], mu[1:], mu[2:])):
                    if m2 - m1 < factor*(m3 - m2):
                        idx = i+1
                if idx is None: idx = len(params) - 1
                idxs.append(idx)

        results[k] = idxs
    return results


def _expl_stability_results_to_table(var_stability, bin_counts, tasks):
    explstr = ['SG', 'IG', 'KS', 'LI']
    _task_names = _variables._tasks_paper_names
    lineending = "\\\\ \midrule\n"
    F = ""
    for i, t in enumerate(tasks):
        F += "\n\n" + _task_names[t] + "\n"
        table = ""
        for exp in explstr:
            line = f"{exp.upper()} &"
            line += "&".join([f" ${v[0]:.4f} \pm {v[1]:.4f}$ " for v in var_stability[i][exp.lower()]]) + "\\\\ \n"
            line += "& " + "&".join([f" ${v}$ " for v in bin_counts[i][exp.lower()]]) + lineending
            table += line
        table = table[:-len(lineending)] + "\n"  # remove last break and midrule
        F += table
    print(F)

    return F


def jsds_as_table(jsds, tasks):
    F = ""
    F += " & ".join([_variables._tasks_paper_names[t] for t in tasks]) + "\\\\ \hline \n"
    F += "&".join([f" ${np.mean(jsds):.4f} \pm {np.std(jsds):.4f}$" for jsds in jsds]) + "\n"
    return F


def _make_011_table(data_task_wise):
    F = ""
    lineending = "\\\\ \hline\n"
    F += "& " + " & ".join([_variables._tasks_paper_names[t] for t in _variables.tasks]) + lineending
    for _method in _variables.explanation_abbreviations:
        F += _method.upper() + " & "
        for _task in _variables.tasks:
            F += "  \makecell{"
            for _metric in _variables.metric_names:
                color = _variables._map_metric_to_color[_metric]
                v = data_task_wise[_task][_metric][_method]
                F += f" ${{\color{{{color}}} {v[0]:.2f} \pm {v[1]:.2f} }}$ "
                F += "\\\\ "
                F += " "
            F += '} &'
        F = F[:-1]
        F += '\\\\ \midrule \n'
    print(F)
    return F


def _make_offdiag_table(data_task_wise, line_starts_with_method=True):
    # same format for both 110, 010
    # off-diag diag
    exps = _variables.explanation_abbreviations
    _method_comps = [f'{exps[i]}-{exps[j]}' for i in range(len(exps)) for j in range(i+1, len(exps))]
    F = ""
    lineending = "\\\\ \hline\n"
    F += "& " + " & ".join([_variables._tasks_paper_names[t] for t in _variables.tasks]) + lineending
    for c in range(len(_method_comps)):# in _method_comps:
        if line_starts_with_method:
            F += _method_comps[c].upper()
        F += " & "
        for t in range(len(_variables.tasks)):
            F += "  \makecell{"
            for m in range(len(_variables.metric_names)):
                color = _variables._map_metric_to_color[_variables.metric_names[m]]
                v = data_task_wise[t][m][c]
                v = (np.mean(v), np.std(v))  # fock
                F += f" ${{\color{{{color}}} {v[0]:.2f} \pm {v[1]:.2f} }}$ "
                F += "\\\\ "
                F += " "
            F += '} &'
        F = F[:-1]
        F += '\\\\ \midrule \n'
    print(F)
    return F


def _make_offdiag_table_rk_methodwise(data_task_wise):
    # same format for both 110, 010
    # off-diag diag
    exps = _variables.explanation_abbreviations
    _method_comps = [f'{exps[i]}-{exps[j]}' for i in range(len(exps)) for j in range(i+1, len(exps))]
    F = ""
    lineending = "\\\\ \hline\n"
    F += "& " + " & ".join([_variables._tasks_paper_names[t] for t in _variables.tasks]) + lineending
    for c in range(len(_method_comps)):# in _method_comps:
        F += _method_comps[c].upper()
        F += " & "
        for t in range(len(_variables.tasks)):
            F += "  \makecell{"
            for m in range(len(_variables.metric_names)):
                color = _variables._map_metric_to_color[_variables.metric_names[m]]
                v = data_task_wise[t][m][c]
                F += f" ${{\color{{{color}}} {v[0]:.2f} \pm {v[1]:.2f} }}$ "
                F += "\\\\ "
                F += " "
            F += '} &'
        F = F[:-1]
        F += '\\\\ \midrule \n'
    print(F)
    return F


def _make_ranking_table(rankings):
    # [task[metric]
    exps = _variables.explanation_abbreviations
    F = ""
    lineending = "\\\\ \hline\n"
    F += "& " + " & ".join([_variables._tasks_paper_names[t] for t in _variables.tasks]) + lineending

    # cols tasks, rows metrics
    for m in range(len(_variables.metric_names)):
        m_name = _variables._metric_paper_names[_variables.metric_names[m]]
        F += m_name + "&"
        for t in range(len(_variables.tasks)):
            color = _variables._map_metric_to_color[_variables.metric_names[m]]
            v = rankings[t][m]
            v = (np.mean(v), np.std(v))
            F += f" ${{\color{{{color}}} {v[0]:.2f} \pm {v[1]:.2f} }}$ &"
        F = F[:-1]
        F += '\\\\ \midrule \n'

    print(F)

    return F


def _compare_offdiag_rankings(offdiag_distances, return_p_vals=False, n_jobs=14):
    taus = []
    p_vals = []
    with parallel_backend(backend='loky', n_jobs=n_jobs):
        for t in offdiag_distances:
            _tau_by_met = []
            _p_by_met = []
            for met in t:
                tau = []
                model_expl_comb = np.argsort(np.stack(met).T)[:, ::-1]

                tau = Parallel(verbose=1)(delayed(kendalltau)
                                 (model_expl_comb[i], model_expl_comb[j])
                                 for i in range(model_expl_comb.shape[0]) for j in range(i+1, model_expl_comb.shape[0]))
                _tau_significant = [_tau for _tau in tau if _tau[1] < 0.05]
                _tau_stat = [_tau[0] for _tau in _tau_significant]
                _p_vals = [_tau[1] for _tau in _tau_significant]
                _mean_tau, _std_tau = np.mean(_tau_significant), np.std(_tau_significant)
                _p_mean, _p_std = np.mean(_p_vals), np.std(_p_vals)
                print(f"tau=({_mean_tau}+-{_std_tau}), pval=({_p_mean}+-{_p_std})")
                _tau_by_met.append(_tau_stat)
                _p_by_met.append(_p_vals)

            taus.append(_tau_by_met)
            p_vals.append(_p_by_met)

    if return_p_vals:
        return taus, p_vals
    return taus


def _comp_avg_ranks_(offdiag_distances):
    stat_rankings = _comp_ranks(offdiag_distances)
    # with parallel_backend(backend='loky', n_jobs=n_jobs):
    for t in offdiag_distances:
        stat_metrics = []
        for met in t:
            stat_metrics.append(
                [(m, s) for m, s in zip(np.mean(met, axis=0), np.std(met, axis=0))]
            )
        stat_rankings.append(stat_metrics)

    return stat_rankings


def _comp_ranks(offdiag_distances):
    _rankings = []
    # with parallel_backend(backend='loky', n_jobs=n_jobs):
    for t in offdiag_distances:
        _t_rks = []
        for met in t:
            model_expl_comb = np.argsort(np.stack(met).T)+1
            _t_rks.append(model_expl_comb)
        _rankings.append(_t_rks)

    return _rankings


def save_table(table: str, pth, fname):
    with open(str(Path(pth, fname)), 'w') as f:
        f.write(table)


if __name__ == '__main__':

    mpl.use("pgf")

    _rcParams = {
        "pgf.texsystem": "pdflatex",  # or any other engine you want to use
        "text.usetex": True,  # use TeX for all texts
        "font.family": "serif",
        "font.serif": [],  # empty entries should cause the usage of the document fonts
        "font.sans-serif": [],
        "font.monospace": [],
        "font.size": 12,  # control font sizes of different elements
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pgf.preamble": [  # specify additional preamble calls for LaTeX's run
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{siunitx}",
        ],
    }

    mpl.rcParams.update(_rcParams)
    format = 'pgf'
    plt.rcParams['figure.dpi'] = 600

    tasks = _variables.tasks
    print(f"tasks: {tasks}")
    stds = []

    all_trius = []  # 011, collect all distances for all tasks/attributions/metrics for histograms
    all_offdiag_diag = []  # 110
    all_accuracies = []
    all_lor_curves = []
    all_js_dists = []
    all_cm_dists = []

    all_js_noisy = []


# ---- 111 -----
    _bin_counts_111 = []
    _var_111 = []
    for task in tqdm(tasks, position=0, desc="task stability"):

        print(f'processing stability data on {task} dataset')
        data_dir = _variables.get_data_dir(task)
        result_dir = _variables.get_result_dir(task)
        print(f'result dir: {result_dir}')
        plot_dir = _variables.get_plot_dir(task)
        print(f'plot dir: {plot_dir}')

        # expl_stability = None
        try:
            expl_stability = load_pkl(Path(result_dir, f"{task}_stability_new.pkl"))
        except FileNotFoundError:
            expl_stability = load_pkl(Path(result_dir, f"{task}_stability.pkl"))

        if task == 'agnews': # sum up over embedding dimension to obtain one val per word
            for model in expl_stability:
                for method in model.keys():
                    if method in ['ks', 'li']:
                        continue
                    expls = model[method]
                    for i in range(len(expls)):
                        exp = expls[i]
                        exp = [torch.sum(e, dim=-1) for e in exp]
                        expls[i] = exp
                    model[method] = expls

        _result, modelwise = eval_stability(expl_stability)
        _var_111.append(_result)
        variance_best_parameter_choice = calc_model_param_choice(modelwise, _variables._eval_hyper_param_expl_n_samples)
        n_params = len(_variables._eval_hyper_param_expl_n_samples['sg'])
        counts = {k: np.bincount(v, minlength=n_params) for k, v in variance_best_parameter_choice.items()}
        _bin_counts_111.append(counts)

    plot_dir = _variables.get_plot_dir('')
    _table_str = _expl_stability_results_to_table(_var_111, _bin_counts_111, tasks)
    save_table(_table_str, plot_dir, "stability_table.txt")


# --- 110 & 011 ---
    # create plots that show tasks individually
    tasks = _variables.tasks
    for task in tqdm(tasks, position=0, desc="tasks"):

        result_dir = _variables.get_result_dir(task)
        plot_dir = _variables.get_plot_dir(task)

        all_cm_dists.append(load_pkl(Path(result_dir, 'confmatdists.pkl')))
        all_js_dists.append(load_pkl(Path(result_dir, 'jsd.pkl')))

        _plot_dir_full_dists = Path(plot_dir, 'fullDists'); _plot_dir_full_dists.mkdir(exist_ok=True, parents=True)

        # load for later use after task-specific plots are finished ---------
        _acc = load_pkl(Path(result_dir, 'accuracies.pkl'))
        all_accuracies.append(_acc)

        _lor = load_pkl(Path(result_dir, 'lor_curves.pkl'))
        all_lor_curves.append(np.array([_lor[abb].detach().numpy() for abb in _variables.explanation_abbreviations]))
        # -------------------------------------------------------------------

        cmap = plt.cm.viridis

        dists = load_pkl(Path(result_dir, 'expl_dists.pkl'))
        n_expl_methods = len(_variables.explanation_abbreviations)
        n_models = int(squareform(np.array(dists[0])).shape[0] / n_expl_methods)
        rows_distance_matrix_triangles = []

        _idxs_diagonal_matrices = [(i * n_models, (i + 1) * n_models) for i in range(n_expl_methods)]
        __idxs = [n_models] + np.arange(n_expl_methods * n_models)[::n_models] + [n_models*n_expl_methods]
        _triu_x, _triu_y = np.triu_indices(n_expl_methods, k=1)
        _triu_x, _triu_y = _triu_x * n_models, _triu_y * n_models
        __idxs_off_diagonal_matrices = [(a,b) for a,b in zip(_triu_x, _triu_y)]
        _diagonals = []
        _trius = []
        _offdiag_diag = []

        for d, metric_name in tqdm(zip(dists, _variables.metric_names), position=1):
            _diagonals.append([])
            _trius.append([])
            # _offdiagonals.append([])
            _offdiag_diag.append([])
            _d = squareform(np.array(d))
            dim, n_expls = _d.shape[0], len(_variables.explanation_abbreviations)

            # f = plot_box_plot_grid(_d, plot_dir, metric_name, task,
            #                    _variables.explanation_abbreviations, n_models)
            # plt.savefig(Path(_plot_dir_full_dists, f'{task}_{metric_name}_box.pdf')); plt.close(f)
            f = plot_standard_distance_matrix(_d, plot_dir, metric_name, task,
                                          _variables.explanation_abbreviations, n_models)
            plt.savefig(Path(_plot_dir_full_dists, f'{task}_{metric_name}_dist.{format}'), format=format); plt.close(f)

            for (s, e) in _idxs_diagonal_matrices:
                square = _d[s:e, s:e]
                if 'eucl' in metric_name:
                    if np.max(square) > 1:
                        square = square / np.max(square)
                _diagonals[-1].append(square)
                _trius[-1].append(square[np.triu_indices_from(square, k=1)])

            for (s, e) in __idxs_off_diagonal_matrices:
                # print(f"rows {s}:{e}, cols {e}:{e+n_models}")
                square = _d[s:e, e:e+n_models]
                if 'eucl' in metric_name:
                    if np.max(square) > 1:
                        square = square / np.max(square)
                # _offdiagonals[-1].append(square)
                _offdiag_diag[-1].append(np.diag(square))

        all_trius.append(_trius)
        all_offdiag_diag.append(_offdiag_diag)

        f = plot_sailplot(_diagonals, _variables.metric_names, _variables.explanation_abbreviations)
        plt.savefig(Path(plot_dir, f'{task}_sailplot.{format}'), format=format)



    print("dataset | accuracy (min, mean, max)")
    print("----- | -----")
    mean_accs = []
    for e, n in zip(all_accuracies, _variables.tasks):
        e = np.array(e)[:, -1]
        mean_accs.append((np.mean(e), np.std(e)))
    print(_variables.tasks)
    print(" & ".join([f"${v[0]:.2f} \pm {v[1]:.2f} $" for v in mean_accs]))
    print("-------------------")

    # ------------------------------------------------------------------------------------------------------------------

    plot_dir_root = _variables.get_plot_dir('')

    print("plot jsd")
    plot_jsd(all_js_dists, _variables.tasks)
    plt.savefig(Path(plot_dir_root, f'jsd.{format}'), format=format)
    _table = jsds_as_table(all_js_dists, _variables.tasks)
    print(_table)
    save_table(_table, plot_dir_root, "jsd_table.txt")

    # accuracy curves ---------------------------------------------------------------------------------------
    f = plot_accuracy_curves(all_accuracies, _variables.tasks)
    plt.savefig(Path(plot_dir_root, f'accuracy_curves.{format}'), format=format); plt.close(f)

    # dist histograms ---------------------------------------------------------------------------------------
    print("plot histograms ..")
    cmap = plt.get_cmap('tab10')
    n_tasks = len(all_trius)
    n_metrics = len(all_trius[0])
    n_expls = len(all_trius[0][0])
    assert n_tasks == len(_variables.tasks) and \
           n_metrics == len(_variables.metric_names) and \
           n_expls == len(_variables.explanation_abbreviations)
    # Calc table report of the mean standard deviation of a metric across all datsets and explanation methods
    # swap [task][metric][expl] -> [task][expl][metric]
    _dists_by_metric = np.array(all_trius).swapaxes(1, 2)

    print("plot dist_hists")
    _rcParams_disthist = {
        "font.size": 11,  # control font sizes of different elements
        "axes.labelsize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
    mpl.rcParams.update(_rcParams_disthist)
    f = plot_dist_hists(_dists_by_metric, _variables.tasks, _variables.explanation_abbreviations,
                        _variables.metric_names)
    plt.savefig(Path(plot_dir_root, f'dist_hists.{format}'), format=format, bbox_inches='tight')
    mpl.rcParams.update(_rcParams)
    print("")

    _011_tasks = {t: None for t in _variables.tasks}
    for _t, t in enumerate(all_trius):
        _011_metric = {m: None for m in _variables.metric_names}
        for _m, metric in enumerate(t):
            _011_method = {e: None for e in _variables.explanation_abbreviations}
            for _method, method in enumerate(metric):
                _011_method[_variables.explanation_abbreviations[_method]] = (np.mean(method), np.std(method))
            _011_metric[_variables.metric_names[_m]] = _011_method
        _011_tasks[_variables.tasks[_t]] = _011_metric


    _method_pairings = [
        f'{_variables.explanation_abbreviations[i].upper()}-{_variables.explanation_abbreviations[j].upper()}'
        for i in range(len(_variables.explanation_abbreviations)) for j in
        range(i + 1, len(_variables.explanation_abbreviations))]
    print("comp avg rankings")

    print("110")
    rk = _comp_ranks(all_offdiag_diag)  # how often models agree on order in which method pairs disagree
    for i, t in enumerate(tasks):
        F = _plot_110_boxplots(rk[i], _method_pairings, _variables._tasks_paper_names[t], _variables.metric_names)
        plt.savefig(Path(_variables.get_plot_dir(t), f'{t}_110_rk_bp.pgf'), format=format, bbox_inches='tight')

    print("110")
    rk = _comp_ranks(all_offdiag_diag)  #  how often models agree on ordere in which method pairs disagree
    for i, t in enumerate(tasks):
        F = _plot_110_boxplots(rk[i], _method_pairings, _variables._tasks_paper_names[t], _variables.metric_names)
        plt.savefig(Path(_variables.get_plot_dir(t), f'110_rk_bp.pdf'), format='pdf', bbox_inches='tight')

    _metric_stds = []
    for m in range(len(_variables.metric_names)):
        _metric_stds.append([np.std(x) for x in list(_dists_by_metric[m])])

    _metric_mean_std = []
    for i in range(len(_metric_stds)):
        ms = _metric_stds[i]
        _metric_mean_std.append((np.mean(ms), np.std(ms)))

    print("metric | mean(std), std(mean(std))")
    print("----- | -----")
    for ms, m in zip(_metric_mean_std, _variables.metric_names):
        print(f"{m} |  {ms[0]:.4f} ± {ms[1]:.4f}")

    print("make rashomon table 011")
    F = _make_011_table(_011_tasks)
    save_table(F, plot_dir_root, "011_rashomon_table.txt")

    print("110 continued")
    rk = _comp_avg_ranks_(all_offdiag_diag)
    F = _make_offdiag_table_rk_methodwise(rk)
    save_table(F, plot_dir_root, "110_disagreement_pair_ranking.txt")

    print("make offdiag diag table 110")
    F = _make_offdiag_table(all_offdiag_diag)
    save_table(F, plot_dir_root, "110_disagreement_table.txt")  # distances

    print("make offdiag diag ranking 110")
    rk = _compare_offdiag_rankings(all_offdiag_diag)# rank correlation between rankings given by models
    F = _make_ranking_table(rk)
    save_table(F, plot_dir_root, "110_disagreement_rankings.txt")  # rankings
