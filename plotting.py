import random
import matplotlib.colors
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import _variables

# --- GLOBALS

cmap = None



def hex_str_to_rgb(hex):
    if '0x' == hex[:2]:
        hex = hex[2:]
    _cast_str = lambda s: int(s, 16)
    r = _cast_str(hex[:2])
    g = _cast_str(hex[2:4])
    b = _cast_str(hex[4:])
    return [r, g, b]

_hex_colors = [
    '1C446F',  # blue
    '8BB445',  # light-green
    'F2C944',  # yellow
    'BE677A'  # red
]
_rgb_colors = np.array([hex_str_to_rgb(c) for c in _hex_colors]) / 255.

_hex_colors_dict = {
    'lamarrBlue': '1C446F',
    'lamarrGreen': '8BB445',
    'lamarrOrange': 'F2C944',
    'lamarrRed': 'BE677A'
}

_rgb_colors_dict = {
    k: np.array(hex_str_to_rgb(v))/255. for k, v in _hex_colors_dict.items()
}

print(_rgb_colors)

_purple = np.array(hex_str_to_rgb('A3A9EF')) / 255.

custom_colormap = matplotlib.colors.ListedColormap(_rgb_colors)

def plot_jsd_cm(jsd, cm, dataset_names, cmap=None):
    plt.clf()

    n_rows = 2
    n_cols = len(dataset_names)

    w, h = plt.figaspect(1. / (n_cols+1))/1.5  # +1 for the legendR

    f, axs = plt.subplots(n_rows, n_cols, sharey=False, figsize=(w, h))

    n_bins = 75
    kwargs = dict(histtype='stepfilled', alpha=1., density=True, bins=n_bins)
    for i, name in enumerate(dataset_names):
        ax_jsd, ax_cm = axs[0][i], axs[1][i ]
        ax_jsd.set_xlim(-.05, 1.05); ax_cm.set_xlim(-.05, 1.05)
        ax_jsd.set_title(name)
        ax_jsd.hist(jsd[i], **kwargs)#, ec='k')
        ax_cm.hist(cm[i], **kwargs)#, ec='k')
        if i == 0:
            ax_jsd.set_ylabel('JSD')
            ax_cm.set_ylabel('CM')


def plot_jsd(jsd, dataset_names):
    plt.clf()

    n_rows = 1
    n_cols = len(dataset_names)

    w, h = plt.figaspect(1. / (n_cols+1))/1.5  # +1 for the legendR

    f, axs = plt.subplots(n_rows, n_cols, sharey=False, figsize=(w, h),)

    n_bins = 75
    kwargs = dict(histtype='stepfilled', alpha=1., density=True, bins=n_bins, color=_purple)
    for i, name in enumerate(dataset_names):
        ax_jsd = axs[i]
        ax_jsd.set_xlim(-.05, 1.05)
        # ax_jsd.set_title(name)
        ax_jsd.hist(jsd[i], **kwargs)#, ec='k')
        if i == 0:
            ax_jsd.set_title('JSD')
    return f


def plot_accuracy_curves(curves, dataset_names, cmap=None):
    plt.clf()

    if cmap is None:
        cmap = plt.get_cmap(name='nipy_spectral').copy()


    n_rows = 1
    n_cols = len(dataset_names)
    w, h = plt.figaspect(1. / n_cols)/2.
    f, axs = plt.subplots(n_rows, n_cols, sharey=True, figsize=(w, h))
    axs[0].set_ylim(-0.05, 1.05)
    for ax, name, curve in zip(axs, dataset_names, curves):
        colors = [cmap(x) for x in np.linspace(0.1, 0.8, len(curve))]
        random.shuffle(colors)
        curve = np.array(curve)
        for cu, co in zip(curve, colors):
            ax.plot(cu.T, linewidth=0.5, color=co)
        ax.set_title(name)
    plt.tight_layout()
    return f


def plot_sailplot(sails, metric_names, expl_names, cmap=None):
    plt.clf()
    if cmap is None:
        cmap = plt.get_cmap(name='viridis').copy()
    cmap.set_bad('white', 1.)
    font_size = plt.rcParams['font.size']
    plt.rcParams['font.size'] = 8

    lower_triangle_indices = np.tril_indices_from(sails[0][0])#, k=-1)  # keep diagonal

    f, axs = plt.subplots(len(metric_names), len(expl_names))
    nrow, ncol = len(metric_names), len(expl_names)
    gs = gridspec.GridSpec(nrow, ncol, wspace=0., hspace=0.)#,
                            # top=0.95, bottom=0.05, left=0.17, right=0.845)
    axs = [[plt.subplot(gs[i, j]) for j in range(ncol)] for i in range(nrow)]
    for i, (metric, dists) in enumerate(zip(metric_names, sails)):
        if i == 0:  # put explanation names on top of first row
            for j in range(len(expl_names)):
                axs[i][j].set_title(expl_names[j])
        for j, (expl, d_triu) in enumerate(zip(expl_names, dists)):
            if j == 0:
                axs[i][j].set_ylabel('\n'.join(metric.replace('- ', '').split(' ')))
            d_triu[lower_triangle_indices] = np.nan
            axs[i][j].imshow(d_triu, cmap=cmap)
            for t in ['top', 'right', 'bottom', 'left']: axs[i][j].spines[t].set_visible(False);
            axs[i][j].set_yticks([]); axs[i][j].set_xticks([])
    # gs.update(hspace=0.3, wspace=-0.5)
    # plt.show()
    plt.rcParams['font.size'] = font_size
    return gs


def plot_dist_hists(dists, tasks, expl_names, metric_names, jsd_noisy=None, cmap=None):

    # dists: nested lists/ iterables of tasks[expls[...]]

    if cmap is None:
        cmap = custom_colormap

    n_bins = 75
    kwargs_outline = dict(histtype='step', alpha=1., density=True, bins=n_bins)#, ec="k")
    kwargs_filled = dict(histtype='stepfilled', alpha=0.15, density=True, bins=n_bins)

    n_cols = len(expl_names)
    n_rows = len(tasks)

    n_metrics = len(metric_names)

    color_idxs = np.linspace(0, len(cmap.colors)-1, n_metrics)
    colors = [cmap.colors[int(i)] for i in color_idxs]

    f, axs = plt.subplots(n_rows, n_cols)

    for i, task in enumerate(tasks):
        ax_row = axs[i]
        ax_row[0].set_ylabel(_variables._tasks_paper_names[task])
        d = dists[i]
        for j, expl in enumerate(expl_names):
            ax = ax_row[j]
            if i == 0:
                ax.set_title(expl.upper())
            if jsd_noisy is None:
                _d = d[j]
            else:
                _d = d[j] + jsd_noisy
            ax.hist(_d, **kwargs_filled, color=colors,
                    label=[_variables._metric_paper_names[m] for m in metric_names])  # plot transparent flling
            ax.hist(_d, **kwargs_outline, color=colors, lw=0.5)  # plot full colored outline
            for t in ['top', 'left', 'right']: axs[i][j].spines[t].set_visible(False);
            axs[i][j].spines['bottom'].set_linewidth(0.43)
            ax.set_yticks([])
            ax.set_xticks([])
    ax.legend(loc='center', bbox_to_anchor=(-2., -0.5),
          ncol=2, fancybox=True)
    return f

# ---

def plot_standard_distance_matrix(D, plot_dir, metric_name, task, expl_names, n_models):
    plt.clf()
    f = plt.figure()
    f.set_dpi(400.)
    im = plt.imshow(D, cmap='viridis')
    pos = [int(n_models* (i - 0.5)) for i in range(1, len(expl_names)+1)]  # put expl names in middle
    plt.colorbar(im)
    plt.xticks(pos, expl_names)
    plt.yticks(pos, expl_names)
    plt.title(f"{task} - {metric_name}")
    return f
    # plt.savefig(Path(plot_dir, f"{task}_{metric_name}_dist.pdf"))


def plot_feature_attribution_mean_var(feature_attrs_classwise, title='', fname=None):
    if title == '' or title is None:
        title = fname.split('.')[0]
    # have a subplot for each class, make a box plot for each feaute
    f, axs = plt.subplots(nrows=1, ncols=len(feature_attrs_classwise),sharey=True, figsize=(12, 9))
    for i, _class in enumerate(feature_attrs_classwise):
        ax = axs[i]
        ax.boxplot(_class, vert=False)
    f.suptitle(title)
    plt.tight_layout()
    plt.savefig(fname, format='pdf')


def get_cmap(n, name='hsv'):
    ''' https://stackoverflow.com/a/25628397
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def _plot_110_boxplots(data, x_labels, task_name, metric_names):
    # for one task only

    plt.clf()
    bp_width = 0.15
    ncols=2
    lw = 2
    w, h = plt.figaspect(1. / (2+1))/1.5  # +1 for the legendR
    f, axs = plt.subplots(1, ncols, sharey=True, figsize=(w, h))
    plt.subplots_adjust(wspace=0.05)
    metric_pairs = [('feature disagreement', 'sign disagreement'), ('euclid', 'euclid abs')]
    metric_idxs = [(0, 1), (2, 3)]
    for ax, (m1, m2), (idx1, idx2) in zip(axs, metric_pairs, metric_idxs):
        for i, label in enumerate(x_labels):
            d1 = data[idx1].T[i]  #
            d2 = data[idx2].T[i]
            bp = ax.boxplot(d1, positions=[i + 1 - 0.15], widths=[bp_width], patch_artist=True,
                            medianprops={'linewidth':lw})
            bp['boxes'][0].set_facecolor(_rgb_colors_dict[_variables._map_metric_to_color[m1]])
            bp = ax.boxplot(d2, positions=[i + 1 + 0.15], widths=[bp_width], patch_artist=True,
                            medianprops={'linewidth':lw})
            bp['boxes'][0].set_facecolor(_rgb_colors_dict[_variables._map_metric_to_color[m2]])
        ax.set_xticks(range(0, len(x_labels) + 1))
        ax.set_xticklabels([None] + x_labels, rotation=45)
    axs[0].set_yticks(range(1, len(x_labels) + 1))
    axs[0].set_ylabel("Rank")
    f.suptitle(task_name)
    return f
