import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_convergence(array, n_trials_list, ax, k=2):
    """
    Plots convergence of the correlation matrix
    array: np.array, matrix of correlations, shape len(n_trials_list) x n_repetitions,
    n_trials_list: list/array, of n_trials used to compute the correlations from
    ax: matplotlib axis
    k: int, half of the forms
    """
    ax.plot(n_trials_list, array.mean(axis=1))
    ax.scatter(n_trials_list, array.mean(axis=1))
    ax.axhline(array.mean(axis=1).max(), c="orange", label=f"max={array.mean(axis=1).max():.2f}")
    ax.set_xlabel("Number of trials used for calculating the score")
    ax.set_ylabel("Mean correlation across " + "$n_{rep}=$" + f"{array.shape[1]} trials")

    # add a line for num of exemplars
    ax.axvline(max(n_trials_list), c="k", ls='--', alpha=0.5, label="full form")
    for l in range(1, k):
        ax.axvline(l * max(n_trials_list) // k, c="k", ls='--', alpha=0.5)

    ax.legend(loc="lower right")


def plot_figure_convergent_tasks_random(n_trials_list, y_sim, y_std, ax, bigN=True, loc='best', alph=0.5,
                                        colour='darkorange', label='', N=None, N_n=None,
                                        total_n_trials=None, total_n_trials_n=None):
    if bigN:
        if not label:
            label = f'N={N}, n trials={total_n_trials}'
    else:
        if not label:
            label = f'random data\nN={N_n}, n trials={total_n_trials_n}'
    # plot random
    ax.errorbar(n_trials_list, y_sim, yerr=y_std, fmt='o', capsize=2, c=colour, alpha=alph, label=label)
    ax.set_title(f'Random data with no structure')
    ax.legend(loc=loc)
    ax.set_xlabel('Number of trials, L')
    # add numbering from zero
    ax.set_xlim(0, ax.get_xlim()[1])


def label_correlation(x, y, ax, xy=(.08, .85), plot_corr=True, **kwargs):
    """
    Label plot with correlation, returns the correlation values and the p-value
    Parameters
    ----------
    x: array, values to be correlated
    y: array, values to be correlated
    ax: matplotlib axis to be annotated
    xy: tuple of floats <0,1>, position of the label of the correlation
    plot_corr: bool, default True, whether to plot the correlation

    Returns
    -------
    r, p -- correlation coefficient and p-value

    """
    # find nans
    nans = np.logical_or(np.isnan(x), np.isnan(y))
    # add correlation label
    r, p = stats.pearsonr(x[~nans], y[~nans])
    if plot_corr:
        ax = ax or plt.gca()
        ax.annotate(f"r = {r:.2f}\nρ = {p:.2e}", xy=xy, xycoords=ax.transAxes, **kwargs)
    return r, p


def corrfunc(x, y, **kws):
    try:
        r, p = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate(f"r = {r:.2f}\nρ = {p:.2e}",
                xy=(.1, .9), xycoords=ax.transAxes)
    except ValueError:
#         print(f"Values {x} and {y} contain nan or infinite values, no correlation calculated.")
        print("Not plotted")


def plot_correlation_scatter(array, ax):
    """
    Plots scatter plot of the correlation matrix
    array: np.array, matrix of correlations, shape n_repetitions,
    n_trials_list: list/array, of n_trials used to compute the correlations from
    ax: matplotlib axis
    """
    ax.scatter(range(array.shape[0]), array)
    ax.axhline(array.mean(), c="orange", label=f"mean={array.mean():.2f}")

    ax.set_xlabel("Sample number")
    ax.set_ylabel("Correlation between accuracy per subject in two halves")
    ax.legend()


def plot_correlation_hist(array, ax):
    """
    Plots scatter plot of the correlation matrix
    array: np.array, matrix of correlations, shape n_repetitions,
    n_trials_list: list/array, of n_trials used to compute the correlations from
    ax: matplotlib axis
    """
    sns.histplot(array, bins='auto', ax=ax)
    ax.axvline(array.mean(), c="orange", label=f"mean={array.mean():.2f}")

    ax.set_xlabel("Sample number")
    ax.set_ylabel("Correlation between accuracy per subject in two halves")
    ax.legend()


def label_column_height(ax, x=0.35, y=0.03, size=12, rot=0, n=2):
    """
    adds a text label of how height the bar is
    x: float, vertical position of the label
    y: float, how much above the bar the label should be
    size: int, fontsize
    rot: int, rotation
    n: int, number of decimal places to show
    """
    for i, bar in enumerate(ax.patches):
        h = bar.get_height()
        # control for negative labelling
        if h < 0:
            height = h - y
        else:
            height = h + y
        ax.text(
            bar.get_x() + x,  # bar index (x coordinate of text)
            height,  # y coordinate of text
            str('{:.' + str(n) + 'f}').format(h),  # y label
            ha='center',
            va='center',
            rotation=rot,
            size=size)


def change_width(ax, new_value):
    """
    Set the width nicely, see https://stackoverflow.com/questions/34888058/changing-width-of-bars-in-bar-chart-created-using-seaborn-factorplot
    """
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def sharey_ax(a, source):
    """
    Takes axis a that we want to share with axis source, for CanD
    """
    # sharey, take the first axis and sharey with the last plotted one
    a.sharey(source)
    # need to rescale all axis
    a.autoscale()
    # remove ticks from shared y axes
    plt.setp(a.get_yticklabels(), visible=False)
    # remove ylabel
    a.set_ylabel('')


def sharey_name(a, source, c):
    """
    Takes name a of an axis that we want to share with an axis named source, need CanD canvas c, for CanD
    """
    # sharey, take the first axis and sharey with the last plotted one
    c.ax(a).sharey(c.ax(source))
    # need to rescale all axis
    c.ax(a).autoscale()
    # remove ticks from shared y axes
    plt.setp(c.ax(a).get_yticklabels(), visible=False)
    # remove ylabel
    c.ax(a).set_ylabel('')


def plot_RC_distributions_line(corr_distribution_array, df_neeed_trials, max_rel_thr, title,
                               axs=None, colour=None, legend_position=(.41, 1.05), frameon=True, real_corr=None,
                               legend_title='Reliability of the two tasks', columnspacing=0.8, x_l=1.1, y_l=10,
                               corr_label=None, SMALL_SIZE=12, MEDIUM_SIZE=14, BIGGER_SIZE=16, plot_in_jupyter=True):
    if colour is None:
        colour = sns.color_palette("Spectral", n_colors=corr_distribution_array.shape[0])

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.1)
        show_plt = True
    else:
        show_plt = False

    # check what the data is and process it accordingly
    # print(df_neeed_trials)
    if isinstance(df_neeed_trials, (list, np.ndarray)):
        x_rel = np.array(df_neeed_trials)[np.array(df_neeed_trials) <= max_rel_thr]
    elif isinstance(df_neeed_trials, pd.core.frame.DataFrame):
        x_rel = df_neeed_trials.loc[df_neeed_trials["thr"] <= max_rel_thr, "thr"].unique()
    else:
        print(f'df_needed_trials is neither list, np array or pandas dataframe, it is {type(df_neeed_trials)}. Quitting.')
        exit()

    for j, thr in enumerate(x_rel):
        sns.kdeplot(corr_distribution_array[j], label=f'{thr:.2f}', color=colour[j], multiple='stack', alpha=0.2,
                    ax=axs[0])

    if real_corr is not None:
        axs[0].axvline(real_corr, c='crimson')
        if corr_label is None:
            corr_label = f'full-sample\ncorrelation\nr={real_corr:.3f}'
        axs[0].annotate(corr_label, (real_corr*x_l, y_l))

    #         axs[0].axvline(real_corr, c='y', linestyle='--')

    axs[0].legend(title=legend_title, bbox_to_anchor=legend_position, ncol=2, frameon=frameon, columnspacing=columnspacing)
    axs[0].set_xlabel('Correlation')

    #######
    y_corr = np.nanmean(corr_distribution_array, axis=1)

    axs[1].errorbar(x_rel, y_corr, yerr=np.nanstd(corr_distribution_array, axis=1), capsize=2, c='gray')
    axs[1].scatter(x_rel, y_corr, c='b', zorder=4)

    axs[1].set_xlabel('Reliabilty R')
    axs[1].set_ylabel('Mean Correlation')

    for ax in axs.flatten():
        sns.despine(ax=ax)

    plt.suptitle(title)
    if show_plt:
        plt.show()

    # reset back
    # if on Jupyter, https://stackoverflow.com/questions/42656668/matplotlibrc-rcparams-modified-for-jupyter-inline-plots
    if plot_in_jupyter:
        plt.rcParams.update(
            {'figure.figsize': (6.0, 4.0),
             'figure.facecolor': (1, 1, 1, 0),  # play nicely with white background in the Qt and notebook
             'figure.edgecolor': (1, 1, 1, 0),
             'font.size': 10,  # 12pt labels get cutoff on 6x4 logplots, so use 10pt.
             'figure.dpi': 72,  # 72 dpi matches SVG/qtconsole
             'figure.subplot.bottom': .125  # 10pt still needs a little more room on the xlabel
             }
        )
    else:
        plt.rcParams.update(plt.rcParamsDefault)


def plot_sampling_curves(n_trials_list, reliability_arr_mu, reliability_arr_sd, ax,
                         color='r', alpha=0.8, label='', **kwargs):
    """
    Plot reliability curves from simulations of sampling error (with their error)

    Parameters
    ----------
    n_trials_list: array/list, of Ls, number of trials that were sampled
    reliability_arr_mu: array, of means of reliabilities per given L
    reliability_arr_sd: array, of standard deviations of reliabilities per given L
    ax: matplotlib axis to plot on
    color: str, default 'r'
    alpha: float, default 0.8
    label: str, default ''

    Returns
    -------

    """
    ax.fill_between(n_trials_list,
                    reliability_arr_mu-reliability_arr_sd,
                    reliability_arr_mu+reliability_arr_sd,
                    color=color, alpha=alpha, label=label
                   )
