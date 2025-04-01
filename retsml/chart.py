from matplotlib import pyplot as plt
from matplotlib.pyplot import axvline


def plot_fvr_curve(prediction_list, labels=None, save_path=None, fig_size=(12, 8), sample_split=None, title=None):
    if labels is None:
        labels = [f'model_{i}' for i in range(len(prediction_list))]
    assert len(labels) == len(prediction_list)

    fig, ax = plt.subplots(figsize=fig_size)
    for j, pred in enumerate(prediction_list):
        fvr = pred.FvR.sum(axis=1)
        ax.plot(pred.time, fvr, label=labels[j])
    ax.grid(True)
    if sample_split is not None:
        axvline(sample_split, linestyle='dashed')
    ax.legend(loc='upper left')
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def plot_feature_stats():
    from scipy.stats import pearsonr, spearmanr
