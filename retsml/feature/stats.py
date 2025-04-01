
class FeatureStats:
    @classmethod
    def build(cls, feature_name, volatile=True):
        from ..data import load_sample_data
        sample = load_sample_data()

        if volatile:
            from .base import feature_builder
            fb = feature_builder(feature_name)
            x = fb.compute()
            _, _, y = sample.raw_dataset(y_type='norm5')
            keys = fb.feature_keys
        else:
            sample.load_features([feature_name])
            _, x, y = sample.raw_dataset(y_type='norm5')
            keys = sample.feature_keys
        return FeatureStats(keys, x, y)

    def __init__(self, keys, x, y):
        self.keys = keys
        self.x = x
        self.y = y

    def plot_histogram(self):
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for i, key in enumerate(self.keys):
            ax.hist(self.x[:, :, i].flatten(), bins=100, density=True, label=key, histtype='step')
        ax.legend()
        ax.grid(True)
        return fig

    def _get_corr_func(self, stats):
        from scipy.stats import pearsonr, spearmanr
        return {
            'pearson': pearsonr,
            'spearman': spearmanr,
        }[stats]

    def show_autocorrelation(self, stats='pearson', n_lags=8, target_ix=0, horizon_ix=0, **table_args):
        x, y = self.x[:, target_ix, :], self.y[:, target_ix, horizon_ix]
        corr = self._get_corr_func(stats)
        tbl = []
        headers = ['']
        for j in range(x.shape[-1]):
            row = [self.keys[j]]
            for k in range(n_lags):
                headers.append(f'lag={k}')
                s, p = corr(x[:len(x)-k, j], y[k:])
                tag = ''
                if abs(s) > 0.02 and p < 0.05:
                    tag = '**'
                elif abs(s) > 0.01 and p < 0.2:
                    tag = '*'
                row.append(f's={s:+.03f},p={p:.02f}{tag}')
            tbl.append(row)

        from tabulate import tabulate
        from ..const import TICKERS
        table_args['colalign'] = 'r'
        print(tabulate(tbl, headers=headers, **table_args))

    def show_correlation(self, stats='pearson', horizon_ix=0, **table_args):
        x, y = self.x, self.y[:, :, horizon_ix]

        corr = self._get_corr_func(stats)
        tbl = []
        for j in range(x.shape[-1]):
            row = [self.keys[j]]
            for i in range(y.shape[1]):
                s, p = corr(x[:, i, j], y[:, i])
                tag = ''
                if abs(s) > 0.02 and p < 0.05:
                    tag = '**'
                elif abs(s) > 0.01 and p < 0.1:
                    tag = '*'
                row.append(f's={s:+.03f},p={p:.02f}{tag}')
            tbl.append(row)

        from tabulate import tabulate
        from ..const import TICKERS
        table_args['colalign'] = 'r'
        print(tabulate(tbl, headers=[''] + TICKERS, **table_args))
