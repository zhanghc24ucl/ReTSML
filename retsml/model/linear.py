from sklearn.linear_model import Lars, Lasso, LinearRegression

from .base import ModelBase, model

CONFIGS = {
    'lasso': [f'lasso_{f}_{j:.2f}' for j in [0.01, 0.03, 0.1, 0.3, 1] for f in ('f1', 'f2')],
    'lars': [f'lars_{f}_{j}' for j in [10, 15, 25, 50] for f in ('f1', 'f2')],
    'lasso_p': [f'lasso_{f}_{j:.2f}_p' for j in [0.01, 0.03, 0.1, 0.3, 1] for f in ('f1', 'f2')],
    'lars_p': [f'lars_{f}_{j}_p' for j in [10, 15, 25, 50] for f in ('f1', 'f2')],
}
FEATURE_SETS = {
    'f1': ('ret',),
    'f2': ('ret', 'time'),
}
CONST_ARGS = {
    'train_size': None,
    'label_type': 'norm5',
    'pred_type': 'norm_ret',
}


@model(name='linear')
class SkLearnLinearRegression(ModelBase):
    def __init__(self, config):
        self.config = config

        # parse config
        parts = config.split('_')
        self.method = parts[0]
        feature_set = FEATURE_SETS[parts[1]]
        self.args = tuple(parts[2:])

        train_mode = 'pooling' if parts[-1] == 'p' else 'single'
        super().__init__(config, feature_set, train_mode=train_mode, **CONST_ARGS)

    def _create(self):
        # no trained model found, create a new one
        if self.method == 'lasso':
            alpha = self.args[0]
            return Lasso(alpha=float(alpha))
        elif self.method == 'lars':
            n_nonzero_coefs = self.args[0]
            return Lars(n_nonzero_coefs=int(n_nonzero_coefs))
        elif self.method == 'ols':
            return LinearRegression()
        raise ValueError('Unknown method: {}'.format(self.method))

    def _load_or_train(self, split_id, train_x, train_y):
        mpath = self._model_path(split_id)
        from pickle import load, dump
        try:
            with open(mpath, 'rb') as fh:
                return load(fh)
        except:
            pass

        assert self.train_mode in ('single', 'pooling')

        if self.train_mode == 'single':
            n_targets = train_x.shape[1]
            models = []
            for i in range(n_targets):
                m = self._create()
                m.fit(train_x[:, i, :], train_y[:, i])
                models.append(m)
        elif self.train_mode == 'pooling':
            m = self._create()
            F = train_x.shape[-1]
            train_x = train_x.reshape(-1, F)
            train_y = train_y.flatten()
            m.fit(train_x, train_y)
            models = [m]
        else:
            raise ValueError('Unknown train mode: {}'.format(self.train_mode))

        with open(mpath, 'wb') as fh:
            dump([m], fh)
        return models

    def _train_predict(self, split_id, train_x, train_y, test_x):
        n_test, n_tgts, n_features = test_x.shape

        models = self._load_or_train(split_id, train_x, train_y)
        if self.train_mode == 'pooling':
            models = [models[0]] * n_tgts

        from numpy import empty
        pred = empty((n_test, n_tgts), dtype=float)
        for j, m in enumerate(models):
            pred[:, j] = m.predict(test_x[:, j])
        return pred
