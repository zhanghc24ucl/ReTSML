from .base import PoolingModelBase, model


CONFIGS = {
    'gbm': [
        f'{f}_{lr}_{nE}_{depth}_{nL}'
        for lr in [0.07, 0.1]  # default 0.1
        for f in ['f2', 'f3']
        for depth, nL in [(4, 10), (5, 24), (6, 40)]  # default (-1, 32)
        for nE in [100, 200, 300]  # default (100, 0)
    ],
    'f5': [
        f'{f}_{lr}_{nE}_{depth}_{nL}'
        for lr in [0.07, 0.1]  # default 0.1
        for f in ['f5']
        for depth, nL in [(4, 10), (5, 24), (6, 40)]  # default (-1, 32)
        for nE in [100, 200, 300]  # default (100, 0)
    ]
}

FEATURE_SETS = {
    'f1': ('ret',),
    'f2': ('ret', 'time'),
    'f3': ('ret', 'lag_ret', 'time'),
    'f5': ('ret', 'macd', 'vol', 'hfreq', 'time'),
}
CONST_ARGS = {
    'train_size': None,
    'label_type': 'norm5',
    'pred_type': 'norm_ret',
    'train_mode': 'pooling',
}


@model(name='GBM')
class LightGBM(PoolingModelBase):
    def __init__(self, config):
        # parse config
        parts = config.split('_')
        feature_set = FEATURE_SETS[parts[0]]
        self.args = {
            'learning_rate': float(parts[1]),
            'n_estimators': int(parts[2]),
            'max_depth': int(parts[3]),
            'num_leaves': int(parts[4]),
        }
        super().__init__(config, feature_set, **CONST_ARGS)

    def _create(self):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**self.args)

    def _load_or_train(self, split_id, train_x, train_y):
        mpath = self._model_path(split_id)
        try:
            from lightgbm import Booster
            return Booster(model_file=mpath)
        except:
            pass

        assert self.train_mode == 'pooling'
        m = self._create()
        k = train_x.shape[-1]
        m.fit(train_x.reshape(-1, k), train_y.flatten())
        m.booster_.save_model(mpath)
        return m
