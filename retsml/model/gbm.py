from .base import ModelBase, model


CONFIGS = {
    'gbm': ['f2'],
}

FEATURE_SETS = {
    'f1': ('ret',),
    'f2': ('ret', 'time'),
}
CONST_ARGS = {
    'train_size': None,
    'label_type': 'norm5',
    'pred_type': 'norm_ret',
    'train_mode': 'pooling',
}


@model(name='GBM')
class LightGBM(ModelBase):
    def __init__(self, config):
        self.config = config

        # parse config
        parts = config.split('_')
        feature_set = FEATURE_SETS[parts[0]]
        self.args = tuple(parts[1:])

        super().__init__(config, feature_set, **CONST_ARGS)
    def _create(self):
        from lightgbm import LGBMRegressor
        return LGBMRegressor()

    def _load_or_train(self, split_id, train_x, train_y):
        mpath = self._model_path(split_id)
        from pickle import load, dump
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

    def _train_predict(self, split_id, train_x, train_y, test_x):
        model = self._load_or_train(split_id, train_x, train_y)
        assert self.train_mode == 'pooling'
        from numpy import empty
        n_test, n_tgts, n_features = test_x.shape
        pred = empty((n_test, n_tgts), dtype=float)
        for j in range(n_tgts):
            pred[:, j] = model.predict(test_x[:, j])
        return pred
