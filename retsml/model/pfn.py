from .base import PoolingModelBase, model

CONFIGS = {
    'pfn': ['f3'],
}
FEATURE_SETS = {
    'f1': ('ret',),
    'f2': ('ret', 'time'),
    'f3': ('ret', 'lag_ret', 'time'),
}
CONST_ARGS = {
    'train_size': 8*240,
    'label_type': 'norm5',
    'pred_type': 'norm_ret',
    'train_mode': 'pooling',
}


@model(name='tabpfn')
class TabPFN(PoolingModelBase):
    def __init__(self, config):
        # parse config
        parts = config.split('_')
        feature_set = FEATURE_SETS[parts[0]]
        super().__init__(config, feature_set, **CONST_ARGS)

    def _create(self):
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(ignore_pretraining_limits=True)

    def _load_or_train(self, split_id, train_x, train_y):
        mpath = self._model_path(split_id)
        from pickle import dump, load

        try:
            with open(mpath, 'rb') as fh:
                return load(fh)
        except:
            pass

        assert self.train_mode == 'pooling'
        m = self._create()
        print(train_x.shape)
        k = train_x.shape[-1]
        print('start to train')
        m.fit(train_x.reshape(-1, k), train_y.flatten())
        print('done')
        with open(mpath, 'wb') as fh:
            dump(m, fh)
        return m
