from typing import Type


class ModelConfig:
    def __init__(self, **kwargs):
        self.attrs = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.attrs.add(k)

    def set_config(self, key, value):
        setattr(self, key, value)
        self.attrs.add(key)

    def __getstate__(self):
        state = {}
        for attr in self.attrs:
            state[attr] = getattr(self, attr)
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
            self.attrs.add(k)

    def __reduce__(self):
        return type(self), (), self.__getstate__()

    def to_str(self, keys=None):
        if keys is None:
            keys = self.attrs
        else:
            keys = set(keys) & self.attrs
        return '_'.join([str(getattr(self, k)) for k in sorted(keys)])


class ModelBase:
    def __init__(
            self, config, feature_set,
            train_size=None,
            label_type='norm5',
            pred_type='norm_ret',
            horizon=0,
            train_mode='single',
    ):
        self.config = config

        self.feature_set = feature_set
        self.train_size = train_size

        self.label_type = label_type
        self.pred_type = pred_type

        self.horizon = horizon

        assert train_mode in ('single', 'pooling', 'cross')
        self.train_mode = train_mode

    def _model_path(self, split_id):
        from ..data import DATA_ROOT
        from os import mkdir
        folder = f'{DATA_ROOT}/model/{self.name}'
        try:
            mkdir(folder)
        except FileExistsError:
            pass
        return f'{folder}/{self.config}_{split_id}.mod'

    def _train_predict(self, split_id, train_x, train_y, test_x):
        raise NotImplementedError()

    def train_predict(self, in_sample=False):
        from ..data import load_sample_data

        sample_data = load_sample_data()
        sample_data.load_features(self.feature_set)

        scope = 'in-sample' if in_sample else 'all'
        all_time = sample_data.samples['time']

        res_time = []
        res_label = []
        res_pred = []
        args = dict(
            train_size=self.train_size,
            scope=scope,
            stack_feature=self.train_mode=='cross',
            y_type=self.label_type,
        )
        for test0_ix, train_x, train_y, test_x, test_y in sample_data.dataset(**args):
            train_y = train_y[:, :, self.horizon]
            test_y = test_y[:, :, self.horizon]
            res_label.append(test_y)

            t0 = time_ns()
            pred_y = self._train_predict(test0_ix, train_x, train_y, test_x)
            res_pred.append(pred_y)

            test_size = test_x.shape[0]
            res_time.append(all_time[test0_ix:test0_ix+test_size])

        from numpy import concatenate
        res_time = concatenate(res_time, axis=0)
        res_label = concatenate(res_label, axis=0)
        res_pred = concatenate(res_pred, axis=0)

        from ..data import PredictionData
        return PredictionData(time=res_time, norm_ret_pred=res_pred, norm_ret_label=res_label)


class PoolingModelBase(ModelBase):
    def _load_or_train(self, split_id, train_x, train_y):
        raise NotImplementedError()

    def _train_predict(self, split_id, train_x, train_y, test_x):
        model = self._load_or_train(split_id, train_x, train_y)
        assert self.train_mode == 'pooling'
        from numpy import empty
        n_test, n_tgts, n_features = test_x.shape
        pred = empty((n_test, n_tgts), dtype=float)
        for j in range(n_tgts):
            pred[:, j] = model.predict(test_x[:, j])
        return pred


_REGISTERED_MODELS = {}

def model(name: str):
    def decorator(cls: Type[ModelBase]):
        _REGISTERED_MODELS[name] = cls
        cls.name = name
        return cls
    return decorator


def create_model(name: str, config) -> ModelBase:
    return _REGISTERED_MODELS[name](config)
