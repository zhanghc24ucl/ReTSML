from typing import Type

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
        from ..data import load_feature_data, load_sample_data

        sample_data = load_sample_data()
        v, keys = load_feature_data(self.feature_set)
        sample_data.attach_features(v, keys)

        mode = 'in-sample' if in_sample else 'all'
        all_time = sample_data.samples['time']

        res_time = []
        res_label = []
        res_pred = []
        for test0_ix, train_x, train_y, test_x, test_y in \
                sample_data.dataset(self.train_size, self.label_type, mode):
            train_x = train_x[:, :, :]
            train_y = train_y[:, :, self.horizon]
            test_x = test_x[:, :, :]

            test_y = test_y[:, :, self.horizon]
            res_label.append(test_y)

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


_REGISTERED_MODELS = {}

def model(name: str):
    def decorator(cls: Type[ModelBase]):
        _REGISTERED_MODELS[name] = cls
        cls.name = name
        return cls
    return decorator


def create_model(name: str, config) -> ModelBase:
    return _REGISTERED_MODELS[name](config)
