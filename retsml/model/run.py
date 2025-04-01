from .base import create_model
from .linear import CONFIGS as LR_CONFIGS
from .gbm import CONFIGS as GBM_CONFIGS
from .pfn import CONFIGS as PFN_CONFIGS
from ..data import DATA_ROOT
from time import time_ns

ALL_CONFIGS = {
    'linear': LR_CONFIGS,
    'GBM': GBM_CONFIGS,
    'tabpfn': PFN_CONFIGS,
}


def tune(model_name, config_grid, metric='RMSE', verbose=True):
    tpath = f'{DATA_ROOT}/model/tuned_{model_name}_{config_grid}_{metric}.bin'
    from pickle import load, dump
    try:
        with open(tpath, 'rb') as fh:
            return load(fh)
    except FileNotFoundError:
        pass

    configs = ALL_CONFIGS[model_name][config_grid]
    values = []

    for f in configs:
        t0 = time_ns()
        model = create_model(model_name, f)
        res = model.train_predict(in_sample=True)
        mv = getattr(res, metric)
        values.append(mv)
        t1 = time_ns()
        if verbose:
            print(f, f'done with metric: {res.RMSE} with {(t1-t0)/1e9:.1f}s.')

    from numpy import array, argmin
    values = array(values, dtype=float)
    best_cfg = configs[argmin(values)]

    with open(tpath, 'wb') as fh:
        rv = best_cfg, configs, values
        dump(rv, fh)
    return rv


def predict(model_name, config, in_sample=False):
    model = create_model(model_name, config)
    return model.train_predict(in_sample=in_sample)
