
def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    Args:
        seed (int): The random seed value.
    """
    from torch import manual_seed, cuda, backends

    manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed_all(seed)
        backends.cudnn.deterministic = True
        backends.cudnn.benchmark = False

    from numpy.random import seed as np_seed
    np_seed(seed)

    from random import seed as native_seed
    native_seed(seed)


def split_dataset(X, Y, split_ix, begin_ix=0, end_ix=None, rolling_size=None, to_tensor=False):
    end_ix = end_ix or len(X)

    if rolling_size is None:
        trainX = X[begin_ix:split_ix]
        trainY = Y[begin_ix:split_ix]
        testX = X[split_ix:end_ix]
        testY = Y[split_ix:end_ix]
    else:
        trainXs = []
        trainYs = []
        testXs = []
        testYs = []

        for i in range(begin_ix + rolling_size, split_ix):
            trainXs.append(X[i - rolling_size:i])
            trainYs.append(Y[i])
        for i in range(split_ix, end_ix):
            testXs.append(X[i - rolling_size:i])
            testYs.append(Y[i])

        from numpy import array
        trainX = array(trainXs)
        trainY = array(trainYs)
        testX = array(testXs)
        testY = array(testYs)

    if to_tensor:
        from torch import tensor, float32
        trainX = tensor(trainX, dtype=float32)
        trainY = tensor(trainY, dtype=float32)
        testX = tensor(testX, dtype=float32)
        testY = tensor(testY, dtype=float32)
    return trainX, trainY, testX, testY
