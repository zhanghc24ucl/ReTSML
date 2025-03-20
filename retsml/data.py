"""
retsml/data.py

This module provides functions and classes for handling financial data, including loading raw data,
preprocessing it into samples, generating dataset splits, and managing features.

Classes:
- RawData: Represents and manipulates financial raw data at different frequencies and its covariance matrix.
- SampleData: Represents and manipulates preprocessed sample data and its splits.

Functions:
- raw_to_sample: Converts raw financial data into sampled data suitable for analysis or modeling.
- load_raw_data: Loads raw financial data from disk.
- load_sample_data: Loads preprocessed sample data from disk.
- load_feature_data: Loads feature data from disk based on the provided feature names.
"""
from os import environ
DATA_ROOT = environ.get('DATA_ROOT', './data')


__all__ = ['raw_to_sample',
           'load_raw_data', 'load_feature_data', 'load_sample_data',
           'RawData', 'SampleData', 'PredictionData']

def _add_months(date, months):
    """
    Add a specified number of months to a given date.

    :param date: The starting date.
    :param months: The number of months to add.
    :return: The resulting date after adding the specified number of months.
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    from numpy import datetime64
    return datetime64(date.astype(datetime) + relativedelta(months=months), 's')


def _generate_splits(sample_time, frequency: int = 3):
    """
    Generate training and testing split points for time series data.

    This function takes a list of sample times and a frequency, and generates a series of split points for training
    and testing. The split points are determined based on a moving time window, with the training end point advancing
    by the specified months each time. The function also identifies if the split point has reached or exceeded the
    year 2022, which is marked as out-sample.

    :param sample_time: A numpy.ndarray[int] of sample times.
    :param frequency: The months of movement for the training and testing split points, default is 3.
    :return: An array containing information about each split, including the index of the training end point,
             the index of the testing end point, and whether it is out-of-sample (after 2022).
    """
    from datetime import datetime
    from numpy import empty, datetime64

    # Use first 2 years from 2016-01-01 to 2017-12-31 as warm-up period
    train_end = datetime64('2018-01-01', 's')

    train_end_ix, test_end_ix = [], []
    out_sample = []
    while train_end < sample_time[-1]:
        # Find the index corresponding to the training end time
        train_end_ix.append(sample_time.searchsorted(train_end, side='right'))
        # Determine if the current split is out-of-sample
        out_sample.append(train_end.astype(datetime).year >= 2022)

        # Calculate the testing end time based on the frequency
        test_end = _add_months(train_end, frequency)
        # Find the index corresponding to the testing end time
        test_end_ix.append(sample_time.searchsorted(test_end, side='right'))

        # Move the training end time to the testing end time for the next iteration
        train_end = test_end

    # Create a structured array to store split information
    splits = empty(len(train_end_ix), dtype=[
        ('train_end_ix', int),
        ('test_end_ix', int),
        ('out-sample', bool),
    ])
    # Populate the structured array with split information
    splits['train_end_ix'] = train_end_ix
    splits['test_end_ix'] = test_end_ix
    splits['out-sample'] = out_sample
    return splits


def raw_to_sample():
    """
    Convert raw financial data into sampled data suitable for analysis or modeling.

    This function processes minute-level (m1), daily (d1), and 5-minute (m5) raw data to create samples
    at 30-minute intervals. It calculates target returns with different horizons and normalizes these returns
    using variance data.
    """
    raw_data = load_raw_data()
    from numpy import arange, clip, stack, empty, log, repeat, sqrt, timedelta64

    # Sample every 30 minutes, at 09:30, 10:00, ..., 14:30
    # Since the time in raw_data is the end_time of the interval,
    # we subtract the time span of each bar to get the proper sample time.
    _1min = timedelta64(1, 'm')
    per_1m = 30  # Number of 1-minute intervals per sample grid
    per_5m = 6   # Number of 5-minute intervals per sample grid

    sample_time = raw_data['m1:time'][::per_1m] - _1min

    n_d1 = len(raw_data.raw_d1)
    n_m1 = len(raw_data.raw_m1)
    n_m5 = len(raw_data.raw_m5)
    s = len(sample_time)
    m = raw_data.covariance.shape[1]  # Fixed number of instruments (m == 3)
    assert m == 3

    # There are 240 minutes per trading day, so:
    # n_d1 == n_m1 * 240 == n_m5 * 240 // 5
    # len(sample_time) == n_d1 * 8 == n_m1 // 30
    # Each 8 samples belong to 1 day
    assert s == 8 * n_d1

    # Find indices of raw_d1 where time <= sample_time
    d1_ix = raw_data['d1:time'].searchsorted(sample_time, side='right') - 1
    # Find indices of raw_m1 where time <= sample_time
    m1_ix = raw_data['m1:time'].searchsorted(sample_time, side='right') - 1
    # Find indices of raw_m5 where time <= sample_time
    m5_ix = raw_data['m5:time'].searchsorted(sample_time, side='right') - 1

    # Here, we use a clever and efficient method to determine the covar_ix for each sample.
    # Since there are exactly 8 data points per day, we can compute day_ix directly,
    # which also serves as covar_ix, by repeating the ascending order.
    # This approach works because we have already addressed the causality
    # of the covariance data.
    covar_ix = day_ix = repeat(arange(n_d1), 8)

    # We can further verify this by directly comparing the search-sorted data indices
    # with the indices obtained using our optimized method.
    # While these approaches are efficient, they rely on the assumption
    # that all data points follow strict, regular patterns without edge cases.
    assert all(d1_ix == day_ix - 1)
    assert all(m1_ix == arange(-1, n_m1-1)[::per_1m])
    assert all(m5_ix == arange(-1, n_m5-1)[::per_5m])

    # Drop first 8 samples where d1_ix is -1.
    bix = 8
    # Drop last 16 samples which cannot get sufficient
    # data to compute the target return with horizon=2d.
    eix = s - 16

    sample_time = sample_time[bix:eix]
    m1_ix = m1_ix[bix:eix]
    m5_ix = m5_ix[bix:eix]
    d1_ix = d1_ix[bix:eix]
    covar_ix = covar_ix[bix:eix]

    # Update sample size
    s = len(sample_time)

    # Calculate target returns with different horizons
    close_m1 = raw_data['m1:close']

    p0 = log(close_m1[m1_ix])

    p30m = log(close_m1[m1_ix+30])
    p1h = log(close_m1[m1_ix+60])
    p4h = log(close_m1[m1_ix+240])
    p8h = log(close_m1[m1_ix+480])

    # Calculate targets with horizon=30m,2h,4h(1d),8h(2d)
    tgt_ret = empty((s, 3, 4), dtype=float)
    tgt_ret[:, :, 0] = p30m - p0
    tgt_ret[:, :, 1] = p1h - p0
    tgt_ret[:, :, 2] = p4h - p0
    tgt_ret[:, :, 3] = p8h - p0

    # Validate causality
    assert all(sample_time >= raw_data['d1:time'][d1_ix])
    assert all(sample_time >= raw_data['m1:time'][m1_ix])
    assert all(sample_time >= raw_data['m5:time'][m5_ix])

    samples = empty(s, dtype=[
        ('time'     , 'datetime64[s]'),
        ('ix_d1'    , int),
        ('ix_m1'    , int),
        ('ix_m5'    , int),
        ('ix_covar' , int),
        ('y'        , float, (m, 4)),
        ('norm_y'   , float, (m, 4)),
        ('norm_y^5' , float, (m, 4)),
    ])
    samples['time'] = sample_time
    samples['ix_d1'] = d1_ix
    samples['ix_m1'] = m1_ix
    samples['ix_m5'] = m5_ix
    samples['ix_covar'] = covar_ix
    samples['y'] = tgt_ret

    sigma_4h = raw_data['covar:vol'][covar_ix]
    # `sigma` is at 1-day scale, or 4-hour scale. It should be
    # rescaled to match the timescale of the prediction horizon.
    sigma_30m = sigma_4h * sqrt(1/8)  # 4hour -> 30min
    sigma_1h  = sigma_4h * sqrt(1/4)  # 4hour -> 1hour
    sigma_8h  = sigma_4h * sqrt(2)    # 4hour -> 8hour

    sigma = stack([sigma_30m, sigma_1h, sigma_4h, sigma_8h], axis=2)
    samples['norm_y'] = ny = tgt_ret / sigma
    samples['norm_y^5'] = clip(ny, -5., 5.)

    # Generate quarterly rolled splits
    splits = _generate_splits(sample_time, frequency=3)

    from numpy import savez_compressed
    savez_compressed(f'{DATA_ROOT}/input/sample.npz', samples=samples, splits=splits)


class RawData:
    """
    A class for representing and manipulating financial raw data at different frequencies,
    along with its daily covariance matrix.

    This class includes three levels of market data and a daily covariance matrix.

    Users can:
    * Load raw data using `load_raw_data()`.
    * Access values using keys in the format '${grid}:${field}', e.g., `raw['m1:high']`.
    * Retrieve covariance, variance, and volatility using keys like `raw['covar:vol']`.
    """

    def __init__(self, raw_d1, raw_m1, raw_m5, covariance):
        """
        Initialize the RawData object with daily, minute, 5-minute raw data, and covariance matrix.

        :param raw_d1: Daily raw data.
        :param raw_m1: Minute raw data.
        :param raw_m5: 5-minute raw data.
        :param covariance: Covariance matrix.
        """
        self.raw_d1 = raw_d1
        self.raw_m1 = raw_m1
        self.raw_m5 = raw_m5
        self.covariance = covariance

        from numpy import diag, empty
        n, m, _ = covariance.shape
        self.variance = var = empty((n, m), dtype=float)
        for j in range(n):
            var[j] = diag(covariance[j])

    def __reduce__(self):
        """
        Support pickling of the RawData object.

        :return: Tuple containing class type and constructor arguments.
        """
        return type(self), (self.raw_d1, self.raw_m1, self.raw_m5, self.covariance)

    def __getitem__(self, key):
        """
        Retrieves the specified field from the corresponding raw grid data.

        Examples:
        >>> raw = load_raw_data()
        >>> raw['d1:open']       # (N_d1, M)
        >>> raw['m5:high']       # (N_m5, M)
        >>> raw['m1:time']       # (N_m1, M)
        * special cases for covariance/variance/volatility
        >>> raw['covar:vol']     # (N_d1, M)
        >>> raw['covar:var']     # (N_d1, M)
        >>> raw['covar:cov']     # (N_d1, M, M)
        """
        grid, field = key.split(':', 1)

        if field == 'vol':
            from numpy import sqrt
            return sqrt(self.variance)
        elif field == 'var':
            return self.variance
        elif field == 'cov':
            return self.covariance

        return getattr(self, f'raw_{grid}')[field]

def load_raw_data() -> 'RawData':
    """
    Load raw financial data from disk.

    :return: A RawData object containing loaded data.
    """
    from numpy import load
    keys = ['raw_d1', 'raw_m1', 'raw_m5', 'covariance']
    rv = {}
    with load(f'{DATA_ROOT}/input/cnIXfuts_raw.npz') as data:
        for k in keys:
            rv[k] = data[k]
    return RawData(**rv)


class SampleData:
    """
    A class for representing and managing preprocessed sample data and its splits.

    * `samples` contains indices mapping to `raw_data`, along with labels for `y` and `norm_y` across four horizons.
    * `splits` stores a series of training and testing dataset pairs, updated quarterly.

    `SampleData` is a core data structure for feature computation, model training, and evaluation.

    It supports two primary use cases:
    1. Loading raw data and upstream features to compute new features.
    2. Loading predefined features and generating train/test datasets.
    """
    def __init__(self, samples, splits):
        """
        Initialize the SampleData object with preprocessed samples and split information.

        :param samples: Structured array of preprocessed samples.
        :param splits: Splits of the train-test pairs.
        """
        self.samples = samples
        self.splits = splits

        self.raw_data = None

        self.feature_value = None
        self.feature_keys = None
        self.feature_key_index = {}

    def __len__(self):
        return len(self.samples)

    @property
    def label_shape(self):
        return self.samples['y'].shape

    def set_raw_data(self, raw_data: 'RawData'):
        """
        Set raw data to the SampleData object.

        :param raw_data: RawData object containing raw financial data.
        """
        self.raw_data = raw_data

    def attach_features(self, value, feature_keys):
        """
        Attach feature values and keys to the SampleData object.

        If features already exist, the newly added feature keys and value
        will be appended to the existing features.

        :param value: Feature values.
        :param feature_keys: Feature keys corresponding to the feature values.
        """
        if self.feature_value is None:
            self.feature_value = value
            self.feature_keys = tuple(feature_keys)
        else:
            from numpy import concatenate
            self.feature_value = concatenate((self.feature_value, value), axis=2)
            self.feature_keys = self.feature_keys + tuple(feature_keys)

        # Update feature key index
        self.feature_key_index = {k: j for j, k in enumerate(self.feature_keys)}
        assert len(self.feature_value) == len(self)

    def __getitem__(self, key):
        """
        Retrieve data based on the provided key.

        If the key corresponds to a feature, return the feature values.
        Otherwise, parse the key to retrieve raw data fields.

        All retreved data will be aligned to the sample grids.

        Here are some examples:
        >>> sample = load_sample_data()
        >>> sample.set_raw_data(load_raw_data())
        >>> v, keys = load_feature_data(['nret'])
        >>> sample.attach_features(v, keys)

        * value of feature `nret:1m:vol`
        >>> sample['nret:1m:vol']
        * value of feature `mock^5`
        >>> sample['mock^5']
        * value of raw_d1 `high`
        >>> sample['d1:high']
        * value of raw_m1 `low`
        >>> sample['m1:low']
        * volatility calculated from covariance matrix
        >>> sample['covar:vol']
        * covariance for each sample
        >>> sample['covar:cov']

        :param key: Key for retrieving data.
        :return: Retrieved data.
        """
        if key in self.feature_key_index:
            return self.feature_value[:, :, self.feature_key_index[key]]

        if self.raw_data is None:
            raise KeyError('Please set_raw_data first.')

        raw = self.raw_data[key]
        ix = self.raw_data_indices(grid=key.split(':', 1)[0])
        return raw[ix]

    def raw_data_indices(self, grid):
        """
        Get indices of the given grid raw data for each sample.

        :param grid: 'd1', 'm1', or 'm5'
        :return:
        """
        return self.samples[f'ix_{grid}']


    _Y_FIELD = {
        'raw': 'y',
        'norm': 'norm_y',
        'norm5': 'norm_y^5',
    }

    def dataset(self, train_size=None, y_type='norm5', mode='in-sample'):
        """
        Generates datasets based on the specified mode and training set size.

        :param train_size: The size of the training set. If None, use the default size.
        :param y_type: Which type of y is used, which can be 'raw', 'norm', 'norm5'
        :param mode: The dataset mode, which can be 'in-sample', 'out-sample', or 'all'.

        :return: generator providing (test0_ix, trainX, trainY, testX, testY)
        """
        # Ensure the mode parameter is valid
        assert mode in ('in-sample', 'out-sample', 'all')
        assert y_type in ('raw', 'norm', 'norm5')

        # Retrieve the dataset splits information
        splits = self.splits

        # Filter the dataset based on the specified mode
        if mode == 'in-sample':
            splits = splits[~splits['out-sample']]
        elif mode == 'out-sample':
            splits = splits[splits['out-sample']]

        # Retrieve feature and label data
        x = self.feature_value
        y = self.samples[self._Y_FIELD[y_type]]

        # Extract indices for training and testing sets
        train_ix = splits['train_end_ix']
        test_ix = splits['test_end_ix']

        # Iterate over each dataset split to generate training and testing sets
        for j in range(len(splits)):
            ix1 = train_ix[j]
            # Calculate the starting index for the training set if a size is specified
            ix0 = 0 if train_size is None else max(0, ix1 - train_size)
            ix2 = test_ix[j]

            # Slice the feature and label data for training and testing sets
            train_x = x[ix0:ix1]
            train_y = y[ix0:ix1]
            test_x = x[ix1:ix2]
            test_y = y[ix1:ix2]

            # Yield the training and testing sets
            yield ix1, train_x, train_y, test_x, test_y


def load_sample_data():
    """
    Load preprocessed sample data from disk.

    :return: A SampleData object containing loaded samples and split information.
    """
    from numpy import load

    with load(f'{DATA_ROOT}/input/sample.npz') as data:
        samples, splits = data['samples'], data['splits']
    return SampleData(samples, splits)


def load_feature_data(feature_names):
    """
    Load feature data from disk based on the provided feature names.

    :param feature_names: List of feature names to load.
    :return: Tuple containing concatenated feature values and combined feature keys.
    """
    from numpy import concatenate, load
    values = []
    all_keys = []

    n, m, f = None, None, None

    for fname in feature_names:
        with load(f'{DATA_ROOT}/feature/{fname}.npz') as data:
            value, keys = data['value'], data['feature_keys']
        if n is None:
            n, m, f = value.shape
            assert len(keys) == f
        else:
            n1, m1, f1 = value.shape
            assert n == n1 and m == m1
            assert len(keys) == f1
            f += f1

        values.append(value)
        all_keys.extend(keys)

    values = concatenate(values, axis=2)
    assert values.shape == (n, m, f)
    return values, tuple(all_keys)


class PredictionData:
    __slots__ = ['time', 'position_pred', 'norm_ret_pred', 'norm_ret_label', 'ret_pred', 'ret_label']

    def __init__(
            self, time, position_pred=None,
            norm_ret_pred=None, norm_ret_label=None,
            ret_pred=None, ret_label=None):
        self.time = time
        self.position_pred = position_pred
        self.norm_ret_pred = norm_ret_pred
        self.norm_ret_label = norm_ret_label
        self.ret_pred = ret_pred
        self.ret_label = ret_label

    def __reduce__(self):
        return self.__class__, (
            self.time,
            self.position_pred,
            self.norm_ret_pred,
            self.norm_ret_label,
            self.ret_pred,
            self.ret_label,
        )

    @property
    def RMSE(self):
        from sklearn.metrics import mean_squared_error
        y_true = self.norm_ret_label.ravel()
        y_pred = self.norm_ret_pred.ravel()
        return mean_squared_error(y_true, y_pred)

    @property
    def FvR(self):
        return (self.norm_ret_pred * self.norm_ret_label).cumsum(axis=0)
