from .base import FeatureBuilder, feature


@feature(name='ret')
class ReturnFeatureBuilder(FeatureBuilder):
    def __init__(self):
        r1g_keys = ('nret:1g^5',)

        # 6 lags of 5m for the 30m sample grid
        self.r5m_size = 6
        r5m_keys = tuple(f'nret:5m_{k}^5' for k in range(self.r5m_size))

        self.retNd_spans = [1, 5, 20, 60, 120, 240]
        rNd_keys = tuple(f'nret:{i}d^5' for i in self.retNd_spans)

        # total 13 features
        keys = r1g_keys + r5m_keys + rNd_keys
        super().__init__(keys, upstreams=None, use_raw_data=True)

    def values(self, input_data):
        rv = self.zero_array(input_data)

        from numpy import sqrt, log, clip

        column_ptr = 0
        sigma_1d = input_data['covar:vol']

        # nret:1g^5
        # log return for the sample grid (30 minutes), normalized and clipped.
        sigma_1g = sigma_1d * sqrt(1./8)  # 30 minutes for 1 grid
        close_1g = log(input_data['m1:close'])
        logr_1g = (close_1g[1:] - close_1g[:-1]) / sigma_1g[1:]
        logr_1g = clip(logr_1g, -5., 5.)
        rv[1:, :, column_ptr] = logr_1g
        column_ptr += 1

        # nret:5m_i^5
        # lagged log return for every 5 minutes, normalized and clipped.
        raw_m5_log_return = input_data.raw_data['m5:norm_log_ret']
        raw_m5_log_return = clip(raw_m5_log_return, -5., 5.)
        m5_ix = input_data.raw_data_indices('m5')
        for i in range(self.r5m_size):
            rv[:, :, column_ptr] = raw_m5_log_return[clip(m5_ix-i, 0, None)]
            column_ptr += 1

        # nret:{span}d^5
        raw_close_1d = log(input_data.raw_data['d1:close'])
        d1_ix = input_data.raw_data_indices('d1')
        close_1d = raw_close_1d[d1_ix]
        for span in self.retNd_spans:
            dN_ix = clip(d1_ix - span, 0, None)
            logr = close_1d - raw_close_1d[dN_ix]
            sigma_Nd = sigma_1d * sqrt(span) if span > 1 else sigma_1d
            nret = logr / sigma_Nd
            rv[:, :, column_ptr] = clip(nret, -5., 5.)
            column_ptr += 1

        assert column_ptr == rv.shape[-1]
        return rv


@feature(name='lag_ret')
class LaggedReturnFeatureBuilder(FeatureBuilder):
    def __init__(self):
        self.r1g_lags = 16  # 16*30=480m, 8h, 2d
        r1g_keys = tuple(f'nret:1g^5[{n}]' for n in range(1, self.r1g_lags))

        self.r5m_size = 6
        self.r5m_lags = 8
        r5m_keys = tuple(f'nret:5m_{k}^5[{6*n}]'
                         for k in range(self.r5m_size)
                         for n in range(1, self.r5m_lags))

        self.r1d_lags = 5
        rNd_keys = tuple(f'nret:1d^5[{n}]' for n in range(1, self.r1d_lags))

        # total 61 features
        keys = r1g_keys + r5m_keys + rNd_keys
        super().__init__(keys, upstreams=['ret'], use_raw_data=False)

    def values(self, input_data):
        rv = self.zero_array(input_data)

        column_ptr = 0

        raw1g = input_data['nret:1g^5']
        for i in range(1, self.r1g_lags):
            rv[i:, :, column_ptr] = raw1g[:-i, :]
            column_ptr += 1

        for j in range(self.r5m_size):
            raw5mj = input_data[f'nret:5m_{j}^5']
            for i in range(1, self.r5m_lags):
                rv[i:, :, column_ptr] = raw5mj[:-i, :]
                column_ptr += 1

        raw1d = input_data['nret:1d^5']
        for i in range(1, self.r1d_lags):
            rv[i:, :, column_ptr] = raw1d[:-i, :]
            column_ptr += 1

        assert column_ptr == rv.shape[-1]
        return rv
