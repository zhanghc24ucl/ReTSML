from .base import FeatureBuilder, feature


@feature(name='ret')
class ReturnFeatureBuilder(FeatureBuilder):
    def __init__(self):
        self.ret1g_lags = 16  # 16*30m, 2d
        r1g_keys = tuple(f'nret:1g^5[{i}]' for i in range(self.ret1g_lags))

        self.ret5m_lags = 48  # 48*5m, 1d
        r5m_keys = tuple(f'nret:5m^5[{i}]' for i in range(self.ret5m_lags))

        self.ret1d_lags = 15  # 15d
        r1d_keys = tuple(f'nret:1d^5[{i}]' for i in range(self.ret1d_lags))

        self.retNd_spans = [2, 5, 20, 60, 120, 240]
        rNd_keys = tuple(f'nret:{i}d^5' for i in self.retNd_spans)

        keys = r1g_keys + r5m_keys + r1d_keys + rNd_keys
        super().__init__(keys, None, True)


    def values(self, input_data):
        rv = self.zero_array(input_data)

        from numpy import sqrt, log, clip

        column_ptr = 0
        sigma_1d = input_data['covar:vol']

        # nret:1g^5[i]
        # lagged log return for each sample (30 minutes), normalized and clipped.
        sigma_1g = sigma_1d * sqrt(1./8)  # 30 minutes for 1 grid
        close_1g = log(input_data['m1:close'])
        logr_1g = (close_1g[1:] - close_1g[:-1]) / sigma_1g[1:]
        logr_1g = clip(logr_1g, -5., 5.)
        for i in range(1, self.ret1g_lags+1):
            rv[i:, :, column_ptr] = logr_1g[i-1:]
            column_ptr += 1

        # nret:5m^5[i]
        # lagged log return for every 5 minutes, normalized and clipped.
        raw_m5_log_return = input_data.raw_data['m5:norm_log_ret']
        raw_m5_log_return = clip(raw_m5_log_return, -5., 5.)
        m5_ix = input_data.raw_data_indices('m5')
        for i in range(self.ret5m_lags):
            rv[:, :, column_ptr] = raw_m5_log_return[clip(m5_ix-i, 0, None)]
            column_ptr += 1

        # nret:1d^5[i]
        # lagged daily log return, normalized and clipped.
        raw_d1_log_return = input_data.raw_data['d1:norm_log_ret']
        raw_d1_log_return = clip(raw_d1_log_return, -5., 5.)
        d1_ix = input_data.raw_data_indices('d1')
        for i in range(self.ret1d_lags):
            rv[:, :, column_ptr] = raw_d1_log_return[clip(d1_ix-i, 0, None)]
            column_ptr += 1

        # nret:{i}d^5
        raw_close_1d = log(input_data.raw_data['d1:close'])
        # d1_ix = input_data.raw_data_indices('d1')
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
