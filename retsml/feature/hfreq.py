from .base import FeatureBuilder, feature


@feature(name='hfreq')
class HighFreqFeatureBuilder(FeatureBuilder):
    def __init__(self):
        # 1. volatility
        # 2. bull percentage
        # 3. agreement
        # 4. return to VWAP
        keys = ['sigma_1m/1g', 'bull_1m/1g', 'agr_1m/1g', 'vwr_1m/1g']
        super().__init__(keys, upstreams=None, use_raw_data=True)

    def _hfreq(self, bar):
        # 1. volatility
        from numpy import log
        S = len(bar)
        logr = log(bar['close']) - log(bar['open'])    # (n, m)
        vol_1m = logr.std(axis=0)                      # (m,)

        # 2. bull percentage
        bull_sgn = bar['close'] > bar['open']          # (n, m)
        bull_perc = (bull_sgn.sum(axis=0) / S - 0.5) * 2 # (m,)

        # 3. agreement
        size = bar['volume']                           # (n, m)
        sgn_vol = (size * bull_sgn).sum(axis=0)        # (m,)
        sum_vol = size.sum(axis=0) + 1                 # (m,)
        agr = sgn_vol / sum_vol                        # (m,)

        # 4. VWAP
        w = size / sum_vol[None, :]                    # (n, m)
        avgp = (bar['close'] + bar['open'] + bar['high'] + bar['low']) / 4
        vwap = (avgp * w).sum(axis=0)                  # (m,)
        err_ix = vwap < 1e-10
        if err_ix.any():
            vwap[err_ix] = avgp[:, err_ix].mean(axis=0)
        return vol_1m, bull_perc, agr, vwap

    def values(self, input_data):
        rv = self.zero_array(input_data)

        from numpy import log, zeros
        data_1m = input_data.raw_data.raw_m1
        close_1g = log(input_data['m1:close'])

        N, M, _ = rv.shape
        ixs = input_data.raw_data_indices('m1')
        for i in range(N):
            eix = ixs[i]+1  # eix is not inclusive
            bar = data_1m[eix-30:eix]
            fv = self._hfreq(bar)
            rv[i, :, 0] = fv[0]  # volatility
            rv[i, :, 1] = fv[1]  # bull percentage
            rv[i, :, 2] = fv[2]  # agreement
            rv[i, :, 3] = close_1g[i] - log(fv[3])  # logret to VWAP

        return rv
