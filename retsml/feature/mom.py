from .base import FeatureBuilder, feature


@feature(name='macd')
class MACDFeatureBuilder(FeatureBuilder):
    def __init__(self):
        keys = ('macd^1g', 'macd^1d')
        super().__init__(keys, upstreams=None, use_raw_data=True)

    def _macd(self, c):
        from .func import ewm
        dif = ewm(c, span=12) - ewm(c, span=26)
        dea = ewm(dif, span=9)
        return dif, dea

    def values(self, input_data):
        rv = self.zero_array(input_data)
        from numpy import log
        close_1g = log(input_data['m1:close'])
        dif_1g, dea_1g = self._macd(close_1g)
        rv[:, :, 0] = dif_1g - dea_1g

        raw_close_1d = log(input_data.raw_data['d1:close'])
        dif_1d, dea_1d = self._macd(raw_close_1d)
        d1_ix = input_data.raw_data_indices('d1')
        rv[:, :, 1] = (dif_1d - dea_1d)[d1_ix]
        return rv
