from .base import FeatureBuilder, feature


@feature(name='vol')
class VolumeFeatureBuilder(FeatureBuilder):
    def __init__(self):
        keys = ('vsurp^1g', 'vsurp^1d')
        super().__init__(keys, upstreams=None, use_raw_data='True')

    def values(self, input_data):
        rv = self.zero_array(input_data)
        from numpy import log
        from .func import ewm
        vol_1g = log(input_data['m1:volume']+1)
        vol_ewm = ewm(vol_1g, span=24)  # 3d
        rv[:, :, 0] = vol_1g - vol_ewm

        raw_vol_1d = log(input_data.raw_data['d1:volume']+1)
        raw_vol_ewm = ewm(raw_vol_1d, span=20)
        d1_ix = input_data.raw_data_indices('d1')
        rv[:, :, 1] = (raw_vol_1d - raw_vol_ewm)[d1_ix]
        return rv
