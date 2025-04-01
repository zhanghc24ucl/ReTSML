from .base import FeatureBuilder, feature


@feature(name='time')
class TimeFBuilder(FeatureBuilder):
    def __init__(self):
        keys = (
            'time:DoM', # day of month
            'time:MoY', # month of year
            'time:DoW', # weekday
            'time:IoD', # intraday index of day, from 0 to 7
            'time:BoD', # is beginning grid of the day
            'time:EoD', # is end grid of the day
        )
        super().__init__(keys, None, False, True)

    def values(self, input_data):
        from datetime import datetime
        rv = self.zero_array(input_data)
        sample_time = input_data.samples['time']
        for i in range(len(sample_time)):
            t = sample_time[i].astype(datetime)
            rv[i, 0] = t.day / 31.      # day of month
            rv[i, 1] = t.month / 12.    # month of year
            rv[i, 2] = t.weekday() / 7. # weekday

            # This is tricky since we know there are 8 samples for each day
            ix_of_day = i % 8
            rv[i, 3] = ix_of_day / 8.   # index of day
            rv[i, 4] = ix_of_day == 0   # is first grid
            rv[i, 5] = ix_of_day == 7   # is last grid
        return rv
