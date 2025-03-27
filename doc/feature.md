## Feature Description

### Return

code: `retsml.feature.ret`

feature_name: `ret`

feature_size: 13

feature_keys:
* `nret:1g^5`: normalized log return of the current grid (30 minutes), clipped within `[-5, 5]`.
    * `ix=0` 
* `nret:5m_${lag}^5`: normalized 5-minute log return of the current grid, clipped within `[-5, 5]`
    * `ix=1,2,3,4,5,6` corresponding to lags from 0 to 5.
* `nret:1d^5`: normalized log return of the last full trading day, clipped within `[-5, 5]`.
    * `ix=7`
    * If the sample grid is at T days' 10:30, the `nret:1d^5` is computed by closing prices at T-1 and T-2.
    * 'nret:1d^5' keeps the same value for a particular trading day.
* `nret:${span}^5`: normalized log return of the last `${span}` trading days, clipped within `[-5, 5]`.
    * `ix=8,9,10,11,12` corresponding to spans from `5d, 20d, 60d, 120d, 240d`
    * The spans correspond to "a week", "a month", "a quanter", "half a year", and "a year", respectively.

### Lagged Return

code: `retsml.feature.ret`

feature_name: `lag_ret`

feature_size: 61

feature_keys:
* `nret:1g^5[$lag]` with lag from 1 to 15.
    * Cover the last 2 days with 16*30=480 minutes.
    * `ix=0~14`, 15 features
* `nret:5m_$n^5[$lag]` with `lag=[6, 12, 18, 24, 30, 36, 42]`
    * `n=[0,1,2,3,4,5]`, and `n+$lag` is the actual lag of the 5m log return.
    * Cover the last 1 day with 8*6*5=240 minutes.
    * `ix=15~56`, 7*6=42 features
* `nret:1d^5[$lag]` with lag from 1 to 4.
    * Cover the last 5 days
    * `ix=57~60`, 4 features

### Time

code: `retsml.feature.time`

feature_name: `time`

feature_keys:
* `time:DoM`:`ix=0`: day of month
* `time:MoY`:`ix=1`: month of year
* `time:DoW`:`ix=2`: weekday
* `time:IoD`:`ix=3`: intraday index of day
    * Value from 0 to 7, which is tricky that there are fixed 8 samples per day.
* `time:BoD`:`ix=4`: 1 if it is the beginning of day, 0 otherwise.
* `time:EoD`:`ix=5`: 1 if it is the end of day, 0 otherwise.