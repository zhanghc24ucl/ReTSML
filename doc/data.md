## Input Data

* `N`: the data length
* `M`: the number of instruments
    * `M=3` with `['IH', 'IF', 'IC']`
* `S`: the size of samples
    * Sampled every 30 minutes.
* `K`: the size of data splits
    * Each data split will be trained as a model.
    * Used for cross-validation and model calibration.

### Raw

**`input/cnIXfuts_raw.npz` including `raw_d1`, `raw_m5`, `raw_m1` and `covariance`.**

* `raw_${datatype}` is a `numpy.recarray` with fields:

| field         | dtype       | shape    |
|---------------| ----------- | -------- |
| time          | datetime[s] | `(N,)`   |
| open          | float       | `(N, M)` |
| high          | float       | `(N, M)` |
| low           | float       | `(N, M)` |
| close         | float       | `(N, M)` |
| value         | float       | `(N, M)` |
| volume        | float       | `(N, M)` |
| open_interest | float       | `(N, M)` |
| log_ret       | float       | `(N, M)` |
| norm_log_ret  | float       | `(N, M)` |

* `covariance` is a 3-dimension `numpy.array` with shape of `(N, M, M)`.
    * `(M, M)` covariance matrix of the instruments for each day.
    * The covariance is available before the beginning of the corresponding day.
* `log_ret` and `norm_log_ret` are calculated as `close-to-close`
    * `norm_log_ret = log_ret / scaled_sigma`
    * `log_ret` is addable

### Sample

**`input/sample.npz` including `samples` and `splits`.**

* `samples` is a `numpy.recarray` with fields:

| field    | dtype       | shape       |                                                             |
|----------| ----------- | ----------- |-------------------------------------------------------------|
| time     | datetime[s] | `(S,)`      |                                                             |
| ix_d1    | int         | `(S,)`      | effective `raw_d1` indices                                  |
| ix_m1    | int         | `(S,)`      | effective `raw_m1` indices                                  |
| ix_m5    | int         | `(S,)`      | effective `raw_m5` indices                                  |
| ix_covar | int         | `(S,)`      | effective `covariance` indices                              |
| y        | float       | `(S, M, 4)` | log return of `horizon=30m, 1h, 4h, 8h` for each instrument |
| norm_y   | float       | `(S, M, 4)` | normalized log return                                       |
| norm_y^5 | float       | `(S, M, 4)` | normalized log return clipped to [-5, 5]                    |

* `splits` is a `numpy.recarray` with fields:

| field        | dtype | shape  | description                                                  |
| ------------ | ----- | ------ |--------------------------------------------------------------|
| train_end_ix | int   | `(K,)` | `samples[train_end_ix-train_size:train_end_ix]` as train set |
| test_end_ix  | int   | `(K,)` | `samples[train_end_ix:test_end_ix]` as test set              |
| out-sample   | bool  | `(K,)` | `true` for out-sample and `false` for in-sample              |

## Feature Data

**`feature/${name}.npz` including `value` and `feature_keys`.**

* `value` is a float `numpy.array` with size of `(S, M, F)`:
    * Calculated for all samples and instruments.
    * `F` is the number of features.
* `feature_keys` is a `unicode/str` `numpy.array` with size of `(F,)`.
* Please refer to `feature.md` for more details of building features.

## Data Stats

* **size**

| key                                                       | value  |
|-----------------------------------------------------------|--------|
| `N_d1`: data length of `raw_d1`, i.e. how many trade days | 2223   |
| `N_m1`: data length of `raw_m1`, and `N_m1=N_d1*240`      | 533520 |
| `N_m5`: data length of `raw_m5`, and `N_m5=N_d1*240/5`    | 106704 |
| `S`: total sample size                                    | 17760  |

* **splits**

| split_id | train_end_ix | train_end_date | test_end_ix | out-sample |
|----------|--------------|----------------|-------------|------------|
| 0        | 3896         | 2017-12-29     | 4368        | 0          |
| 1        | 4368         | 2018-03-30     | 4848        | 0          |
| ...      |              |                |             | ...        |
| 14       | 10680        | 2021-06-30     | 11192       | 0          |
| 15       | 11192        | 2021-09-30     | 11680       | 0          |
| 16       | 11680        | 2021-12-31     | 12144       | 1          |
| 17       | 12144        | 2022-03-31     | 12616       | 1          |
| ...      |              |                |             | ...        |
| 27       | 17000        | 2024-09-30     | 17488       | 1          |
| 28       | 17488        | 2024-12-31     | 17760       | 1          |

* In-sample dataset contains 16 splits(quarters).
* Out-sample dataset contains 13 splits(quarters).
