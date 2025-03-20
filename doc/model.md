## Configurations

There are 3 different type of configurations: `model_cfg`, `model_cfg_grid` and `search_cfg`
* All these configs are indexed by a symbol.
* `model_cfg` is a specified model config.
* `model_cfg_grid` is consist of a group of potential hyperparameters.
  * Each hyperparameter has a specific `model_cfg`.
  * `model_cfg_grid` is actual a list of `model_cfg`.
* `search_cfg` is a config for tuning, including:
  * How to divide splits into different folds.
  * Which evaluation metric is used.
  * How to search the best parameters.

## General tasks

### Tune hyperparameters

```bash
run model $name $model_cfg_grid tuning $search_cfg
```

1. Train models of all hyperparameters on all splits of the in-sample data.
2. Test all models and assemble the results according to the specified fold aggregating method.
3. Evaluate results of all folds.
4. Find the best hyperparameters according to the search config.
5. Save the selected hyperparameters.

### Train and test models

```bash
run model $name $model_task_cfg prediction
```

1. `model_task_cfg` actually can be either a `model_cfg_grid` or a `model_cfg`.
    * If it is a `model_cfg_grid`, then the tuned `model_cfg` will be used.
        * If no tuned result is found, the task will be failed.
    * If it is a `model_cfg`, then the `model_cfg` will be used.
2. Use the `model_cfg` to train all models on the whole data, including in-sample and out-sample data.
3. Test all corresponding models on all the splits and get prediction results.
4. Assemble all splits' prediction results into a full sequence.
    * The result sequence has the same length as the sample data.
    * There are 3 types of results:
        * Normalized return sequence corresponding to different horizons.
        * Position sequence for each sample.
    * The prediction is consist of 3 keys:
        * `ret`, `norm_ret` or `position` 
            * `(#size, #instr, #horizon)` for `ret` and `norm_ret`
            * `(#size, #instr)` for `position`
        * `norm_ret` is recommended for all types of evaluation.
        * Any key could be None.
5. Save the prediction sequence indexed by the `model_name` and `model_cfg`

### Storage

* **Model**: store at `$DATA_ROOT/model/${model_name}_${model_cfg}_${split_id}.bin`.
* **Tuned**: store at `$DATA_ROOT/model/tuned_${model_name}_${model_cfg_grid}.txt`.
    * Just a text-file with string of `model_cfg` 
* **Prediction**: store at `$DATA_ROOT/prediction/${model_name}_${model_cfg}.npz`
    * Merged all splits and should with the same size of `sample_data`

### Evaluation

Evaluation is conducted with the `sample_data` and `test_results`.

| Metric | ResultType            | SampleData Field |
|--------|-----------------------|------------------|
| RMSE   | norm_ret              | norm_y           |
| R2     | norm_ret              | norm_y           |
| FvR    | norm_ret              | norm_y           |
| Sharpe | ret/norm_ret/position | y                |

**RMSE**

$$
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
$$

**R2**

$$
\begin{split}
R^2&=1-\frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}\\
\bar{y}&=\frac{1}{n}\sum_{i=1}^{N}y_i
\end{split}
$$

**FvR**

$$
\text{FvR}=\sum_{i=1}^{N}y_i\hat{y}_i
$$

* $y_i$ and $\hat{y}_i$ are the normalized return for the grid
* Both of them should be with `horizon=30m`, which is the same value as the sample span.
    * If `horizon` does not match, the result should be scaled, but still could be wrong.
* **FvR** is an approximate measure of the total profit.
    * Optimized position with expected return $f_i$ is $p_i=f_i\sigma_i^{-2}$ according to a simple MPT model.
    * Profit of a single sample is $p_i r_i=f_i r_i \sigma_i^{-2}$.
    * Normalized return is $y_i=r_i \sigma^{-1}$ and normalized prediction is $\hat{y}_i=f_i \sigma^{-1}$.
    * Approximate profit of a single sample can be calculated as $y_i\hat{y}_i=f_ir_i\sigma_i^{-2}$.

**Sharpe**

$$
\begin{split}
x_i&=p_i r_i-(p_i-p_{i-1})\delta\\
\text{Sharpe}&=\frac{\mathbb{E}[x]}{\sigma_{x}}
\end{split}
$$

* $p_i$ is the position (in notional value).
    * It can be calculated from the expected return by portfolio optimization.
* $r_i$ is the sample return.
    * It can be calculated from the log return of the `sample_data`.
* $\delta$ is the trade cost.
