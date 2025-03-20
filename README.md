# ReTSML

## How to use

`ReTSML` command is used to complete all predefined work flows including ETL, model 
training, hyperparameter tuning and model evaluation.

Here is a typical work flow:

1. The input data `cnIXfuts_raw.npz` should be downloaded and placed in the `data/input` folder.
2. Run ETL task and get the processed sample data `data/input/sample.npz`
3. Build features with specific pre-defined configuration to `data/feature/${feature_name}.npz`
4. Train models and tune hyperparameters on the in-sample dataset, with specific pre-defined configuration set.
    * All trained models are stored in `data/model/${model_name}/${config}_${model_id}.mod`.
    * Tuned model hyperparameters and the best model config are stored in `data/model/tuned_${model_name}_${config}_${METRIC}.bin`.
5. Evaluate tuned model on in-sample or the entire datasets.

```bash
# show help and usage
./ReTSML --help

# run ETL task
./ReTSML ETL

# build `ret` and `time` features
./ReTSML feature ret,time

# train all models under config_grid and tune best config
./ReTSML tune GBM gbm --metric=RMSE

# evaluate models
# FIXME: not implemented yet
./ReTSML evaluate GBM gbm --dataset=all
```

## Documents

Please refer to 
* `doc/data.md`: data details and specifications.
* `doc/feature.md`: how to implement features (Not implemented yet).
* `doc/model.md`: how models are defined and organized.
