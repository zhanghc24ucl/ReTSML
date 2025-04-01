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
bin/ReTSML --help

# run ETL task
bin/ReTSML ETL

# build `ret` and `time` features
bin/ReTSML feature ret,time

# train all models under config_grid and tune best config
bin/ReTSML tune GBM gbm --metric=RMSE

# evaluate models
# FIXME: not implemented yet
bin/ReTSML evaluate GBM gbm --dataset=all
```

## Data folder

The default data folder is `data`, with a structure like:
```bash
$ tree data
data
├── feature
│   ├── ${feature_name}.npz
├── input
│   ├── cnIXfuts_raw.npz
│   └── sample.npz
├── model
│   ├── ${model_name}
│   │   ├── ${config_grid}_${split_id}.mod
│   └── tuned_${model_name}_${config_grid}_${metric}.bin
└── prediction
```

For a newly cloned workspace, it is recommended to create an empty data folder:
```bash
$ mkdir data data/input data/feature data/model data/prediction
$ tree data
data
├── feature
├── input
├── model
└── prediction
```

Other data folders are supported, with environment variables `DATA_ROOT`.
For example, if another data folder located at `/content/drive/$data_path` is
used, one should set the environment variable `DATA_ROOT=/content/drive/$data_path`
while running scripts.

## Documents

Please refer to 
* `doc/data.md`: data details and specifications.
* `doc/feature.md`: how to implement features (Not implemented yet).
* `doc/model.md`: how models are defined and organized.
