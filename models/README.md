# Folder containing various sized Branch-ELM models pre-trained on NeuronIO:

1) For an overview of each models performance see __eval_results.csv__.
2) For a script to reproduce these results see __/notebooks/neuronio_train_script.py__.

## Provided files per model:
- __neuronio_best_model_state.pt__: the serialized model weights
- __eval_results.json__: the train, valid and test set results
- __model_config.pt__: settings to load model weights
- __train_config.pt__: setting to reproduce results
- __model_stats.pt__: info such as number of parameters

## Evaluate or infer using a pre-trained model:
=> __/notebooks/eval_elm_on_neuronio.ipynb__

## Run your own ELM training on NeuronIO:
=> __/notebooks/train_elm_on_neuronio.ipynb__
