# Reproduce Results

0. Setup a conda environment using the conda_env_rashomon.yml
1. Use a run_* script to start training, eg ```./run_small.sh``` to obtain models and compute explanations on all tabular datasets
   - this calls ```python 1_train_models_collect_explanations.py``` with command line args specified in script
   - this produces a subfolder 'data' containing models, data about the training and the explanations
   - everything is seeded rigorously and with the provided seeds you should be able to replicate results to the digit
2. Run ```python 2_assess_hyperparameters.py``` to produce results for numerical stability
   - it accesses _variables.py and will compute the evaluation for all tasks, methods and explanation methods listed in variables tasks, explanation_abbreviations and metric_names
   - for AG News this took several days on 14 cores
3. Run ```python 3_calculate_evaluation.py``` to compute distances for 011 and 110
   - as the script before, it accesses _variables.py and will compute the evaluation for all tasks, methods and explanation methods listed in variables tasks, explanation_abbreviations and metric_names
4. Run ```python 4_plot_evaluation.py``` calculates rankings (can take long), produces plots and prints tables in tex.

