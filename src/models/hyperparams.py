import sys
sys.path.append('..')
import pickle
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice
import matplotlib.pyplot as plt
import torch
from data.data import LoFiDataset
from optimization.optuna_vae import objective

path_to_save = r"../data/processed/best_hyperparams.pkl"
n_trials = 30
models = ["lstm"]
n_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = LoFiDataset('../../data')
print(f"Total de arquivos: {len(dataset)}")

studies = {model:None for model in models}
for model_name in studies.keys():
    print(f"Otimização do modelo {model_name} iniciada!")
    studies[model_name] = optuna.create_study(direction="minimize")
    studies[model_name].optimize(lambda trial: objective(trial, model_name, n_epochs, dataset, device), n_trials=n_trials, gc_after_trial=True)
    pruned_trials = [t for t in studies[model_name].trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in studies[model_name].trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Número de trials finalizadas: {len(studies[model_name].trials)}")
    print(f"Número de trials podadas: {len(pruned_trials)}")
    print(f"Número de trials completas: {len(complete_trials)}")
    print(f"Otimização do modelo {model_name} finalizada!")

print(studies["lstm"].best_params)

with open('best_params_lstm.txt', 'w') as f:
    f.write(f'LSTM: {studies["lstm"].best_params}\n')
