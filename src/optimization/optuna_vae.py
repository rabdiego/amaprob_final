import optuna
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split
from src.models.vae import DenseVAE, Conv1DVAE, LSTMVAE
from src.utils.metrics import VAELoss
from src.utils.utils import train

# Espaço de busca dos hiperparâmetros
hyperparam_spaces = {
    "dense":{
        "num_layers": lambda trial: trial.suggest_int("num_layers", 2, 4),
        "latent_dim": lambda trial: trial.suggest_int("latent_dim", 30000, 40000),
        "input_neurons": lambda trial: trial.suggest_int("input_neurons", 1, 1000),
        "output_neurons": lambda trial: trial.suggest_int("output_neurons", 1, 1000),
        "num_epochs": lambda trial: trial.suggest_categorical("num_epochs", [100])
    },
    "conv":{},
    "lstm":{}}

# Modelos
# models = {"dense":DenseVAE()}
"""
def dense_objective(trial, hyperparam_spaces["dense"]):
    model = DenseVAE(**hyperparam_spaces)
    loss = VAELoss()

    
    return evaluation_score
"""
def lstm_objective(trial, dataset, device, n_epochs):
    # Hiperparâmetro
    num_layers = trial.suggest_int("num_layers", 2, 4)
    original_size = trial.suggest_categorical("original_size", [29999])
    latent_dim = trial.suggest_int("latent_dim", 2, 40000)
    output_neurons = trial.suggest_categorical("output_neurons", [300])
    middle_ground = trial.suggest_int("middle_ground", 300, 1000)

    # Dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    batch_size = 32

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Modelo e cost function
    model = LSTMVAE(num_layers=num_layers, original_size=original_size, latent_dim=latent_dim, output_neurons=output_neurons, middle_ground=middle_ground, device=device).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_function = VAELoss()
    loss_curve = train(model, train_loader, test_loader, loss_function, optimizer, epochs=n_epochs, device=device, save_path='best_model.pth')

    return loss_curve[0][-1]
