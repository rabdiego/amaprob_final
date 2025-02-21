import optuna
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split
from src.models.vae import DenseVAE, Conv1DVAE, LSTMVAE
from src.utils.metrics import VAELoss
from src.utils.utils import train

# Espaço de busca dos hiperparâmetros pros 3 modelos
hyperparam_spaces = {
    "dense":{
        "num_layers": lambda trial: trial.suggest_categorical("num_layers", [2, 3, 4, 5]),
        "latent_dim": lambda trial: trial.suggest_categorical("latent_dim", [2, 5, 10, 20]),
        "input_neurons": lambda trial: trial.suggest_categorical("input_neurons", [1000, 750, 500]),
        "output_neurons": lambda trial: trial.suggest_categorical("output_neurons", [50, 100, 200])
    },
    "conv":{
        "num_conv_layers": lambda trial: trial.suggest_categorical("num_conv_layers", [1, 2, 3]),
        "num_dense_layers": lambda trial: trial.suggest_categorical("num_dense_layers", [1, 2, 3]),
        "latent_dim": lambda trial: trial.suggest_categorical("latent_dim", [2, 5, 10]),
        "input_neurons": lambda trial: trial.suggest_categorical("input_neurons", [500, 200]),
        "output_neurons": lambda trial: trial.suggest_categorical("output_neurons", [50, 100]),
        "initial_channels": lambda trial: trial.suggest_categorical("initial_channels", [2, 4]),
        "factor": lambda trial: trial.suggest_categorical("factor", [2, 3])
    },
    "lstm":{
        "num_layers": lambda trial: trial.suggest_categorical("num_layers", [1, 2, 3]),
        "latent_dim": lambda trial: trial.suggest_categorical("latent_dim", [2, 5]),
        "output_neurons": lambda trial: trial.suggest_categorical("output_neurons", [50, 100, 200]),
        "middle_ground": lambda trial: trial.suggest_categorical("middle_ground", [75, 100, 150])
    }
}

def objective(trial, model_name, n_epochs, dataset, device):
    """Função objetivo a ser maximizada (ou função custo a ser minimizada) pelo conjunto de hiperparâmetros

    Args:
        trial (trial): trial
        model_name (str): Nome do modelo
        n_epochs (int): Número de épocas
        dataset (_type_): Dataset avaliado
        device (_type_): Dispositivo a ser considerado

    Returns:
        float: Valor da função custo
    """
    # Inicialização do modelo e dos seus candidatos a hiperparâmetros
    model_hyperparams = {hyperparam:candidate(trial) for hyperparam, candidate in hyperparam_spaces[model_name].items()}
    model_hyperparams["device"] = device
    model_hyperparams["original_size"] = dataset[0].shape[0]
    if model_name == "dense":
        model = DenseVAE(**model_hyperparams).to(device)
    elif model_name == "conv":
        model_hyperparams["original_size"] = dataset[0].shape[0]
        model = Conv1DVAE(**model_hyperparams).to(device)
    elif model_name == "lstm":
        model = LSTMVAE(**model_hyperparams).to(device)

    # Inicialização do otimizador
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # Inicialização dos dataloaders
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    batch_size = 32
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Cálculo da função de custo (métrica que se deseja minimizar)
    loss_function = VAELoss()
    loss_curve = train(model, train_loader, test_loader, loss_function, optimizer, epochs=n_epochs, device=device)
    return min(loss_curve[1])