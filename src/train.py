import sys
sys.path.append('..')
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from metrics import VAELoss
from data import LoFiDataset
from vae import DenseVAE, Conv1DVAE, LSTMVAE
from utils import train

model_name = sys.argv[1]

match model_name:
    case 'dense':
        num_layers, latent_dim, input_neurons, output_nerons, num_epochs = map(int, sys.argv[2:])
    case 'conv':
        num_conv_layers, num_dense_layers, latent_dim, input_neurons, output_neurons, initial_channels, factor, num_epochs = map(int, sys.argv[2:])
    case 'lstm':
        num_layers, latent_dim, output_neurons, middle_ground, num_epochs = map(int, sys.argv[2:])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\t[INFO] Running on {device}\n')

dataset = LoFiDataset('../data')

original_size = dataset[0].shape[0]

batch_size = 32
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

match model_name:
    case 'dense':
        model = DenseVAE(num_layers, original_size, latent_dim, input_neurons, output_nerons, device).to(device)
    case 'conv':
        model = Conv1DVAE(
            num_conv_layers,
            num_dense_layers,
            original_size,
            latent_dim,
            input_neurons,
            output_neurons,
            initial_channels,
            factor,
            device
        ).to(device)
    case 'lstm':
        model = LSTMVAE(num_layers, original_size, latent_dim, output_neurons, middle_ground, device).to(device)

print(f'\t[INFO] Model architecture:\n')
print(model)
print()

optimizer = Adam(model.parameters(), lr=1e-3)
loss_function = VAELoss()

print(f'\t[INFO] Training started')
loss_curve = train(model, train_loader, loss_function, optimizer, epochs=num_epochs, device=device, save_path=f'../models/{model_name}.pth')

plt.plot(loss_curve)
plt.title(f'Função custo - {model_name}')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.savefig(f'../plots/loss/{model_name}.png')
print(f'\n\t[INFO] Loss curve saved at \'../plots/loss/{model_name}.png\'')
