import sys
sys.path.append('..')
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.adam import Adam
from utils.metrics import VAELoss
from data.data import LoFiDataset
from vae import DenseVAE, Conv1DVAE, LSTMVAE
from utils.utils import train

model_name = sys.argv[1]
model_path = sys.argv[2]

match model_name:
    case 'dense':
        num_layers, latent_dim, input_neurons, output_nerons, num_epochs = map(int, sys.argv[3:])
    case 'conv':
        num_conv_layers, num_dense_layers, latent_dim, input_neurons, output_neurons, initial_channels, factor, num_epochs = map(int, sys.argv[3:])
    case 'lstm':
        num_layers, latent_dim, output_neurons, middle_ground, num_epochs = map(int, sys.argv[3:])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\t[INFO] Running on {device}\n')

dataset = LoFiDataset('../../data')

generator = torch.Generator().manual_seed(0)
train_set, test_set = random_split(dataset, [0.8, 0.2], generator=generator)

original_size = dataset[0].shape[0]

batch_size = 32
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

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

optimizer = Adam(model.parameters(), lr=6e-4)
loss_function = VAELoss()

print(f'\t[INFO] Training started')
loss_curve = train(model, train_loader, test_loader, loss_function, optimizer, epochs=num_epochs, device=device, save_path=f'../../models/{model_path}.pth')

plt.plot(loss_curve[0])
plt.plot(loss_curve[1])
plt.title(f'Função custo - {model_path}')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend(['Treino', 'Validação'])
plt.savefig(f'../../plots/loss/{model_path}.png')
print(f'\n\t[INFO] Loss curve saved at \'../../plots/loss/{model_path}.png\'')
