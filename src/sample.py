import torch
from data import LoFiDataset
from vae import DenseVAE, Conv1DVAE, LSTMVAE
from utils import save_sample
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

model_name = sys.argv[1]

match model_name:
    case 'dense':
        num_layers, latent_dim, input_neurons, output_nerons  = map(int, sys.argv[2:])
    case 'conv':
        num_conv_layers, num_dense_layers, latent_dim, input_neurons, output_neurons, initial_channels, factor  = map(int, sys.argv[2:])
    case 'lstm':
        num_layers, latent_dim, output_neurons = map(int, sys.argv[2:])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\t[INFO] Running on {device}')

dataset = LoFiDataset('../data')

original_size = dataset[0].shape[0]

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
        model = LSTMVAE(num_layers, original_size, latent_dim, output_neurons, device).to(device)

model.load_state_dict(torch.load(f'../models/{model_name}.pth', weights_only=True))

a = model.sample(scale=2)
b = save_sample(a, f'../samples/{model_name}.wav')
print(f'\t[INFO] Audio saved at \'../samples/{model_name}.wav\'')

plt.plot(b[0])
plt.savefig(f'../plots/soundwave/{model_name}.png')
print(f'\t[INFO] Soundwave saved at \'../plots/soundwave/{model_name}.png\'')
