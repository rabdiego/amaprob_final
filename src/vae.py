import torch
import torch.nn as nn
from random import random
from encoders import *
from decoders import *

class AbstractVAE(nn.Module):
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar


    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z


    def decode(self, x):
        return self.decoder(x)


    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    

    def sample(self, scale=2):
        z_sample = (scale*(torch.rand((2, self.latent_dim)) - 1)).to(self.device)
        x_decoded = self.decode(z_sample)
        return x_decoded


class DenseVAE(AbstractVAE):
    def __init__(self, num_layers, original_size, latent_dim, input_neurons, output_neurons, device):
        """
        num_layers: Número de camadas no encoder e no decoder
        original_size: Tamanho do vetor 1D
        latent_dim: Tamanho da dimensão latente
        input_neurons: Número de neurônios na primeira camada oculta do encoder / penultima camada do decoder
        output_neurons: Número de neurônios na última camada oculta do encoder / primeira camada do decoder
        """
        super(DenseVAE, self).__init__()

        self.encoder = DenseEncoder(num_layers, original_size, input_neurons, output_neurons)

        self.latent_dim = latent_dim
        self.mean_layer = nn.Linear(output_neurons, latent_dim)
        self.logvar_layer = nn.Linear(output_neurons, latent_dim)
        
        self.decoder = DenseDecoder(num_layers, original_size, latent_dim, output_neurons, input_neurons)

        self.device = device


class Conv1DVAE(AbstractVAE):
    def __init__(self,
                 num_conv_layers,
                 num_dense_layers,
                 original_size,
                 latent_dim,
                 input_neurons,
                 output_neurons,
                 initial_channels,
                 factor,
                 device,
                 kernel_size=3, stride=2, padding=1,):
        """
        num_conv_layers: Número de camadas convolucionais no encoder e no decoder
        num_dense_layers: Número de camadas densas no encoder e no decoder
        original_size: Tamanho do vetor 1D
        latent_dim: Tamanho da dimensão latente
        input_neurons: Número de neurônios na primeira camada oculta densa do encoder / penultima camada do decoder
        output_neurons: Número de neurônios na última camada oculta do encoder / primeira camada densa do decoder
        initial_channels: Número de canais que a primeira camada convolucional gerará
        factor: Fator de multiplicação para aumentar o número de canais a cada camada convolucional
        """
        
        super(Conv1DVAE, self).__init__()

        self.encoder = Conv1DEncoder(
            num_conv_layers,
            num_dense_layers,
            original_size,
            input_neurons,
            output_neurons,
            initial_channels,
            factor,
            kernel_size, stride, padding
        )

        self.latent_dim = latent_dim
        self.mean_layer = nn.Linear(output_neurons, latent_dim)
        self.logvar_layer = nn.Linear(output_neurons, latent_dim)

        self.decoder = Conv1DDecoder(
            num_conv_layers,
            num_dense_layers,
            original_size,
            latent_dim,
            input_neurons,
            output_neurons,
            initial_channels,
            factor,
            kernel_size, stride, padding
        )
        
        self.device = device


class LSTMVAE(AbstractVAE):
    def __init__(self, num_layers, original_size, latent_dim, output_neurons, middle_ground, device):
        """
        num_layers: Número de camadas no encoder e no decoder
        original_size: Tamanho do vetor 1D
        latent_dim: Tamanho da dimensão latente
        output_neurons: Número de neurônios na última camada oculta do encoder / primeira camada do decoder
        middle_ground: Número de neurônios na camada intermediária entre a saída do LSTM e o tamanho original
        """
        super(LSTMVAE, self).__init__()

        self.encoder = LSTMEncoder(num_layers, original_size, output_neurons)

        self.latent_dim = latent_dim
        self.mean_layer = nn.Linear(output_neurons, latent_dim)
        self.logvar_layer = nn.Linear(output_neurons, latent_dim)
        
        self.decoder = LSTMDecoder(num_layers, original_size, latent_dim, output_neurons, middle_ground)

        self.device = device

