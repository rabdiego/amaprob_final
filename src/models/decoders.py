import torch
import torch.nn as nn

class DesFlatten(nn.Module):
    def __init__(self, c, b):
        super(DesFlatten, self).__init__()
        self.c = c
        self.d = b // c

    def forward(self, x):
        a, _ = x.shape
        return x.view(a, self.c, self.d)

class DenseDecoder(nn.Module):
    def __init__(self, num_layers, original_size, latent_dim, input_neurons, output_neurons, tanh=True):
        super(DenseDecoder, self).__init__()
        
        out_in_ratio = (output_neurons / input_neurons)
        if num_layers >= 2:
            oir_scaled = out_in_ratio**(1/(num_layers-1))
        else:
            oir_scaled = 1
        
        num_neurons = [int(input_neurons * oir_scaled**i) for i in range(num_layers)]
        num_neurons[-1] = output_neurons
        num_neurons.append(original_size)
        num_neurons.insert(0, latent_dim)

        layers = list()
        
        for i in range(num_layers+1):
            layers.append(nn.Linear(num_neurons[i], num_neurons[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        
        if tanh:
            layers.pop()
            layers.append(nn.Tanh())
        
        self.block = nn.Sequential(*layers)


    def forward(self, x):
        return self.block(x)


class Conv1DDecoder(nn.Module):
    def __init__(self,
                 num_conv_layers,
                 num_dense_layers,
                 original_size,
                 latent_dim,
                 input_neurons,
                 output_neurons,
                 initial_channels,
                 factor,
                 kernel_size=3, stride=2, padding=1):
        
        super(Conv1DDecoder, self).__init__()

        channels = [initial_channels * factor**i for i in range(num_conv_layers)]
        channels.insert(0, 1)

        layers = list()

        out_neurons = self.__compute_out_neurons_conv(original_size, kernel_size, stride, padding, num_conv_layers) * channels[-1]

        dense_decoder = DenseDecoder(num_dense_layers, out_neurons, latent_dim, output_neurons, input_neurons, tanh=False)
        layers.append(dense_decoder)
        layers.append(DesFlatten(channels[-1], out_neurons))
        
        channels.reverse()
        for i in range(num_conv_layers):
            layers.append(nn.ConvTranspose1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.LeakyReLU(0.2))
        
        layers.append(nn.Flatten())
        layers.append(nn.Tanh())

        self.block = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.block(x)


    def __compute_out_neurons_conv(self, input_length, kernel_size, stride, padding, num_layers):
        out_neurons = int(((input_length + 2*padding - kernel_size)/stride) + 1)
        if num_layers == 1:
            return out_neurons
        return self.__compute_out_neurons_conv(out_neurons, kernel_size, stride, padding, num_layers-1)


class LSTMDecoder(nn.Module):
    def __init__(self, num_layers, original_size, latent_size, output_neurons, middle_ground):
        super(LSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(latent_size, output_neurons, num_layers=num_layers)
        self.block = nn.Sequential(
            nn.Linear(output_neurons, middle_ground),
            nn.LeakyReLU(0.2),
            nn.Linear(middle_ground, original_size),
            nn.Tanh()
        )
    

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.block(x)

