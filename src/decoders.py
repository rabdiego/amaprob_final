import torch
import torch.nn as nn

class DenseDecoder(nn.Module):
    def __init__(self, num_layers, original_size, latent_dim, input_neurons, output_neurons):
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
        
        layers.pop()
        layers.append(nn.Tanh())
        
        self.block = nn.Sequential(*layers)


    def forward(self, x):
        return self.block(x)

