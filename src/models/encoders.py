import torch
import torch.nn as nn

class DenseEncoder(nn.Module):
    def __init__(self, num_layers, original_size, input_neurons, output_neurons):
        super(DenseEncoder, self).__init__()
        
        out_in_ratio = (output_neurons / input_neurons)
        if num_layers >= 2:
            oir_scaled = out_in_ratio**(1/(num_layers-1))
        else:
            oir_scaled = 1
        
        num_neurons = [int(input_neurons * oir_scaled**i) for i in range(num_layers)]
        num_neurons[-1] = output_neurons
        num_neurons.insert(0, original_size)

        layers = list()
        
        for i in range(num_layers):
            layers.append(nn.Linear(num_neurons[i], num_neurons[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        
        self.block = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.block(x)


class Conv1DEncoder(nn.Module):
    def __init__(self, 
                 num_conv_layers, 
                 num_dense_layers, 
                 original_size, 
                 input_neurons, 
                 output_neurons, 
                 initial_channels, 
                 factor, 
                 kernel_size=3, stride=2, padding=1):
        super(Conv1DEncoder, self).__init__()

        channels = [initial_channels * factor**i for i in range(num_conv_layers)]
        channels.insert(0, 1)

        layers = list()
        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.LeakyReLU(0.2))
        
        layers.append(nn.Flatten())

        out_neurons = self.__compute_out_neurons_conv(original_size, kernel_size, stride, padding, num_conv_layers) * channels[-1]

        dense_encoder = DenseEncoder(num_dense_layers, out_neurons, input_neurons, output_neurons)
        layers.append(dense_encoder)

        self.block = nn.Sequential(*layers)
        

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.block(x)


    def __compute_out_neurons_conv(self, input_length, kernel_size, stride, padding, num_layers):
        out_neurons = int(((input_length + 2*padding - kernel_size)/stride) + 1)
        if num_layers == 1:
            return out_neurons
        else:
            return self.__compute_out_neurons_conv(out_neurons, kernel_size, stride, padding, num_layers-1)

