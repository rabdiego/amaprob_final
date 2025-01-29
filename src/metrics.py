import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = self.reconstruction_loss(recon_x, x)

        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_div

