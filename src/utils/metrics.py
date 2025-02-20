import torch
import torch.nn as nn

class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.softplus = nn.Softplus()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = self.reconstruction_loss(recon_x, x)

        scale = self.softplus(logvar) + 1e-8
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        z = dist.rsample()

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1)
        )
        kl_loss = torch.distributions.kl.kl_divergence(dist, std_normal).mean()

        return recon_loss + kl_loss

