from vae import *

a = Conv1DVAE(1, 2, 32921, 2, 800, 200, 4, 2, 'cuda').to('cuda')
b = torch.rand([32, 32921]).to('cuda')
print(a)