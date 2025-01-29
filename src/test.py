from vae import *

a = Conv1DEncoder(3, 4, 32921, 800, 100, 4, 2)
print(a)
b = torch.rand([32, 32921])
print(a(b).shape)