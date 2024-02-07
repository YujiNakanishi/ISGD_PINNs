import torch
import torch.nn as nn
import config as c

class VanillaPINNs(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = nn.Linear(in_features = 1, out_features = 200)
        self.act1 = nn.Tanh()
        self.lin2 = nn.Linear(in_features = 200, out_features = 200)
        self.act2 = nn.Tanh()
        self.lin3 = nn.Linear(in_features = 200, out_features = 200)
        self.act3 = nn.Tanh()
        self.lin4 = nn.Linear(in_features = 200, out_features = 200)
        self.act4 = nn.Tanh()
        self.lin5 = nn.Linear(in_features = 200, out_features = 200)
        self.act5 = nn.Tanh()
        self.lin6 = nn.Linear(in_features = 200, out_features = 1)
    
    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        x = self.act4(self.lin4(x))
        x = self.act5(self.lin5(x))

        return self.lin6(x)
    
    def BC_loss(self):
        _input = torch.tensor([[0.], [1.]], dtype = torch.float32).to("cuda")
        u = self(_input)

        loss = torch.mean(u**2)

        return loss
    
    def PDE_loss(self):
        _input = torch.rand(c.PDE_num, 1, dtype = torch.float32).to("cuda")
        u = self(_input)

        dxu = torch.autograd.grad(torch.sum(u), _input, create_graph=True)[0]
        dxxu = torch.autograd.grad(torch.sum(dxu), _input, create_graph=True)[0]

        residual = dxxu + 4.*(torch.pi**2)*torch.sin(2.*torch.pi*_input) + 250.*(torch.pi**2)*torch.sin(50.*torch.pi*_input)

        loss = torch.mean(residual**2)

        return loss