import torch.nn


class LinearLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation_fun,representations_number, device):
        super(LinearLayer, self).__init__()
        if activation_fun == 'tanh':
            self.activation_fun = torch.nn.Tanh()
        elif activation_fun == 'relu':
            self.activation_fun = torch.nn.ReLU()
        elif activation_fun == 'sigmoid':
            self.activation_fun = torch.nn.Sigmoid()
        elif activation_fun == 'leaky_relu':
            self.activation_fun = torch.nn.LeakyReLU()
        else:
            self.activation_fun = torch.nn.Tanh()
        self.representations_number = representations_number
        self.device=device

        self.r = torch.nn.Parameter(torch.randn(representations_number, in_channels)).to(device)
        self.s = torch.nn.Parameter(torch.randn(representations_number, out_channels)).to(device)

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=False).to(device)

    def forward(self, x):
        s_repeated = torch.repeat_interleave(self.s, int(x.shape[0] / self.representations_number), dim=0)
        r_repeated = torch.repeat_interleave(self.r, int(x.shape[0] / self.representations_number), dim=0)

        output = torch.mul(x,r_repeated)
        output = self.linear(output)
        output = torch.mul(output,s_repeated)
        output = self.activation_fun(output)

        return output
