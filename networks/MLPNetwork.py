import torch.nn

from networks.LinearLayer import LinearLayer


class MLPNetwork(torch.nn.Module):
    def __init__(self, n_features, network_hidden_dimensions=[500, 100], representation_dimensionality=20,
                 representations_number=50, activation_fun='tanh', device='cuda'):
        super(MLPNetwork, self).__init__()
        self.device = device

        layers_number = len(network_hidden_dimensions)
        self.representations_number = representations_number
        self.layers = []

        for i in range(0, layers_number + 1):
            in_channels = n_features if i == 0 else network_hidden_dimensions[i - 1]
            out_channels = representation_dimensionality if i == layers_number else network_hidden_dimensions[i]
            self.layers += [LinearLayer(in_channels, out_channels,
                                        activation_fun=activation_fun,
                                        representations_number=representations_number, device=self.device)]
        self.mlp_network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.repeat(self.representations_number, 1)
        x = self.mlp_network(x)
        return x
