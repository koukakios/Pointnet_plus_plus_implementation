import torch


class UnitPointnet(torch.nn.Module):

    def __init__(self, dims_1, dims_2):
        super(UnitPointnet, self).__init__()

        #optional Tnet here for later

        layers = []
        for v, w in zip(dims_1, dims_1[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())

        self.mlp_1 = torch.nn.Sequential(*layers)

        #optinal TNet here


        layers = []
        for v, w in zip(dims_2, dims_2[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())

        self.mlp_2 = torch.nn.Sequential(*layers)

    def forward(self, x):
        """

        :param x: (B, N, dims_1[0])
        :return: (B, N , dims_2[-1])
        """
        #optinal Tnet here
        x = self.mlp_1(x)
        # optinal Tnet here
        x = self.mlp_2(x)
        return x
