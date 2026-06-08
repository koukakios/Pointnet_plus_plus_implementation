import torch


class UnitPointnetSegm(torch.nn.Module):

    def __init__(self, dims_1, dims_2, dims_3, dims_4):
        super(UnitPointnetSegm, self).__init__()

        # optional Tnet here for later

        layers = []
        for v, w in zip(dims_1, dims_1[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())
        self.mlp_1 = torch.nn.Sequential(*layers)

        # optinal TNet here

        layers = []
        for v, w in zip(dims_2, dims_2[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())
        self.mlp_2 = torch.nn.Sequential(*layers)

        layers = []
        for v, w in zip(dims_3, dims_3[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())
        self.mlp_3 = torch.nn.Sequential(*layers)

        layers = []
        for v, w in zip(dims_4, dims_4[1:]):
            layers.append(torch.nn.Linear(v, w))
            layers.append(torch.nn.ReLU())
        self.mlp_4 = torch.nn.Sequential(*layers)

    def forward(self, x):
        #optional Tnet here
        x = self.mlp_1(x)
        #optional Tnet here
        x_concat_segm = torch.clone(x)
        x = self.mlp_2(x)
        x = self.mlp_3(torch.cat([x, x_concat_segm], dim = - 1))
        x = self.mlp_4(x)
        return x
