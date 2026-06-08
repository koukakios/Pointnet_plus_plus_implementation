import torch


class BatchNormLastDim(torch.nn.BatchNorm1d):
    def forward(self, x):
        shape = x.shape
        x = super().forward(x.reshape(-1, shape[-1]))
        return x.reshape(shape)
