import torch

import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    def __init__(self, phos_w=4.5, phoc_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w

    def forward(self, y: dict, targets: torch.Tensor):
        phos_loss = self.phos_w * F.mse_loss(y['phos'], targets['phos'])
        phoc_loss = self.phoc_w * F.cross_entropy(y['phoc'], targets['phoc'])

        loss = phos_loss + phoc_loss
        return loss
    
class PHOSCCosineLoss(nn.Module):
    def __init__(self, phos_w=4.5, phoc_w=1, cosine_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w
        self.cosine_w = cosine_w

        self.cosine_criterion = nn.CosineEmbeddingLoss()

    def forward(self, y: dict, targets: dict):
        phos_loss = self.phos_w * F.mse_loss(y['phos'], targets['phos'])
        phoc_loss = self.phoc_w * F.cross_entropy(y['phoc'], targets['phoc'])

        phosc_out = torch.cat((y['phoc'], y['phos']), dim=1)
        phosc_target = targets['phosc']
        similarities = targets['sim']

        cosine_loss = self.cosine_w * F.cosine_embedding_loss(phosc_out, phosc_target, similarities)

        loss = phos_loss + phoc_loss + cosine_loss
        return loss


if __name__ == '__main__':

    y = {
        'phos': torch.randn((64, 165)),
        'phoc': torch.randn((64, 604))
    }
    target = {
        'phos': torch.randn((64, 165)),
        'phoc': torch.randn((64, 604)),
        'phosc': torch.randn((64, 769)),
        'sim': torch.ones((64))
    }

    print(y['phos'].shape, y['phoc'].shape)
    print(target['phos'].shape, target['phoc'].shape)

    criterion = PHOSCCosineLoss()

    loss = criterion(y, target)
    print(loss)