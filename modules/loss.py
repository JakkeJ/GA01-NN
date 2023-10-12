import torch

import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    def __init__(self, phos_w=4.5, phoc_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w

    def forward(self, y: dict, targets: torch.Tensor):
    	
    	# Apply the loss on PHOS features this is a regression loss
    	# Note: This loss should be applicable to the PHOS part of the 
    	# output which is the first part of the output.
        mse_loss = nn.MSELoss()
        phos_loss = self.phos_w * mse_loss(targets[:, :y['phos'].shape[1]], y['phos'])
        
        # Apply the loss on PHOC features this is a classification loss
    	# Note: This loss should be applicable to the PHOC part of the 
    	# output which is the later part of the output.
        bce_loss = nn.BCEWithLogitsLoss()
        phoc_loss = self.phoc_w * bce_loss(y['phoc'], targets[:, y['phos'].shape[1]:])

        loss = phos_loss + phoc_loss
        return loss
