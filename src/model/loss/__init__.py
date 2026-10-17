"""Loss functions for ConfigurableEncoder training."""

import torch.nn as nn

from .composite import CompositeLoss
from .infonce import InfoNCE
from .siglip import SigLIP
from .sigreg import SIGReg


def get_loss_function(type, logit_scale=None, **kwargs):

    if type == "l2":
        return nn.MSELoss()
    if type == "l1":
        return nn.L1Loss()
    if type == "infonce":
        return InfoNCE(logit_scale=logit_scale, **kwargs)
    if type == "siglip":
        return SigLIP(**kwargs)
    if type == "sigreg":
        return SIGReg(**kwargs)
    if type == "composite":
        return CompositeLoss(logit_scale=logit_scale, **kwargs)
    raise ValueError(f"Unsupported loss type: {type}")
