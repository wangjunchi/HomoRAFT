import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss


def sequence_loss(pf_preds, pf_gt, mask=None, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(pf_preds)
    seq_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = smooth_l1_loss(pf_preds[i], pf_gt, reduction='none')
        i_loss = i_loss.permute(0, 2, 3, 1)
        if mask is not None:
            i_loss = i_loss * mask[:, :, :, None]
        i_loss = i_loss.mean()
        seq_loss += i_weight * i_loss

    return seq_loss