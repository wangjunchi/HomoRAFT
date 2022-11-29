import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.raft_block.update import BasicUpdateBlock, UpdateModule
from Models.raft_block.extractor import BasicEncoder
from Models.raft_block.corr import CorrBlock
from Models.raft_block.utils import bilinear_sampler, coords_grid, upflow8
from Models.raft_block.flow_viz import flow_to_image

from easydict import EasyDict as edict


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.corr_radius = 4
        self.corr_radius = 3
        self.dropout = 0.0

        self.test_mode = True

        self.args = edict({'small': False, 'mixed_precision': False, 'alternate_corr': False})

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.args.corr_levels = 4
        self.args.corr_radius = 4

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)


    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0

            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


if __name__ == '__main__':
    model = Model()
    model.cuda()

    # using fake input
    image1 = torch.randn(1, 3, 240, 320).cuda()
    image2 = torch.randn(1, 3, 240, 320).cuda()

    # forward pass
    flow_predictions = model(image1, image2, iters=3)
    print(len(flow_predictions))
    print(flow_predictions[0].shape)
