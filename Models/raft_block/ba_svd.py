import torch
import torch.nn.functional as F
from kornia.geometry.homography import find_homography_dlt_iterated, find_homography_dlt

def compute_h_dlt(flow, weight=None):
    """
    bundle adjustment layer, solving the least square problem
    :param target: the flow map predicted by the update module
    :param weight: the weight map of the flow map
    :return: the regressed homography, 8 parameters
    """

    # source points
    B, C, H, W = flow.shape
    x = torch.arange(0, W).view(1, 1, 1, W).repeat(B, 1, H, 1).to(flow.device)
    y = torch.arange(0, H).view(1, 1, H, 1).repeat(B, 1, 1, W).to(flow.device)
    source = torch.cat([x, y], dim=1).float()
    source = source.view(B, 2, -1)
    source = source.permute(0, 2, 1)

    # target points
    # flow = flow.permute(0, 3, 1, 2)
    flow = flow.view(B, 2, -1)
    flow = flow.permute(0, 2, 1)
    target = flow + source

    # weight
    if weight is not None:
        weight = weight.view(B, -1)

    # solve homography
    homography = find_homography_dlt(source, target, weight)

    return homography


if __name__ == '__main__':
    target = torch.ones(1, 2, 240, 320)
    target *= 10
    weight = torch.rand(1, 240, 320, 1)
    homography = BA(target, weight)
    print(homography)





