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


def corner_loss(pred_h, gt_h, four_corners, gamma=0.8):
    """
    Loss function for corner prediction
    :param pred_h:  B*3*3
    :param gt_h: B*3*3
    :param four_corners: B*4*2
    :return:
    """

    # convert 4 points to homogeneous coordinates
    # B*4*3
    four_corners_h = torch.cat([four_corners, torch.ones((four_corners.shape[0], four_corners.shape[1], 1), device=four_corners.device, dtype=four_corners.dtype)], dim=2)


    n_predictions = len(pred_h)
    total_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)

        gt_h_inv = torch.inverse(gt_h)

        # convert 4 points to homogeneous coordinates
        # B*4*3
        # four_corners_h = torch.cat([four_corners, torch.ones((four_corners.shape[0], four_corners.shape[1], 1), device=four_corners.device, dtype=four_corners.dtype)], dim=2)

        reproject_matrix = torch.bmm(gt_h_inv, pred_h[i])
        reprojected_corners = torch.bmm(reproject_matrix, four_corners_h.permute(0, 2, 1)).permute(0, 2, 1)

        # compute reprojection error
        reprojected_corners = reprojected_corners[:, :, :2] / reprojected_corners[:, :, 2:]
        i_loss = smooth_l1_loss(reprojected_corners, four_corners, reduction='mean')
        # take log to make the loss more stable
        i_loss = torch.log(i_loss + 1)
        # i_loss = torch.clamp(i_loss, min=0, max=5)
        total_loss += i_weight * i_loss

    # seq_loss = torch.clamp(seq_loss, min=0, max=50)

    return total_loss


    # # wrap the four corners using the predicted homography
    # loss = torch.tensor(0.0, device=four_corners.device, dtype=torch.float64)
    # for i in range(four_corners.shape[0]):
    #     pred_target[i] = torch.matmul(pred_h[i], four_corners_h[i].T).T
    #     gt_target[i] = torch.matmul(gt_h[i], four_corners_h[i].T).T
    #     pred_target[i] = pred_target[i] / pred_target[i, :, 2:]
    #     gt_target[i] = gt_target[i] / gt_target[i, :, 2:]
    #     loss += smooth_l1_loss(pred_target[i], gt_target[i], reduction='mean')
    #
    # return loss / four_corners.shape[0]



if __name__ == '__main__':
    # test corner loss
    pred_h = torch.rand((2, 3, 3))
    gt_h = torch.rand((2, 3, 3))
    four_corners = torch.rand((2, 4, 2))
    loss = corner_loss(pred_h, gt_h, four_corners)
    print(loss)