import torch
import numpy as np
import cv2
import traceback


def _postprocess(perspective_field):
    # Move data to numpy?
    if torch.is_tensor(perspective_field):
        perspective_field = perspective_field.cpu().detach().numpy()

    # Create field of the coordinates
    y_patch_grid, x_patch_grid = np.mgrid[0:perspective_field.shape[-2], 0:perspective_field.shape[-1]]
    x_patch_grid = np.tile(x_patch_grid.reshape(1, -1), (perspective_field.shape[0], 1))
    y_patch_grid = np.tile(y_patch_grid.reshape(1, -1), (perspective_field.shape[0], 1))
    coordinate_field = np.stack((x_patch_grid, y_patch_grid), axis=1).transpose(0, 2, 1)

    # Create prediction field
    # perspective_field: b*2*h*w
    pf_x1_img_coord, pf_y1_img_coord = np.split(perspective_field, 2, axis=1)
    pf_x1_img_coord = pf_x1_img_coord.reshape(perspective_field.shape[0], -1)
    pf_y1_img_coord = pf_y1_img_coord.reshape(perspective_field.shape[0], -1)
    mapping_field = coordinate_field + np.stack((pf_x1_img_coord, pf_y1_img_coord), axis=1).transpose(0, 2, 1)

    return coordinate_field, mapping_field


def compute_homography(flow_pred, mask=None):

    coordinate_field, mapping_field = _postprocess(flow_pred)
    # Find best homography fit and its delta
    predicted_h = []
    predicted_delta = []

    for i in range(flow_pred.shape[0]):
        if mask is not None:
            mask_i = mask[i].reshape(-1).cpu().detach().numpy()
            src_pts = coordinate_field[i][mask_i == 1]
            dst_pts = mapping_field[i][mask_i == 1]
        else:
            src_pts = coordinate_field[i]
            dst_pts = mapping_field[i]
        h = np.eye(3)
        try:
            h = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts), cv2.RANSAC, 1)[0]
        except Exception:
            print("valid points in mask = ", mask_i.sum())
            traceback.print_exc()
            print("using identity matrix instead")
            h = np.eye(3)
        if h is None:
            print("opencv returns None as honography, using identity matrix instead")
            h = np.eye(3)
        predicted_h.append(h)

    predicted_h = np.array(predicted_h)

    return predicted_h


def compute_mace(pred_h, gt_h, four_points):
    # Compute MACE
    mace = []
    gt_h = gt_h.cpu().detach().numpy()
    for i in range(gt_h.shape[0]):
        try:
            delta_gt = cv2.perspectiveTransform(np.asarray([four_points]), gt_h[i]).squeeze() - four_points
            delta_pred = cv2.perspectiveTransform(np.asarray([four_points]), pred_h[i]).squeeze() - four_points
            mace.append(np.mean(np.abs(delta_gt - delta_pred)))
        except:
            print("Error when compiting mace")
            print('homography: ', pred_h[i])
            traceback.print_exc()
            exit()

    return np.mean(mace)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Create a random flow field
    flow = torch.ones(1, 2, 240, 320)
    flow[:, 0, :, :] = flow[:, 0, :, :] * 10
    flow[:, 1, :, :] = flow[:, 1, :, :] * 20

    # Create a mask
    mask = torch.rand(1, 240, 320) > 0.5

    # Compute homography
    h = compute_homography(flow, mask)

    # Create a four point
    four_points = np.array([[0, 0], [0, 240], [320, 240], [320, 0]], dtype=float).reshape(-1, 2)

    # Compute MACE
    h_gt = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    mace = compute_mace(h, h_gt, four_points)

    print('MACE: {}'.format(mace))

