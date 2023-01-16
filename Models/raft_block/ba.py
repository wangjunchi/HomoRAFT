import torch
from functorch import vmap, jacrev, jacfwd
import time

def BA_Homography_Residual(x, y, coefficients):
    one = torch.ones([1], dtype=torch.float32, device=x.device)
    h = torch.cat([coefficients, one])
    h = h.reshape([3, 3])
    pts_dst = h @ x
    pts_dst = pts_dst / pts_dst[-1, :]
    return (pts_dst - y).flatten()

def BA_Homography_Pseudoinverse(x, w):
    return torch.linalg.inv((x.T * w) @ x) @ (x.T * w)

def BA_Homography(coords0, coords1, weights, init_guess):

    max_iter = 5
    tolerance_difference = 10 ** (-4)
    tolerance = 10 ** (-4)

    rmse_prev = torch.inf
    coefficients = init_guess
    if weights is None:
        weights = torch.ones([coords0.shape[0], 3600], dtype=torch.float32, device=coords0.device)
    weights = weights.view(weights.shape[0], -1)
    # prepare points
    src_pts = coords0.permute(0, 3, 2, 1).reshape(coords0.shape[0], -1, 2)
    dst_pts = coords1.permute(0, 3, 2, 1).reshape(coords1.shape[0], -1, 2)
    # convert to homogenous coordinates
    src_pts = torch.cat([src_pts, torch.ones(src_pts.shape[0], src_pts.shape[1], 1).to(src_pts.device)], dim=2)
    dst_pts = torch.cat([dst_pts, torch.ones(dst_pts.shape[0], dst_pts.shape[1], 1).to(dst_pts.device)], dim=2)
    src_pts = src_pts.permute(0, 2, 1)
    dst_pts = dst_pts.permute(0, 2, 1)

    for k in range(max_iter):
        residual = vmap(BA_Homography_Residual)(src_pts, dst_pts, coefficients)

        jac = vmap(jacfwd(BA_Homography_Residual, argnums=2))(src_pts, dst_pts, coefficients)

        # jac = torch.autograd.functional.jacobian(_calculate_residual, coefficients, vectorize=True)
        # jacobian = BA_Homography_Jacobian(src_pts, dst_pts, coefficients)
        pseudoinverse = vmap(BA_Homography_Pseudoinverse)(jac, weights)
        coefficients = coefficients - (pseudoinverse @ residual[:, :, None]).squeeze(dim=-1)
        rmse = torch.sqrt(torch.mean(residual ** 2))
        # print(f"Iteration {k}: RMSE = {rmse}")
        if torch.abs(rmse - rmse_prev) < tolerance_difference:
            # print("Terminating iteration due to small RMSE difference.")
            break
        if rmse < tolerance:
            # print("Terminating iteration due to small RMSE.")
            break
        rmse_prev = rmse

    return coefficients
