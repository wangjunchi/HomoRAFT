import torch
from functorch import vmap, jacrev, jacfwd
import numpy as np
from torch.autograd.functional import jacobian
import time

def BA_Homography_Residual(x, y, coefficients):
    one = torch.ones([1], dtype=torch.float32, device=x.device)
    h = torch.cat([coefficients, one])
    h = h.reshape([3, 3])
    pts_dst = h @ x
    pts_dst = pts_dst / pts_dst[-1, :]
    return (pts_dst - y).flatten()

# def BA_Homography_Jacobian(x, y, coefficients):
#     jacobian = torch.autograd.functional.jacobian(BA_Homography_Residual, (x, y, coefficients), vectorize=True)
#     return jacobian

def BA_Homography_Pseudoinverse(x):
    return torch.linalg.inv(x.T @ x) @ x.T

def BA_Homography(src_pts, dst_pts, init_guess):

    max_iter = 10
    tolerance_difference = 10 ** (-4)
    tolerance = 10 ** (-4)

    rmse_prev = torch.inf
    coefficients = init_guess

    for k in range(max_iter):
        residual = vmap(BA_Homography_Residual)(src_pts, dst_pts, coefficients)

        jac = vmap(jacfwd(BA_Homography_Residual, argnums=2))(src_pts, dst_pts, coefficients)

        # jac = torch.autograd.functional.jacobian(_calculate_residual, coefficients, vectorize=True)
        # jacobian = BA_Homography_Jacobian(src_pts, dst_pts, coefficients)
        pseudoinverse = vmap(BA_Homography_Pseudoinverse)(jac)
        coefficients = coefficients - (pseudoinverse @ residual[:, :, None]).squeeze(dim=-1)
        rmse = torch.sqrt(torch.mean(residual ** 2))
        print(f"Iteration {k}: RMSE = {rmse}")
        if torch.abs(rmse - rmse_prev) < tolerance_difference:
            # print("Terminating iteration due to small RMSE difference.")
            break
        if rmse < tolerance:
            # print("Terminating iteration due to small RMSE.")
            break
        rmse_prev = rmse

    return coefficients

def main():
    h_gt = torch.tensor([[1, 0.2, 10],
                         [0, 1, 20],
                         [0.5, 0, 1]]).float()

    # sample points
    xs = torch.linspace(0, 40 - 1, 40)
    ys = torch.linspace(0, 30 - 1, 30)
    gridx, gridy = torch.meshgrid(xs, ys)

    src_pts = torch.stack([gridx, gridy])
    src_pts = torch.concat((src_pts, torch.ones(src_pts.shape[1:])[None, :, :]), dim=0)

    src_pts = src_pts.reshape(3, -1)

    # apply homography
    dst_pts = h_gt @ src_pts
    dst_pts = dst_pts / dst_pts[-1, :]

    # add noise to the first two coordinates
    noise = 3
    dst_pts[:, :2] += noise * torch.randn(dst_pts[:, :2].shape)

    # theta0 = torch.ones((3,3)).flatten()[:-1]
    theta0 = torch.tensor([1.0, 0, 0, 0, 1.0, 0, 0, 0]).float()

    # batch all the input
    batch_size = 16
    src_pts = src_pts[None, :, :].repeat(batch_size, 1, 1).cuda()
    dst_pts = dst_pts[None, :, :].repeat(batch_size, 1, 1).cuda()
    theta0 = theta0[None, :].repeat(batch_size, 1).cuda()

    st = time.time()
    iter_num = 1
    for i in range(iter_num):
        h_est = BA_Homography(src_pts, dst_pts, theta0)
    et = time.time()
    print(f"Batch size: {batch_size}")
    print(f"Running for {iter_num} iterations")
    print(f"Time taken: {et - st} seconds")
    h_est = h_est[0].cpu()
    h_est = torch.cat([h_est, torch.ones([1], dtype=torch.float32)]).reshape([3, 3])
    print(f"Ground truth homography: {h_gt}")
    print(f"Estimated homography: {h_est}")


if __name__ == "__main__":
    main()