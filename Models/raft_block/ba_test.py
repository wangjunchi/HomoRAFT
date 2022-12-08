import logging
from typing import Callable
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def BA_Homography_Residual(x, y, coefficients):
    h = torch.cat([coefficients, torch.ones([1], dtype=torch.float32)])
    h = h.reshape([3, 3])
    pts_dst = h @ x
    pts_dst = pts_dst / pts_dst[-1, :]
    return (pts_dst - y).flatten()

def BA_Homography_Jacobian(x, y, coefficients):
    jacobian = torch.autograd.functional.jacobian(BA_Homography_Residual, (x, y, coefficients), vectorize=True)
    return jacobian

def BA_Homography_Pseudoinverse(x):
    return torch.linalg.inv(x.T @ x) @ x.T

def BA_Homography(src_pts, dst_pts, init_guess):

    max_iter = 10
    tolerance_difference = 10 ** (-4)
    tolerance = 10 ** (-4)

    rmse_prev = torch.inf
    coefficients = init_guess

    def _calculate_residual(coefficients):
        return BA_Homography_Residual(src_pts, dst_pts, coefficients)

    for k in range(max_iter):
        residual = BA_Homography_Residual(src_pts, dst_pts, coefficients)
        jacobian = torch.autograd.functional.jacobian(_calculate_residual, coefficients, vectorize=True)
        # jacobian = BA_Homography_Jacobian(src_pts, dst_pts, coefficients)
        pseudoinverse = BA_Homography_Pseudoinverse(jacobian)
        coefficients = coefficients - pseudoinverse @ residual
        rmse = torch.sqrt(torch.sum(residual ** 2))
        print(f"Iteration {k}: RMSE = {rmse}")
        if torch.abs(rmse - rmse_prev) < tolerance_difference:
            print("Terminating iteration due to small RMSE difference.")
            break
        if rmse < tolerance:
            print("Terminating iteration due to small RMSE.")
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

    st = time.time()
    h_est = BA_Homography(src_pts, dst_pts, theta0)
    et = time.time()
    print(f"Time taken: {et - st} seconds")
    h_est = torch.cat([h_est, torch.ones([1], dtype=torch.float32)]).reshape([3, 3])
    print(f"Ground truth homography: {h_gt}")
    print(f"Estimated homography: {h_est}")


if __name__ == "__main__":
    main()