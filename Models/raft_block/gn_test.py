import logging
from typing import Callable
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import pinv

logger = logging.getLogger(__name__)


class GNSolver:
    """
    Gauss-Newton solver.
    Given response vector y, dependent variable x and fit function f,
    Minimize sum(residual^2) where residual = f(x, coefficients) - y.
    """

    def __init__(self,
                 fit_function: Callable,
                 max_iter: int = 1000,
                 tolerance_difference: float = 10 ** (-16),
                 tolerance: float = 10 ** (-9),
                 init_guess: torch.tensor = None,
                 ):
        """
        :param fit_function: Function that needs be fitted; y_estimate = fit_function(x, coefficients).
        :param max_iter: Maximum number of iterations for optimization.
        :param tolerance_difference: Terminate iteration if RMSE difference between iterations smaller than tolerance.
        :param tolerance: Terminate iteration if RMSE is smaller than tolerance.
        :param init_guess: Initial guess for coefficients.
        """
        self.fit_function = fit_function
        self.max_iter = max_iter
        self.tolerance_difference = tolerance_difference
        self.tolerance = tolerance
        self.coefficients = None
        self.x = None
        self.y = None
        self.init_guess = None
        if init_guess is not None:
            self.init_guess = init_guess

    def fit(self,
            x: torch.tensor,
            y: torch.tensor,
            init_guess: torch.tensor = None) -> torch.tensor:
        """
        Fit coefficients by minimizing RMSE.
        :param x: Independent variable.
        :param y: Response vector.
        :param init_guess: Initial guess for coefficients.
        :return: Fitted coefficients.
        """

        self.x = x
        self.y = y
        if init_guess is not None:
            self.init_guess = init_guess

        if init_guess is None:
            raise Exception("Initial guess needs to be provided")

        self.coefficients = self.init_guess
        rmse_prev = torch.inf
        for k in range(self.max_iter):
            st = time.time()
            residual = self.get_residual()
            jacobian = self._calculate_jacobian(self.coefficients)

            self.coefficients = self.coefficients - self._calculate_pseudoinverse(jacobian) @ residual
            rmse = torch.sqrt(torch.sum(residual ** 2))
            logger.info(f"Round {k}: RMSE {rmse}")
            if self.tolerance_difference is not None:
                diff = torch.abs(rmse_prev - rmse)
                if diff < self.tolerance_difference:
                    logger.info("RMSE difference between iterations smaller than tolerance. Fit terminated.")
                    return self.coefficients
            if rmse < self.tolerance:
                logger.info("RMSE error smaller than tolerance. Fit terminated.")
                return self.coefficients
            rmse_prev = rmse
            logger.info(f"Round {k} took {time.time() - st} seconds")
        logger.info("Max number of iterations reached. Fit didn't converge.")

        return self.coefficients

    def predict(self, x: torch.tensor):
        """
        Predict response for given x based on fitted coefficients.
        :param x: Independent variable.
        :return: Response vector.
        """
        return self.fit_function(x, self.coefficients)

    def get_residual(self) -> torch.tensor:
        """
        Get residual after fit.
        :return: Residual (y_fitted - y).
        """
        return self._calculate_residual(self.coefficients)

    def get_estimate(self) -> torch.tensor:
        """
        Get estimated response vector based on fit.
        :return: Response vector
        """
        return self.fit_function(self.x, self.coefficients)

    def _calculate_residual(self, coefficients: torch.tensor) -> torch.tensor:
        y_fit = self.fit_function(self.x, coefficients)
        return (y_fit - self.y).flatten()

    def _calculate_jacobian(self,
                            x0: torch.tensor):

        jacobian = torch.autograd.functional.jacobian(self._calculate_residual, x0, vectorize=True)

        return jacobian

    @staticmethod
    def _calculate_pseudoinverse(x: torch.tensor) -> torch.tensor:
        """
        Moore-Penrose inverse.
        """
        return torch.linalg.inv(x.T @ x) @ x.T

logging.basicConfig(level=logging.INFO)

NOISE = 3
COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]


def func(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * torch.sin(x)

def apply_homography(H, pts_src):
    pts_dst = H @ pts_src
    pts_dst = pts_dst / pts_dst[-1, :]
    return pts_dst

def fun(src_pts, theta):
    theta = torch.cat([theta, torch.ones([1], dtype=torch.float32)])
    return apply_homography(theta.reshape(3, 3), src_pts)

def main():
    # compute homography

    h_gt = torch.tensor([[1, 0, 10],
                         [0, 1, 20],
                         [0, 0, 1]]).float()

    # sample points
    xs = torch.linspace(0, 40 - 1, 40)
    ys = torch.linspace(0, 30 - 1, 30)
    gridx, gridy = torch.meshgrid(xs, ys)

    src_pts = torch.stack([gridx, gridy])
    src_pts = torch.concat((src_pts, torch.ones(src_pts.shape[1:])[None, :, :]), dim=0)
    # apply homography
    src_pts = src_pts.reshape(3, -1)
    dst_pts = apply_homography(h_gt, src_pts)
    # dst_pts = dst_pts[:2, :]

    # add noise to the first two coordinates
    noise = 0.5
    dst_pts[2, :] += noise * torch.randn(dst_pts[2, :].shape)
    # dst_pts += noise * torch.rand(dst_pts.shape)

    # theta0 = torch.ones((3,3)).flatten()[:-1]
    theta0 = torch.tensor([1.0, 0, 0, 0, 1.0, 0, 0, 0]).float()


    solver = GNSolver(fit_function=fun, max_iter=5, tolerance_difference=10 ** (-4))

    init_guess = theta0
    theta = solver.fit(src_pts, dst_pts, init_guess)
    print(theta)

# def main():
#     # set seed for reproducibility
#     torch.manual_seed(0)
#
#     x = torch.arange(1, 100)
#
#     y = func(x, COEFFICIENTS)
#     yn = y + NOISE * torch.rand(len(x))
#
#     solver = GNSolver(fit_function=func, max_iter=100, tolerance_difference=10 ** (-4))
#     init_guess = 100000 * torch.rand(len(COEFFICIENTS))
#     _ = solver.fit(x, yn, init_guess)
#     fit = solver.get_estimate()
#     residual = solver.get_residual()
#
#     plt.figure()
#     plt.plot(x, y, label="Original, noiseless signal", linewidth=2)
#     plt.plot(x, yn, label="Noisy signal", linewidth=2)
#     plt.plot(x, fit, label="Fit", linewidth=2)
#     plt.plot(x, residual, label="Residual", linewidth=2)
#     plt.title("Gauss-Newton: curve fitting example")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.grid()
#     plt.legend()
#     plt.show()


if __name__ == "__main__":
    main()