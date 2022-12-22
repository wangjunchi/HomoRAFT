import torch
from torch.linalg import pinv
from typing import Callable

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

    def fit(self, x: torch.tensor, y: torch.tensor, init_guess: torch.tensor = None):
        """
        :param x: Dependent variable.
        :param y: Response vector.
        :param init_guess: Initial guess for coefficients.
        :return: Fitted coefficients.
        """
        self.x = x
        self.y = y
        if init_guess is not None:
            self.init_guess = init_guess
        self.coefficients = self.init_guess
        rmse_prev = torch.inf

        for k in range(self.max_iter):
            residual = self.get_residual()
            jacobian = self._calculate_jacobian(self.coefficients, eps=1e-6)
            pseudoinverse = self._calculate_pseudoinverse(jacobian)
            self.coefficients = self.coefficients - pseudoinverse @ residual
            rmse = torch.sqrt(torch.mean(residual ** 2))
            print(f"Iteration {k}: RMSE = {rmse}")
            if torch.abs(rmse - rmse_prev) < self.tolerance_difference:
                print("Terminating iteration due to small RMSE difference.")
                break
            if rmse < self.tolerance:
                print("Terminating iteration due to small RMSE.")
                break
            rmse_prev = rmse
        return self.coefficients

    def predict(self, x: torch.tensor):
        """
        :param x: Dependent variable.
        :return: Predicted response vector.
        """
        return self.fit_function(x, self.coefficients)

    def get_residual(self):
        """
        :return: Residual vector.
        """
        return self._calculate_residual(self.coefficients)

    def _calculate_residual(self, coefficients):
        y_fit = self.fit_function(self.x, coefficients)
        return y_fit - self.y

    def _calculate_jacobian(self, x0, eps=1e-6):
        """
        :param coefficients: Coefficients.
        :param eps: Step size for numerical differentiation.
        :return: Jacobian matrix.
        """
        y0 = self._calculate_residual(x0)

        jacobian = []
        for i, parameter in enumerate(x0):
            x = x0.clone()
            x[i] += eps
            y = self._calculate_residual(x)
            jacobian.append((y - y0) / eps)
        return torch.stack(jacobian).T


    def _calculate_pseudoinverse(self, x):
        """
        :param jacobian: Jacobian matrix.
        :return: Pseudoinverse of Jacobian matrix.
        """
        return pinv(x.T @ x) @ x.T


def func(x, coeff):
    return coeff[0] * x ** 3 + coeff[1] * x ** 2 + coeff[2] * x + coeff[3] + coeff[4] * torch.sin(x)

def main():
    NOISE = 3
    COEFFICIENTS = [-0.001, 0.1, 0.1, 2, 15]

    x = torch.arange(1, 100).double()

    y = func(x, COEFFICIENTS)
    yn = y + NOISE * torch.randn(len(x))

    solver = GNSolver(fit_function=func, max_iter=100, tolerance_difference=10 ** (-6))
    init_guess = 10 * torch.rand(len(COEFFICIENTS))

    solver.fit(x, yn, init_guess=init_guess)
    print(solver.coefficients)

    residual = solver.get_residual()
    print(f"RMSE = {torch.sqrt(torch.mean(residual ** 2))}")


if __name__ == "__main__":
    main()