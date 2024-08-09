import pandas as pd
import random
import numpy as np
from typing import Union, Literal, Optional, Callable


class LinearRegression:
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: Union[int, float, Callable] = 0.1,
        metric: Optional[Literal["mae", "mse", "rmse", "mape", "r2"]] = None,
        regularisation: Optional[Literal["l1", "l2", "elasticnet"]] = None,
        l1_coef: Optional[Union[int, float]] = None,
        l2_coef: Optional[Union[int, float]] = None,
        sgd_sample: Optional[Union[int, float]] = None,
        random_state: Optional[int] = 42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.reg = regularisation
        self.random_state = random_state
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self._best_metric = None
        self._weights = None

    def __repr__(self):
        return f"LinearRegression class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, metric={self.metric}, regularisation={self.reg}"

    def calc_metric(self, y, y_pred):
        metrics = {
            "mae": ((y - y_pred).abs()).mean(),
            "mse": ((y - y_pred) ** 2).mean(),
            "rmse": ((y - y_pred) ** 2).mean() ** 0.5,
            "mape": (((y - y_pred) / y).abs() * 100).mean(),
            "r2": 1 - (((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        }
        if self.metric is None:
            return None
        return metrics[self.metric]

    def calc_regulatisation(self, marker: Literal["loss", "grad"]):
        if self.reg is None:
            return 0
        if (
            (self.reg == "l1" and self.l1_coef is None)
            or (self.reg == "l2" and self.l2_coef is None)
            or (
                self.reg == "elssticnet"
                and (self.l1_coef is None or self.l2_coef is None)
            )
        ):
            raise ValueError("Not enough arguments for estimation")
        if self.reg == "l1":
            return (
                self.l1_coef * np.abs(self._weights).sum()
                if marker == "loss"
                else self.l1_coef * np.sign(self._weights)
            )
        if self.reg == "l2":
            return (
                self.l2_coef * (self._weights**2).sum()
                if marker == "loss"
                else self.l2_coef * 2 * self._weights
            )
        if self.reg == "elasticnet":
            return (
                self.l1_coef * np.abs(self._weights).sum()
                + self.l2_coef * (self._weights**2).sum()
                if marker == "loss"
                else self.l1_coef * np.sign(self._weights)
                + self.l2_coef * 2 * self._weights
            )
        raise ValueError("Invalid regularisation method")

    @property
    def weights(self):
        return self._weights[1:]

    @property
    def best_metric(self):
        return self._best_metric

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Optional[int] = 10):
        X.insert(loc=0, column="temp", value=np.ones(len(X)))
        lines_count, feature_count = X.shape
        self._weights = np.ones(feature_count)
        random.seed(self.random_state)
        for i in range(1, self.n_iter + 1):
            sample_rows_idx = (
                random.sample(
                    range(lines_count),
                    (
                        self.sgd_sample
                        if isinstance(self.sgd_sample, int)
                        else int(lines_count * self.sgd_sample)
                    ),
                )
                if self.sgd_sample
                else None
            )
            t_X = X.iloc[sample_rows_idx] if self.sgd_sample else X
            t_y = y.iloc[sample_rows_idx] if self.sgd_sample else y
            y_pred = t_X.dot(self._weights)
            mse = ((t_y - y_pred) ** 2).mean() + self.calc_regulatisation("loss")
            if verbose and (not i % verbose or i == 0):
                print(
                    f"{i} | loss: {mse} | metric: {self.calc_metric(y, X.dot(self._weights))}"
                )
            grad = (
                ((y_pred - t_y) @ t_X) * 2 / t_X.shape[0]
            ) + self.calc_regulatisation("grad")
            learning_rate = (
                self.learning_rate
                if isinstance(self.learning_rate, float)
                or isinstance(self.learning_rate, int)
                else self.learning_rate(i)
            )
            if isinstance(self.learning_rate, float) or isinstance(
                self.learning_rate, int
            ):
                self._weights = self._weights - self.learning_rate * grad
            else:
                self._weights = self._weights - self.learning_rate(i) * grad
        self._best_metric = self.calc_metric(y, X.dot(self._weights))

    def predict(self, X):
        X.insert(loc=0, column="temp", value=np.ones(len(X)))
        return X.dot(self._weights)
