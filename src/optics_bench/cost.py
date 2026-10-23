"""C = N_forward + alpha * N_adjoint. One wavelength, one 2D FDFD solve, counts as 1."""

from __future__ import annotations


class CostMeter:
    def __init__(self, alpha: float = 1.0):
        if alpha != 1.0:
            raise ValueError("alpha is locked at 1")
        self.alpha = alpha
        self.n_forward = 0
        self.n_adjoint = 0
        # "forward" while the primal is traced; "adjoint" while the vjp runs.
        self.mode = "forward"

    @property
    def C(self) -> float:
        return self.n_forward + self.alpha * self.n_adjoint

    def note_solve(self) -> None:
        if self.mode == "forward":
            self.n_forward += 1
        elif self.mode == "adjoint":
            self.n_adjoint += 1
        else:
            raise RuntimeError(f"unknown cost mode: {self.mode}")

    def as_dict(self) -> dict[str, float]:
        return {
            "n_forward": self.n_forward,
            "n_adjoint": self.n_adjoint,
            "C": self.C,
        }
