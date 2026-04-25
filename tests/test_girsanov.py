"""Unit tests for the Girsanov density computation.

Tests verify mathematical properties of Z that must hold regardless of
the trained network: Z_0 = 1, Z >= 0, and Z = 1 everywhere when phi = 0.
Two reference implementations (multiplicative Milstein and log-space Euler)
are tested side-by-side so regressions are caught on both.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# Reference implementations (standalone, no model dependency)
# ---------------------------------------------------------------------------

def _z_multiplicative(phi_eval: torch.Tensor, dw: torch.Tensor, dt: float,
                       milstein: bool = True) -> torch.Tensor:
    """Multiplicative discretisation of Z with optional Milstein correction.

    Args:
        phi_eval: market price of risk, shape (N, T, 1).
        dw: Brownian increments, shape (N, T, 1).
        dt: time step size.
        milstein: whether to include the second-order correction.

    Returns:
        Z paths, shape (N, T, 1).
    """
    N = phi_eval.shape[0]
    factors = 1.0 - phi_eval * dw
    if milstein:
        factors = factors + 0.5 * phi_eval ** 2 * (dw ** 2 - dt)
    ones = torch.ones(N, 1, 1)
    delta_z = torch.cat([ones, factors], dim=1)[:, :-1, :]
    return torch.cumprod(delta_z, dim=1)


def _z_log_space(phi_eval: torch.Tensor, dw: torch.Tensor, dt: float) -> torch.Tensor:
    """Log-space (Euler-Maruyama on log Z) discretisation.

    Always positive by construction; no clamping needed.

    Args:
        phi_eval: market price of risk, shape (N, T, 1).
        dw: Brownian increments, shape (N, T, 1).
        dt: time step size.

    Returns:
        Z paths, shape (N, T, 1).
    """
    N = phi_eval.shape[0]
    log_factors = -phi_eval * dw - 0.5 * phi_eval ** 2 * dt
    zeros = torch.zeros(N, 1, 1)
    log_delta_z = torch.cat([zeros, log_factors], dim=1)[:, :-1, :]
    return torch.exp(torch.cumsum(log_delta_z, dim=1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    torch.manual_seed(0)
    return None


def test_z0_multiplicative_equals_one(rng):
    """Z_0 must equal 1 for all paths (multiplicative form)."""
    N, T = 200, 50
    phi = torch.randn(N, T, 1) * 0.3
    dw = torch.randn(N, T, 1) * 0.01
    Z = _z_multiplicative(phi, dw, dt=0.01)
    assert torch.allclose(Z[:, 0, 0], torch.ones(N)), \
        "Z_0 should be 1 for all paths"


def test_z0_log_space_equals_one(rng):
    """Z_0 must equal 1 for all paths (log-space form)."""
    N, T = 200, 50
    phi = torch.randn(N, T, 1) * 0.3
    dw = torch.randn(N, T, 1) * 0.01
    Z = _z_log_space(phi, dw, dt=0.01)
    assert torch.allclose(Z[:, 0, 0], torch.ones(N)), \
        "Z_0 should be 1 for all paths"


def test_log_space_always_positive(rng):
    """Log-space Z is positive for any phi, including extreme values."""
    N, T = 500, 100
    phi = torch.randn(N, T, 1) * 10.0   # deliberately extreme
    dw = torch.randn(N, T, 1) * 0.1
    Z = _z_log_space(phi, dw, dt=0.01)
    assert (Z > 0).all(), "Log-space Z must be strictly positive"


def test_multiplicative_can_go_negative():
    """Multiplicative Z can produce negative values when phi is large.

    A single Milstein factor is negative when phi > 1/sqrt(dt).
    With dt=0.0025 (n_steps=200 over T=0.5) this threshold is phi=20.
    The kappa parameterisation reaches phi_2 >> 20 near V=0 when the
    Feller condition is violated, making this a realistic failure mode.

    Root analysis: factor = 0.5*phi²*dW² - phi*dW + (1 - 0.5*phi²*dt)
    has real roots iff phi²*dt > 1, i.e. phi > 1/sqrt(dt).
    For phi=25, dt=0.0025: factor < 0 for dW in (0.01, 0.07),
    which has probability ~34% per step under dW ~ N(0, dt).
    """
    torch.manual_seed(1)
    N, T = 500, 10
    dt = 0.0025
    # phi=25 >> 1/sqrt(0.0025)=20 threshold; dW ~ N(0, dt) as in the model
    phi = torch.ones(N, T, 1) * 25.0
    dw = torch.randn(N, T, 1) * (dt ** 0.5)
    Z = _z_multiplicative(phi, dw, dt=dt)
    has_negative = (Z < 0).any()
    assert has_negative, \
        "Expected multiplicative Z to produce negative values for phi=25 > 1/sqrt(dt)=20 — " \
        "this test documents the known instability in the kappa parameterisation near V=0"


def test_phi_zero_gives_z_one(rng):
    """When phi = 0 (no measure change), Z must equal 1 everywhere."""
    N, T = 100, 50
    phi = torch.zeros(N, T, 1)
    dw = torch.randn(N, T, 1) * 0.01
    dt = 0.01

    Z_mult = _z_multiplicative(phi, dw, dt)
    Z_log = _z_log_space(phi, dw, dt)

    assert torch.allclose(Z_mult, torch.ones_like(Z_mult)), \
        "phi=0 → Z=1 everywhere (multiplicative)"
    assert torch.allclose(Z_log, torch.ones_like(Z_log)), \
        "phi=0 → Z=1 everywhere (log-space)"


def test_milstein_shifts_z_vs_euler():
    """Milstein correction changes Z relative to plain Euler.

    For non-zero phi the two must differ — if they're identical
    the Milstein term isn't being applied.
    """
    torch.manual_seed(2)
    N, T = 100, 20
    phi = torch.randn(N, T, 1) * 1.0
    dw = torch.randn(N, T, 1) * 0.05
    dt = 0.0025

    Z_euler = _z_multiplicative(phi, dw, dt, milstein=False)
    Z_milstein = _z_multiplicative(phi, dw, dt, milstein=True)
    assert not torch.allclose(Z_euler, Z_milstein), \
        "Milstein and Euler should produce different Z values"


def test_old_buggy_milstein_corrupts_z0():
    """Reproduce the old Milstein alignment bug.

    Old code: shift delta_Z first, then add milstein_term.  This
    adds milstein_term[0] to the Z_0=1 slot, corrupting the initial condition.
    The fixed code adds milstein_term before the shift.
    """
    torch.manual_seed(3)
    N, T = 50, 10
    phi = torch.ones(N, T, 1)   # constant phi = 1 for easy arithmetic
    dw = torch.ones(N, T, 1) * 0.1
    dt = 0.01

    # --- old (buggy) ordering ---
    delta_z_old = 1.0 - phi * dw
    ones = torch.ones(N, 1, 1)
    delta_z_old = torch.cat([ones, delta_z_old], dim=1)[:, :-1, :]
    milstein_term = 0.5 * phi ** 2 * (dw ** 2 - dt)
    delta_z_old = delta_z_old + milstein_term   # bug: milstein added after shift
    z0_old = delta_z_old[:, 0, 0]

    # Expected corrupt value: 1 + milstein(phi=1, dW=0.1, dt=0.01)
    expected_corrupt = 1.0 + 0.5 * 1.0 ** 2 * (0.1 ** 2 - 0.01)
    assert torch.allclose(z0_old, torch.full((N,), expected_corrupt)), \
        "Old code should corrupt Z_0"

    # --- fixed ordering (milstein before shift) ---
    z_fixed = _z_multiplicative(phi, dw, dt, milstein=True)
    assert torch.allclose(z_fixed[:, 0, 0], torch.ones(N)), \
        "Fixed code should have Z_0 = 1"
