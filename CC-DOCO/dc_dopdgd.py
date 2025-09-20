#!/usr/bin/env python3
"""
DC-DOPDGD implementation built on top of CC-DOCO utilities.

The routines mirror the metrics returned by `adpds_cc_doco.run_adpds_eq17`
and `run_adpds_eq18` so downstream plotting/comparison code can operate on
a common interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from compressor import RandSparse, TopSparse, RandGossip, RandQuant
from adpds_cc_doco import (
    project_l1_ball,
    project_l2_ball,
    best_fixed_wstar_l1_ball,
)


# =============================================================================
# Compressor helpers
# =============================================================================


class CompressorWrapper:
    """Thin wrapper that normalises the interface and statistics."""

    def __init__(self, compressor_type: str, w: Optional[float] = None, s: Optional[int] = None) -> None:
        compressor_type = compressor_type.lower()
        if compressor_type not in {"no", "rand", "top", "gossip", "quant"}:
            raise ValueError(f"Unsupported compressor type: {compressor_type}")
        if compressor_type == "quant":
            if s is None:
                raise ValueError("Quantised compressor requires `s` levels")
            self._impl = RandQuant(s)
        elif compressor_type == "rand":
            if w is None:
                raise ValueError("Random sparse compressor requires `w`")
            self._impl = RandSparse(w)
        elif compressor_type == "top":
            if w is None:
                raise ValueError("Top-k sparse compressor requires `w`")
            self._impl = TopSparse(w)
        elif compressor_type == "gossip":
            if w is None:
                raise ValueError("Random gossip compressor requires `w`")
            self._impl = RandGossip(w)
        else:  # "no" => deterministic gossip with probability 1
            self._impl = RandGossip(1.0)
        self.compressor_type = compressor_type
        self.w = w
        self.s = s

    def reset(self, dimension: int) -> None:
        if hasattr(self._impl, "update_w"):
            self._impl.update_w(dimension)
        if hasattr(self._impl, "total_bit"):
            self._impl.total_bit = 0

    def compress(self, vec: np.ndarray, times: float) -> np.ndarray:
        return self._impl.compress(vec, times)

    @property
    def total_bit(self) -> float:
        return getattr(self._impl, "total_bit", 0.0)


# =============================================================================
# Config dataclasses
# =============================================================================


@dataclass
class DCDOPDGDConfigEq17:
    c: float
    T: int
    rho: float
    proj: str  # 'l1' or 'l2'
    gamma: float = 0.5
    omega: float = 0.0
    Cbeta: float = 1.0
    Ceta: float = 1.0
    Calpha: float = 1.0
    beta_exp: Optional[float] = None
    eta_exp: Optional[float] = None
    alpha_exp: Optional[float] = None
    tol: float = 1e-12


@dataclass
class DCDOPDGDConfigEq18:
    c: float
    T: int
    R: float
    gamma: float = 0.5
    omega: float = 0.0
    Cbeta: float = 1.0
    Ceta: float = 1.0
    Calpha: float = 1.0
    beta_exp: Optional[float] = None
    eta_exp: Optional[float] = None
    alpha_exp: Optional[float] = None


# =============================================================================
# Core algorithmic helpers
# =============================================================================


def _build_schedules(Cbeta: float, Ceta: float, Calpha: float, c: float,
                     beta_exp: Optional[float], eta_exp: Optional[float], alpha_exp: Optional[float],
                     T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(1, T + 1, dtype=float)
    beta_e = beta_exp if beta_exp is not None else (0.5 + c)
    eta_e = eta_exp if eta_exp is not None else c
    alpha_e = alpha_exp if alpha_exp is not None else 0.0
    beta = Cbeta * (t ** (-beta_e))
    eta = Ceta * (t ** (-eta_e))
    alpha = Calpha * (t ** (-alpha_e))
    return beta, eta, alpha


def _compress_diff(compressor: CompressorWrapper, diff: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    out = np.zeros_like(diff)
    for i in range(diff.shape[0]):
        deg = degrees[i]
        out[i] = compressor.compress(diff[i], deg if deg > 0 else 1.0)
    return out


# =============================================================================
# Eq.17 (2D) routine
# =============================================================================


def run_dcdopdgd_eq17(cfg: DCDOPDGDConfigEq17, W: np.ndarray, A: np.ndarray,
                       a: np.ndarray, b: np.ndarray, compressor: CompressorWrapper) -> Dict[str, np.ndarray]:
    T, m = a.shape
    beta, eta, alpha_seq = _build_schedules(cfg.Cbeta, cfg.Ceta, cfg.Calpha, cfg.c,
                                            cfg.beta_exp, cfg.eta_exp, cfg.alpha_exp, T)
    proj_radius = max(1.0 - cfg.omega, 1e-12)

    compressor.reset(dimension=2)
    degrees = A.sum(axis=1)

    x = np.zeros((m, 2))
    x_hat = np.zeros_like(x)
    y = np.zeros_like(x)
    lam = np.zeros(m)

    x_hist = np.zeros((T, m, 2))
    g_pre_perstep = np.zeros((T, m))
    g_post_perstep = np.zeros((T, m))
    bits_history = np.zeros(T)
    prev_bits = 0.0

    for k in range(T):
        x_hist[k] = x

        q = _compress_diff(compressor, x - x_hat, degrees)
        x_hat = x_hat + q
        y = y + W.dot(q)

        grad_f = np.empty_like(x)
        grad_f[:, 0] = 2.0 * cfg.rho * (x[:, 0] - a[k])
        grad_f[:, 1] = 2.0 * (x[:, 1] - b[k])

        g_vals = np.sum(np.abs(x), axis=1) - 1.0
        g_pre_perstep[k] = g_vals
        grad_g = np.sign(x)
        grad_L = grad_f + lam[:, None] * grad_g

        x_candidate = x + cfg.gamma * (y - x_hat) - beta[k] * grad_L
        if cfg.proj == 'l2':
            x_new = project_l2_ball(x_candidate, radius=proj_radius)
        else:
            x_new = project_l1_ball(x_candidate, radius=proj_radius, tol=cfg.tol)

        g_post = np.sum(np.abs(x_new), axis=1) - 1.0
        g_post_perstep[k] = g_post

        grad_lambda = g_vals - alpha_seq[k] * lam
        lam_temp = lam + eta[k] * grad_lambda
        lam_mixed = W.dot(lam_temp)
        lam = np.maximum(lam_mixed, 0.0)
        x = x_new

        bits_history[k] = compressor.total_bit - prev_bits
        prev_bits = compressor.total_bit

    # regret comparator on K (reuse ADPDS helper)
    wstar_T = np.zeros((T, 2))
    ref_cost_Tp = np.zeros(T)
    for Tp in range(1, T + 1):
        wstar = best_fixed_wstar_l1_ball(a[:Tp, :], b[:Tp, :], cfg.rho)
        wstar_T[Tp - 1] = wstar
        ref_cost_Tp[Tp - 1] = cfg.rho * np.sum((wstar[0] - a[:Tp, :]) ** 2) + np.sum((wstar[1] - b[:Tp, :]) ** 2)

    fsum_j_cumsum = np.zeros((T, m))
    for k in range(T):
        xjt = x_hist[k]
        ai = a[k][None, :]
        bi = b[k][None, :]
        f_all = cfg.rho * (xjt[:, 0][:, None] - ai) ** 2 + (xjt[:, 1][:, None] - bi) ** 2
        fsum_j = np.sum(f_all, axis=1)
        fsum_j_cumsum[k] = fsum_j if k == 0 else fsum_j_cumsum[k - 1] + fsum_j

    regret_per_jT = np.zeros((T, m))
    for Tp in range(1, T + 1):
        regret_per_jT[Tp - 1] = fsum_j_cumsum[Tp - 1] - ref_cost_Tp[Tp - 1]

    pre_pos = np.maximum(g_pre_perstep, 0.0)
    post_pos = np.maximum(g_post_perstep, 0.0)
    pre_cum = np.cumsum(g_pre_perstep, axis=0)
    post_cum = np.cumsum(g_post_perstep, axis=0)

    return dict(
        regret_per_jT=regret_per_jT,
        g_pre_perstep=g_pre_perstep,
        g_post_perstep=g_post_perstep,
        pre_pos=pre_pos,
        post_pos=post_pos,
        pre_cum=pre_cum,
        post_cum=post_cum,
        violation_eval_cumsum=np.cumsum(post_pos, axis=0),
        bits_history=bits_history,
    )


# =============================================================================
# Eq.18 (1D) routine
# =============================================================================


def run_dcdopdgd_eq18(cfg: DCDOPDGDConfigEq18, W: np.ndarray, A: np.ndarray,
                       a: np.ndarray, b: np.ndarray, compressor: CompressorWrapper) -> Dict[str, np.ndarray]:
    T, m = a.shape
    beta, eta, alpha_seq = _build_schedules(cfg.Cbeta, cfg.Ceta, cfg.Calpha, cfg.c,
                                            cfg.beta_exp, cfg.eta_exp, cfg.alpha_exp, T)
    proj_radius = max(cfg.R * (1.0 - cfg.omega), 1e-12)

    compressor.reset(dimension=1)
    degrees = A.sum(axis=1)

    x = np.zeros(m)
    x_hat = np.zeros_like(x)
    y = np.zeros_like(x)
    lam = np.zeros(m)

    x_hist = np.zeros((T, m))
    g_post_perstep = np.zeros((T, m))
    g_pre_perstep = np.zeros((T, m))
    bits_history = np.zeros(T)
    prev_bits = 0.0

    c_all = a - 0.5

    for k in range(T):
        x_hist[k] = x

        q = _compress_diff(compressor, (x - x_hat)[:, None], degrees)[:, 0]
        x_hat = x_hat + q
        y = y + W.dot(q)

        g_vals = np.abs(x) - cfg.R
        g_pre_perstep[k] = g_vals
        grad_f = 2.0 * (c_all[k] * x + b[k]) * c_all[k]
        grad_g = np.sign(x)
        grad_L = grad_f + lam * grad_g

        x_candidate = x + cfg.gamma * (y - x_hat) - beta[k] * grad_L
        x_new = np.clip(x_candidate, -proj_radius, proj_radius)

        g_post = np.abs(x_new) - cfg.R
        g_post_perstep[k] = g_post

        grad_lambda = g_vals - alpha_seq[k] * lam
        lam_temp = lam + eta[k] * grad_lambda
        lam_mixed = W.dot(lam_temp)
        lam = np.maximum(lam_mixed, 0.0)
        x = x_new

        bits_history[k] = compressor.total_bit - prev_bits
        prev_bits = compressor.total_bit

    # comparator/regret
    C2_cumsum = np.zeros(T)
    CB_cumsum = np.zeros(T)
    wstar_T = np.zeros(T)
    for Tp in range(T):
        C2_cumsum[Tp] = np.sum(c_all[:Tp + 1] ** 2)
        CB_cumsum[Tp] = np.sum(c_all[:Tp + 1] * b[:Tp + 1])
        w_u = 0.0 if C2_cumsum[Tp] <= 0 else -CB_cumsum[Tp] / C2_cumsum[Tp]
        wstar_T[Tp] = np.clip(w_u, -cfg.R, cfg.R)
    ref_cost_Tp = np.zeros(T)
    for Tp in range(T):
        ref_cost_Tp[Tp] = np.sum((c_all[:Tp + 1] * wstar_T[Tp] + b[:Tp + 1]) ** 2)

    fsum_j_cumsum = np.zeros((T, m))
    for k in range(T):
        xjt = x_hist[k]
        c_it = c_all[k][None, :]
        b_it = b[k][None, :]
        f_all = (c_it * xjt[:, None] + b_it) ** 2
        fsum_j = np.sum(f_all, axis=1)
        fsum_j_cumsum[k] = fsum_j if k == 0 else fsum_j_cumsum[k - 1] + fsum_j

    regret_per_jT = np.zeros((T, m))
    for Tp in range(1, T + 1):
        regret_per_jT[Tp - 1] = fsum_j_cumsum[Tp - 1] - ref_cost_Tp[Tp - 1]

    pre_pos = np.maximum(g_pre_perstep, 0.0)
    post_pos = np.maximum(g_post_perstep, 0.0)
    pre_cum = np.cumsum(g_pre_perstep, axis=0)
    post_cum = np.cumsum(g_post_perstep, axis=0)

    return dict(
        regret_per_jT=regret_per_jT,
        g_pre_perstep=g_pre_perstep,
        g_post_perstep=g_post_perstep,
        pre_pos=pre_pos,
        post_pos=post_pos,
        pre_cum=pre_cum,
        post_cum=post_cum,
        violation_eval_cumsum=np.cumsum(post_pos, axis=0),
        bits_history=bits_history,
    )