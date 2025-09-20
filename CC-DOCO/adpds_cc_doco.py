#!/usr/bin/env python3
"""
ADPDS reproduction script integrated with the CC-DOCO repository.

This file mirrors the functionality of the standalone adpds.py while
living inside CC-DOCO so future extensions (e.g. DC-DOBD) can build on it.
"""
from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Utilities & projections
# =============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def project_l2_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        if n > radius:
            return (radius / n) * x
        return x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.ones_like(n)
    mask = (n > radius)
    scale[mask] = radius / n[mask]
    return x * scale


def _project_l1_ball_vec(v: np.ndarray, radius: float = 1.0, tol: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = np.abs(v).sum()
    if s <= radius:
        return v
    u = np.sort(np.abs(v))[::-1]
    sv = np.cumsum(u)
    rho_idx = np.nonzero(u - (sv - radius) / (np.arange(1, len(u) + 1)) > 0)[0][-1]
    tau = (sv[rho_idx] - radius) / (rho_idx + 1.0)
    vp = np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)
    sp = np.abs(vp).sum()
    if sp > radius + tol:
        vp *= radius / sp
    return vp


def project_l1_ball(X: np.ndarray, radius: float = 1.0, tol: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return _project_l1_ball_vec(X, radius, tol)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = _project_l1_ball_vec(X[i], radius, tol)
    return out


def soft_threshold(x: float | np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

# =============================================================================
# Graph generation & maximum-degree weights (Xiao & Boyd, 2004)
# =============================================================================

def random_geometric_graph_connected(m: int, seed: int = 0, step: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = rng.random((m, 2))

    def build_A(rad: float) -> np.ndarray:
        diff = P[:, None, :] - P[None, :, :]
        D2 = np.sum(diff * diff, axis=2)
        A = (D2 < rad * rad).astype(int)
        np.fill_diagonal(A, 0)
        A = np.maximum(A, A.T)
        return A

    def is_connected(A: np.ndarray) -> bool:
        m = A.shape[0]
        vis = np.zeros(m, dtype=bool)
        stack = [0]; vis[0] = True
        while stack:
            u = stack.pop()
            for v in np.where(A[u] > 0)[0]:
                if not vis[v]:
                    vis[v] = True; stack.append(v)
        return vis.all()

    rad = step
    for _ in range(10000):
        A = build_A(rad)
        if is_connected(A):
            return A, P
        rad += step
    raise RuntimeError("Failed to obtain a connected RGG; increase step or change seed.")


def maximum_degree_weights(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, float]:
    deg = A.sum(axis=1)
    dmax = int(deg.max())
    if dmax <= 0:
        raise ValueError("Isolated nodes found.")
    L = np.diag(deg) - A
    alpha = 1.0 / float(dmax)
    W = np.eye(A.shape[0]) - alpha * L
    return W, L, dmax, alpha

# =============================================================================
# Eq. (17) -- 2D quadratic with ell1-constraint (Figures 2/3/4)
# =============================================================================

@dataclass
class ADPDSConfigEq17:
    c: float
    T: int
    rho: float
    proj: str  # 'l1' or 'l2'
    tol: float = 1e-12
    Ceta: float = 1.0
    Cbeta: float = 1.0
    Cgamma: float = 1.0
    # how to *plot* violation (does not affect updates)
    viol_metric: str = 'post_sum_clip'  # {'post_sum_clip','pre_sum_clip','post_perstep','pre_perstep'}


def generate_data_eq17(T: int, m: int, low: float = 1.0, high: float = 1.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(T, m)), rng.uniform(low, high, size=(T, m))


def best_fixed_wstar_l1_ball(a_vals: np.ndarray, b_vals: np.ndarray, rho: float) -> np.ndarray:
    a_bar = float(np.mean(a_vals)); b_bar = float(np.mean(b_vals))
    if abs(a_bar) + abs(b_bar) <= 1.0:
        return np.array([a_bar, b_bar], dtype=float)
    lam_lo, lam_hi = 0.0, max(rho * abs(a_bar), abs(b_bar))
    for _ in range(80):
        lam = 0.5 * (lam_lo + lam_hi)
        w1 = soft_threshold(a_bar, lam / rho)
        w2 = soft_threshold(b_bar, lam)
        s = abs(w1) + abs(w2)
        if s > 1.0: lam_lo = lam
        else: lam_hi = lam
    lam = lam_hi
    return np.array([soft_threshold(a_bar, lam / rho), soft_threshold(b_bar, lam)], dtype=float)


def run_adpds_eq17(cfg: ADPDSConfigEq17, W: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    T, m = a.shape
    t = np.arange(1, T + 1, dtype=float)
    eta = cfg.Ceta   * t ** (-cfg.c)
    beta= cfg.Cbeta  * t ** (-(0.5 + cfg.c))
    gamma=cfg.Cgamma * t ** (-0.5)

    w = np.zeros((m, 2)); alpha = np.zeros(m)

    # logs for regret & violations (pre/post & per-step)
    w_hist = np.zeros((T, m, 2))
    g_pre_perstep  = np.zeros((T, m))
    g_post_perstep = np.zeros((T, m))

    infeasible = 0

    for k in range(T):
        # decision used at round k
        w_hist[k] = w

        # gradients at current iterate (pre)
        grad_f = np.empty_like(w)
        grad_f[:, 0] = 2.0 * cfg.rho * (w[:, 0] - a[k])
        grad_f[:, 1] = 2.0 * (w[:, 1] - b[k])
        g_w_pre = np.sum(np.abs(w), axis=1) - 1.0
        g_pre_perstep[k] = g_w_pre
        subgrad_g = np.sign(w)

        # dual & primal temporary steps
        grad_alpha = g_w_pre - eta[k] * alpha
        grad_w_L   = grad_f + alpha[:, None] * subgrad_g
        w_hat      = w - beta[k]  * grad_w_L
        alpha_hat  = alpha + gamma[k] * grad_alpha

        # mixing
        w_mixed     = W.dot(w_hat)
        alpha_mixed = W.dot(alpha_hat)

        # projection to B and to R_+
        if cfg.proj == 'l2':
            w_new = project_l2_ball(w_mixed, radius=1.0)
        else:
            w_new = project_l1_ball(w_mixed, radius=1.0, tol=cfg.tol)
        alpha_new = np.maximum(alpha_mixed, 0.0)

        # post-update violation for plotting
        g_w_post = np.sum(np.abs(w_new), axis=1) - 1.0
        g_post_perstep[k] = g_w_post

        # advance
        w, alpha = w_new, alpha_new

        if cfg.proj == 'l1' and np.any(np.sum(np.abs(w), axis=1) > 1.0 + cfg.tol):
            infeasible += 1

    if cfg.proj == 'l1' and infeasible:
        print(f"[Eq17] WARNING: {infeasible} step(s) have ||w||1>1+{cfg.tol:g} after ell1 projection.")

    # regret comparator on K
    wstar_T = np.zeros((T, 2)); ref_cost_Tp = np.zeros(T)
    for Tp in range(1, T + 1):
        wstar = best_fixed_wstar_l1_ball(a[:Tp, :], b[:Tp, :], cfg.rho)
        wstar_T[Tp - 1] = wstar
        ref_cost_Tp[Tp - 1] = cfg.rho * np.sum((wstar[0] - a[:Tp, :]) ** 2) + np.sum((wstar[1] - b[:Tp, :]) ** 2)

    # cumulative losses for regret
    fsum_j_cumsum = np.zeros((T, m))
    for k in range(T):
        wjt = w_hist[k]
        ai = a[k][None, :]; bi = b[k][None, :]
        f_all = cfg.rho * (wjt[:, 0][:, None] - ai) ** 2 + (wjt[:, 1][:, None] - bi) ** 2
        fsum_j = np.sum(f_all, axis=1)
        fsum_j_cumsum[k] = fsum_j if k == 0 else fsum_j_cumsum[k - 1] + fsum_j

    regret_per_jT = np.zeros((T, m))
    for Tp in range(1, T + 1):
        regret_per_jT[Tp - 1] = fsum_j_cumsum[Tp - 1] - ref_cost_Tp[Tp - 1]

    # violation arrays for plotting/aggregation
    pre_pos  = np.maximum(g_pre_perstep,  0.0)
    post_pos = np.maximum(g_post_perstep, 0.0)
    pre_cum  = np.cumsum(g_pre_perstep,  axis=0)
    post_cum = np.cumsum(g_post_perstep, axis=0)

    return dict(
        regret_per_jT=regret_per_jT,
        g_pre_perstep=g_pre_perstep, g_post_perstep=g_post_perstep,
        pre_pos=pre_pos, post_pos=post_pos,
        pre_cum=pre_cum, post_cum=post_cum,
    )

# =============================================================================
# Eq. (18) -- 1D scalar with interval constraint (Figures 6/7)
# =============================================================================

@dataclass
class ADPDSConfigEq18:
    c: float
    T: int
    R: float
    Ceta: float = 1.0
    Cbeta: float = 1.0
    Cgamma: float = 1.0


def generate_data_eq18(T: int, m: int, seed: int = 0,
                       a_low: float = 1.0, a_high: float = 2.0,
                       b_low: float = -0.5, b_high: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(a_low, a_high, size=(T, m))
    b = rng.uniform(b_low, b_high, size=(T, m))
    return a, b


def run_adpds_eq18(cfg: ADPDSConfigEq18, W: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    T, m = a.shape
    t = np.arange(1, T + 1, dtype=float)
    eta = cfg.Ceta   * t ** (-cfg.c)
    beta= cfg.Cbeta  * t ** (-(0.5 + cfg.c))
    gamma=cfg.Cgamma * t ** (-0.5)

    w = np.zeros(m); alpha = np.zeros(m)

    w_hist = np.zeros((T, m))
    g_post_perstep = np.zeros((T, m))

    for k in range(T):
        w_hist[k] = w
        c_it = a[k] - 0.5
        grad_f = 2.0 * (c_it * w + b[k]) * c_it
        subgrad_g = np.sign(w)
        g_w_pre = np.abs(w) - cfg.R
        grad_alpha = g_w_pre - eta[k] * alpha
        grad_w_L = grad_f + alpha * subgrad_g
        w_hat = w - beta[k] * grad_w_L
        alpha_hat = alpha + gamma[k] * grad_alpha
        w_mixed = W.dot(w_hat); alpha_mixed = W.dot(alpha_hat)
        w_new = np.clip(w_mixed, -cfg.R, cfg.R)
        alpha_new = np.maximum(alpha_mixed, 0.0)
        g_post_perstep[k] = np.abs(w_new) - cfg.R
        w, alpha = w_new, alpha_new

    # comparator/regret
    c_all = a - 0.5
    C2_cumsum = np.zeros(T); CB_cumsum = np.zeros(T); wstar_T = np.zeros(T)
    for Tp in range(T):
        C2_cumsum[Tp] = np.sum(c_all[:Tp+1] ** 2)
        CB_cumsum[Tp] = np.sum(c_all[:Tp+1] * b[:Tp+1])
        w_u = 0.0 if C2_cumsum[Tp] <= 0 else - CB_cumsum[Tp] / C2_cumsum[Tp]
        wstar_T[Tp] = np.clip(w_u, -cfg.R, cfg.R)
    ref_cost_Tp = np.zeros(T)
    for Tp in range(T):
        ref_cost_Tp[Tp] = np.sum((c_all[:Tp+1] * wstar_T[Tp] + b[:Tp+1]) ** 2)
    fsum_j_cumsum = np.zeros((T, m))
    for k in range(T):
        wjt = w_hist[k]
        c_it = c_all[k][None, :]; b_it = b[k][None, :]
        f_all = (c_it * wjt[:, None] + b_it) ** 2
        fsum_j = np.sum(f_all, axis=1)
        fsum_j_cumsum[k] = fsum_j if k == 0 else fsum_j_cumsum[k - 1] + fsum_j
    regret_per_jT = np.zeros((T, m))
    for Tp in range(1, T + 1):
        regret_per_jT[Tp - 1] = fsum_j_cumsum[Tp - 1] - ref_cost_Tp[Tp - 1]

    viol_eval_cumsum = np.cumsum(np.maximum(g_post_perstep, 0.0), axis=0)

    return dict(
        regret_per_jT=regret_per_jT,
        violation_eval_cumsum=viol_eval_cumsum,
    )

# =============================================================================
# Plotting helpers
# =============================================================================

def _safe_logy(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 1e-16)


def plot_fig2(min_avg_reg: np.ndarray, max_avg_reg: np.ndarray, T: int, savepath: str) -> None:
    plt.figure(figsize=(6, 4))
    t = np.arange(1, T + 1)
    plt.plot(t, _safe_logy(max_avg_reg), label='Max avg regret')
    plt.plot(t, _safe_logy(min_avg_reg), label='Min avg regret')
    plt.yscale('log')
    plt.xlabel('T (iterations)')
    plt.ylabel('Average regret (log scale)')
    plt.title('Replica of Figure 2 (c = 0.4)')
    plt.grid(True, which='both', linestyle=':')
    plt.legend(); plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()


def plot_fig3(min_avg_viol: np.ndarray, max_avg_viol: np.ndarray, T: int, savepath: str) -> None:
    plt.figure(figsize=(6, 4))
    t = np.arange(1, T + 1)
    plt.plot(t, _safe_logy(max_avg_viol), label='Max avg violation')
    plt.plot(t, _safe_logy(min_avg_viol), label='Min avg violation')
    plt.yscale('log')
    plt.xlabel('T (iterations)')
    plt.ylabel('Average violation (log scale)')
    plt.title('Replica of Figure 3 (c = 0.4)')
    plt.grid(True, which='both', linestyle=':')
    plt.legend(); plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()


def plot_fig4(max_avg_reg_c01: np.ndarray, max_avg_viol_c01: np.ndarray,
              max_avg_reg_c025: np.ndarray, max_avg_viol_c025: np.ndarray,
              T: int, savepath: str) -> None:
    plt.figure(figsize=(6, 4))
    t = np.arange(1, T + 1)
    plt.plot(t, _safe_logy(max_avg_reg_c01), linestyle='-', label='Max avg regret (c=0.1)')
    plt.plot(t, _safe_logy(max_avg_viol_c01), linestyle='--', label='Max avg violation (c=0.1)')
    plt.plot(t, _safe_logy(max_avg_reg_c025), linestyle='-', label='Max avg regret (c=0.25)')
    plt.plot(t, _safe_logy(max_avg_viol_c025), linestyle='--', label='Max avg violation (c=0.25)')
    plt.yscale('log')
    plt.xlabel('T (iterations)')
    plt.ylabel('Average performance (log scale)')
    plt.title('Replica of Figure 4 (c = 0.1 vs 0.25)')
    plt.grid(True, which='both', linestyle=':')
    plt.legend(); plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()


def plot_fig6(avg_reg_node1: np.ndarray, avg_reg_node2: np.ndarray, nodes: Tuple[int,int], T: int, savepath: str, c_val: float) -> None:
    plt.figure(figsize=(6, 4))
    t = np.arange(1, T + 1)
    plt.plot(t, _safe_logy(avg_reg_node1), label=f'Node {nodes[0]}')
    plt.plot(t, _safe_logy(avg_reg_node2), label=f'Node {nodes[1]}')
    plt.yscale('log')
    plt.xlabel('T (iterations)')
    plt.ylabel('Average regret (log scale)')
    plt.title(f'Replica of Figure 6 (scalar case, c={c_val})')
    plt.grid(True, which='both', linestyle=':')
    plt.legend(); plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()


def plot_fig7(avg_viol_node1: np.ndarray, avg_viol_node2: np.ndarray, nodes: Tuple[int,int], T: int, savepath: str, c_val: float) -> None:
    plt.figure(figsize=(6, 4))
    t = np.arange(1, T + 1)
    plt.plot(t, _safe_logy(avg_viol_node1), label=f'Node {nodes[0]}')
    plt.plot(t, _safe_logy(avg_viol_node2), label=f'Node {nodes[1]}')
    plt.yscale('log')
    plt.xlabel('T (iterations)')
    plt.ylabel('Average constraint violation (log scale)')
    plt.title(f'Replica of Figure 7 (scalar case, c={c_val})')
    plt.grid(True, which='both', linestyle=':')
    plt.legend(); plt.tight_layout(); plt.savefig(savepath, dpi=300); plt.close()

# =============================================================================
# Runners for figures
# =============================================================================

def _aggregate_violation(out: Dict[str, np.ndarray], T: int, metric: str) -> np.ndarray:
    tgrid = np.arange(1, T + 1)[:, None]
    if metric == 'post_sum_clip':
        avg = np.maximum(out['post_cum'], 0.0) / tgrid
    elif metric == 'pre_sum_clip':
        avg = np.maximum(out['pre_cum'], 0.0) / tgrid
    elif metric == 'post_perstep':
        avg = np.cumsum(out['post_pos'], axis=0) / tgrid
    elif metric == 'pre_perstep':
        avg = np.cumsum(out['pre_pos'],  axis=0) / tgrid
    else:
        raise ValueError('Unknown viol_metric')
    return avg


def run_eq17_figs(m: int, T: int, rho: float, seed: int, saveprefix: str,
                  proj_fig2: str, proj_fig3: str, proj_fig4: str,
                  tol: float, Ceta: float, Cbeta: float, Cgamma: float,
                  viol_metric: str) -> List[str]:
    # One shared graph+data for all Eq.17 figures
    A, _ = random_geometric_graph_connected(m, seed=seed)
    W, _, _, _ = maximum_degree_weights(A)
    a, b = generate_data_eq17(T, m, low=1.0, high=1.5, seed=seed)

    # Fig.2 (regret)
    out2 = run_adpds_eq17(ADPDSConfigEq17(c=0.4, T=T, rho=rho, proj=proj_fig2, tol=tol,
                                          Ceta=Ceta, Cbeta=Cbeta, Cgamma=Cgamma,
                                          viol_metric=viol_metric), W, a, b)
    tgrid = np.arange(1, T + 1)
    avg_reg_04 = out2['regret_per_jT'] / tgrid[:, None]
    min_avg_reg_04 = np.min(avg_reg_04, axis=1); max_avg_reg_04 = np.max(avg_reg_04, axis=1)

    # Fig.3 (violation)
    out3 = run_adpds_eq17(ADPDSConfigEq17(c=0.4, T=T, rho=rho, proj=proj_fig3, tol=tol,
                                          Ceta=Ceta, Cbeta=Cbeta, Cgamma=Cgamma,
                                          viol_metric=viol_metric), W, a, b)
    avg_viol_04 = _aggregate_violation(out3, T, viol_metric)
    min_avg_viol_04 = np.min(avg_viol_04, axis=1); max_avg_viol_04 = np.max(avg_viol_04, axis=1)

    # Fig.4 (c = 0.1, 0.25)
    out_c01 = run_adpds_eq17(ADPDSConfigEq17(c=0.1, T=T, rho=rho, proj=proj_fig4, tol=tol,
                                             Ceta=Ceta, Cbeta=Cbeta, Cgamma=Cgamma,
                                             viol_metric=viol_metric), W, a, b)
    out_c025 = run_adpds_eq17(ADPDSConfigEq17(c=0.25, T=T, rho=rho, proj=proj_fig4, tol=tol,
                                              Ceta=Ceta, Cbeta=Cbeta, Cgamma=Cgamma,
                                              viol_metric=viol_metric), W, a, b)
    avg_reg_c01 = out_c01['regret_per_jT'] / tgrid[:, None]
    avg_viol_c01 = _aggregate_violation(out_c01, T, viol_metric)
    avg_reg_c025 = out_c025['regret_per_jT'] / tgrid[:, None]
    avg_viol_c025 = _aggregate_violation(out_c025, T, viol_metric)
    max_avg_reg_c01 = np.max(avg_reg_c01, axis=1); max_avg_viol_c01 = np.max(avg_viol_c01, axis=1)
    max_avg_reg_c025 = np.max(avg_reg_c025, axis=1); max_avg_viol_c025 = np.max(avg_viol_c025, axis=1)

    prefix = (saveprefix + '_') if saveprefix else ''
    f1 = f"{prefix}fig1_adpds_fig2.png"; f2 = f"{prefix}fig2_adpds_fig3.png"; f3 = f"{prefix}fig3_adpds_fig4.png"
    plot_fig2(min_avg_reg_04, max_avg_reg_04, T, f1)
    plot_fig3(min_avg_viol_04, max_avg_viol_04, T, f2)
    plot_fig4(max_avg_reg_c01, max_avg_viol_c01, max_avg_reg_c025, max_avg_viol_c025, T, f3)

    return [f1, f2, f3]


def run_eq18_figs(m: int, T: int, R: float, c: float, seed: int, saveprefix: str, nodes: Tuple[int,int] | None) -> Tuple[List[str], Tuple[int,int]]:
    A, _ = random_geometric_graph_connected(m, seed=seed)
    W, _, _, _ = maximum_degree_weights(A)
    a, b = generate_data_eq18(T, m, seed=seed, a_low=1.0, a_high=2.0, b_low=-0.5, b_high=0.5)
    out = run_adpds_eq18(ADPDSConfigEq18(c=c, T=T, R=R), W, a, b)

    t = np.arange(1, T + 1)
    avg_reg = out['regret_per_jT'] / t[:, None]
    avg_viol = out['violation_eval_cumsum'] / t[:, None]

    if nodes is None:
        rng = np.random.default_rng(seed)
        i, j = rng.choice(m, size=2, replace=False).tolist()
        nodes = (i, j)

    i, j = nodes

    prefix = (saveprefix + '_') if saveprefix else ''
    f4 = f"{prefix}fig4_adpds_fig6.png"; f5 = f"{prefix}fig5_adpds_fig7.png"
    plot_fig6(avg_reg[:, i], avg_reg[:, j], nodes, T, f4, c)
    plot_fig7(avg_viol[:, i], avg_viol[:, j], nodes, T, f5, c)

    return [f4, f5], nodes

# =============================================================================
# Main
# =============================================================================

def parse_nodes(arg: str | None, m: int) -> Tuple[int,int] | None:
    if arg is None: return None
    parts = arg.split(',')
    if len(parts) != 2: raise ValueError('--nodes must be "i,j"')
    i, j = int(parts[0]), int(parts[1])
    if not (0 <= i < m and 0 <= j < m) or i == j:
        raise ValueError('--nodes indices invalid or identical')
    return (i, j)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--m', type=int, default=40)
    p.add_argument('--rho', type=float, default=1.5)
    p.add_argument('--R', type=float, default=0.1)
    p.add_argument('--c18', type=float, default=0.4)
    p.add_argument('--nodes', type=str, default=None)
    p.add_argument('--run', type=str, nargs='*', default=['eq17','eq18'], choices=['eq17','eq18'])
    p.add_argument('--saveprefix', type=str, default='')

    # Eq.17 figure-specific projection sets
    p.add_argument('--eq17_proj_fig2', type=str, default='l1', choices=['l1','l2'])
    p.add_argument('--eq17_proj_fig3', type=str, default='l2', choices=['l1','l2'])
    p.add_argument('--eq17_proj_fig4', type=str, default='l2', choices=['l1','l2'])

    # step-size multipliers
    p.add_argument('--Ceta',   type=float, default=1.0)
    p.add_argument('--Cbeta',  type=float, default=1.0)
    p.add_argument('--Cgamma', type=float, default=1.0)

    # plotting-time violation metric
    p.add_argument('--viol_metric', type=str, default='post_sum_clip',
                   choices=['post_sum_clip','pre_sum_clip','post_perstep','pre_perstep'])

    # tolerance for ell1 feasibility warnings
    p.add_argument('--eq17_tol', type=float, default=1e-12)

    args = p.parse_args()

    set_seed(args.seed)

    outputs: List[str] = []

    if 'eq17' in args.run:
        outs = run_eq17_figs(m=args.m, T=args.T, rho=args.rho, seed=args.seed, saveprefix=args.saveprefix,
                              proj_fig2=args.eq17_proj_fig2, proj_fig3=args.eq17_proj_fig3,
                              proj_fig4=args.eq17_proj_fig4, tol=args.eq17_tol,
                              Ceta=args.Ceta, Cbeta=args.Cbeta, Cgamma=args.Cgamma,
                              viol_metric=args.viol_metric)
        outputs.extend(outs)

    if 'eq18' in args.run:
        nodes = parse_nodes(args.nodes, args.m)
        outs, nodes_used = run_eq18_figs(m=args.m, T=args.T, R=args.R, c=args.c18, seed=args.seed, saveprefix=args.saveprefix, nodes=nodes)
        outputs.extend(outs)
        print('Eq.18 picked nodes:', nodes_used)

    print('Saved files:')
    for f in outputs:
        print('  ', f)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(1)