#!/usr/bin/env python3
"""
Comparison driver for ADPDS and DC-DOPDGD under various compression rates
and network topologies.

The script relies on the standalone `adpds_cc_doco.py` baseline and the new
`dc_dopdgd.py` module so that both algorithms can be run on identical data
and communication graphs.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from adpds_cc_doco import (
    ADPDSConfigEq17,
    ADPDSConfigEq18,
    _aggregate_violation,
    generate_data_eq17,
    generate_data_eq18,
    maximum_degree_weights,
    random_geometric_graph_connected,
    run_adpds_eq17,
    run_adpds_eq18,
)
from dc_dopdgd import (
    CompressorWrapper,
    DCDOPDGDConfigEq17,
    DCDOPDGDConfigEq18,
    run_dcdopdgd_eq17,
    run_dcdopdgd_eq18,
)


SUPPORTED_ALGOS = {"adpds", "dc-dopdgd"}
SUPPORTED_TOPOLOGIES = {"geom", "ring", "star", "full", "random"}


def build_adjacency(topology: str, m: int, seed: int, edge_num: int | None) -> np.ndarray:
    topology = topology.lower()
    if topology == "geom":
        A, _ = random_geometric_graph_connected(m, seed=seed)
        return A.astype(int)
    if topology == "ring":
        A = np.zeros((m, m), dtype=int)
        for i in range(m):
            A[i, (i - 1) % m] = 1
            A[i, (i + 1) % m] = 1
        np.fill_diagonal(A, 0)
        return A
    if topology == "star":
        A = np.zeros((m, m), dtype=int)
        for i in range(1, m):
            A[0, i] = 1
            A[i, 0] = 1
        return A
    if topology == "full":
        A = np.ones((m, m), dtype=int)
        np.fill_diagonal(A, 0)
        return A
    if topology == "random":
        if edge_num is None:
            edge_num = max(m, 2 * m)
        rng = np.random.default_rng(seed)
        for _ in range(100):
            G = nx.gnm_random_graph(m, edge_num, seed=int(rng.integers(0, 1 << 31)))
            if nx.is_connected(G):
                A = nx.to_numpy_array(G, dtype=int)
                np.fill_diagonal(A, 0)
                A = np.maximum(A, A.T)
                return A
        raise RuntimeError("Failed to build a connected random graph after 100 attempts")
    raise ValueError(f"Unsupported topology: {topology}")


def label_for(algo: str, compressor_type: str, rate: float | None) -> str:
    if algo == "dc-dopdgd" and rate is not None:
        return f"{algo} (rate={rate:.2f})"
    return algo


def label_for_compressor(comp: str, rate: float | None, quant_levels: int) -> str:
    comp = comp.lower()
    if comp == 'quant':
        return f"{comp} (s={quant_levels})"
    if rate is not None:
        return f"{comp} (rate={rate:.2f})"
    return comp


def plot_curves(t: np.ndarray, curves: Iterable[Tuple[str, np.ndarray]], ylabel: str,
                title: str, savepath: Path, logscale: bool = True) -> None:
    plt.figure(figsize=(6, 4))
    for label, values in curves:
        vals = np.maximum(values, 1e-16) if logscale else values
        plt.plot(t, vals, label=label)
    if logscale:
        plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


def summarise_eq17(out: Dict[str, np.ndarray], T: int, viol_metric: str) -> Dict[str, np.ndarray]:
    t = np.arange(1, T + 1)
    avg_reg = out['regret_per_jT'] / t[:, None]
    reg_max = np.max(avg_reg, axis=1)
    reg_min = np.min(avg_reg, axis=1)
    reg_avg = np.mean(avg_reg, axis=1)
    viol_matrix = _aggregate_violation(out, T, viol_metric)
    viol_max = np.max(viol_matrix, axis=1)
    viol_min = np.min(viol_matrix, axis=1)
    viol_avg = np.mean(viol_matrix, axis=1)
    bits_curve = np.cumsum(out.get('bits_history', np.zeros(T)))
    return dict(reg_max=reg_max, reg_min=reg_min, reg_avg=reg_avg,
                viol_max=viol_max, viol_min=viol_min, viol_avg=viol_avg,
                bits=bits_curve)


def summarise_eq18(out: Dict[str, np.ndarray], T: int) -> Dict[str, np.ndarray]:
    t = np.arange(1, T + 1)
    avg_reg = out['regret_per_jT'] / t[:, None]
    reg_max = np.max(avg_reg, axis=1)
    reg_min = np.min(avg_reg, axis=1)
    reg_avg = np.mean(avg_reg, axis=1)
    viol_cum = out.get('violation_eval_cumsum')
    if viol_cum is None:
        viol_cum = np.maximum(out['post_cum'], 0.0)
    avg_viol = np.maximum(viol_cum, 0.0) / t[:, None]
    viol_max = np.max(avg_viol, axis=1)
    viol_min = np.min(avg_viol, axis=1)
    viol_avg = np.mean(avg_viol, axis=1)
    bits_curve = np.cumsum(out.get('bits_history', np.zeros(T)))
    return dict(reg_max=reg_max, reg_min=reg_min, reg_avg=reg_avg,
                viol_max=viol_max, viol_min=viol_min, viol_avg=viol_avg,
                bits=bits_curve)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--problem', choices=['eq17', 'eq18'], default='eq17')
    p.add_argument('--algos', nargs='*', default=['adpds', 'dc-dopdgd'])
    p.add_argument('--topologies', nargs='*', default=['geom', 'ring', 'star', 'full'])
    p.add_argument('--compressor', type=str, default='rand',
                   choices=['no', 'rand', 'top', 'gossip', 'quant'])
    p.add_argument('--compress-rates', nargs='*', type=float, default=[1.0])
    p.add_argument('--quant-levels', type=int, default=2,
                   help='Used when compressor=quant.')
    p.add_argument('--edge-num', type=int, default=None,
                   help='Number of edges for random topology (optional).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--T', type=int, default=50)
    p.add_argument('--m', type=int, default=40)
    p.add_argument('--rho', type=float, default=1.5)
    p.add_argument('--proj', choices=['l1', 'l2'], default='l1')
    p.add_argument('--adpds-proj-regret', choices=['l1', 'l2'], default=None,
                   help='Override ADPDS projection for regret curves (Eq.17 only).')
    p.add_argument('--adpds-proj-violation', choices=['l1', 'l2'], default=None,
                   help='Override ADPDS projection for violation curves (Eq.17 only).')
    p.add_argument('--R', type=float, default=0.1)
    p.add_argument('--c', type=float, default=0.4)
    p.add_argument('--c18', type=float, default=0.4)
    p.add_argument('--adpds-Ceta', type=float, default=1.0)
    p.add_argument('--adpds-Cbeta', type=float, default=1.0)
    p.add_argument('--adpds-Cgamma', type=float, default=1.0)
    p.add_argument('--eq17-tol', type=float, default=1e-12)
    p.add_argument('--viol-metric', type=str, default='post_sum_clip',
                   choices=['post_sum_clip', 'pre_sum_clip', 'post_perstep', 'pre_perstep'])
    p.add_argument('--dc-Cbeta', type=float, default=1.0)
    p.add_argument('--dc-Ceta', type=float, default=1.0)
    p.add_argument('--dc-Calpha', type=float, default=1.0)
    p.add_argument('--dc-gamma', type=float, default=0.5)
    p.add_argument('--dc-omega', type=float, default=0.0)
    p.add_argument('--dc-nu', type=float, default=0.1,
                   help='Exploration radius ν for DC-DOPDGD zero-order estimator.')
    p.add_argument('--dc-nu-exp', type=float, default=0.0,
                   help='Decay exponent p for ν_t = ν / t^p.')
    p.add_argument('--dc-batch', type=int, default=1,
                   help='Number of directions per step for bandit averaging.')
    p.add_argument('--dc-common-random', action='store_true',
                   help='Use common random direction across nodes each step.')
    p.add_argument('--dc-beta-exp', type=float, default=None)
    p.add_argument('--dc-eta-exp', type=float, default=None)
    p.add_argument('--dc-alpha-exp', type=float, default=None)
    p.add_argument('--output-dir', type=str, default='comparison_outputs')
    p.add_argument('--combine-rates', action='store_true',
                   help='Plot all compressor rates on shared regret/bits axes (Eq.17).')
    p.add_argument('--combine-viol', action='store_true',
                   help='Plot average violation curves across rates on shared axes (Eq.17).')
    p.add_argument('--compare-compressors', nargs='*', default=[],
                   help='Compare specified compressor types for DC-DOPDGD (Eq.17 only).')
    p.add_argument('--compare-topologies', action='store_true',
                   help='Compare DC-DOPDGD across the provided topologies (Eq.17 only).')
    return p.parse_args()



def main() -> None:
    args = parse_args()
    algos = [a.lower() for a in args.algos]
    for algo in algos:
        if algo not in SUPPORTED_ALGOS:
            raise ValueError(f"Unknown algorithm '{algo}'")
    for topo in args.topologies:
        if topo.lower() not in SUPPORTED_TOPOLOGIES:
            raise ValueError(f"Unsupported topology '{topo}'")

    output_dir = Path(args.output_dir)
    compare_compressors = [c.lower() for c in args.compare_compressors] if args.compare_compressors else []
    invalid_comp = [c for c in compare_compressors if c not in {'no', 'rand', 'top', 'gossip', 'quant'}]
    if invalid_comp:
        raise ValueError(f"Unsupported compressor(s) for comparison: {invalid_comp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.problem == 'eq17':
        data = generate_data_eq17(args.T, args.m, seed=args.seed)
        adpds_dim = 2
    else:
        data = generate_data_eq18(args.T, args.m, seed=args.seed)
        adpds_dim = 1

    results: Dict[Tuple[str, str, float], Dict[str, np.ndarray]] = {}
    topo_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for topology in args.topologies:
        topo_key = topology.lower()
        A = build_adjacency(topo_key, args.m, args.seed, args.edge_num)
        W, _, _, _ = maximum_degree_weights(A)
        topo_cache[topo_key] = (A, W)

        if args.problem == 'eq17':
            a, b = data
        else:
            a, b = data

        for rate in args.compress_rates:
            for algo in algos:
                key = (algo, topo_key, rate)
                if algo == 'adpds':
                    if args.problem == 'eq17':
                        proj_reg = args.adpds_proj_regret or args.proj
                        proj_viol = args.adpds_proj_violation or args.proj
                        cfg_reg = ADPDSConfigEq17(c=args.c, T=args.T, rho=args.rho, proj=proj_reg,
                                                  tol=args.eq17_tol, Ceta=args.adpds_Ceta,
                                                  Cbeta=args.adpds_Cbeta, Cgamma=args.adpds_Cgamma,
                                                  viol_metric=args.viol_metric)
                        out_reg = dict(run_adpds_eq17(cfg_reg, W, a, b))
                        out_reg['bits_history'] = np.zeros(args.T)
                        summary_reg = summarise_eq17(out_reg, args.T, args.viol_metric)
                        if proj_viol == proj_reg:
                            summary_viol = summary_reg
                        else:
                            cfg_viol = ADPDSConfigEq17(c=args.c, T=args.T, rho=args.rho, proj=proj_viol,
                                                       tol=args.eq17_tol, Ceta=args.adpds_Ceta,
                                                       Cbeta=args.adpds_Cbeta, Cgamma=args.adpds_Cgamma,
                                                       viol_metric=args.viol_metric)
                            out_viol = dict(run_adpds_eq17(cfg_viol, W, a, b))
                            out_viol['bits_history'] = np.zeros(args.T)
                            summary_viol = summarise_eq17(out_viol, args.T, args.viol_metric)
                        summary = {
                            'reg_max': summary_reg['reg_max'],
                            'reg_min': summary_reg['reg_min'],
                            'reg_avg': summary_reg['reg_avg'],
                            'viol_max': summary_viol['viol_max'],
                            'viol_min': summary_viol['viol_min'],
                            'viol_avg': summary_viol['viol_avg'],
                            'bits': summary_reg['bits'],
                        }
                        edge_count = int(np.count_nonzero(A) // 2)
                        bits_per_iter = edge_count * adpds_dim * 2 * 64.0
                        summary['bits'] = np.cumsum(np.full(args.T, bits_per_iter))
                    else:
                        cfg = ADPDSConfigEq18(c=args.c18, T=args.T, R=args.R)
                        out = dict(run_adpds_eq18(cfg, W, a, b))
                        out['bits_history'] = np.zeros(args.T)
                        summary = summarise_eq18(out, args.T)
                        edge_count = int(np.count_nonzero(A) // 2)
                        bits_per_iter = edge_count * 1 * 2 * 64.0
                        summary['bits'] = np.cumsum(np.full(args.T, bits_per_iter))
                else:
                    comp_rate = None if args.compressor == 'quant' else rate
                    compressor = CompressorWrapper(args.compressor, w=comp_rate,
                                                    s=args.quant_levels if args.compressor == 'quant' else None)
                    if args.problem == 'eq17':
                        cfg = DCDOPDGDConfigEq17(c=args.c, T=args.T, rho=args.rho, proj=args.proj,
                                                 gamma=args.dc_gamma, omega=args.dc_omega,
                                                 Cbeta=args.dc_Cbeta, Ceta=args.dc_Ceta,
                                                 Calpha=args.dc_Calpha, beta_exp=args.dc_beta_exp,
                                                 eta_exp=args.dc_eta_exp, alpha_exp=args.dc_alpha_exp,
                                                 tol=args.eq17_tol, nu=args.dc_nu,
                                                 nu_exp=args.dc_nu_exp, batch=args.dc_batch,
                                                 common_random=args.dc_common_random)
                        out = run_dcdopdgd_eq17(cfg, W, A, a, b, compressor)
                        summary = summarise_eq17(out, args.T, args.viol_metric)
                    else:
                        cfg = DCDOPDGDConfigEq18(c=args.c18, T=args.T, R=args.R,
                                                 gamma=args.dc_gamma, omega=args.dc_omega,
                                                 Cbeta=args.dc_Cbeta, Ceta=args.dc_Ceta,
                                                 Calpha=args.dc_Calpha, beta_exp=args.dc_beta_exp,
                                                 eta_exp=args.dc_eta_exp, alpha_exp=args.dc_alpha_exp,
                                                 nu=args.dc_nu, nu_exp=args.dc_nu_exp,
                                                 batch=args.dc_batch,
                                                 common_random=args.dc_common_random)
                        out = run_dcdopdgd_eq18(cfg, W, A, a, b, compressor)
                        summary = summarise_eq18(out, args.T)
                results[key] = summary

    t = np.arange(1, args.T + 1)
    for topology, rate in product(args.topologies, args.compress_rates):
        topo_key = topology.lower()
        reg_curves = []
        viol_curves = []
        bits_curves = []
        for algo in algos:
            key = (algo, topo_key, rate)
            summary = results[key]
            label = label_for(algo, args.compressor, rate if algo == 'dc-dopdgd' else None)
            reg_curves.append((f"{label} (max)", summary['reg_max']))
            reg_curves.append((f"{label} (min)", summary['reg_min']))
            viol_curves.append((f"{label} (max)", summary['viol_max']))
            viol_curves.append((f"{label} (min)", summary['viol_min']))
            bits_curves.append((label, summary['bits']))

        prefix = f"{args.problem}_{topo_key}_rate{rate:.2f}".replace('.', 'p')
        plot_curves(t, reg_curves, 'Average regret',
                    f'{args.problem.upper()} - {topology} (rate={rate:.2f})',
                    output_dir / f'{prefix}_regret.png', logscale=True)
        plot_curves(t, viol_curves, 'Average violation',
                    f'{args.problem.upper()} - {topology} (rate={rate:.2f})',
                    output_dir / f'{prefix}_violation.png', logscale=True)
        if any(np.any(curve[1] > 0) for curve in bits_curves if curve[0].startswith('dc-dopdgd') or curve[0].startswith('dc-dopdgd ')):
            plot_curves(t, bits_curves, 'Cumulative bits',
                        f'Transmitted bits - {topology} (rate={rate:.2f})',
                        output_dir / f'{prefix}_bits.png', logscale=False)

    if args.problem == 'eq17':
        for topology in args.topologies:
            topo_key = topology.lower()
            cache_entry = topo_cache.get(topo_key)
            if cache_entry is None:
                continue
            A, W = cache_entry
            a, b = data

            if args.combine_rates:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for ax in axes:
                    ax.grid(True, which='both', linestyle=':')
                plotted_adpds = False
                for algo in algos:
                    for rate in args.compress_rates:
                        if algo == 'adpds' and plotted_adpds:
                            continue
                        key = (algo, topo_key, rate) if algo != 'adpds' else (algo, topo_key, args.compress_rates[0])
                        summary = results.get(key)
                        if summary is None:
                            continue
                        label = label_for(algo, args.compressor, rate if algo == 'dc-dopdgd' else None)
                        axes[0].plot(t, summary['reg_avg'], label=label)
                        bits = np.maximum(summary['bits'], 1e-12)
                        axes[1].plot(bits, summary['reg_avg'], label=label)
                        if algo == 'adpds':
                            plotted_adpds = True
                axes[0].set_xlabel('Iterations')
                axes[0].set_ylabel('Average regret')
                axes[0].set_yscale('log')
                axes[0].legend()
                axes[1].set_xlabel('Transmitted bits')
                axes[1].set_ylabel('Average regret')
                axes[1].set_xscale('log')
                axes[1].set_yscale('log')
                axes[1].legend()
                axes[0].set_title(f"{args.problem.upper()} - {topology} (regret vs. iterations)")
                axes[1].set_title(f"{args.problem.upper()} - {topology} (regret vs. bits)")
                fig.tight_layout()
                combined_name = f"{args.problem}_{topo_key}_combined.png"
                fig.savefig(output_dir / combined_name, dpi=300)
                plt.close(fig)

            if args.combine_viol:
                fig_v, axes_v = plt.subplots(1, 2, figsize=(10, 4))
                for ax in axes_v:
                    ax.grid(True, which='both', linestyle=':')
                plotted_adpds_v = False
                for algo_v in algos:
                    for rate_v in args.compress_rates:
                        if algo_v == 'adpds' and plotted_adpds_v:
                            continue
                        key_v = (algo_v, topo_key, rate_v) if algo_v != 'adpds' else (algo_v, topo_key, args.compress_rates[0])
                        summary_v = results.get(key_v)
                        if summary_v is None:
                            continue
                        label_v = label_for(algo_v, args.compressor, rate_v if algo_v == 'dc-dopdgd' else None)
                        axes_v[0].plot(t, summary_v['viol_avg'], label=label_v)
                        bits_v = np.maximum(summary_v['bits'], 1e-12)
                        axes_v[1].plot(bits_v, summary_v['viol_avg'], label=label_v)
                        if algo_v == 'adpds':
                            plotted_adpds_v = True
                axes_v[0].set_xlabel('Iterations')
                axes_v[0].set_ylabel('Average violation')
                axes_v[0].set_yscale('log')
                axes_v[0].legend()
                axes_v[1].set_xlabel('Transmitted bits')
                axes_v[1].set_ylabel('Average violation')
                axes_v[1].set_xscale('log')
                axes_v[1].set_yscale('log')
                axes_v[1].legend()
                axes_v[0].set_title(f"{args.problem.upper()} - {topology} (violation vs. iterations)")
                axes_v[1].set_title(f"{args.problem.upper()} - {topology} (violation vs. bits)")
                fig_v.tight_layout()
                combined_name_v = f"{args.problem}_{topo_key}_viol_combined.png"
                fig_v.savefig(output_dir / combined_name_v, dpi=300)
                plt.close(fig_v)

            if compare_compressors:
                for rate_cmp in args.compress_rates:
                    fig_c, axes_c = plt.subplots(1, 2, figsize=(10, 4))
                    for ax in axes_c:
                        ax.grid(True, which='both', linestyle=':')
                    for comp_type in compare_compressors:
                        comp_rate = None if comp_type == 'quant' else rate_cmp
                        compressor_cmp = CompressorWrapper(comp_type, w=comp_rate,
                                                           s=args.quant_levels if comp_type == 'quant' else None)
                        cfg_cmp = DCDOPDGDConfigEq17(c=args.c, T=args.T, rho=args.rho, proj=args.proj,
                                                     gamma=args.dc_gamma, omega=args.dc_omega,
                                                     Cbeta=args.dc_Cbeta, Ceta=args.dc_Ceta,
                                                     Calpha=args.dc_Calpha, beta_exp=args.dc_beta_exp,
                                                     eta_exp=args.dc_eta_exp, alpha_exp=args.dc_alpha_exp,
                                                     tol=args.eq17_tol, nu=args.dc_nu,
                                                     nu_exp=args.dc_nu_exp, batch=args.dc_batch,
                                                     common_random=args.dc_common_random)
                        out_cmp = run_dcdopdgd_eq17(cfg_cmp, W, A, a, b, compressor_cmp)
                        summary_cmp = summarise_eq17(out_cmp, args.T, args.viol_metric)
                        label_cmp = label_for_compressor(comp_type, rate_cmp if comp_type != 'quant' else None, args.quant_levels)
                        axes_c[0].plot(t, summary_cmp['reg_avg'], label=label_cmp)
                        bits_cmp = np.maximum(summary_cmp['bits'], 1e-12)
                        axes_c[1].plot(bits_cmp, summary_cmp['reg_avg'], label=label_cmp)
                    axes_c[0].set_xlabel('Iterations')
                    axes_c[0].set_ylabel('Average regret')
                    axes_c[0].set_yscale('log')
                    axes_c[0].legend()
                    axes_c[1].set_xlabel('Transmitted bits')
                    axes_c[1].set_ylabel('Average regret')
                    axes_c[1].set_xscale('log')
                    axes_c[1].set_yscale('log')
                    axes_c[1].legend()
                    axes_c[0].set_title(f"{args.problem.upper()} - {topology} (compressor vs. iterations)")
                    axes_c[1].set_title(f"{args.problem.upper()} - {topology} (compressor vs. bits)")
                    fig_c.tight_layout()
                    comp_name = f"{args.problem}_{topo_key}_rate{rate_cmp:.2f}_compressors.png".replace('.', 'p')
                    fig_c.savefig(output_dir / comp_name, dpi=300)
                    plt.close(fig_c)


    if args.problem == 'eq17' and args.compare_topologies and 'dc-dopdgd' in algos:
        for rate_cmp in args.compress_rates:
            fig_t, axes_t = plt.subplots(1, 2, figsize=(10, 4))
            for ax in axes_t:
                ax.grid(True, which='both', linestyle=':')
            plotted_any = False
            for topology in args.topologies:
                key = ('dc-dopdgd', topology.lower(), rate_cmp)
                summary = results.get(key)
                if summary is None:
                    continue
                label = topology
                axes_t[0].plot(t, summary['reg_avg'], label=label)
                bits = np.maximum(summary['bits'], 1e-12)
                axes_t[1].plot(bits, summary['reg_avg'], label=label)
                plotted_any = True
            if plotted_any:
                axes_t[0].set_xlabel('Iterations')
                axes_t[0].set_ylabel('Average regret')
                axes_t[0].set_yscale('log')
                axes_t[0].legend()
                axes_t[1].set_xlabel('Transmitted bits')
                axes_t[1].set_ylabel('Average regret')
                axes_t[1].set_xscale('log')
                axes_t[1].set_yscale('log')
                axes_t[1].legend()
                axes_t[0].set_title(f"{args.problem.upper()} - Topology comparison (iterations)")
                axes_t[1].set_title(f"{args.problem.upper()} - Topology comparison (bits)")
                fig_t.tight_layout()
                topo_name = f"{args.problem}_rate{rate_cmp:.2f}_topologies.png".replace('.', 'p')
                fig_t.savefig(output_dir / topo_name, dpi=300)
            plt.close(fig_t)
    
    print('Summary (final iteration metrics):')
    idx = args.T - 1
    for topology, rate in product(args.topologies, args.compress_rates):
        topo_key = topology.lower()
        for algo in algos:
            key = (algo, topo_key, rate)
            summary = results[key]
            label = label_for(algo, args.compressor, rate if algo == 'dc-dopdgd' else None)
            print(
                f"{topology:<6} rate={rate:>4.2f} | {label:<18} => "
                f"regret[max]={summary['reg_max'][idx]:.4e}, regret[min]={summary['reg_min'][idx]:.4e}, "
                f"viol[max]={summary['viol_max'][idx]:.4e}, viol[min]={summary['viol_min'][idx]:.4e}, "
                f"bits={summary['bits'][idx]:.2f}"
            )


if __name__ == '__main__':  # pragma: no cover
    main()
