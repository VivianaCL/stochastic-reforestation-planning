# sample_nodes.py
"""
Helper to generate a nodes.csv with a triangular ('tres bolillos') lattice,
tagging a spatially random subset of nodes as 'existing' and assigning species
according to a Binomial + Dirichlet–Multinomial model calibrated from field data.

Output schema (CSV): node_id,x,y,status,species_id

Usage as a library:
-------------------
from existing_nodes_helper import generate_nodes_csv
generate_nodes_csv(n_nodes=625, out_path="nodes_625.csv", seed=42)

CLI usage:
----------
# variabilidad realista (Dirichlet–Multinomial) con ϕ inferido desde std
python sample_nodes.py --n 625 --out nodes_625.csv --seed 42 --spacing 3.2 --mode dirichlet --tol-l1 0.10

# composición determinista (proporciones objetivo exactas por restos mayores)
python sample_nodes.py --n 625 --out nodes_625_exact.csv --seed 42 --mode exact_means

Notes:
- Por defecto, la densidad de nodos por hectárea se calcula desde el `spacing`
  usando la fórmula del retículo triangular ideal. Si quieres reproducir un supuesto
  fijo (p. ej. 658 nodos/ha), pásalo con --nodes-per-ha 658.
- El total de existentes por ha se modela como Binomial(N*=145, p*=0.8927) -> E[λ]=129.51/ha.
- La composición por especie se modela con Dirichlet(α=ϕ·p̄) y luego Multinomial(X_tot, p),
  o bien en modo determinista con redondeo por restos mayores ("Hamilton") para clavar las medias.
"""

import math
import argparse
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

SPECIES_IDS = [
    "AG_LEC", "AG_SAL", "AG_SCB", "AG_STR",
    "OP_CAN", "OP_ENG", "OP_ROB", "OP_STR",
    "PR_LAE", "YU_FIL"
]

P_MEAN_VEC = np.array([
    0.06237857016,  # AG_LEC
    0.2941802345,   # AG_SAL
    0.0637873077,   # AG_SCB
    0.06019351159,  # AG_STR
    0.07732665946,  # OP_CAN
    0.06321196499,  # OP_ENG
    0.1152252671,   # OP_ROB
    0.09551359296,  # OP_STR
    0.1287178888,   # PR_LAE
    0.03946500276   # YU_FIL
], dtype=float)

P_STD_VEC = np.array([
    0.01092496674, 0.01508438895, 0.01056107563, 0.01091126803,
    0.01070779767, 0.009848872888, 0.01109275802, 0.0144683688,
    0.01818201083, 0.007618690728
], dtype=float)

P_VEC = P_MEAN_VEC.copy()
N_PER_HA = 145.0
P_PER_HA = 0.8927
LAMBDA_MEAN = N_PER_HA * P_PER_HA  # 129.51


def triangular_lattice_xy(n_nodes: int, spacing: float = 3.2, origin: Tuple[float, float] = (0.0, 0.0)) -> List[Tuple[float, float]]:
    """
    Build a triangular (hexagonal) lattice with approximately n_nodes points.
    Points are laid out row by row. The vertical spacing is spacing * sqrt(3)/2.
    Odd rows are offset by spacing/2 in x to achieve the 'tres bolillos' pattern.
    """
    if n_nodes < 1:
        return []
    sx, sy = origin
    dx = spacing
    dy = spacing * math.sqrt(3) / 2.0

    cols = max(1, int(round(math.sqrt(n_nodes))))
    rows = max(1, int(math.ceil(n_nodes / cols)))

    pts = []
    for r in range(rows):
        x_offset = (dx / 2.0) if (r % 2 == 1) else 0.0
        for c in range(cols):
            if len(pts) >= n_nodes:
                break
            x = sx + c * dx + x_offset
            y = sy + r * dy
            pts.append((x, y))
    return pts

def nodes_per_ha_from_spacing(spacing: float) -> float:
    """
    Densidad de un retículo triangular ideal (puntos por hectárea) dada la separación (m).
    Fórmula: densidad = 2 / (sqrt(3) * spacing^2) por m^2; multiplicar por 10,000 para ha.
    """
    return (2.0 / (math.sqrt(3.0) * spacing * spacing)) * 10000.0


def largest_remainder_counts(total: int, proportions: np.ndarray) -> np.ndarray:
    """
    Asigna enteros que suman 'total' preservando al máximo las proporciones.
    Método de restos mayores (Hamilton).
    """
    x = np.asarray(proportions, dtype=float)
    x = x / x.sum()
    raw = x * total
    base = np.floor(raw).astype(int)
    rem = int(total - base.sum())
    if rem > 0:
        frac_idx = np.argsort(-(raw - base))[:rem]
        base[frac_idx] += 1
    return base

def infer_phi_from_std(p_mean: np.ndarray, p_std: np.ndarray, n_eff: Optional[float] = None) -> float:
    """
    Calibra ϕ a partir de std por especie.
    Var Dirichlet: Var(q_e) = p(1-p)/(ϕ+1).
    Si se pasa n_eff (= E[X_tot]), resta el ruido muestral Multinomial ~ p(1-p)/n_eff.
    Devuelve una ϕ robusta (mediana de ϕ_e por especie, con recorte mínimo a 1.0).
    """
    eps = 1e-12
    var_obs = np.maximum(p_std**2, eps)
    if n_eff is not None:
        var_obs = np.maximum(var_obs - (p_mean * (1.0 - p_mean)) / float(n_eff), eps)
    phi_e = (p_mean * (1.0 - p_mean) / var_obs) - 1.0
    phi_e = np.maximum(phi_e, 1.0)  # robustez
    return float(np.median(phi_e))

def simulate_existing_counts(n_nodes: int,
                             nodes_per_ha: Optional[float] = None,
                             spacing: float = 3.2,
                             seed: Optional[int] = None,
                             mode: str = "dirichlet",
                             target_means: np.ndarray = P_MEAN_VEC,
                             target_stds: np.ndarray = P_STD_VEC,
                             phi: Optional[float] = None,
                             tol_l1: Optional[float] = None,
                             max_tries: int = 200) -> np.ndarray:
    """
    Simula el vector de existentes por especie (len=10).

    - Total por hectárea: Binomial(N*=145, p*=0.8927) -> E[λ]=129.51/ha.
      Escala a área A según n_nodes y densidad de la malla.

    - 'nodes_per_ha':
        * Si es None, se calcula a partir de 'spacing' con la fórmula del retículo triangular.
        * Si viene explícito, se usa tal cual.

    - 'mode':
        * "exact_means": ignora Dirichlet, usa target_means y reparte con redondeo por restos mayores.
        * "dirichlet": usa Dirichlet(α = ϕ * target_means) y luego Multinomial.
                       Si 'tol_l1' está definido, reintenta hasta que ||p_hat - target_means||_1 ≤ tol_l1
                       (máx. 'max_tries'; introduce sesgo a favor de estar cerca).

    - 'phi':
        * Si None, se calibra automáticamente desde 'target_stds' (corrigiendo el ruido Multinomial con n_eff=E[X_tot]).
    """
    rng = np.random.default_rng(seed)

    if nodes_per_ha is None:
        nodes_per_ha = nodes_per_ha_from_spacing(spacing)
    A = max(1e-9, float(n_nodes) / float(nodes_per_ha))

    N_scaled = int(round(N_PER_HA * A))
    X_tot = int(rng.binomial(N_scaled, P_PER_HA))
    X_tot = max(0, X_tot)

    if X_tot == 0:
        return np.zeros_like(target_means, dtype=int)

    if phi is None and mode == "dirichlet":
        n_eff = max(1.0, LAMBDA_MEAN * A)  # ≈ E[X_tot]
        phi = infer_phi_from_std(target_means, target_stds, n_eff=n_eff)
    elif phi is None:
        phi = infer_phi_from_std(target_means, target_stds, n_eff=None)

    if mode == "exact_means":
        counts = largest_remainder_counts(X_tot, target_means)
        return counts.astype(int)

    alpha = float(phi) * (target_means / target_means.sum())

    def one_draw():
        p_vec = rng.dirichlet(alpha)
        return rng.multinomial(X_tot, p_vec)

    if tol_l1 is None:
        return one_draw().astype(int)

    best = None
    best_l1 = float("inf")
    target = target_means / target_means.sum()
    for _ in range(max_tries):
        c = one_draw().astype(int)
        phat = c / max(1, c.sum())
        l1 = float(np.abs(phat - target).sum())
        if l1 < best_l1:
            best, best_l1 = c, l1
            if l1 <= tol_l1:
                break
    return best.astype(int)

def build_nodes_dataframe(n_nodes: int,
                          spacing: float = 3.2,
                          origin: Tuple[float, float] = (0.0, 0.0),
                          seed: Optional[int] = None,
                          nodes_per_ha: Optional[float] = None,
                          mode: str = "dirichlet",
                          tol_l1: Optional[float] = None,
                          phi: Optional[float] = None) -> pd.DataFrame:
    """
    Create a DataFrame with columns: node_id,x,y,status,species_id
    - Place n_nodes on a triangular lattice ('tres bolillos')
    - Simulate existing counts and pick that many nodes uniformly at random to be 'existing'
    - Assign species to existing nodes according to the simulated counts
    - Remaining nodes are 'available' with empty species_id
    """
    rng = np.random.default_rng(seed)
    coords = triangular_lattice_xy(n_nodes, spacing=spacing, origin=origin)
    df = pd.DataFrame(coords, columns=["x", "y"])
    df.insert(0, "node_id", range(1, len(df) + 1))
    df["status"] = "available"
    df["species_id"] = ""

    counts = simulate_existing_counts(
        n_nodes=n_nodes,
        nodes_per_ha=nodes_per_ha,
        spacing=spacing,
        seed=seed,
        mode=mode,
        phi=phi,
        tol_l1=tol_l1
    )

    total_existing = int(counts.sum())
    total_existing = min(total_existing, n_nodes)

    if total_existing > 0:
        existing_idx = rng.choice(df.index.values, size=total_existing, replace=False)
        df.loc[existing_idx, "status"] = "existing"
        species_pool: List[str] = []
        for sp_id, cnt in zip(SPECIES_IDS, counts.tolist()):
            species_pool.extend([sp_id] * int(cnt))

        species_pool = species_pool[:total_existing]
        while len(species_pool) < total_existing:
            species_pool.append(str(rng.choice(SPECIES_IDS, p=P_VEC / P_VEC.sum())))

        rng.shuffle(species_pool)
        df.loc[existing_idx, "species_id"] = species_pool

    return df[["node_id", "x", "y", "status", "species_id"]]

def generate_nodes_csv(n_nodes: int,
                       out_path: str,
                       spacing: float = 3.2,
                       origin: Tuple[float, float] = (0.0, 0.0),
                       seed: Optional[int] = None,
                       nodes_per_ha: Optional[float] = None,
                       mode: str = "dirichlet",
                       tol_l1: Optional[float] = None,
                       phi: Optional[float] = None) -> str:
    """
    High-level helper that creates and writes the nodes CSV.
    By default, nodes_per_ha is inferred from spacing unless provided explicitly.
    Returns the path written.
    """
    if nodes_per_ha is None:
        nodes_per_ha = nodes_per_ha_from_spacing(spacing)

    df = build_nodes_dataframe(
        n_nodes=n_nodes,
        spacing=spacing,
        origin=origin,
        seed=seed,
        nodes_per_ha=nodes_per_ha,
        mode=mode,
        tol_l1=tol_l1,
        phi=phi
    )
    df.to_csv(out_path, index=False)
    return out_path

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate nodes.csv with simulated existing plants.")
    p.add_argument("--n", type=int, required=True, help="Total number of nodes to generate.")
    p.add_argument("--out", type=str, required=True, help="Output CSV path.")
    p.add_argument("--spacing", type=float, default=3.2, help="Inter-plant spacing (m). Default: 3.2")
    p.add_argument("--origin-x", type=float, default=0.0, help="Origin X coordinate.")
    p.add_argument("--origin-y", type=float, default=0.0, help="Origin Y coordinate.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--nodes-per-ha", type=float, default=None,
                   help="Node density per hectare. If omitted, inferred from spacing (triangular lattice).")
    p.add_argument("--mode", type=str, choices=["dirichlet", "exact_means"], default="dirichlet",
                   help="Composition strategy: dirichlet (realistic variability) or exact_means (deterministic).")
    p.add_argument("--tol-l1", type=float, default=None,
                   help="(dirichlet only) L1 tolerance target vs. realized; retries up to an internal cap.")
    p.add_argument("--phi", type=float, default=None,
                   help="Dirichlet concentration. If not provided, inferred from species stds correcting multinomial noise.")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    path = generate_nodes_csv(
        n_nodes=args.n,
        out_path=args.out,
        spacing=args.spacing,
        origin=(args.origin_x, args.origin_y),
        seed=args.seed,
        nodes_per_ha=args.nodes_per_ha,
        mode=args.mode,
        tol_l1=args.tol_l1,
        phi=args.phi
    )
    print(path)
