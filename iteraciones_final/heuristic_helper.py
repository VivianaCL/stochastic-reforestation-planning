# heuristic_helper.py
# GA de asignación CONAFOR (NN+NE) + corrida en batch con muestreo de existentes.
# Sin gráficas. Compatible con sample_nodes.generate_nodes_csv.

from __future__ import annotations
import os, math, random, time
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

SPECIES_IDS = [
    "AG_LEC","AG_SAL","AG_SCB","AG_STR",
    "OP_CAN","OP_ENG","OP_ROB","OP_STR",
    "PR_LAE","YU_FIL"
]
SIDX = {s:i for i,s in enumerate(SPECIES_IDS)}

def _load_species_targets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    req = {"species_id","f_target","v_tol"}
    if not req.issubset(df.columns):
        raise ValueError(f"[{path}] Debe contener columnas {sorted(req)}")
    df["species_id"] = df["species_id"].astype(str).str.strip()
    extra = set(df["species_id"]) - set(SPECIES_IDS)
    miss  = set(SPECIES_IDS) - set(df["species_id"])
    if extra or miss:
        raise ValueError(f"[{path}] Especies desalineadas. Sobran={sorted(extra)} Faltan={sorted(miss)}")
    df = (df.set_index("species_id").loc[SPECIES_IDS].reset_index())
    df["f_target"] = pd.to_numeric(df["f_target"], errors="raise")
    df["v_tol"]    = pd.to_numeric(df["v_tol"],    errors="raise")
    s = float(df["f_target"].sum())
    if s <= 0:
        raise ValueError(f"[{path}] f_target debe sumar > 0")
    df["f_target"] /= s
    return df

def _load_W(path: str) -> np.ndarray:
    W_df = pd.read_csv(path).copy()
    W_df.columns = [c.strip() for c in W_df.columns]
    if W_df.columns[0].lower() != "species_id":
        raise ValueError("[W_matrix.csv] La primera columna debe ser 'species_id'.")
    W_df["species_id"] = W_df["species_id"].astype(str).str.strip()
    W_df = W_df.set_index("species_id")
    miss_rows = set(SPECIES_IDS) - set(W_df.index)
    miss_cols = set(SPECIES_IDS) - set(W_df.columns)
    if miss_rows or miss_cols:
        raise ValueError(f"[W_matrix.csv] faltan ids. filas={miss_rows} cols={miss_cols}")
    W_df = W_df.loc[SPECIES_IDS, SPECIES_IDS].astype(float)
    W_sym = 0.5*(W_df + W_df.T)
    W_sym[W_sym < 0.0] = 0.0
    return W_sym.to_numpy()

def _build_edges_radius(df_all: pd.DataFrame, radius: float) -> List[Tuple[str,str]]:
    """Vecindad por radio, sin doble conteo, devuelve (u,v) con u<v sobre node_id (str)."""
    pts = df_all[["x","y"]].to_numpy(float)
    ids = df_all["node_id"].astype(str).tolist()
    cell = {}
    s = radius; r2 = radius*radius
    for k,(x,y) in enumerate(pts):
        cx, cy = int(math.floor(x/s)), int(math.floor(y/s))
        cell.setdefault((cx,cy), []).append(k)
    edges = set()
    neigh = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1)]
    for (cx,cy), idxs in cell.items():
        cand = set()
        for dx,dy in neigh:
            cand.update(cell.get((cx+dx,cy+dy), []))
        for i in idxs:
            xi, yi = pts[i]
            for j in cand:
                if j <= i: continue
                dx = xi - pts[j,0]; dy = yi - pts[j,1]
                if dx*dx + dy*dy <= r2:
                    u, v = ids[i], ids[j]
                    if v < u: u, v = v, u
                    edges.add((u,v))
    return sorted(edges)

def _prepare_graph(nodes_df: pd.DataFrame, adj_csv: Optional[str], radius: float):
    exist_df = nodes_df[nodes_df["status"]=="existing"].copy()
    avail_df = nodes_df[nodes_df["status"]=="available"].copy()
    exist_df["node_id"] = exist_df["node_id"].astype(str)
    avail_df["node_id"] = avail_df["node_id"].astype(str)
    exist_df = exist_df.reset_index(drop=True)
    avail_df = avail_df.reset_index(drop=True)

    aidx = {nid:i for i,nid in enumerate(avail_df["node_id"])}
    eidx = {nid:i for i,nid in enumerate(exist_df["node_id"])}

    active_df = pd.concat([exist_df, avail_df], ignore_index=True)
    active_df["node_id"] = active_df["node_id"].astype(str)
    active_ids = set(active_df["node_id"])

    if adj_csv and os.path.exists(adj_csv):
        adj_df = pd.read_csv(adj_csv)
        adj_df = adj_df.rename(columns={adj_df.columns[0]:"u_node_id", adj_df.columns[1]:"v_node_id"})
        adj_df["u_node_id"] = adj_df["u_node_id"].astype(str)
        adj_df["v_node_id"] = adj_df["v_node_id"].astype(str)
        adj_df = adj_df[adj_df["u_node_id"].isin(active_ids) & adj_df["v_node_id"].isin(active_ids)].copy()
        u = np.where(adj_df["u_node_id"] <= adj_df["v_node_id"], adj_df["u_node_id"], adj_df["v_node_id"])
        v = np.where(adj_df["u_node_id"] <= adj_df["v_node_id"], adj_df["v_node_id"], adj_df["u_node_id"])
        adj_df = pd.DataFrame({"u_node_id":u,"v_node_id":v}).drop_duplicates()
    else:
        edges = _build_edges_radius(active_df, radius)
        adj_df = pd.DataFrame(edges, columns=["u_node_id","v_node_id"])

    set_av = set(avail_df["node_id"]); set_ex = set(exist_df["node_id"])
    edges_nn, edges_ne = [], []
    for u,v in adj_df.itertuples(index=False):
        inA_u, inA_v = (u in set_av), (v in set_av)
        inE_u, inE_v = (u in set_ex), (v in set_ex)
        if inA_u and inA_v:
            edges_nn.append((u,v))
        elif inA_u and inE_v:
            edges_ne.append((u,v))
        elif inE_u and inA_v:
            edges_ne.append((v,u))

    edges_nn = sorted(set(edges_nn))
    edges_ne = sorted(set(edges_ne))


    if edges_nn:
        nn_u = np.array([aidx[u] for (u,_) in edges_nn], dtype=np.int32)
        nn_v = np.array([aidx[v] for (_,v) in edges_nn], dtype=np.int32)
    else:
        nn_u = np.zeros(0, dtype=np.int32); nn_v = np.zeros(0, dtype=np.int32)

    exist_species_idx = np.array([SIDX.get(sp, -1) for sp in exist_df["species_id"]], dtype=np.int32)
    if edges_ne:
        ne_i  = np.array([aidx[u] for (u,_) in edges_ne], dtype=np.int32)
        ne_jE = np.array([exist_species_idx[eidx[v]] for (_,v) in edges_ne], dtype=np.int32)
    else:
        ne_i  = np.zeros(0, dtype=np.int32); ne_jE = np.zeros(0, dtype=np.int32)

    return exist_df, avail_df, nn_u, nn_v, ne_i, ne_jE

def _compute_bounds(T: int, P_exist: np.ndarray, f_vec: np.ndarray, v_vec: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    lower_final = np.maximum(0.0, (f_vec - v_vec) * T)
    upper_final = np.minimum(T,   (f_vec + v_vec) * T)
    L = np.maximum(0, np.ceil (lower_final - P_exist)).astype(int)
    U = np.maximum(0, np.floor(upper_final - P_exist)).astype(int)
    return L, U

def _project_counts_to_feasible(N_new: int, L: np.ndarray, U: np.ndarray, x_target: np.ndarray, species_ids=SPECIES_IDS) -> np.ndarray:
    x = np.clip(x_target, L, U).astype(int)
    for _ in range(2):
        s = int(x.sum())
        if s < N_new:
            deficit = N_new - s
            order = np.argsort(-(U - x))  
            for idx in order:
                if deficit <= 0: break
                add = int(min(deficit, U[idx] - x[idx]))
                if add > 0:
                    x[idx] += add; deficit -= add
        elif s > N_new:
            excess = s - N_new
            order = np.argsort(-(x - L))  
            for idx in order:
                if excess <= 0: break
                dec = int(min(excess, x[idx] - L[idx]))
                if dec > 0:
                    x[idx] -= dec; excess -= dec
        else:
            break
    if int(x.sum()) != N_new:
        raise ValueError("No fue posible proyectar a una composición factible con las bandas dadas.")
    return x


def _evaluate_cost(W: np.ndarray, nn_u: np.ndarray, nn_v: np.ndarray, ne_i: np.ndarray, ne_jE: np.ndarray, assign_idx: np.ndarray) -> float:
    cost = 0.0
    if nn_u.size:
        cost += float(np.sum(W[assign_idx[nn_u], assign_idx[nn_v]]))
    if ne_i.size:
        cost += float(np.sum(W[assign_idx[ne_i],  ne_jE]))
    return cost

def _prepare_adj_lists(N: int, nn_u: np.ndarray, nn_v: np.ndarray, ne_i: np.ndarray, ne_jE: np.ndarray):
    adj_nn_out = [[] for _ in range(N)]
    for u,v in zip(nn_u, nn_v):
        adj_nn_out[u].append(v); adj_nn_out[v].append(u)
    adj_ne_out = [[] for _ in range(N)]
    for i, eE in zip(ne_i, ne_jE):
        adj_ne_out[i].append(int(eE))
    return adj_nn_out, adj_ne_out

def _delta_swap(W: np.ndarray, adj_nn_out: List[List[int]], adj_ne_out: List[List[int]], assign_idx: np.ndarray, p: int, q: int) -> float:
    if p == q: return 0.0
    e_p, e_q = int(assign_idx[p]), int(assign_idx[q])
    if e_p == e_q: return 0.0
    d = 0.0
    for t in adj_nn_out[p]:
        if t == q: continue
        d += W[e_q, assign_idx[t]] - W[e_p, assign_idx[t]]

    for t in adj_nn_out[q]:
        if t == p: continue
        d += W[e_p, assign_idx[t]] - W[e_q, assign_idx[t]]
    for eE in adj_ne_out[p]:
        d += W[e_q, eE] - W[e_p, eE]
    for eE in adj_ne_out[q]:
        d += W[e_p, eE] - W[e_q, eE]
    return float(d)

def _greedy_initial(N: int, x_feasible: Dict[str,int], e2idx: Dict[str,int], W: np.ndarray, adj_nn_out, adj_ne_out) -> np.ndarray:
    remaining = x_feasible.copy()
    assign = -np.ones(N, dtype=np.int16)
    deg = np.array([len(adj_nn_out[i]) + len(adj_ne_out[i]) for i in range(N)])
    order = np.argsort(-deg)
    for i in order:
        best_e_str, best_inc = None, 1e18
        for e_str, cnt in remaining.items():
            if cnt <= 0: continue
            e = e2idx[e_str]
            inc = 0.0
            for t in adj_nn_out[i]:
                if assign[t] != -1:
                    inc += W[e, assign[t]]
            for eE in adj_ne_out[i]:
                inc += W[e, eE]
            if inc < best_inc:
                best_inc = inc; best_e_str = e_str
        if best_e_str is None:
            best_e_str = max(remaining, key=lambda k: remaining[k])
        assign[i] = np.int16(e2idx[best_e_str])
        remaining[best_e_str] -= 1
    return assign

def _random_initial(species_tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(species_tokens)

def _crossover_preserve(a: np.ndarray, b: np.ndarray, x_feasible: Dict[str,int], species_ids=SPECIES_IDS, rng: Optional[np.random.Generator]=None) -> np.ndarray:
    n = len(a)
    child = np.where((rng.random(n) < 0.5), a, b).astype(np.int16, copy=True)
    desired = np.array([x_feasible[e] for e in species_ids], dtype=int)
    cur = np.bincount(child, minlength=len(species_ids)).astype(int)
    surplus_idx = np.where(cur > desired)[0].tolist()
    deficit_idx = np.where(cur < desired)[0].tolist()
    if not surplus_idx and not deficit_idx:
        return child
    surplus_pos = {e: [] for e in surplus_idx}
    for pos,e in enumerate(child):
        if e in surplus_pos: surplus_pos[e].append(pos)
    for e_def in deficit_idx:
        need = desired[e_def] - cur[e_def]
        while need > 0 and surplus_idx:
            e_sup = max(surplus_idx, key=lambda k: cur[k]-desired[k])
            have = cur[e_sup] - desired[e_sup]
            if have <= 0 or not surplus_pos[e_sup]:
                surplus_idx.remove(e_sup); continue
            pos = surplus_pos[e_sup].pop()
            child[pos] = np.int16(e_def)
            cur[e_sup] -= 1; cur[e_def] += 1; need -= 1
            if cur[e_sup] <= desired[e_sup] and e_sup in surplus_idx:
                surplus_idx.remove(e_sup)
    if not np.array_equal(np.bincount(child, minlength=len(species_ids)), desired):
        raise RuntimeError("crossover_preserve no logró ajustar exactamente a x_feasible.")
    return child

def _mutate_swap(assign: np.ndarray, rng: np.random.Generator, p_swap: float = 0.2) -> np.ndarray:
    child = assign.copy()
    if rng.random() < p_swap and len(assign) >= 2:
        i, j = rng.integers(0, len(assign), size=2)
        if i != j: child[i], child[j] = child[j], child[i]
    return child

def _local_search(assign: np.ndarray, W: np.ndarray, adj_nn_out, adj_ne_out, rng: np.random.Generator, max_iters: int = 200):
    cur = assign.copy()
    cur_cost = None
    for _ in range(max_iters):
        i = rng.integers(0, len(assign))
        cand = adj_nn_out[i]
        j = random.choice(cand) if cand else rng.integers(0, len(assign))
        if i == j: continue
        d = _delta_swap(W, adj_nn_out, adj_ne_out, cur, i, j)
        if d < -1e-8:
            cur[i], cur[j] = cur[j], cur[i]
            if cur_cost is not None:
                cur_cost += d
    if cur_cost is None:
        cur_cost = _evaluate_cost(W, np.array([],dtype=np.int32), np.array([],dtype=np.int32), np.array([],dtype=np.int32), np.array([],dtype=np.int32), cur)  # placeholder

    return cur, cur_cost

def solve_instance_with_ga(
    nodes_csv: str,
    species_csv: str = "species_targets.csv",
    w_csv: str = "W_matrix.csv",
    adj_csv: Optional[str] = None,
    radius: float = 3.2,
    rng_seed: Optional[int] = None,
    pop: int = 40, gens: int = 200, elite: int = 4, tour: int = 3, ls_rate: float = 0.3, mut_p: float = 0.3
) -> Dict[str, object]:
    """
    Resuelve una instancia (nodes_csv) con el GA y devuelve:
      - fitness (float)
      - totals: total_existing, total_planted
      - exist_{sid}, new_{sid} por especie (dicts)
      - assignments_df (DataFrame) con available_assigned
      - T_counts (10x10) conteos dirigidos (para Markov), acumulados en esta corrida
    """
    rng = np.random.default_rng(rng_seed)

    species_df = _load_species_targets(species_csv)
    W = _load_W(w_csv)
    nodes_df = pd.read_csv(nodes_csv).copy()
    nodes_df["status"]     = nodes_df["status"].astype(str).str.lower().str.strip()
    nodes_df["species_id"] = nodes_df["species_id"].fillna("").astype(str).str.strip()
    nodes_df["x"] = pd.to_numeric(nodes_df["x"], errors="coerce")
    nodes_df["y"] = pd.to_numeric(nodes_df["y"], errors="coerce")

    exist_df, avail_df, nn_u, nn_v, ne_i, ne_jE = _prepare_graph(nodes_df, adj_csv, radius)
    N = len(avail_df); T = len(avail_df) + len(exist_df)

    P_exist_vec = np.zeros(len(SPECIES_IDS), dtype=int)
    if len(exist_df):
        ex_idx = np.array([SIDX[s] for s in exist_df["species_id"]], dtype=int)
        uniq, cnt = np.unique(ex_idx, return_counts=True)
        P_exist_vec[uniq] = cnt

    f_vec = species_df["f_target"].to_numpy(float)
    v_vec = species_df["v_tol"].to_numpy(float)
    L, U = _compute_bounds(T, P_exist_vec, f_vec, v_vec)

    if (L > U).any():
        raise ValueError(f"Infeasible bands per species: {[(SPECIES_IDS[i], int(L[i]), int(U[i])) for i in np.where(L>U)[0]]}")
    if int(L.sum()) > N or N > int(U.sum()):
        raise ValueError(f"Infeasible totals: sum L={int(L.sum())}, N={N}, sum U={int(U.sum())}")

    x_target = np.round(f_vec*T - P_exist_vec).astype(int)
    x_feasible = _project_counts_to_feasible(N, L, U, x_target)

    species_tokens = np.concatenate([np.repeat(i, x_feasible[i]) for i in range(len(SPECIES_IDS))]).astype(np.int16)

    adj_nn_out, adj_ne_out = _prepare_adj_lists(N, nn_u, nn_v, ne_i, ne_jE)

    g0 = _greedy_initial(N, {SPECIES_IDS[i]: int(x_feasible[i]) for i in range(len(SPECIES_IDS))}, SIDX, W, adj_nn_out, adj_ne_out)
    pop_list = [g0] + [ _random_initial(species_tokens, rng) for _ in range(max(0, pop-1)) ]
    fitness = np.array([_evaluate_cost(W, nn_u, nn_v, ne_i, ne_jE, ind) for ind in pop_list], dtype=float)
    best_idx = int(np.argmin(fitness))
    best, best_cost = pop_list[best_idx].copy(), float(fitness[best_idx])

    def tournament_select() -> np.ndarray:
        idx = rng.choice(len(pop_list), size=tour, replace=False)
        j = idx[np.argmin(fitness[idx])]
        return pop_list[j]

    for _gen in range(gens):
        order = np.argsort(fitness)
        pop_list = [pop_list[i] for i in order]
        fitness  = fitness[order]
        new_pop  = pop_list[:elite]
        new_fit  = fitness[:elite].tolist()
        while len(new_pop) < pop:
            p1, p2 = tournament_select(), tournament_select()
            child = _crossover_preserve(p1, p2, {SPECIES_IDS[i]: int(x_feasible[i]) for i in range(len(SPECIES_IDS))}, SPECIES_IDS, rng)
            child = _mutate_swap(child, rng, p_swap=mut_p)
            if rng.random() < ls_rate:
                child, _ = _local_search(child, W, adj_nn_out, adj_ne_out, rng, max_iters=100)
            new_pop.append(child)
            new_fit.append(_evaluate_cost(W, nn_u, nn_v, ne_i, ne_jE, child))
        pop_list = new_pop
        fitness  = np.array(new_fit, dtype=float)
        k = int(np.argmin(fitness))
        if fitness[k] < best_cost - 1e-8:
            best, best_cost = pop_list[k].copy(), float(fitness[k])

    assign_species = [SPECIES_IDS[int(k)] for k in best]
    out = avail_df[["node_id","x","y"]].copy()
    out["node_id"] = out["node_id"].astype(str)
    out["status"] = "available_assigned"
    out["species_id"] = assign_species
    exist_out = exist_df[["node_id","x","y","status","species_id"]].copy()
    exist_out["node_id"] = exist_out["node_id"].astype(str)
    assignments_df = pd.concat([exist_out, out], ignore_index=True)

    counts_new = np.bincount(best, minlength=len(SPECIES_IDS)).astype(int)
    counts_exist = P_exist_vec.copy()
    counts_final = counts_exist + counts_new
    total_existing = int(counts_exist.sum())
    total_planted  = int(counts_new.sum())

    T_counts = np.zeros((len(SPECIES_IDS), len(SPECIES_IDS)), dtype=np.int64)

    for u,v in zip(nn_u, nn_v):
        a, b = int(best[u]), int(best[v])
        T_counts[a,b] += 1; T_counts[b,a] += 1
  
    for i, b in zip(ne_i, ne_jE):
        a = int(best[i])
        if b >= 0:
            T_counts[a,b] += 1; T_counts[b,a] += 1

    res = {
        "fitness": float(best_cost),
        "total_existing": total_existing,
        "total_planted":  total_planted,
        "exist_counts": {SPECIES_IDS[i]: int(counts_exist[i]) for i in range(len(SPECIES_IDS))},
        "new_counts":   {SPECIES_IDS[i]: int(counts_new[i])   for i in range(len(SPECIES_IDS))},
        "final_counts": {SPECIES_IDS[i]: int(counts_final[i]) for i in range(len(SPECIES_IDS))},
        "assignments_df": assignments_df,
        "T_counts": T_counts,
    }
    return res

def run_batch_with_sampling(
    N: int,
    n_nodes: int,
    out_dir: str,
    sample_kwargs: Optional[Dict] = None,
    ga_kwargs: Optional[Dict] = None,
    species_csv: str = "species_targets.csv",
    w_csv: str = "W_matrix.csv",
) -> Dict[str,str]:
    """
    Repite N veces:
      1) Muestra nodos con sample_nodes.generate_nodes_csv(...)
      2) Resuelve GA
      3) Guarda por iteración: fitness, totales, conteos por especie
    Devuelve paths a:
      - results.csv
      - results_averages.csv
      - T_matrix.csv   (probabilidades condicionales por fila)
    """
    os.makedirs(out_dir, exist_ok=True)

    import importlib.util as _util
    spec = _util.spec_from_file_location("sample_nodes", os.path.join(os.getcwd(), "sample_nodes.py"))
    mod  = _util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)

    sample_kwargs = dict(sample_kwargs or {})
    ga_kwargs     = dict(ga_kwargs or {})

    rows = []
    T_accum = np.zeros((len(SPECIES_IDS), len(SPECIES_IDS)), dtype=np.int64)
    feasible_runs = 0

    for it in range(1, N+1):
        seed = int(np.random.default_rng().integers(0, 1_000_000))
        nodes_path = os.path.join(out_dir, f"nodes_iter_{it}.csv")
        mod.generate_nodes_csv(
            n_nodes=n_nodes, out_path=nodes_path, seed=seed,
            spacing=sample_kwargs.get("spacing", 3.2),
            nodes_per_ha=sample_kwargs.get("nodes_per_ha", None),
            mode=sample_kwargs.get("mode", "dirichlet"),
            tol_l1=sample_kwargs.get("tol_l1", None),
            phi=sample_kwargs.get("phi", None),
        )

        try:
            res = solve_instance_with_ga(
                nodes_csv=nodes_path,
                species_csv=species_csv,
                w_csv=w_csv,
                adj_csv=None,
                radius=ga_kwargs.get("radius", 3.2),
                rng_seed=seed,
                pop=ga_kwargs.get("pop", 40),
                gens=ga_kwargs.get("gens", 200),
                elite=ga_kwargs.get("elite", 4),
                tour=ga_kwargs.get("tour", 3),
                ls_rate=ga_kwargs.get("ls_rate", 0.3),
                mut_p=ga_kwargs.get("mut_p", 0.3),
            )
            feasible_runs += 1
            T_accum += res["T_counts"]

            row = {
                "iteration": it,
                "seed": seed,
                "fitness": res["fitness"],
                "total_existing": res["total_existing"],
                "total_planted":  res["total_planted"],
            }
            for sid in SPECIES_IDS:
                row[f"exist_{sid}"] = res["exist_counts"][sid]
            for sid in SPECIES_IDS:
                row[f"new_{sid}"]   = res["new_counts"][sid]
            rows.append(row)

            assign_path = os.path.join(out_dir, f"assignments_iter_{it}.csv")
            res["assignments_df"].to_csv(assign_path, index=False)

        except Exception as ex:
            row = {"iteration": it, "seed": seed, "fitness": np.nan, "total_existing": np.nan, "total_planted": np.nan}
            for sid in SPECIES_IDS:
                row[f"exist_{sid}"] = np.nan
                row[f"new_{sid}"]   = np.nan
            rows.append(row)
            

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(out_dir, "results.csv")
    results_df.to_csv(results_csv, index=False)

    ok = results_df.dropna(subset=["fitness"]).copy()
    avg = {
        "metric": "mean_over_feasible",
        "feasible_runs": feasible_runs,
        "fitness": float(ok["fitness"].mean()) if feasible_runs else np.nan,
        "total_existing": float(ok["total_existing"].mean()) if feasible_runs else np.nan,
        "total_planted":  float(ok["total_planted"].mean())  if feasible_runs else np.nan,
    }
    for sid in SPECIES_IDS:
        avg[f"exist_{sid}"] = float(ok[f"exist_{sid}"].mean()) if feasible_runs else np.nan
        avg[f"new_{sid}"]   = float(ok[f"new_{sid}"].mean())   if feasible_runs else np.nan

    averages_df = pd.DataFrame([avg])
    averages_csv = os.path.join(out_dir, "results_averages.csv")
    averages_df.to_csv(averages_csv, index=False)

    T = T_accum.astype(float)
    row_sums = T.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        T_prob = np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums>0)

    T_df = pd.DataFrame(T_prob, index=SPECIES_IDS, columns=SPECIES_IDS)
    T_csv = os.path.join(out_dir, "T_matrix.csv")
    T_df.to_csv(T_csv, float_format="%.6f")

    return {"results_csv": results_csv, "averages_csv": averages_csv, "T_matrix_csv": T_csv}
