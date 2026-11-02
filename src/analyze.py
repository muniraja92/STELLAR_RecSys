"""Aggregate multi-seed results, significance tests, beyond-accuracy and subgroup analyses."""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Dataset

SEEDS = [42, 1, 7]   # the three seeds of the study
OUT = os.path.join(os.path.dirname(RES), 'tables'); os.makedirs(OUT, exist_ok=True)


def load(dsn, model, seeds=SEEDS):
    S, P = [], []
    for s in seeds:
        j = os.path.join(RES, dsn, model, f'seed{s}.json')
        if not os.path.exists(j): continue
        S.append(json.load(open(j))); P.append(np.load(os.path.join(RES, dsn, model, f'seed{s}.npz')))
    return S, P


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    # chunk to save memory
    means = np.concatenate([d[idx[i:i + 500]].mean(1) for i in range(0, n, 500)])
    return np.percentile(means, [2.5, 97.5])


def main_table(dsn, models, metrics=('HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR@10')):
    rows = []
    for m in models:
        S, _ = load(dsn, m)
        if not S: continue
        r = {'model': m, 'n_seeds': len(S)}
        for k in list(metrics) + ['s_HR@10', 's_NDCG@10', 'ARP@10', 'APLT@10', 'Coverage@10', 'Gini@10',
                                  'NDCG@10_tail_target', 'NDCG@10_head_target', 'NDCG@10_env_low', 'NDCG@10_env_mid',
                                  'NDCG@10_env_high', 'train_time_s', 'infer_time_s', 'val_NDCG@10']:
            v = np.array([s.get(k, np.nan) for s in S], float)
            r[k] = np.nanmean(v); r[k + '_std'] = np.nanstd(v, ddof=1) if len(v) > 1 else 0.0
        rows.append(r)
    return pd.DataFrame(rows)


def per_user(dsn, model, metric='NDCG@10'):
    S, P = load(dsn, model)
    if not P: return None
    return np.mean([p[metric] for p in P], 0), np.array([p[metric].mean() for p in P])


def compare(dsn, target, others, metric='NDCG@10'):
    t = per_user(dsn, target)
    rows = []
    for o in others:
        b = per_user(dsn, o)
        if b is None or t is None: continue
        d = t[0] - b[0]
        nz = d[d != 0]
        w = stats.wilcoxon(nz) if len(nz) > 10 else None
        ci = boot_ci(d)
        k = min(len(t[1]), len(b[1]))
        seed_t = stats.ttest_rel(t[1][:k], b[1][:k]) if k >= 3 else None
        rows.append({'comparison': f'{target} vs {o}', 'mean_target': t[0].mean(), 'mean_other': b[0].mean(),
                     'delta': d.mean(), 'rel_delta_pct': 100 * d.mean() / max(b[0].mean(), 1e-12),
                     'ci_low': ci[0], 'ci_high': ci[1], 'wilcoxon_p': w.pvalue if w else np.nan,
                     'dz': d.mean() / (d.std(ddof=1) + 1e-12),
                     'seed_wins': int((t[1][:k] > b[1][:k]).sum()), 'n_seeds': k,
                     'seed_t_p': seed_t.pvalue if seed_t else np.nan})
    df = pd.DataFrame(rows)
    if len(df):
        df['wilcoxon_p_holm'] = holm(df['wilcoxon_p'].fillna(1).values)
    return df


def user_groups(dsn, models, n_groups=5):
    ds = Dataset(dsn)
    L = np.array([len(h) for h in ds.history('test')])
    q = np.quantile(L, np.linspace(0, 1, n_groups + 1))
    g = np.clip(np.searchsorted(q, L, side='right') - 1, 0, n_groups - 1)
    rows = []
    for m in models:
        pu = per_user(dsn, m)
        if pu is None: continue
        for k in range(n_groups):
            rows.append({'model': m, 'group': k, 'len_min': int(L[g == k].min()), 'len_max': int(L[g == k].max()),
                         'n_users': int((g == k).sum()), 'NDCG@10': pu[0][g == k].mean()})
    return pd.DataFrame(rows), g


if __name__ == '__main__':
    print('use from notebook / scripts')
