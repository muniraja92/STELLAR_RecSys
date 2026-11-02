"""Compact, self-contained report of all results for one dataset (printed as JSON, also saved as CSV)."""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Dataset
from analyze import load, holm, boot_ci, main_table, per_user, SEEDS

BASELINES = ['pop', 'itemknn', 'ease', 'bpr', 'lightgcn', 'kgat', 'kgin', 'sasrec', 'bert4rec', 'bsarec']
COMPONENTS = ['sgw', 'tpr']
ABLATIONS = ['tpr_tok_sgw_nokg', 'tpr_tok_lightgcn', 'stellar_fixed', 'stellar_nokg', 'sgw_nokg', 'sgw_band', 'sgw_fixrel', 'tpr_nokg', 'tpr_nopcl',
             'tpr_nocir', 'tpr_notime', 'tpr_norope', 'tpr_nodecay']
ALL = BASELINES + COMPONENTS + ['stellar'] + ABLATIONS
PAIRS = [('tpr', 'tpr_tok_sgw_nokg'), ('tpr', 'tpr_tok_lightgcn'), ('tpr_tok_sgw_nokg', 'tpr_nokg'), ('tpr_tok_lightgcn', 'tpr_nokg'), ('stellar', 'tpr'), ('stellar', 'stellar_nokg'), ('stellar', 'stellar_fixed'),
         ('tpr', 'tpr_nokg'), ('tpr', 'tpr_nopcl'), ('tpr', 'tpr_nocir'), ('tpr', 'tpr_notime'), ('tpr', 'tpr_norope'),
         ('tpr', 'tpr_nodecay'), ('tpr', 'sasrec'), ('sgw', 'sgw_nokg'), ('sgw', 'sgw_band'), ('sgw', 'sgw_fixrel'), ('sgw', 'lightgcn')]


def r4(x):
    if isinstance(x, (float, np.floating)):
        return float(f'{x:.4g}') if abs(x) < 1e-3 and x != 0 else round(float(x), 4)
    return x


def seeds_of(dsn, m):
    return [s for s in SEEDS if os.path.exists(os.path.join(RES, dsn, m, f'seed{s}.json'))]


def per_user_seeds(dsn, m, seeds, metric):
    P = [np.load(os.path.join(RES, dsn, m, f'seed{s}.npz'))[metric] for s in seeds]
    return np.mean(P, 0), np.array([p.mean() for p in P])


def pairwise(dsn, a, b, mask=None, metric='NDCG@10'):
    common = [s for s in seeds_of(dsn, a) if s in seeds_of(dsn, b)]   # paired over identical seeds
    if not common: return None
    ta, tb = per_user_seeds(dsn, a, common, metric), per_user_seeds(dsn, b, common, metric)
    x, y = ta[0], tb[0]
    if mask is not None: x, y = x[mask], y[mask]
    d = x - y; nz = d[d != 0]
    p = stats.wilcoxon(nz).pvalue if len(nz) > 10 else np.nan
    ci = boot_ci(d, n=5000)
    return dict(a=a, b=b, n=int(len(d)), mean_a=x.mean(), mean_b=y.mean(), delta=d.mean(),
                rel=100 * d.mean() / max(y.mean(), 1e-12), ci=[ci[0], ci[1]], p=p, dz=d.mean() / (d.std(ddof=1) + 1e-12),
                seed_wins=int((ta[1] > tb[1]).sum()), n_seeds=int(min(len(ta[1]), len(tb[1]))))


def main(dsn):
    ds = Dataset(dsn)
    out = {'dataset': dsn}
    mt = main_table(dsn, ALL)
    out['main'] = [{k: r4(v) for k, v in row.items()} for row in mt.to_dict('records')]
    # STELLAR vs every baseline and component (Holm within this family)
    fam = [pairwise(dsn, 'stellar', b) for b in BASELINES + COMPONENTS]
    fam = [f for f in fam if f]
    if fam:
        adj = holm([f['p'] if not np.isnan(f['p']) else 1.0 for f in fam])
        for f, q in zip(fam, adj): f['p_holm'] = q
    out['stellar_vs'] = [{k: (r4(v) if not isinstance(v, list) else [r4(z) for z in v]) for k, v in f.items()} for f in fam]
    # best baseline comparison on other metrics
    abl = [pairwise(dsn, a, b) for a, b in PAIRS]
    abl = [f for f in abl if f]
    if abl:
        adj = holm([f['p'] if not np.isnan(f['p']) else 1.0 for f in abl])
        for f, q in zip(abl, adj): f['p_holm'] = q
    out['pairs'] = [{k: (r4(v) if not isinstance(v, list) else [r4(z) for z in v]) for k, v in f.items()} for f in abl]
    # user-sparsity groups (history length quintiles at test time)
    L = np.array([len(h) for h in ds.history('test')])
    g = pd.qcut(L, 5, labels=False, duplicates='drop'); G = int(g.max()) + 1   # length quantile groups
    grp = {'bounds': [[int(L[g == k].min()), int(L[g == k].max()), int((g == k).sum())] for k in range(G)]}
    for m in ['stellar', 'tpr', 'tpr_nopcl', 'tpr_nokg', 'sasrec', 'bsarec', 'sgw', 'sgw_nokg', 'lightgcn', 'kgin', 'kgat', 'ease', 'itemknn']:
        pu = per_user(dsn, m)
        if pu is not None:
            grp[m] = [r4(pu[0][g == k].mean()) for k in range(G)]
    for a, b in [('tpr', 'tpr_nopcl'), ('stellar', 'tpr'), ('tpr', 'tpr_nokg'), ('stellar', 'stellar_nokg'), ('sgw', 'sgw_nokg')]:
        rows = []
        for k in range(G):
            f = pairwise(dsn, a, b, mask=(g == k))
            if f: rows.append([r4(f['delta']), r4(f['p'])])
        grp[f'{a}-{b}'] = rows
    out['user_groups'] = grp
    # target-item popularity groups (tertiles of training popularity)
    env = ds.pop_env[ds.test]
    ig = {'n': [int((env == e).sum()) for e in range(3)]}
    for a, b in [('tpr', 'tpr_nocir'), ('stellar', 'tpr'), ('tpr', 'tpr_nokg'), ('stellar', 'stellar_nokg')]:
        ig[f'{a}-{b}'] = []
        for e in range(3):
            f = pairwise(dsn, a, b, mask=(env == e))
            if f: ig[f'{a}-{b}'].append([r4(f['mean_a']), r4(f['mean_b']), r4(f['delta']), r4(f['p'])])
    out['item_groups'] = ig
    # module diagnostics
    diag = {}
    for m in ['sgw', 'sgw_band', 'sgw_fixrel', 'sgw_nokg']:
        S, _ = load(dsn, m)
        if S and 'describe' in S[0]:
            d = {}
            for key in ('relation_weights', 'kernel_weights_user', 'kernel_weights_item'):
                if key in S[0]['describe']:
                    d[key] = {k: [r4(np.mean([s['describe'][key][k] for s in S])), r4(np.std([s['describe'][key][k] for s in S]))]
                              for k in S[0]['describe'][key]}
            d['cheb_err'] = r4(S[0]['describe']['cheb_max_abs_err'])
            diag[m] = d
    for m in ['tpr']:
        S, _ = load(dsn, m)
        if S:
            diag[m] = {k: r4(np.mean([s.get(k) or 0 for s in S])) for k in ('gamma', 'kg_proj_norm', 'id_emb_norm')}
    for m in ['stellar', 'stellar_nokg']:
        S, _ = load(dsn, m)
        if S:
            W = np.array([s['gate_w'] for s in S])
            diag[m] = {'gate_w_mean': np.round(W.mean(0), 4).tolist(), 'gate_w_std': np.round(W.std(0), 4).tolist(),
                       'gate_mean': r4(np.mean([s['gate_mean_test'] for s in S])), 'gate_std': r4(np.mean([s['gate_std_test'] for s in S])),
                       'temp': r4(np.mean([s['temp'] for s in S]))}
            # correlation of the gate with user history length
            gs = [np.load(os.path.join(RES, dsn, m, f'gates_seed{sd}.npy')) for sd in SEEDS
                  if os.path.exists(os.path.join(RES, dsn, m, f'gates_seed{sd}.npy'))]
            if gs:
                gm = np.mean(gs, 0)
                diag[m]['gate_by_len_group'] = [r4(gm[g == k].mean()) for k in range(G)]
                diag[m]['spearman_gate_len'] = r4(stats.spearmanr(gm, L).correlation)
    out['diagnostics'] = diag
    bc = os.path.join(RES, dsn, 'best_configs.json')
    out['best_configs'] = json.load(open(bc)) if os.path.exists(bc) else {}
    os.makedirs(os.path.join(RES, 'tables'), exist_ok=True)
    mt.to_csv(os.path.join(RES, 'tables', f'{dsn}_main.csv'), index=False)
    json.dump(out, open(os.path.join(RES, 'tables', f'{dsn}_report.json'), 'w'), indent=1, default=float)
    return out


if __name__ == '__main__':
    o = main(sys.argv[1])
    print('REPORT_JSON_BEGIN')
    print(json.dumps(o, separators=(',', ':'), default=float))
    print('REPORT_JSON_END')
