"""Stratified evaluation of the MovieLens-1M results.

The knowledge graph of this study (Satori relations, genre and release decade) was built for
MovieLens-1M only, so all conclusions are drawn on that dataset.  This script examines whether
they hold across sub-populations of MovieLens-1M users and test targets that differ in sparsity,
demographics, timing and the amount of knowledge available about the target item.

Inputs
  results/strat/ml1m_peruser.npz  per-user test metrics, keys '<model>|<seed>|<metric>' (export_peruser.py)
  users.dat of MovieLens-1M       gender and age (results/strat/ml1m_users.dat or $STELLAR_RAW/ml1m/users.dat)
  data/ml1m.pkl                   prepared dataset (sequences, timestamps, knowledge graph)
Outputs
  results/strat/ml1m_strat.json   all stratified statistics
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(ROOT, 'results', 'strat')
SEEDS = (42, 1, 7)
METRIC = 'NDCG@10'
B = 2000                      # bootstrap resamples
RNG = np.random.default_rng(2026)


def reseed(*keys):
    """Independent, reproducible bootstrap stream per analysis, unaffected by which other analyses are run."""
    import zlib
    global RNG
    RNG = np.random.default_rng([2026, zlib.crc32('|'.join(map(str, keys)).encode())])

MODELS = ['stellar', 'tpr', 'tpr_nokg', 'stellar_nokg', 'tpr_tok_sgw_nokg', 'tpr_tok_lightgcn', 'sasrec', 'bsarec', 'bert4rec', 'kgin', 'kgat',
          'lightgcn', 'sgw', 'ease', 'itemknn', 'bpr', 'pop']
PAIRS = [('stellar', 'sasrec'), ('stellar', 'bsarec'), ('stellar', 'bert4rec'), ('stellar', 'kgin'), ('stellar', 'tpr'),
         ('tpr', 'tpr_nokg'), ('stellar', 'stellar_nokg'), ('tpr_tok_sgw_nokg', 'tpr_nokg'), ('tpr_tok_lightgcn', 'tpr_nokg'), ('tpr', 'tpr_tok_sgw_nokg')]


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def r4(x):
    x = float(x)
    return float(f'{x:.3g}') if (abs(x) < 1e-3 and x != 0) else round(x, 4)


def load():
    z = np.load(os.path.join(SD, 'ml1m_peruser.npz'))
    X = {}
    for m in sorted({k.split('|')[0] for k in z.files}):
        ks = [f'{m}|{s}|{METRIC}' for s in SEEDS if f'{m}|{s}|{METRIC}' in z.files]
        if len(ks) == len(SEEDS):
            X[m] = np.mean([z[k] for k in ks], 0).astype(np.float64)
    return X


def strata(d):
    """Return {dimension: (labels array, ordered list of level names)} for the 5950 users."""
    n = d['n_users']
    S = {}
    # 1. history length at test time (training sequence + validation item), quintiles
    L = np.array([len(t) + 1 for t in d['train']])
    q = pd.qcut(L, 5, labels=False, duplicates='drop')
    S['history'] = (q, [f'{L[q == k].min()}-{L[q == k].max()}' for k in range(q.max() + 1)])
    # 2-3. demographics from the MovieLens-1M user file
    uf = os.path.join(SD, 'ml1m_users.dat')
    if not os.path.exists(uf):     # MovieLens-1M user file from the raw release (not redistributed here)
        raw = os.environ.get('STELLAR_RAW', os.path.join(ROOT, 'raw'))
        uf = next(f for f in (os.path.join(raw, 'ml1m', 'users.dat'), os.path.join(raw, 'ml-1m', 'users.dat')) if os.path.exists(f))
    u = pd.read_csv(uf, sep='::', engine='python',
                    names=['u', 'gender', 'age', 'occ', 'zip']).set_index('u').loc[d['user_raw']]
    S['gender'] = ((u.gender.values == 'M').astype(int), ['female', 'male'])
    a = u.age.values
    S['age'] = (np.select([a < 25, a < 35, a < 45], [0, 1, 2], 3), ['<25', '25-34', '35-44', '>=45'])
    # 4. time between the validation and the test interaction
    gap = np.array([t[-1] - t[-2] for t in d['ts']])
    S['gap'] = (np.select([gap == 0, gap < 60, gap < 3600], [0, 1, 2], 3), ['0 s', '1-59 s', '1-59 min', '>=1 h'])
    # 5. calendar year of the test interaction
    yr = pd.to_datetime(np.array([t[-1] for t in d['ts']]), unit='s').year.values
    S['period'] = ((yr >= 2001).astype(int), ['2000', '2001-2003'])
    # 6. training popularity of the test target, item tertiles (as used by CIR)
    pop = np.zeros(d['n_items'])
    for t in d['train']: np.add.at(pop, t, 1)
    q1, q2 = np.quantile(pop, [1 / 3, 2 / 3])
    te = np.asarray(d['test'])
    S['popularity'] = (np.where(pop[te] < q1, 0, np.where(pop[te] < q2, 1, 2)), ['low', 'mid', 'high'])
    # 7. knowledge available about the test target: Satori links (actor, director, writer, ...)
    kg = d['kg']; rel = d['relations']
    sat = [i for i, r in enumerate(rel) if r not in ('genre(MovieLens)', 'release_decade')]
    c = np.bincount(kg[np.isin(kg[:, 1], sat), 0], minlength=d['n_items'])[te]
    med = np.median(c[c > 0])
    S['knowledge'] = (np.where(c == 0, 0, np.where(c <= med, 1, 2)),
                      ['none', f'1-{int(med)}', f'>{int(med)}'])
    for k, (g, _) in S.items(): assert len(g) == n, k
    return S


def signed_rank_parts(dk):
    """Wilcoxon signed-rank pieces with zero differences dropped (exact null moments under ties)."""
    nz = dk[dk != 0]
    if len(nz) == 0: return 0.0, 0.0, 0.0
    r = stats.rankdata(np.abs(nz))
    T = r[nz > 0].sum()
    return T - r.sum() / 2, (r ** 2).sum() / 4, len(nz)


def van_elteren(d, g, K):
    """Stratified (van Elteren) signed-rank test: strata statistics weighted by 1/(n_k+1)."""
    num = var = 0.0
    for k in range(K):
        c, v, nk = signed_rank_parts(d[g == k])
        if nk == 0: continue
        w = 1.0 / (nk + 1); num += w * c; var += w * w * v
    z = num / np.sqrt(var) if var > 0 else 0.0
    return z, 2 * stats.norm.sf(abs(z))


def cochran_q(d, g, K):
    m = np.array([d[g == k].mean() for k in range(K)])
    se = np.array([d[g == k].std(ddof=1) / np.sqrt((g == k).sum()) for k in range(K)])
    w = 1 / np.maximum(se, 1e-12) ** 2
    mw = (w * m).sum() / w.sum()
    Q = (w * (m - mw) ** 2).sum(); df = K - 1
    return Q, stats.chi2.sf(Q, df), max(0.0, (Q - df) / Q) if Q > 0 else 0.0


def strat_boot_means(v, g, K, weights, b=B):
    """Stratified bootstrap of a reweighted mean: resample users within each stratum."""
    idx = [np.flatnonzero(g == k) for k in range(K)]
    out = np.zeros(b)
    for k in range(K):
        x = v[idx[k]]
        mk = x[RNG.integers(0, len(x), (b, len(x)))].mean(1)
        out += weights[k] * mk
    return out


def main():
    d = pickle.load(open(os.path.join(ROOT, 'data', 'ml1m.pkl'), 'rb'))
    X = load()
    S = strata(d)
    res = {'n_users': int(d['n_users']), 'metric': METRIC, 'dimensions': {}}
    for dim, (g, names) in S.items():
        K = len(names)
        n_k = np.array([(g == k).sum() for k in range(K)])
        D = {'levels': names, 'n': n_k.tolist(),
             'ndcg': {m: [r4(X[m][g == k].mean()) for k in range(K)] for m in MODELS if m in X},
             'pairs': {}}
        # which model is best in each stratum
        D['best'] = [max((m for m in MODELS if m in X), key=lambda m: X[m][g == k].mean()) for k in range(K)]
        for a, b in PAIRS:
            if a not in X or b not in X: continue
            reseed(dim, a, b)
            dd = X[a] - X[b]
            rows = []
            for k in range(K):
                x = dd[g == k]; nz = x[x != 0]
                p = stats.wilcoxon(nz).pvalue if len(nz) > 10 else 1.0
                bm = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
                rows.append(dict(delta=r4(x.mean()), ci=[r4(np.percentile(bm, 2.5)), r4(np.percentile(bm, 97.5))],
                                 rel=r4(100 * x.mean() / max(X[b][g == k].mean(), 1e-12)), p=p))
            for r_, q in zip(rows, holm([r_['p'] for r_ in rows])):
                r_['p'] = r4(r_['p']); r_['p_holm'] = r4(q)
            z, pv = van_elteren(dd, g, K)
            Q, pq, I2 = cochran_q(dd, g, K)
            # generalisation: equal-weight (balanced) population and the least favourable stratum
            eq = strat_boot_means(dd, g, K, np.full(K, 1 / K))
            worst = int(np.argmin([dd[g == k].mean() for k in range(K)]))
            D['pairs'][f'{a}-{b}'] = dict(strata=rows, vanelteren_z=r4(z), vanelteren_p=r4(pv),
                                         Q=r4(Q), Q_p=r4(pq), I2=r4(I2),
                                         equal_weight=r4(np.mean([dd[g == k].mean() for k in range(K)])),
                                         equal_weight_ci=[r4(np.percentile(eq, 2.5)), r4(np.percentile(eq, 97.5))],
                                         n_positive=int(sum(dd[g == k].mean() > 0 for k in range(K))), K=K,
                                         worst_level=names[worst], worst_delta=r4(dd[g == worst].mean()))
        res['dimensions'][dim] = D
    # Holm across the seven dimensions for each pair's stratified test and heterogeneity test
    for key in [f'{a}-{b}' for a, b in PAIRS]:
        dims = [dm for dm in res['dimensions'] if key in res['dimensions'][dm]['pairs']]
        for fld in ('vanelteren_p', 'Q_p'):
            adj = holm([res['dimensions'][dm]['pairs'][key][fld] for dm in dims])
            for dm, q in zip(dims, adj): res['dimensions'][dm]['pairs'][key][fld + '_holm'] = r4(q)
    # ranking stability under a stratified bootstrap over history quintile x target popularity (15 cells)
    cell = S['history'][0] * 3 + S['popularity'][0]
    cand = [m for m in MODELS if m in X and m not in ('tpr_nokg', 'stellar_nokg', 'tpr_tok_sgw_nokg', 'tpr_tok_lightgcn')]
    idx = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    reseed('stability')
    first = {m: 0 for m in cand}; beat = {b: 0 for b in cand if b != 'stellar'}
    M = np.stack([X[m] for m in cand])
    for _ in range(B):
        sel = np.concatenate([ix[RNG.integers(0, len(ix), len(ix))] for ix in idx])
        mu = M[:, sel].mean(1)
        first[cand[int(np.argmax(mu))]] += 1
        for j, m in enumerate(cand):
            if m != 'stellar': beat[m] += int(mu[cand.index('stellar')] > mu[j])
    res['stability'] = dict(cells=int(len(idx)), B=B, first={m: r4(v / B) for m, v in first.items() if v},
                            stellar_beats={m: r4(v / B) for m, v in beat.items()})
    # joint balanced population: every history x popularity cell equally weighted
    jw = {}
    for a, b in PAIRS:
        if a in X and b in X:
            reseed('balanced', a, b)
            dd = X[a] - X[b]; cells = np.unique(cell)
            bt = strat_boot_means(dd, np.searchsorted(cells, cell), len(cells), np.full(len(cells), 1 / len(cells)))
            jw[f'{a}-{b}'] = dict(delta=r4(np.mean([dd[cell == c].mean() for c in cells])),
                                  ci=[r4(np.percentile(bt, 2.5)), r4(np.percentile(bt, 97.5))])
    res['balanced_history_x_popularity'] = jw
    # knowledge strata adjusted for the confounders popularity, history length and time gap:
    # (a) within each knowledge level a van Elteren test stratified by target popularity,
    # (b) OLS of the per-user difference on the knowledge level (linear trend) with stratum dummies, HC3 errors
    import statsmodels.formula.api as smf
    kn, po = S['knowledge'][0], S['popularity'][0]
    adj = {}
    for a, b in [('tpr', 'tpr_nokg'), ('stellar', 'stellar_nokg'), ('tpr_tok_sgw_nokg', 'tpr_nokg'), ('tpr_tok_lightgcn', 'tpr_nokg'), ('tpr', 'tpr_tok_sgw_nokg')]:
        dd = X[a] - X[b]
        lv = []
        for k in range(3):
            m = kn == k; z, pv = van_elteren(dd[m], po[m], 3); lv.append(dict(delta=r4(dd[m].mean()), p=r4(pv)))
        df = pd.DataFrame(dict(d=dd, k=kn, p=po, h=S['history'][0], g=S['gap'][0]))
        fit = smf.ols('d ~ k + C(p) + C(h) + C(g)', df).fit(cov_type='HC3')
        ci = fit.conf_int().loc['k'].tolist()
        adj[f'{a}-{b}'] = dict(levels=lv, trend=r4(fit.params['k']), trend_ci=[r4(ci[0]), r4(ci[1])], trend_p=r4(fit.pvalues['k']))
    res['knowledge_adjusted'] = adj
    json.dump(res, open(os.path.join(SD, 'ml1m_strat.json'), 'w'), indent=1)
    return res


if __name__ == '__main__':
    r = main()
    for dim, D in r['dimensions'].items():
        print(f"\n== {dim}  levels={D['levels']}  n={D['n']}  best={D['best']}")
        for m in ('stellar', 'tpr', 'tpr_nokg', 'sasrec', 'bsarec', 'kgin'):
            print(f'  {m:10s}', D['ndcg'][m])
        for key, P in D['pairs'].items():
            print(f"  {key:22s} d={[s['delta'] for s in P['strata']]} p={[s['p_holm'] for s in P['strata']]} "
                  f"vE_p={P['vanelteren_p']} Q_p={P['Q_p']} I2={P['I2']} eq={P['equal_weight']} {P['equal_weight_ci']} pos={P['n_positive']}/{P['K']}")
    print('\nstability', r['stability'])
    print('balanced', r['balanced_history_x_popularity'])
