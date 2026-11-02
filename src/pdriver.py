"""Parallel version of driver.py for a multi-core machine (same grids, same selection rule, same runs).

Usage: python pdriver.py <dataset> <workers> [seeds]
Phase 1: every grid point (except TPR, whose knowledge tokens need the selected KG-SGW) in parallel,
         plus the boundary extensions of the ItemKNN / EASE grids; selection on validation NDCG@10.
Phase 2: TPR grid (after the seed-42 KG-SGW run with the selected configuration exists).
Phase 3: all final runs for the requested seeds, respecting dependencies
         (KG-SGW variants -> TPR variants with knowledge tokens -> STELLAR variants)."""
import os, sys, json
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RES
import driver as D

STAGE_A = ['pop', 'itemknn', 'ease', 'bpr', 'lightgcn', 'kgat', 'kgin', 'sgw', 'sgw_nokg', 'sgw_band', 'sgw_fixrel',
           'sasrec', 'bert4rec', 'bsarec', 'tpr_nokg']
STAGE_B = ['tpr', 'tpr_nopcl', 'tpr_nocir', 'tpr_notime']
STAGE_C = ['stellar', 'stellar_fixed', 'stellar_nokg']


def run_retry(*job, tries=3):
    # a run killed from outside (e.g. by the system's memory daemon) is restarted; completed runs are skipped by run.py
    for _ in range(tries):
        r = D.run(*job)
        if r is not None: return r
    return None


def pmap(pool, jobs):
    futs = [pool.submit(run_retry, *j) for j in jobs]
    return [f.result() for f in futs]


def tune(ds, pool):
    bestf = os.path.join(RES, ds, 'best_configs.json')
    best = json.load(open(bestf)) if os.path.exists(bestf) else {}
    jobs, grids = [], {}
    for m, grid in D.GRIDS.items():
        if m == 'tpr' or m in best: continue
        if m == 'bert4rec' and ds != 'ml1m':
            grid = [dict(c, mask_prob=0.6) for c in grid]
        grid = list(grid) + D.EXTRA.get(m, [])
        grids[m] = grid
        for k, cfg in enumerate(grid):
            jobs.append((ds, m, 42, cfg, f'tune_{m}_{k}'))
    res = pmap(pool, jobs)
    i = 0
    for m, grid in grids.items():
        scores = []
        for cfg in grid:
            r = res[i]; i += 1
            scores.append((r['val_NDCG@10'] if r else -1, cfg))
        sel = max(scores, key=lambda x: x[0]); best[m] = sel[1]
        print(f'TUNED {ds} {m}: {sel[1]} (val NDCG@10 {sel[0]:.4f}) grid={[round(s, 4) for s, _ in scores]}', flush=True)
    json.dump(best, open(bestf, 'w'), indent=1)
    if 'tpr' not in best:
        D.reuse_tuned(ds, 'sgw', best['sgw'])
        D.run(ds, 'sgw', 42, best['sgw'], 'sgw')
        grid = D.GRIDS['tpr']
        res = pmap(pool, [(ds, 'tpr', 42, cfg, f'tune_tpr_{k}') for k, cfg in enumerate(grid)])
        scores = [(r['val_NDCG@10'] if r else -1, cfg) for r, cfg in zip(res, grid)]
        sel = max(scores, key=lambda x: x[0]); best['tpr'] = sel[1]
        print(f'TUNED {ds} tpr: {sel[1]} (val NDCG@10 {sel[0]:.4f}) grid={[round(s, 4) for s, _ in scores]}', flush=True)
        json.dump(best, open(bestf, 'w'), indent=1)
    return best


def final(ds, pool, seeds):
    best = json.load(open(os.path.join(RES, ds, 'best_configs.json')))
    def job(m, seed):
        if m in D.STELLAR:
            return (ds, 'stellar', seed, D.STELLAR[m], m)
        cfg = dict(best[D.INHERIT.get(m, m)])
        if seed == 42 and m in D.GRIDS:
            D.reuse_tuned(ds, m, cfg)
        return (ds, m, seed, cfg, m)
    for stage in (STAGE_A, STAGE_B, STAGE_C):
        pmap(pool, [job(m, s) for s in seeds for m in stage])


if __name__ == '__main__':
    ds = sys.argv[1]; W = int(sys.argv[2])
    seeds = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [42, 1, 7]
    os.makedirs(os.path.join(RES, ds), exist_ok=True)
    with ThreadPoolExecutor(W) as pool:
        tune(ds, pool)
        final(ds, pool, seeds)
    print('PDRIVER FINISHED', ds, flush=True)
