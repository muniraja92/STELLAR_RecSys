"""Tuning + multi-seed driver.

Usage: python driver.py <dataset> [phase]   phase in {tune, final, all}
Tuning: every model gets a small grid, evaluated on validation NDCG@10 with seed 42 only.
The selected configuration is stored in results/<dataset>/best_configs.json and then run with
all three seeds (42, 1, 7).  Test metrics are never used for selection."""
import os, sys, json, subprocess, time, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RES

SEEDS = [42, 1, 7]   # the three seeds of the study
GRIDS = {
    'pop': [{}],
    'itemknn': [{'k': k} for k in (20, 50, 100, 200, 500)],
    'ease': [{'lam': l} for l in (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)],
    'bpr': [{'lr': 1e-3}, {'lr': 5e-3}],
    'lightgcn': [{'lr': 1e-3}, {'lr': 5e-3}],
    'kgat': [{'lr': 1e-3}, {'lr': 5e-3}],
    'kgin': [{'lr': 1e-3}, {'lr': 5e-3}],
    'sgw': [{'lr': 1e-3}, {'lr': 5e-3}],
    'sasrec': [{'drop': d} for d in (0.2, 0.5)],
    'bert4rec': [{'drop': d} for d in (0.2, 0.5)],
    'bsarec': [{'drop': d, 'alpha': 0.7, 'c': 5} for d in (0.2, 0.5)],
    'tpr': [{'drop': d, 'pcl': 0.1, 'cir': 0.1} for d in (0.2, 0.5)],
}
# boundary extensions of the cheap grids (run after the main grid, same selection rule)
EXTRA = {'itemknn': [{'k': 5}, {'k': 10}, {'k': 1000}], 'ease': [{'lam': 10.0}, {'lam': 20.0}, {'lam': 5000.0}]}
# ablation variants inherit the configuration selected for their parent model
INHERIT = {'sgw_nokg': 'sgw', 'sgw_band': 'sgw', 'sgw_fixrel': 'sgw', 'tpr_nokg': 'tpr', 'tpr_nopcl': 'tpr', 'tpr_nocir': 'tpr',
           'tpr_notime': 'tpr', 'tpr_norope': 'tpr', 'tpr_nodecay': 'tpr'}
STELLAR = {'stellar': {}, 'stellar_fixed': {'gated': False},
           'stellar_nokg': {'tpr': 'tpr_nokg', 'sgw': 'sgw_nokg'}}
ORDER = ['pop', 'itemknn', 'ease', 'bpr', 'lightgcn', 'kgat', 'kgin', 'sgw', 'sgw_nokg', 'sgw_band', 'sgw_fixrel',
         'sasrec', 'bert4rec', 'bsarec', 'tpr', 'tpr_nokg', 'tpr_nopcl', 'tpr_nocir', 'tpr_notime',
         'tpr_norope', 'tpr_nodecay']


def run(ds, model, seed, cfg, tag):
    cmd = [sys.executable, os.path.join(HERE, 'run.py'), '--dataset', ds, '--model', model, '--seed', str(seed),
           '--cfg', json.dumps(cfg), '--tag', tag]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    os.makedirs(os.path.join(RES, ds, 'logs'), exist_ok=True)
    open(os.path.join(RES, ds, 'logs', f'{tag}_s{seed}.log'), 'w').write(out)
    line = [l for l in out.splitlines() if '== DONE' in l or 'already done' in l]
    if p.returncode != 0 or not line:
        print(f'ERROR {ds} {tag} s{seed} rc={p.returncode}: ' + ' | '.join(out.strip().splitlines()[-3:]), flush=True)
        return None
    print(line[-1].split('== ')[-1] + f' [{time.time() - t:.0f}s]', flush=True)
    j = os.path.join(RES, ds, tag, f'seed{seed}.json')
    return json.load(open(j)) if os.path.exists(j) else None


def tune(ds):
    bestf = os.path.join(RES, ds, 'best_configs.json')
    best = json.load(open(bestf)) if os.path.exists(bestf) else {}
    for m, grid in GRIDS.items():
        if m in best: continue
        if m == 'bert4rec' and ds != 'ml1m':   # masking rate of the original BERT4Rec paper for Amazon data
            grid = [dict(c, mask_prob=0.6) for c in grid]
        scores = []
        for k, cfg in enumerate(grid):
            if len(grid) == 1:
                scores.append((0.0, cfg)); break
            if m == 'tpr' and not os.path.exists(os.path.join(RES, ds, 'sgw', 'seed42.json')):
                reuse_tuned(ds, 'sgw', best['sgw'])
                run(ds, 'sgw', 42, best['sgw'], 'sgw')
            r = run(ds, m, 42, cfg, f'tune_{m}_{k}')
            scores.append((r['val_NDCG@10'] if r else -1, cfg))
        sel = max(scores, key=lambda x: x[0])
        best[m] = sel[1]
        print(f'TUNED {ds} {m}: {sel[1]} (val NDCG@10 {sel[0]:.4f}) grid={[round(s, 4) for s, _ in scores]}', flush=True)
        json.dump(best, open(bestf, 'w'), indent=1)
    return best


def extend(ds):
    bestf = os.path.join(RES, ds, 'best_configs.json')
    best = json.load(open(bestf))
    for m, extra in EXTRA.items():
        base = len(GRIDS[m])
        for i, cfg in enumerate(extra):
            if not os.path.exists(os.path.join(RES, ds, f'tune_{m}_{base + i}', 'seed42.json')):
                run(ds, m, 42, cfg, f'tune_{m}_{base + i}')
        scores = []
        for k in range(base + len(extra)):
            j = os.path.join(RES, ds, f'tune_{m}_{k}', 'seed42.json')
            if os.path.exists(j):
                r = json.load(open(j)); scores.append((r['val_NDCG@10'], r['cfg']))
        sel = max(scores, key=lambda x: x[0])
        best[m] = sel[1]
        print(f'TUNED+ {ds} {m}: {sel[1]} (val NDCG@10 {sel[0]:.4f}) grid={[(round(s, 4), c) for s, c in scores]}', flush=True)
        json.dump(best, open(bestf, 'w'), indent=1)


def reuse_tuned(ds, m, cfg):
    """The selected seed-42 tuning run is the seed-42 final run (identical configuration and seed)."""
    dst = os.path.join(RES, ds, m)
    if os.path.exists(os.path.join(dst, 'seed42.json')): return
    for k in range(10):
        src = os.path.join(RES, ds, f'tune_{m}_{k}')
        j = os.path.join(src, 'seed42.json')
        if os.path.exists(j) and json.load(open(j)).get('cfg') == cfg:
            shutil.copytree(src, dst, dirs_exist_ok=True); print(f'REUSED {ds} {m} seed 42 from tune_{m}_{k}', flush=True); return


def extended(ds):
    return all(os.path.exists(os.path.join(RES, ds, f'tune_{m}_{len(GRIDS[m]) + i}', 'seed42.json'))
               for m, extra in EXTRA.items() for i in range(len(extra)))


def final(ds, models=None, seeds=SEEDS):
    if not extended(ds): extend(ds)
    best = json.load(open(os.path.join(RES, ds, 'best_configs.json')))
    for seed in seeds:
        for m in (models or ORDER):
            cfg = dict(best[INHERIT.get(m, m)])
            if seed == 42 and m in GRIDS:
                reuse_tuned(ds, m, cfg)
            run(ds, m, seed, cfg, m)
        for name, cfg in STELLAR.items():
            if models is None or name in models:
                run(ds, 'stellar', seed, cfg, name)


if __name__ == '__main__':
    ds = sys.argv[1]; phase = sys.argv[2] if len(sys.argv) > 2 else 'all'
    models = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    seeds = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else SEEDS
    os.makedirs(os.path.join(RES, ds), exist_ok=True)
    if phase in ('tune', 'all'): tune(ds)
    if phase in ('extend', 'all'): extend(ds)
    if phase in ('final', 'all'): final(ds, models, seeds)
    print('DRIVER FINISHED', ds, phase, flush=True)
