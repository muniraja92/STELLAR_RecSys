"""Global dependency-driven scheduler (same grids, same selection rule and same runs as driver.py / pdriver.py).

Usage: python gdriver.py <workers> <ds1,ds2,...> [seeds]
Every run starts as soon as its inputs exist, so no dataset waits for a straggling tuning run of another model.
Runs already in progress (e.g. started by an earlier driver) are detected from the process table and not duplicated."""
import os, sys, json, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import RES, done
import driver as D

STAGE_A = ['pop', 'itemknn', 'ease', 'bpr', 'lightgcn', 'kgat', 'kgin', 'sgw', 'sgw_nokg', 'sgw_band', 'sgw_fixrel',
           'sasrec', 'bert4rec', 'bsarec', 'tpr_nokg']
STAGE_B = ['tpr', 'tpr_nopcl', 'tpr_nocir', 'tpr_notime']
FINAL = STAGE_A + STAGE_B + list(D.STELLAR)
TRIES = 3


def log(*a):
    print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)


def grid(ds, m):
    g = list(D.GRIDS[m])
    if m == 'bert4rec' and ds != 'ml1m':
        g = [dict(c, mask_prob=0.6) for c in g]
    return g + D.EXTRA.get(m, [])


def running_tags():
    out = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    tags = set()
    for l in out.splitlines():
        if 'run.py' not in l or '--dataset' not in l: continue
        w = l.split()
        try:
            tags.add((w[w.index('--dataset') + 1], w[w.index('--tag') + 1], int(w[w.index('--seed') + 1])))
        except (ValueError, IndexError):
            pass
    return tags


class State:
    def __init__(self, ds, seeds):
        self.ds, self.seeds = ds, seeds
        self.bestf = os.path.join(RES, ds, 'best_configs.json')
        self.best = json.load(open(self.bestf)) if os.path.exists(self.bestf) else {}

    def select(self, m):
        if m in self.best: return True
        g = grid(self.ds, m)
        if not all(done(self.ds, f'tune_{m}_{k}', 42) for k in range(len(g))): return False
        scores = []
        for k, cfg in enumerate(g):
            r = json.load(open(os.path.join(RES, self.ds, f'tune_{m}_{k}', 'seed42.json')))
            scores.append((r.get('val_NDCG@10', -1), cfg))
        sel = max(scores, key=lambda x: x[0])
        self.best[m] = sel[1]
        json.dump(self.best, open(self.bestf, 'w'), indent=1)
        log(f'TUNED {self.ds} {m}: {sel[1]} (val NDCG@10 {sel[0]:.4f}) grid={[round(s, 4) for s, _ in scores]}')
        return True

    def jobs(self):
        """All runs whose inputs exist now: (priority, ds, model, seed, cfg, tag)."""
        ds, out = self.ds, []
        for m in D.GRIDS:
            if m == 'tpr': continue
            for k, cfg in enumerate(grid(ds, m)):
                out.append((0, ds, m, 42, cfg, f'tune_{m}_{k}'))
        for m in D.GRIDS:
            if m != 'tpr': self.select(m)
        if 'sgw' in self.best and done(ds, 'sgw', 42) or 'sgw' in self.best and self.reuse('sgw'):
            for k, cfg in enumerate(D.GRIDS['tpr']):
                out.append((1, ds, 'tpr', 42, cfg, f'tune_tpr_{k}'))
            self.select('tpr')
        for i, m in enumerate(FINAL):
            for s in self.seeds:
                if m in D.STELLAR:
                    cfg = D.STELLAR[m]
                    t, g = cfg.get('tpr', 'tpr'), cfg.get('sgw', 'sgw')
                    if done(ds, t, s) and done(ds, g, s):
                        out.append((3, ds, 'stellar', s, cfg, m))
                    continue
                parent = D.INHERIT.get(m, m)
                if parent not in self.best: continue
                if m in STAGE_B and not done(ds, 'sgw', s): continue
                if s == 42 and m in D.GRIDS and self.reuse(m): continue
                out.append((2 + (s != 42), ds, m, s, dict(self.best[parent]), m))
        return out

    def reuse(self, m):
        if done(self.ds, m, 42): return True
        if m not in self.best: return False
        D.reuse_tuned(self.ds, m, self.best[m])
        return done(self.ds, m, 42)


def main():
    W = int(sys.argv[1]); dss = sys.argv[2].split(',')
    seeds = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [42, 1, 7]
    for d in dss: os.makedirs(os.path.join(RES, d), exist_ok=True)
    st = [State(d, seeds) for d in dss]
    procs, fails = {}, {}
    while True:
        for key, (p, t0) in list(procs.items()):
            if p.poll() is None: continue
            del procs[key]
            ok = done(key[0], key[1], key[2])
            if not ok: fails[key] = fails.get(key, 0) + 1
            log(('DONE' if ok else f'FAIL#{fails.get(key)}'), *key, f'[{time.time() - t0:.0f}s]')
        busy = running_tags()
        pending = []
        for s in st:
            for pr, ds, m, seed, cfg, tag in s.jobs():
                key = (ds, tag, seed)
                if done(ds, tag, seed) or key in busy or key in procs or fails.get(key, 0) >= TRIES: continue
                pending.append((pr, dss.index(ds), key, m, cfg))
        pending.sort(key=lambda x: (x[0], x[1]))
        n_running = len(busy)
        for pr, _, key, m, cfg in pending:
            if n_running >= W: break
            ds, tag, seed = key
            lf = open(os.path.join(RES, ds, 'logs', f'{tag}_s{seed}.out'), 'a') if os.path.isdir(os.path.join(RES, ds, 'logs')) else subprocess.DEVNULL
            cmd = [sys.executable, os.path.join(HERE, 'run.py'), '--dataset', ds, '--model', m, '--seed', str(seed),
                   '--cfg', json.dumps(cfg), '--tag', tag]
            procs[key] = (subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT), time.time())
            n_running += 1
            log('START', ds, tag, seed, json.dumps(cfg))
        if not procs and not pending and not busy:
            left = [(s.ds, m, x) for s in st for m in FINAL for x in seeds if not done(s.ds, m, x)]
            log('GDRIVER FINISHED; missing:', left); break
        time.sleep(30)


if __name__ == '__main__':
    main()
