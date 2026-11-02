"""Collect the per-user test metrics of every MovieLens-1M run into results/strat/ml1m_peruser.npz (input of stratify.py)."""
import os, glob
import numpy as np
from common import RES

if __name__ == '__main__':
    R = os.path.join(RES, 'ml1m'); out = {}
    for m in sorted(os.listdir(R)):
        fs = sorted(glob.glob(os.path.join(R, m, 'seed*.npz')))
        if not fs or m.startswith('tune'): continue
        for f in fs:
            z = np.load(f); s = os.path.basename(f)[4:-4]
            for k in ('NDCG@10', 'HR@10', 'NDCG@20', 'HR@20'):
                if k in z.files: out[f'{m}|{s}|{k}'] = z[k].astype(np.float32)
    os.makedirs(os.path.join(RES, 'strat'), exist_ok=True)
    np.savez_compressed(os.path.join(RES, 'strat', 'ml1m_peruser.npz'), **out)
    print('saved', len(out), 'arrays')
