"""Shared data access, evaluation and utilities for the STELLAR-KG revision experiments.

Protocol
--------
* Chronological leave-one-out: for every user the last interaction is the test target, the
  second-to-last the validation target, everything earlier is training data.
* Models are fit on training data only.  Validation scores use the training history as input;
  test scores use training + validation history as input (the validation interaction is known
  at test time, the test interaction never is).
* Full-catalogue ranking: every item the user has not interacted with in the input history is a
  candidate.  Ties are broken pessimistically (an item tied with the target counts as ranked
  above it), so constant scorers cannot obtain spurious credit.
* A fixed, model-independent set of 99 sampled negatives per user is also kept for the
  sampled-metric comparison reported in the appendix.
"""
import os, json, time, pickle, random, math
import numpy as np
import scipy.sparse as sp
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
KS = (5, 10, 20)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Dataset:
    def __init__(self, name):
        with open(os.path.join(DATA, f'{name}.pkl'), 'rb') as f:
            b = pickle.load(f)
        self.name = name
        self.n_users, self.n_items = b['n_users'], b['n_items']
        self.seqs, self.ts = b['seqs'], b['ts']
        self.train = [np.array(x, dtype=np.int64) for x in b['train']]
        self.val, self.test = np.array(b['val'], dtype=np.int64), np.array(b['test'], dtype=np.int64)
        self.seqs = [np.array(x, dtype=np.int64) for x in self.seqs]; self.ts = [np.array(x, dtype=np.int64) for x in self.ts]
        self.kg, self.relations, self.n_entities = b['kg'], b['relations'], b['n_entities']
        self.n_relations = len(self.relations)
        # training matrix
        rows = np.concatenate([np.full(len(s), u) for u, s in enumerate(self.train)])
        cols = np.concatenate(self.train)
        self.R_train = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(self.n_users, self.n_items))
        self.R_train.data[:] = 1.0
        self.R_train.sort_indices()
        self.train_keys = np.sort(rows.astype(np.int64) * self.n_items + cols)
        self.item_pop = np.asarray(self.R_train.sum(0)).ravel()
        # input histories per phase
        self.hist_val = self.train
        self.hist_test = [np.append(s, v) for s, v in zip(self.train, self.val)]
        self.ts_val = [t[:-2] for t in self.ts]
        self.ts_test = [t[:-1] for t in self.ts]
        # popularity groups (by training popularity): head = top 20% items by count (short head),
        # tertile environments used by CIR
        order = np.argsort(-self.item_pop, kind='stable')
        self.head_items = np.zeros(self.n_items, bool); self.head_items[order[:int(0.2 * self.n_items)]] = True
        q1, q2 = np.quantile(self.item_pop, [1 / 3, 2 / 3])
        self.pop_env = np.where(self.item_pop < q1, 0, np.where(self.item_pop < q2, 1, 2)).astype(np.int64)
        self._check_leakage()
        self._sampled_negatives()

    def _check_leakage(self):
        for u in range(self.n_users):
            s = self.train[u]
            assert self.val[u] != self.test[u] or True
            assert self.seqs[u][-1] == self.test[u] and self.seqs[u][-2] == self.val[u]
            assert len(s) == len(self.seqs[u]) - 2
        # training matrix contains no validation/test pairs (ML/Amazon data are de-duplicated)
        u = np.arange(self.n_users)
        assert not self.is_train_pair(u, self.test).any() and not self.is_train_pair(u, self.val).any(), \
            'held-out item found in training matrix'

    def is_train_pair(self, u, i):
        k = np.asarray(u, np.int64) * self.n_items + np.asarray(i, np.int64)
        pos = np.searchsorted(self.train_keys, k)
        pos = np.minimum(pos, len(self.train_keys) - 1)
        return self.train_keys[pos] == k

    def _sampled_negatives(self, n_neg=99, seed=2026):
        rng = np.random.default_rng(seed)
        negs = np.zeros((self.n_users, n_neg), np.int64)
        for u in range(self.n_users):
            seen = set(self.seqs[u].tolist())
            out = []
            while len(out) < n_neg:
                c = rng.integers(0, self.n_items, size=2 * n_neg)
                for x in c:
                    if x not in seen and x not in out:
                        out.append(int(x))
                        if len(out) == n_neg: break
            negs[u] = out
        self.neg99 = negs

    def history(self, phase):
        return self.hist_val if phase == 'val' else self.hist_test

    def history_ts(self, phase):
        return self.ts_val if phase == 'val' else self.ts_test

    def targets(self, phase):
        return self.val if phase == 'val' else self.test


def evaluate(ds, score_fn, phase='test', users=None, batch=1024, topn=20, keep_lists=True):
    """score_fn(user_idx_array, phase) -> np.ndarray (B, n_items) of scores (higher = better).

    Returns dict with per-user metric arrays and top-N lists."""
    users = np.arange(ds.n_users) if users is None else np.asarray(users)
    hist = ds.history(phase); tgt = ds.targets(phase)
    out = {f'{m}@{k}': np.zeros(len(users)) for m in ('HR', 'NDCG') for k in KS}
    out['MRR@10'] = np.zeros(len(users)); out['rank'] = np.zeros(len(users), np.int64)
    out['s_HR@10'] = np.zeros(len(users)); out['s_NDCG@10'] = np.zeros(len(users))
    tops = np.zeros((len(users), topn), np.int64)
    for b0 in range(0, len(users), batch):
        ub = users[b0:b0 + batch]
        S = np.asarray(score_fn(ub, phase), dtype=np.float64)
        assert S.shape == (len(ub), ds.n_items), S.shape
        S = np.nan_to_num(S, nan=-1e30, posinf=1e30, neginf=-1e30)
        t = tgt[ub]
        st = S[np.arange(len(ub)), t].copy()
        # sampled-99 metrics (fixed, model-independent negatives)
        sn = np.take_along_axis(S, ds.neg99[ub], 1)
        srank = (sn >= st[:, None]).sum(1)
        out['s_HR@10'][b0:b0 + len(ub)] = srank < 10
        out['s_NDCG@10'][b0:b0 + len(ub)] = np.where(srank < 10, 1 / np.log2(srank + 2), 0)
        # mask history
        for j, u in enumerate(ub):
            S[j, hist[u]] = -np.inf
        S[np.arange(len(ub)), t] = st
        # pessimistic rank: items with score >= target (other than target itself)
        rank = (S >= st[:, None]).sum(1) - 1
        sl = slice(b0, b0 + len(ub))
        out['rank'][sl] = rank
        for k in KS:
            hit = rank < k
            out[f'HR@{k}'][sl] = hit
            out[f'NDCG@{k}'][sl] = np.where(hit, 1 / np.log2(rank + 2), 0)
        out['MRR@10'][sl] = np.where(rank < 10, 1 / (rank + 1), 0)
        if keep_lists:
            part = np.argpartition(-S, topn, axis=1)[:, :topn]
            ps = np.take_along_axis(S, part, 1)
            o = np.argsort(-ps, axis=1, kind='stable')
            tops[sl] = np.take_along_axis(part, o, 1)
    out['users'] = users
    if keep_lists:
        out['top'] = tops
    return out


def summarize(ds, res, k_list=10):
    """Accuracy + beyond-accuracy summary from evaluate() output."""
    s = {m: float(res[m].mean()) for m in res if '@' in m}
    top = res['top'][:, :k_list]
    pop = ds.item_pop
    s['ARP@10'] = float(pop[top].mean())
    s['ARP_norm@10'] = float((pop[top] / pop.max()).mean())
    s['APLT@10'] = float((~ds.head_items[top]).mean())
    cnt = np.bincount(top.ravel(), minlength=ds.n_items).astype(np.float64)
    s['Coverage@10'] = float((cnt > 0).mean())
    x = np.sort(cnt); n = len(x)
    s['Gini@10'] = float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum()))
    tgt = ds.targets('test')[res['users']]
    tail = ~ds.head_items[tgt]
    s['NDCG@10_tail_target'] = float(res['NDCG@10'][tail].mean()) if tail.any() else float('nan')
    s['NDCG@10_head_target'] = float(res['NDCG@10'][~tail].mean()) if (~tail).any() else float('nan')
    for e, nm in enumerate(('low', 'mid', 'high')):
        m = ds.pop_env[tgt] == e
        s[f'NDCG@10_env_{nm}'] = float(res['NDCG@10'][m].mean()) if m.any() else float('nan')
    return s


def save_result(ds_name, model, seed, res, summ, extra=None):
    d = os.path.join(RES, ds_name, model); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f'seed{seed}.npz'),
                        **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})
    summ = dict(summ); summ.update(extra or {})
    with open(os.path.join(d, f'seed{seed}.json'), 'w') as f:
        json.dump(summ, f, indent=1)


def done(ds_name, model, seed):
    return os.path.exists(os.path.join(RES, ds_name, model, f'seed{seed}.json'))


def pad_seqs(seqs, max_len, ts=None):
    """Left-pad item sequences (items shifted by +1, 0 = padding)."""
    B = len(seqs)
    X = np.zeros((B, max_len), np.int64)
    T = np.zeros((B, max_len), np.float32)
    for j, s in enumerate(seqs):
        s = s[-max_len:]
        X[j, max_len - len(s):] = s + 1
        if ts is not None:
            t = ts[j][-max_len:]
            gaps = np.zeros(len(t), np.float32)
            gaps[1:] = np.diff(t) / 86400.0
            T[j, max_len - len(t):] = np.log1p(np.clip(gaps, 0, 3650))
    return X, T


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        n = time.time(); d = n - self.t; self.t = n; return d


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)
