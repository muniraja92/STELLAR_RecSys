"""Non-sequential baselines: Popularity, ItemKNN, EASE, BPR-MF, LightGCN."""
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F
from common import evaluate, log, set_seed, DEVICE


def hist_matrix(ds, phase):
    hist = ds.history(phase)
    rows = np.concatenate([np.full(len(s), u) for u, s in enumerate(hist)])
    cols = np.concatenate(hist)
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(ds.n_users, ds.n_items))
    M.data[:] = 1.0
    return M


class Pop:
    def __init__(self, ds, **kw):
        self.ds = ds

    def fit(self):
        return self

    def score(self, users, phase):
        return np.tile(self.ds.item_pop.astype(np.float64) + 1e-6 * np.arange(self.ds.n_items)[::-1] / self.ds.n_items, (len(users), 1))


class ItemKNN:
    """Item-based k-nearest-neighbour CF (Sarwar et al., 2001; Deshpande & Karypis, 2004):
    cosine similarity on the binary training matrix, top-k neighbours per item, shrinkage."""
    def __init__(self, ds, k=100, shrink=10.0, **kw):
        self.ds, self.k, self.shrink = ds, k, shrink

    def fit(self):
        X = self.ds.R_train.tocsc().astype(np.float32)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel())
        S = (X.T @ X).toarray()
        np.fill_diagonal(S, 0)
        S = S / (np.outer(norms, norms) + self.shrink + 1e-9)
        if self.k < S.shape[0]:
            idx = np.argpartition(-S, self.k, axis=0)[self.k:]
            np.put_along_axis(S, idx, 0, axis=0)
        self.S = S.astype(np.float32)
        self.H = {p: hist_matrix(self.ds, p) for p in ('val', 'test')}
        return self

    def score(self, users, phase):
        return (self.H[phase][users] @ self.S)


class EASE:
    """Embarrassingly Shallow Autoencoder (Steck, 2019)."""
    def __init__(self, ds, lam=500.0, **kw):
        self.ds, self.lam = ds, lam

    def fit(self):
        import fcntl   # the dense inverse needs several GB: only one EASE fit at a time on a machine
        with open('/tmp/stellar_ease.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            X = self.ds.R_train
            G = (X.T @ X).toarray().astype(np.float64)
            G[np.diag_indices_from(G)] += self.lam
            P = torch.linalg.inv(torch.as_tensor(G, device=DEVICE)).cpu().numpy(); del G
            fcntl.flock(lk, fcntl.LOCK_UN)
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0
        self.B = B.astype(np.float32)
        self.H = {p: hist_matrix(self.ds, p) for p in ('val', 'test')}
        return self

    def score(self, users, phase):
        return self.H[phase][users] @ self.B


def sample_bpr(ds, rng, n):
    R = ds.R_train
    u = rng.integers(0, ds.n_users, n)
    indptr, indices = R.indptr, R.indices
    lens = indptr[u + 1] - indptr[u]
    keep = lens > 0; u = u[keep]; lens = lens[keep]
    pos = indices[indptr[u] + (rng.random(len(u)) * lens).astype(np.int64)]
    neg = rng.integers(0, ds.n_items, len(u))
    # resample negatives that collide with training positives
    for _ in range(5):
        bad = ds.is_train_pair(u, neg)
        if not bad.any(): break
        neg[bad] = rng.integers(0, ds.n_items, bad.sum())
    return u, pos, neg


def norm_adj_bipartite(ds):
    n = ds.n_users + ds.n_items
    R = ds.R_train.tocoo()
    rows = np.concatenate([R.row, R.col + ds.n_users]); cols = np.concatenate([R.col + ds.n_users, R.row])
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    d = np.asarray(A.sum(1)).ravel(); di = np.power(np.maximum(d, 1e-12), -0.5)
    A = sp.diags(di) @ A @ sp.diags(di)
    return to_torch_sparse(A.tocoo())


def to_torch_sparse(A):
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A.data, dtype=torch.float32), A.shape).coalesce().to_sparse_csr().to(DEVICE)


class EmbTrainer:
    """Shared BPR training loop with early stopping on validation NDCG@10."""
    def __init__(self, ds, lr=1e-3, reg=1e-4, batch=4096, epochs=200, eval_every=5, patience=4,
                 steps_per_epoch=None, seed=42, val_users=None):
        self.ds, self.lr, self.reg, self.batch, self.epochs = ds, lr, reg, batch, epochs
        self.eval_every, self.patience, self.seed = eval_every, patience, seed
        self.steps = steps_per_epoch or max(1, ds.R_train.nnz // batch)
        self.val_users = val_users
        self.history = []

    def run(self, model, extra_loss=None, on_epoch=None):
        ds = self.ds
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, self.epochs + 1):
            model.train(); tot = 0
            if on_epoch: on_epoch(ep)
            for _ in range(self.steps):
                u, p, n = sample_bpr(ds, rng, self.batch)
                u, p, n = (torch.as_tensor(x, device=DEVICE) for x in (u, p, n))
                loss = model.bpr_loss(u, p, n, self.reg)
                if extra_loss is not None:
                    loss = loss + extra_loss(model, rng)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            if ep % self.eval_every == 0 or ep == self.epochs:
                model.eval()
                with torch.no_grad():
                    model.cache()
                    r = evaluate(ds, model.score, 'val', users=self.val_users, keep_lists=False)
                v = r['NDCG@10'].mean()
                self.history.append((ep, tot / self.steps, float(v)))
                log(f'  ep {ep} loss {tot / self.steps:.4f} val NDCG@10 {v:.4f}')
                if v > best:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            model.cache()
        self.best_val = float(best)
        return model


class BPRMF(nn.Module):
    def __init__(self, ds, dim=64, **kw):
        super().__init__()
        self.ds = ds
        self.U = nn.Embedding(ds.n_users, dim); self.I = nn.Embedding(ds.n_items, dim)
        nn.init.normal_(self.U.weight, std=0.1); nn.init.normal_(self.I.weight, std=0.1)

    def embeddings(self):
        return self.U.weight, self.I.weight

    def bpr_loss(self, u, p, n, reg):
        Ue, Ie = self.embeddings()
        eu, ep, en = Ue[u], Ie[p], Ie[n]
        loss = -F.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
        r = (self.U(u).pow(2).sum() + self.I(p).pow(2).sum() + self.I(n).pow(2).sum()) / len(u)
        return loss + reg * r

    def cache(self):
        Ue, Ie = self.embeddings()
        self._U, self._I = Ue.detach(), Ie.detach()

    def score(self, users, phase):
        with torch.no_grad():
            return (self._U[torch.as_tensor(users, device=DEVICE)] @ self._I.T).cpu().numpy()


class LightGCN(BPRMF):
    """LightGCN (He et al., 2020)."""
    def __init__(self, ds, dim=64, layers=3, **kw):
        super().__init__(ds, dim)
        self.A = norm_adj_bipartite(ds); self.layers = layers

    def embeddings(self):
        x = torch.cat([self.U.weight, self.I.weight]); acc = x
        for _ in range(self.layers):
            x = self.A @ x; acc = acc + x
        acc = acc / (self.layers + 1)
        return acc[:self.ds.n_users], acc[self.ds.n_users:]
