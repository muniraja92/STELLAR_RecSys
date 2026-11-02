"""Hierarchical Gated Fusion (HGF), revised.

Score-level, zero-initialised residual gating:

    y_ui = z^TPR_ui + sum_k g_k(c_u) * z^k_ui ,     g_k(c) = w_k^T phi(c),  w_k initialised to 0

where z^k are per-user standardised channel scores over the item catalogue (computed analytically
from embedding moments), phi(c_u) are user-context features computed from the *input* history
(log length, mean/std log-popularity of consumed items, tail share, log last time-gap) and the
gate parameters are fitted on the validation targets with full-softmax cross-entropy.  Because the
gate starts at zero and is early-stopped on held-out validation users, the fused model starts from,
and can only move away from, the temporal channel when doing so improves held-out validation
likelihood -- the structural fix for the earlier configuration in which fusion fell below its
strongest component.  The token-level level of the hierarchy is the KG projection inside TPR.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import log, DEVICE


def standardiser(Wt):
    """Wt: items x d.  Returns mean vector and covariance for analytic per-user score moments."""
    mu = Wt.mean(0)
    C = np.cov(Wt.T, bias=True)
    return mu, C


class Channel:
    def __init__(self, name, U, W, extra_bias=None):
        """U: users x d user representation for this phase, W: items x d item table."""
        self.name = name
        self.U = torch.as_tensor(U, dtype=torch.float32, device=DEVICE); self.W = torch.as_tensor(W, dtype=torch.float32, device=DEVICE)
        mu, C = standardiser(W.astype(np.float64))
        m = U.astype(np.float64) @ mu
        v = np.einsum('ij,jk,ik->i', U.astype(np.float64), C, U.astype(np.float64))
        self.m = torch.as_tensor(m, dtype=torch.float32, device=DEVICE); self.s = torch.as_tensor(np.sqrt(np.maximum(v, 1e-12)), dtype=torch.float32, device=DEVICE)

    def z(self, idx):
        return (self.U[idx] @ self.W.T - self.m[idx, None]) / self.s[idx, None]


def context_features(ds, phase):
    hist = ds.history(phase); hts = ds.history_ts(phase)
    lp = np.log1p(ds.item_pop)
    f = np.zeros((ds.n_users, 5), np.float32)
    for u in range(ds.n_users):
        h = hist[u]
        f[u, 0] = np.log1p(len(h))
        f[u, 1] = lp[h].mean(); f[u, 2] = lp[h].std()
        f[u, 3] = (~ds.head_items[h]).mean()
        t = hts[u]
        f[u, 4] = np.log1p(max(t[-1] - t[-2], 0) / 86400.0) if len(t) > 1 else 0.0
    return f


class HGF(nn.Module):
    def __init__(self, n_channels, n_feat, gated=True):
        super().__init__()
        self.gated = gated
        self.w = nn.Parameter(torch.zeros(n_channels, n_feat + 1))     # zero-initialised gates
        self.log_temp = nn.Parameter(torch.zeros(()))

    def gates(self, phi):
        x = torch.cat([torch.ones(len(phi), 1, device=phi.device), phi], 1)
        if not self.gated:
            x = x[:, :1]
            return x @ self.w[:, :1].T
        return x @ self.w.T                                           # B x K

    def forward(self, base, resid, phi):
        g = self.gates(phi)
        y = base + (g[:, :, None] * resid).sum(1)
        return y * torch.exp(self.log_temp)


def fused_scores(model, base_ch, res_chs, phi, users):
    idx = torch.as_tensor(users, device=DEVICE)
    base = base_ch.z(idx)
    resid = torch.stack([c.z(idx) for c in res_chs], 1) if res_chs else torch.zeros(len(users), 0, base.shape[1])
    with torch.no_grad():
        return model(base, resid, phi[idx]).cpu().numpy()


def fit_hgf(ds, base_val, res_val, phi_val, seed, gated=True, l2=1e-3, lr=0.05, epochs=100, batch=2048, patience=8):
    """Fit gate parameters on validation targets (80% users), early-stop on the remaining 20%."""
    rng = np.random.default_rng(seed)
    users = rng.permutation(ds.n_users); n_tr = int(0.8 * len(users))
    tr, ho = users[:n_tr], users[n_tr:]
    hist = ds.history('val'); tgt = torch.as_tensor(ds.targets('val'), device=DEVICE)
    model = HGF(len(res_val), phi_val.shape[1], gated).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    lens = np.array([len(h) for h in hist]); flat = np.concatenate(hist)
    offs = np.concatenate([[0], np.cumsum(lens)])

    def batch_loss(ub):
        idx = torch.as_tensor(ub, device=DEVICE)
        base = base_val.z(idx); resid = torch.stack([c.z(idx) for c in res_val], 1)
        y = model(base, resid, phi_val[idx])
        rows = np.repeat(np.arange(len(ub)), lens[ub])                 # exclude input history
        cols = np.concatenate([flat[offs[u]:offs[u + 1]] for u in ub])
        y = y.index_put((torch.as_tensor(rows, device=DEVICE), torch.as_tensor(cols, device=DEVICE)),
                        torch.tensor(-1e9, device=DEVICE))
        return F.cross_entropy(y, tgt[idx])

    model.eval()
    with torch.no_grad():   # held-out loss of the zero-initialised gate (= temporal channel alone)
        hl0 = float(np.mean([batch_loss(ho[b0:b0 + batch]).item() for b0 in range(0, len(ho), batch)]))
    best, best_state, bad, hist_log = hl0, {k: v.clone() for k, v in model.state_dict().items()}, 0, [hl0]
    for ep in range(epochs):
        model.train(); rng.shuffle(tr)
        for b0 in range(0, len(tr), batch):
            loss = batch_loss(tr[b0:b0 + batch]) + l2 * model.w.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hl = float(np.mean([batch_loss(ho[b0:b0 + batch]).item() for b0 in range(0, len(ho), batch)]))
        hist_log.append(hl)
        if hl < best - 1e-5:
            best, bad = hl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    log(f'  HGF fitted: held-out val CE {best:.4f} after {len(hist_log)} epochs; w={model.w.detach().cpu().numpy().round(3).tolist()}')
    return model, hist_log
