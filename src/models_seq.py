"""Sequential recommenders: SASRec (CE), BERT4Rec, BSARec and the STELLAR-KG TPR module.

All causal models are trained with full-softmax cross-entropy at every non-padded position of a
random window of the user's *training* sequence (validation and test items are never used as
inputs or targets during training)."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import evaluate, log, pad_seqs, DEVICE


def rope_cache(L, dh, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2).float() / dh))
    t = torch.arange(L).float()
    f = torch.outer(t, inv)                       # L x dh/2
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    # x: B,H,L,dh ; rotate consecutive pairs (x1,x2)->(x1 cos - x2 sin, x1 sin + x2 cos)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1)
    return out.flatten(-2)


class MHA(nn.Module):
    def __init__(self, d, h, drop, rope=False, max_len=200):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop); self.rope = rope
        if rope:
            c, s = rope_cache(max_len, self.dh)
            self.register_buffer('cos', c, persistent=False); self.register_buffer('sin', s, persistent=False)

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.rope:
            q = apply_rope(q, self.cos[:L], self.sin[:L]); k = apply_rope(k, self.cos[:L], self.sin[:L])
        a = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        a = a.masked_fill(~mask, -1e9)
        a = self.drop(torch.softmax(a, -1))
        return self.o((a @ v).transpose(1, 2).reshape(B, L, D))


class FFN(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN transformer block; optional BSARec frequency branch."""
    def __init__(self, d, h, drop, rope=False, max_len=200, bsa=None):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHA(d, h, drop, rope, max_len); self.ffn = FFN(d, drop); self.drop = nn.Dropout(drop)
        self.bsa = bsa
        if bsa is not None:
            self.c = bsa['c'] // 2 + 1; self.alpha = bsa['alpha']
            self.sqrt_beta = nn.Parameter(torch.randn(1, 1, d))
            self.ln_f = nn.LayerNorm(d)

    def freq(self, x):
        L = x.shape[1]
        X = torch.fft.rfft(x, dim=1, norm='ortho')
        mask = torch.zeros(X.shape[1], device=x.device); mask[:self.c] = 1
        low = torch.fft.irfft(X * mask[None, :, None], n=L, dim=1, norm='ortho')
        high = x - low
        return self.ln_f(x + self.drop(low + self.sqrt_beta ** 2 * high))

    def forward(self, x, mask):
        a = x + self.drop(self.att(self.ln1(x), mask))
        if self.bsa is not None:
            a = self.alpha * self.freq(x) + (1 - self.alpha) * a
        return a + self.drop(self.ffn(self.ln2(a)))


class SeqRec(nn.Module):
    """Generic causal/bidirectional transformer recommender.

    kind: 'sasrec' (learned positions, causal), 'bert4rec' (bidirectional, masked items),
          'bsarec' (SASRec + frequency filter), 'tpr' (RoPE + temporal decay + optional KG item
          tokens, optional PCL and CIR regularisers)."""
    def __init__(self, ds, kind='sasrec', dim=64, layers=2, heads=2, drop=0.2, max_len=50,
                 bsa=None, kg_item=None, time_decay=False, rope=False, learned_pos=True, mask_prob=0.2):
        super().__init__()
        self.ds, self.kind, self.max_len, self.dim = ds, kind, max_len, dim
        n = ds.n_items
        self.extra_tok = 1 if kind == 'bert4rec' else 0     # mask token id = n+1
        self.item = nn.Embedding(n + 1 + self.extra_tok, dim, padding_idx=0)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad(): self.item.weight[0].zero_()
        self.learned_pos = learned_pos
        if learned_pos:
            self.pos = nn.Embedding(max_len, dim); nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads, drop, rope, max_len, bsa) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim); self.drop = nn.Dropout(drop)
        self.mask_prob = mask_prob
        self.time_decay = time_decay
        if time_decay:
            self.gamma = nn.Parameter(torch.tensor(-1.0))          # softplus(-1) ~ 0.31
            self.time_enc = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        if kg_item is not None:   # frozen KG-SGW item representations (n x d_kg)
            self.register_buffer('kg_item', torch.as_tensor(kg_item, dtype=torch.float32))
            self.kg_proj = nn.Linear(self.kg_item.shape[1], dim, bias=False)
            nn.init.zeros_(self.kg_proj.weight)
        else:
            self.kg_item = None

    def item_table(self):
        W = self.item.weight[1:self.ds.n_items + 1]
        if self.kg_item is not None:
            W = W + self.kg_proj(self.kg_item)
        return W

    def embed(self, X, T=None):
        W = self.item_table()
        full = torch.cat([self.item.weight[:1], W, self.item.weight[self.ds.n_items + 1:]], 0)
        e = full[X]
        if self.time_decay and T is not None:
            g = F.softplus(self.gamma)
            w = torch.exp(-g * T).unsqueeze(-1)
            e = w * e + (1 - w) * self.time_enc(T.unsqueeze(-1))
        if self.learned_pos:
            e = e + self.pos.weight[None, -X.shape[1]:]
        return self.drop(e)

    def encode(self, X, T=None):
        B, L = X.shape
        pad = X > 0
        if self.kind == 'bert4rec':
            mask = pad[:, None, None, :].expand(B, 1, L, L)
        else:
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=X.device))
            mask = causal[None, None] & pad[:, None, None, :]
            mask = mask | torch.eye(L, dtype=torch.bool, device=X.device)[None, None]   # padded rows attend to self
        h = self.embed(X, T)
        for b in self.blocks:
            h = b(h, mask)
        return self.ln(h)

    # ---------- training batches ----------
    def make_batch(self, users, rng, use_time):
        seqs, tss = [], []
        for u in users:
            s = self.ds.train[u]; t = self.ds.ts[u][:len(s)]
            L = self.max_len + 1
            if len(s) > L:   # random window (covers the whole training history across epochs)
                end = rng.integers(L, len(s) + 1)
                s, t = s[end - L:end], t[end - L:end]
            seqs.append(s); tss.append(t)
        X, T = pad_seqs(seqs, self.max_len + 1, tss if use_time else None)
        return torch.as_tensor(X, device=DEVICE), torch.as_tensor(T, device=DEVICE)

    def loss(self, X, T, cir=0.0, pcl=0.0, centroids=None, pcl_tau=0.5):
        W = self.item_table()
        if self.kind == 'bert4rec':
            x = X[:, 1:].clone(); tgt = x.clone()
            valid = x > 0
            m = (torch.rand(x.shape, device=x.device) < self.mask_prob) & valid
            # ensure the last position is sometimes masked (matches inference)
            m[:, -1] |= (torch.rand(x.shape[0], device=x.device) < self.mask_prob) & valid[:, -1]
            x[m] = self.ds.n_items + 1
            h = self.encode(x, None)
            hs, ys = h[m], tgt[m] - 1
        else:
            inp, tgt = X[:, :-1], X[:, 1:]
            Tin = T[:, :-1] if T is not None else None
            h = self.encode(inp, Tin)
            valid = tgt > 0
            hs, ys = h[valid], tgt[valid] - 1
        logits = hs @ W.T
        ce = F.cross_entropy(logits, ys, reduction='none')
        loss = ce.mean()
        if cir > 0:      # V-REx style risk-variance penalty across popularity environments
            env = torch.as_tensor(self.ds.pop_env, device=ys.device)[ys]
            risks = [ce[env == e].mean() for e in range(3) if (env == e).any()]
            if len(risks) > 1:
                loss = loss + cir * torch.stack(risks).var(unbiased=False)
        if pcl > 0 and centroids is not None and self.kind != 'bert4rec':
            last = h[:, -1]
            z = F.normalize(last, dim=-1); C = F.normalize(centroids, dim=-1)
            with torch.no_grad():
                a = (z @ C.T).argmax(1)
            loss = loss + pcl * F.cross_entropy(z @ C.T / pcl_tau, a)
        return loss

    # ---------- inference ----------
    @torch.no_grad()
    def user_repr(self, users, phase, use_time):
        hist = self.ds.history(phase); hts = self.ds.history_ts(phase)
        seqs = [hist[u] for u in users]; tss = [hts[u] for u in users] if use_time else None
        if self.kind == 'bert4rec':
            X, _ = pad_seqs([s[-(self.max_len - 1):] for s in seqs], self.max_len - 1)
            X = np.concatenate([X, np.full((len(users), 1), self.ds.n_items + 1)], 1)
            h = self.encode(torch.as_tensor(X, device=DEVICE), None)
            return h[:, -1]
        X, T = pad_seqs(seqs, self.max_len, tss)
        h = self.encode(torch.as_tensor(X, device=DEVICE), torch.as_tensor(T, device=DEVICE) if use_time else None)
        return h[:, -1]

    def score_fn(self, use_time):
        with torch.no_grad():
            W = self.item_table().detach()

        def f(users, phase):
            with torch.no_grad():
                return (self.user_repr(users, phase, use_time) @ W.detach().T).cpu().numpy()
        return f


def kmeans_torch(Z, k, iters=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = Z[torch.randperm(len(Z), generator=g)[:k].to(Z.device)].clone()
    for _ in range(iters):
        a = torch.cdist(Z, C).argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = Z[m].mean(0)
    return C


def train_seq(model, ds, seed, lr=1e-3, batch=128, max_epochs=200, patience=5, eval_every=1,
              use_time=False, cir=0.0, pcl=0.0, pcl_k=32, val_users=None, wd=0.0, time_budget=None):
    import time
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    users = np.arange(ds.n_users)
    users = users[np.array([len(s) >= 2 for s in ds.train])]
    best, best_state, bad, hist = -1, None, 0, []
    t0 = time.time(); centroids = None
    for ep in range(1, max_epochs + 1):
        if pcl > 0:   # intent prototypes from current sequence representations (no gradient)
            model.eval()
            with torch.no_grad():
                Z = torch.cat([model.user_repr(users[i:i + 1024], 'val', use_time) for i in range(0, len(users), 1024)])
            centroids = kmeans_torch(F.normalize(Z, dim=-1), pcl_k, seed=seed + ep)
        model.train(); rng.shuffle(users); tot, nb = 0.0, 0
        for b0 in range(0, len(users), batch):
            X, T = model.make_batch(users[b0:b0 + batch], rng, use_time)
            loss = model.loss(X, T if use_time else None, cir=cir, pcl=pcl, centroids=centroids)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        if ep % eval_every == 0:
            model.eval()
            r = evaluate(ds, model.score_fn(use_time), 'val', users=val_users, keep_lists=False, batch=512)
            v = float(r['NDCG@10'].mean()); hist.append((ep, tot / nb, v))
            log(f'  ep {ep} loss {tot / nb:.4f} val NDCG@10 {v:.4f} ({time.time() - t0:.0f}s)')
            if v > best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
            if time_budget and time.time() - t0 > time_budget:
                log('  time budget reached'); break
    model.load_state_dict(best_state); model.eval()
    return model, best, hist
