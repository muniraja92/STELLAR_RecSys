"""Knowledge-graph models: STELLAR-KG spectral graph wavelets (KG-SGW), KGAT and KGIN.

KG-SGW
------
Operates on the collaborative knowledge graph (CKG) whose nodes are users, items and KG entities.
Edges: user--item (training interactions, weight 1) and item--entity for every KG triple, weighted by
a learnable, non-negative relation weight w_r = softplus(theta_r).  With W the weighted adjacency,
D its degree matrix and A_hat = D^{-1/2} W D^{-1/2}, the normalised Laplacian is L = I - A_hat
(spectrum in [0, 2]).  A bank of spectral kernels is applied: by default (kernels='heatbank', the
configuration of the paper) the multi-scale heat kernels exp(-s l), s in {1, 2, 4}; the ablation
kernels='bank' uses the low-pass scaling kernel h(l) = exp(-4 l) with the band-pass wavelets
g(s l), g(x) = x exp(1 - x) (Hammond et al., 2011), and kernels='lgcn' the fixed three-layer LightGCN
response (diagnostic).  The kernels are applied to learnable node embeddings X through an order-K Chebyshev expansion

    Psi_j X  ~=  sum_{k=0}^{K} c_{j,k} T_k(L - I) X ,

which needs only K sparse products shared by all kernels (no eigendecomposition).  Node
representations are a learned convex combination of the kernel responses.
"""
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F
from models_cf import BPRMF, to_torch_sparse
from common import DEVICE


def cheb_coeffs(fn, K, lmax=2.0, n=2000):
    """Chebyshev coefficients of fn on [0, lmax] (Hammond et al., 2011, Eq. 37)."""
    th = (np.arange(n) + 0.5) * np.pi / n
    lam = (np.cos(th) + 1) * lmax / 2
    y = fn(lam)
    c = np.array([2.0 / n * np.sum(y * np.cos(k * th)) for k in range(K + 1)])
    c[0] /= 2
    return c


def cheb_eval(c, lam, lmax=2.0):
    x = 2 * lam / lmax - 1
    T0, T1 = np.ones_like(x), x
    out = c[0] * T0 + (c[1] * T1 if len(c) > 1 else 0)
    for k in range(2, len(c)):
        T0, T1 = T1, 2 * x * T1 - T0
        out = out + c[k] * T1
    return out


def kernel_bank(scales=(1.0, 2.0, 4.0), s0=4.0):
    ks = [('scaling', lambda l, s=s0: np.exp(-s * l))]
    for s in scales:
        ks.append((f'wavelet_s{s:g}', lambda l, s=s: (s * l) * np.exp(1 - s * l)))
    return ks


def build_ckg_parts(ds, use_kg=True):
    nu, ni, ne = ds.n_users, ds.n_items, ds.n_entities
    N = nu + ni + (ne if use_kg else 0)
    R = ds.R_train.tocoo()
    r0 = np.concatenate([R.row, R.col + nu]); c0 = np.concatenate([R.col + nu, R.row])
    A_ui = sp.csr_matrix((np.ones(len(r0), np.float32), (r0, c0)), shape=(N, N))
    parts = []
    if use_kg and len(ds.kg):
        for r in range(ds.n_relations):
            t = ds.kg[ds.kg[:, 1] == r]
            h = t[:, 0] + nu; e = t[:, 2] + nu + ni
            rr = np.concatenate([h, e]); cc = np.concatenate([e, h])
            parts.append(sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(N, N)))
    return N, A_ui, parts


class WeightedAdj(torch.autograd.Function):
    """y -> A(w) y for the symmetric relation-weighted CKG adjacency, with gradients for w and y."""
    @staticmethod
    def forward(ctx, w, y, m):
        wf = torch.cat([torch.ones(1, device=y.device, dtype=y.dtype), w.detach()])
        v = torch.zeros(len(m.ccol), device=y.device, dtype=y.dtype).index_add_(0, m.e_inv, wf[m.e_rel])
        A = torch.sparse_csr_tensor(m.crow, m.ccol, v, (m.N, m.N))
        ctx.A, ctx.m, ctx.nw = A, m, len(w)
        ctx.save_for_backward(y)
        return A @ y

    @staticmethod
    def backward(ctx, g):
        y, = ctx.saved_tensors; m = ctx.m
        gy = ctx.A @ g                                   # A is symmetric
        gw = None
        if ctx.nw and ctx.needs_input_grad[0]:
            gw = torch.zeros(ctx.nw, device=g.device, dtype=g.dtype).index_add_(
                0, m.kg_rel, (g[m.kg_row] * y[m.kg_col]).sum(1))
        return gw, gy, None


class KGSGW(BPRMF):
    def __init__(self, ds, dim=64, K=8, scales=(1.0, 2.0, 4.0), s0=4.0, use_kg=True,
                 learn_rel=True, kernels='heatbank', learn_mix=True, **kw):
        nn.Module.__init__(self)
        self.ds, self.dim, self.K, self.use_kg = ds, dim, K, use_kg
        N, A_ui, parts = build_ckg_parts(ds, use_kg)
        self.N = N
        # All CKG edges in one CSR structure: an edge of relation r carries weight w_r, a user-item edge weight 1.
        # The relation-weighted product is one sparse product per Chebyshev term (identical to summing
        # per-relation products); gradients for w are accumulated over knowledge edges only (WeightedAdj).
        mats = [A_ui.tocoo()] + [p.tocoo() for p in parts]
        rows = np.concatenate([m.row for m in mats]).astype(np.int64); cols = np.concatenate([m.col for m in mats]).astype(np.int64)
        rel = np.concatenate([np.full(m.nnz, r, np.int64) for r, m in enumerate(mats)])
        uniq, inv = np.unique(rows * N + cols, return_inverse=True)
        urow = uniq // N
        crow = np.zeros(N + 1, np.int64); np.add.at(crow, urow + 1, 1); crow = np.cumsum(crow)
        kg = rel > 0
        for name, arr in (('crow', crow), ('ccol', uniq % N), ('e_inv', inv), ('e_rel', rel),
                          ('kg_row', rows[kg]), ('kg_col', cols[kg]), ('kg_rel', rel[kg] - 1)):
            self.register_buffer(name, torch.tensor(arr, dtype=torch.long))
        self.register_buffer('d_ui', torch.tensor(np.asarray(A_ui.sum(1)).ravel(), dtype=torch.float32))
        self.register_buffer('d_r', torch.stack([torch.tensor(np.asarray(p.sum(1)).ravel(), dtype=torch.float32) for p in parts]) if parts else None)
        self.theta = nn.Parameter(torch.full((len(parts),), 0.5413)) if parts else None   # softplus(.5413)=1
        self.learn_rel = learn_rel
        if not learn_rel and parts:
            self.theta.requires_grad_(False)
        self.E = nn.Embedding(N, dim); nn.init.normal_(self.E.weight, std=0.1)
        if kernels == 'bank':
            bank = kernel_bank(scales, s0)
        elif kernels == 'bank_id':     # bank plus the all-pass (identity) kernel, as the layer-0 term of LightGCN
            bank = [('identity', lambda l: np.ones_like(l))] + kernel_bank(scales, s0)
        elif kernels == 'lgcn':        # diagnostic: the fixed LightGCN response (1/4) sum_{m<=3} (1 - lambda)^m
            bank = [('lgcn', lambda l: sum((1 - l) ** m for m in range(4)) / 4)]
        elif kernels == 'heatbank':    # multi-scale heat (low-pass) kernels exp(-s lambda)
            bank = [(f'heat_s{s:g}', lambda l, s=s: np.exp(-s * l)) for s in (1.0, 2.0, 4.0)]
        elif kernels == 'heat':        # single low-pass heat kernel (no band-pass wavelets)
            bank = [('scaling', lambda l, s=s0: np.exp(-s * l))]
        else:
            raise ValueError(kernels)
        self.kernel_names = [n for n, _ in bank]
        C = np.stack([cheb_coeffs(f, K) for _, f in bank])               # J x (K+1)
        lam = np.linspace(0, 2, 401)
        self.cheb_max_err = float(max(np.abs(cheb_eval(C[j], lam) - f(lam)).max() for j, (_, f) in enumerate(bank)))
        self.register_buffer('C', torch.tensor(C, dtype=torch.float32))
        self.beta_u = nn.Parameter(torch.zeros(len(bank))); self.beta_i = nn.Parameter(torch.zeros(len(bank)))
        if not learn_mix:              # fixed uniform mixture of the kernel bank
            self.beta_u.requires_grad_(False); self.beta_i.requires_grad_(False)
        self.U = self.I = None

    def rel_weights(self):
        return F.softplus(self.theta) if self.theta is not None else None

    def propagate(self, x):
        """returns A_hat @ x with relation-weighted, symmetrically normalised adjacency."""
        w = self.rel_weights()
        d = self.d_ui.clone()
        if w is not None:
            d = d + (w[:, None] * self.d_r).sum(0)
        dis = d.clamp_min(1e-12).pow(-0.5)[:, None]
        if w is None:
            w = torch.zeros(0, device=x.device)
        return dis * WeightedAdj.apply(w, dis * x, self)

    def embeddings(self):
        X = self.E.weight
        # T_k(L - I) with L - I = -A_hat
        T_prev, T_cur = X, -self.propagate(X)
        acc = self.C[:, 0, None, None] * T_prev + self.C[:, 1, None, None] * T_cur      # J x N x d
        for k in range(2, self.K + 1):
            T_prev, T_cur = T_cur, -2 * self.propagate(T_cur) - T_prev
            acc = acc + self.C[:, k, None, None] * T_cur
        nu, ni = self.ds.n_users, self.ds.n_items
        au, ai = torch.softmax(self.beta_u, 0), torch.softmax(self.beta_i, 0)
        Zu = (au[:, None, None] * acc[:, :nu]).sum(0)
        Zi = (ai[:, None, None] * acc[:, nu:nu + ni]).sum(0)
        return Zu, Zi

    def bpr_loss(self, u, p, n, reg):
        Zu, Zi = self.embeddings()
        eu, ep, en = Zu[u], Zi[p], Zi[n]
        loss = -F.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
        X = self.E.weight
        r = (X[u].pow(2).sum() + X[p + self.ds.n_users].pow(2).sum() + X[n + self.ds.n_users].pow(2).sum()) / len(u)
        return loss + reg * r

    def cache(self):
        Zu, Zi = self.embeddings()
        self._U, self._I = Zu.detach(), Zi.detach()

    def describe(self):
        d = {'kernel_weights_user': dict(zip(self.kernel_names, torch.softmax(self.beta_u, 0).tolist())),
             'kernel_weights_item': dict(zip(self.kernel_names, torch.softmax(self.beta_i, 0).tolist())),
             'cheb_max_abs_err': self.cheb_max_err}
        if self.theta is not None:
            d['relation_weights'] = dict(zip(self.ds.relations, self.rel_weights().tolist()))
        return d


# --------------------------------------------------------------------------------------------
class KGAT(BPRMF):
    """KGAT (Wang et al., KDD 2019): TransR-based attentive propagation over the CKG with a
    bi-interaction aggregator, layer sizes [64, 32, 16], alternating CF (BPR) and KG (TransR) losses."""
    def __init__(self, ds, dim=64, layers=(64, 32, 16), mess_drop=0.1, kg_batch=4096, **kw):
        nn.Module.__init__(self)
        self.ds = ds
        nu, ni, ne = ds.n_users, ds.n_items, ds.n_entities
        self.nu, self.ni = nu, ni
        self.N = nu + ni + ne
        R = ds.R_train.tocoo()
        # relations: 0 interact, 1..R kg, then inverses
        heads = [R.row, R.col + nu]; tails = [R.col + nu, R.row]; rels = [np.zeros(R.nnz), np.full(R.nnz, 1 + ds.n_relations)]
        nr = ds.n_relations
        if len(ds.kg):
            h = ds.kg[:, 0] + nu; t = ds.kg[:, 2] + nu + ni; r = ds.kg[:, 1] + 1
            heads += [h, t]; tails += [t, h]; rels += [r, r + 1 + nr]
        self.h = torch.as_tensor(np.concatenate(heads), dtype=torch.long, device=DEVICE)
        self.t = torch.as_tensor(np.concatenate(tails), dtype=torch.long, device=DEVICE)
        self.r = torch.as_tensor(np.concatenate(rels), dtype=torch.long, device=DEVICE)
        self.n_rel = 2 * (nr + 1)
        self.ent = nn.Embedding(self.N, dim); nn.init.xavier_uniform_(self.ent.weight)
        self.rel = nn.Embedding(self.n_rel, dim); nn.init.xavier_uniform_(self.rel.weight)
        self.Wr = nn.Parameter(torch.empty(self.n_rel, dim, dim)); nn.init.xavier_uniform_(self.Wr)
        dims = [dim] + list(layers)
        self.W1 = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(layers))])
        self.W2 = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(layers))])
        self.drop = nn.Dropout(mess_drop)
        self.kg_batch = kg_batch
        self.att = None   # computed by update_attention() after the model is moved to its device

    @torch.no_grad()
    def update_attention(self):
        vals = torch.empty(len(self.h), device=self.h.device)
        E = self.ent.weight
        for r in range(self.n_rel):
            idx = (self.r == r).nonzero().squeeze(1)
            if len(idx) == 0: continue
            W = self.Wr[r]
            for s in range(0, len(idx), 500000):
                ii = idx[s:s + 500000]
                eh = E[self.h[ii]] @ W; et = E[self.t[ii]] @ W
                vals[ii] = (et * torch.tanh(eh + self.rel.weight[r])).sum(1)
        # softmax over the neighbours of each head
        mx = torch.full((self.N,), -1e30, device=vals.device).scatter_reduce(0, self.h, vals, 'amax')
        ex = torch.exp(vals - mx[self.h])
        den = torch.zeros(self.N, device=vals.device).index_add_(0, self.h, ex)
        a = ex / den[self.h]
        A = torch.sparse_coo_tensor(torch.stack([self.h, self.t]), a, (self.N, self.N)).coalesce().to_sparse_csr()
        self.att = A

    def embeddings(self):
        x = self.ent.weight; outs = [F.normalize(x, dim=1)]
        for W1, W2 in zip(self.W1, self.W2):
            n = self.att @ x
            x = F.leaky_relu(W1(x + n), 0.2) + F.leaky_relu(W2(x * n), 0.2)
            x = self.drop(x)
            outs.append(F.normalize(x, dim=1))
        z = torch.cat(outs, 1)
        return z[:self.nu], z[self.nu:self.nu + self.ni]

    def bpr_loss(self, u, p, n, reg):
        Zu, Zi = self.embeddings()
        eu, ep, en = Zu[u], Zi[p], Zi[n]
        loss = -F.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
        return loss + reg * (eu.pow(2).sum() + ep.pow(2).sum() + en.pow(2).sum()) / len(u)

    def kg_loss(self, rng, reg=1e-5):
        idx = torch.as_tensor(rng.integers(0, len(self.h), self.kg_batch), device=self.h.device)
        h, r, t = self.h[idx], self.r[idx], self.t[idx]
        tn = torch.as_tensor(rng.integers(0, self.N, self.kg_batch), device=self.h.device)
        W = self.Wr[r]; er = self.rel.weight[r]
        ph = torch.bmm(self.ent.weight[h].unsqueeze(1), W).squeeze(1)
        pt = torch.bmm(self.ent.weight[t].unsqueeze(1), W).squeeze(1)
        pn = torch.bmm(self.ent.weight[tn].unsqueeze(1), W).squeeze(1)
        pos = ((ph + er - pt) ** 2).sum(1); neg = ((ph + er - pn) ** 2).sum(1)
        l = -F.logsigmoid(neg - pos).mean()
        return l + reg * (ph.pow(2).sum() + er.pow(2).sum() + pt.pow(2).sum() + pn.pow(2).sum()) / self.kg_batch


class KGIN(BPRMF):
    """KGIN (Wang et al., WWW 2021): user intents as attentive combinations of KG relations,
    relational-path-aware aggregation, distance-correlation independence among intents."""
    def __init__(self, ds, dim=64, hops=3, n_intents=4, ind_reg=1e-4, mess_drop=0.1, **kw):
        nn.Module.__init__(self)
        self.ds = ds; nu, ni, ne = ds.n_users, ds.n_items, ds.n_entities
        self.nu, self.ni, self.ne = nu, ni, ne
        self.n_ent = ni + ne                                   # items are the first ni entities
        nr = ds.n_relations
        kg = ds.kg
        h = np.concatenate([kg[:, 0], kg[:, 2] + ni]); t = np.concatenate([kg[:, 2] + ni, kg[:, 0]])
        r = np.concatenate([kg[:, 1], kg[:, 1] + nr])
        self.kh, self.kt, self.kr = (torch.as_tensor(x, dtype=torch.long, device=DEVICE) for x in (h, t, r))
        self.n_rel = 2 * nr
        deg = np.bincount(h, minlength=self.n_ent).astype(np.float32)
        self.kdeg = torch.as_tensor(np.maximum(deg, 1), device=DEVICE)
        Rn = ds.R_train.multiply(1.0 / np.maximum(np.asarray(ds.R_train.sum(1)), 1)).tocsr()
        self.Rn = to_torch_sparse(Rn)
        self.user = nn.Embedding(nu, dim); self.entity = nn.Embedding(self.n_ent, dim)
        nn.init.xavier_uniform_(self.user.weight); nn.init.xavier_uniform_(self.entity.weight)
        self.rel = nn.Parameter(torch.empty(self.n_rel, dim)); nn.init.xavier_uniform_(self.rel)
        self.intent_w = nn.Parameter(torch.empty(n_intents, self.n_rel)); nn.init.xavier_uniform_(self.intent_w)
        self.hops, self.ind_reg = hops, ind_reg
        self.drop = nn.Dropout(mess_drop)

    def embeddings(self):
        e, u = self.entity.weight, self.user.weight
        intents = torch.softmax(self.intent_w, 1) @ self.rel             # P x d
        e_res, u_res = e, u
        for _ in range(self.hops):
            msg = self.rel[self.kr] * e[self.kt]
            e_new = torch.zeros_like(e).index_add_(0, self.kh, msg) / self.kdeg[:, None]
            items = e[:self.ni]
            u_agg = self.Rn @ items                                       # mean of item embeddings
            beta = torch.softmax(u @ intents.T, 1)                         # nu x P
            u_new = (beta @ intents) * u_agg + u_agg
            e, u = self.drop(F.normalize(e_new, dim=1)), self.drop(F.normalize(u_new, dim=1))
            e_res, u_res = e_res + e, u_res + u
        return u_res, e_res[:self.ni]

    def independence(self):
        x = self.intent_w          # distance correlation between intent attention rows (official KGIN)
        P = x.shape[0]
        def dcor(a, b):
            a = a[:, None]; b = b[:, None]
            A = (a - a.T).abs(); B = (b - b.T).abs()
            A = A - A.mean(0) - A.mean(1)[:, None] + A.mean(); B = B - B.mean(0) - B.mean(1)[:, None] + B.mean()
            return (A * B).mean().clamp_min(0).sqrt() / ((A * A).mean().sqrt() * (B * B).mean().sqrt()).clamp_min(1e-9).sqrt()
        loss = 0
        for i in range(P):
            for j in range(i + 1, P):
                loss = loss + dcor(x[i], x[j])
        return loss / (P * (P - 1) / 2)

    def bpr_loss(self, u, p, n, reg):
        Zu, Zi = self.embeddings()
        eu, ep, en = Zu[u], Zi[p], Zi[n]
        loss = -F.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
        l2 = (self.user(u).pow(2).sum() + self.entity(p).pow(2).sum() + self.entity(n).pow(2).sum()) / len(u)
        return loss + reg * l2 + self.ind_reg * self.independence()
