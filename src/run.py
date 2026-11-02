"""Experiment runner.  Usage: python run.py --dataset ml1m --model tpr --seed 42 [--cfg JSON]"""
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Dataset, evaluate, summarize, save_result, done, set_seed, log, RES, DEVICE
from models_cf import Pop, ItemKNN, EASE, BPRMF, LightGCN, EmbTrainer
from models_kg import KGSGW, KGAT, KGIN
from models_seq import SeqRec, train_seq
import fusion as FU

torch.set_num_threads(int(os.environ.get("NT", "2")))

SEQ_DEFAULTS = {
    'ml1m':   dict(max_len=100, drop=0.2, lr=1e-3, batch=128),
    'beauty': dict(max_len=50, drop=0.5, lr=1e-3, batch=256),
    'toys':   dict(max_len=50, drop=0.5, lr=1e-3, batch=256),
}
GRAPH_DEFAULTS = dict(lr=1e-3, reg=1e-4, batch=4096, epochs=400, eval_every=5, patience=4)

# variants of the temporal module (TPR) used in the ablation study
TPR_VARIANTS = {
    'tpr':         dict(rope=True, learned_pos=False, time_decay=True, kg=True),
    'tpr_nokg':    dict(rope=True, learned_pos=False, time_decay=True, kg=False),
    'tpr_nopcl':   dict(rope=True, learned_pos=False, time_decay=True, kg=True, pcl=0.0),
    'tpr_nocir':   dict(rope=True, learned_pos=False, time_decay=True, kg=True, cir=0.0),
    'tpr_norope':  dict(rope=False, learned_pos=True, time_decay=True, kg=True),
    'tpr_nodecay': dict(rope=True, learned_pos=False, time_decay=False, kg=True),
    'tpr_notime':  dict(rope=False, learned_pos=True, time_decay=False, kg=True),
    'tpr_core':    dict(rope=True, learned_pos=False, time_decay=True, kg=False, pcl=0.0, cir=0.0),
}
SGW_VARIANTS = {
    'sgw':        dict(use_kg=True, kernels='heatbank', learn_rel=True),    # heat-kernel wavelets, scales 1, 2, 4
    'sgw_nokg':   dict(use_kg=False, kernels='heatbank', learn_rel=True),   # bipartite graph only
    'sgw_band':   dict(use_kg=True, kernels='bank', learn_rel=True),        # scaling + band-pass (Hammond) bank
    'sgw_fixrel': dict(use_kg=True, kernels='heatbank', learn_rel=False),   # relation weights fixed at 1
}


def emb_path(dsn, model, seed):
    return os.path.join(RES, dsn, model, f'emb_seed{seed}.npz')


def run_graph(ds, model, seed, cfg, tag=None):
    tag = tag or model
    g = dict(GRAPH_DEFAULTS); g.update({k: v for k, v in cfg.items() if k in g})
    mk = {k: v for k, v in cfg.items() if k not in g}
    if model == 'bpr': m = BPRMF(ds, **mk)
    elif model == 'lightgcn': m = LightGCN(ds, **mk)
    elif model == 'kgat': m = KGAT(ds, **mk)
    elif model == 'kgin': m = KGIN(ds, **mk)
    elif model in SGW_VARIANTS:
        kw = dict(SGW_VARIANTS[model]); kw.update(mk); m = KGSGW(ds, **kw)
    m = m.to(DEVICE)
    if model == 'kgat':
        m.update_attention()
    tr = EmbTrainer(ds, seed=seed, **g)
    extra, on_epoch = None, None
    if model == 'kgat':
        extra = lambda mm, rng: mm.kg_loss(rng)
        on_epoch = lambda ep: m.update_attention() if ep > 1 else None
    m = tr.run(m, extra_loss=extra, on_epoch=on_epoch)
    info = {'val_NDCG@10': tr.best_val, 'curve': tr.history}
    if hasattr(m, 'describe'): info['describe'] = m.describe()
    if model in SGW_VARIANTS or model in ('lightgcn',):
        os.makedirs(os.path.dirname(emb_path(ds.name, tag, seed)), exist_ok=True)
        np.savez_compressed(emb_path(ds.name, tag, seed), U=m._U.cpu().numpy(), I=m._I.cpu().numpy())
    return m.score, info


def run_seq(ds, model, seed, cfg, tag=None):
    tag = tag or model
    sd = dict(SEQ_DEFAULTS[ds.name]); sd.update(cfg)
    max_len, drop, lr, batch = sd['max_len'], sd['drop'], sd['lr'], sd['batch']
    layers, heads = sd.get('layers', 2), sd.get('heads', 2)
    tb = sd.get('time_budget', 3600 * 3)
    use_time, cir, pcl = False, 0.0, 0.0
    if model == 'sasrec':
        m = SeqRec(ds, 'sasrec', layers=layers, heads=heads, drop=drop, max_len=max_len)
    elif model == 'bert4rec':
        m = SeqRec(ds, 'bert4rec', layers=layers, heads=heads, drop=drop, max_len=max_len, mask_prob=sd.get('mask_prob', 0.2))
    elif model == 'bsarec':
        m = SeqRec(ds, 'bsarec', layers=layers, heads=heads, drop=drop, max_len=max_len,
                   bsa=dict(c=sd.get('c', 5), alpha=sd.get('alpha', 0.7)))
    else:
        v = dict(TPR_VARIANTS[model])
        kg = None
        if v['kg']:
            src = sd.get('kg_source', 'sgw')
            kg = np.load(emb_path(ds.name, src, seed))['I']
            kg = kg / (np.linalg.norm(kg, axis=1, keepdims=True).mean() + 1e-9)
        m = SeqRec(ds, 'tpr', layers=layers, heads=heads, drop=drop, max_len=max_len,
                   kg_item=kg, time_decay=v['time_decay'], rope=v['rope'], learned_pos=v['learned_pos'])
        use_time = v['time_decay']
        cir = v.get('cir', sd.get('cir', 0.0)); pcl = v.get('pcl', sd.get('pcl', 0.0))
    m = m.to(DEVICE)
    m, best, hist = train_seq(m, ds, seed, lr=lr, batch=batch, use_time=use_time, cir=cir, pcl=pcl,
                              pcl_k=sd.get('pcl_k', 32), patience=sd.get('patience', 5),
                              max_epochs=sd.get('max_epochs', 200), time_budget=tb)
    info = {'val_NDCG@10': best, 'curve': hist, 'cfg': sd}
    if m.kind == 'tpr':
        info['gamma'] = float(torch.nn.functional.softplus(m.gamma)) if m.time_decay else None
        if m.kg_item is not None:
            info['kg_proj_norm'] = float(m.kg_proj.weight.norm())
            info['id_emb_norm'] = float(m.item.weight[1:ds.n_items + 1].norm())
        with torch.no_grad():
            W = m.item_table().cpu().numpy()
            Uv = np.concatenate([m.user_repr(np.arange(i, min(i + 1024, ds.n_users)), 'val', use_time).cpu().numpy() for i in range(0, ds.n_users, 1024)])
            Ut = np.concatenate([m.user_repr(np.arange(i, min(i + 1024, ds.n_users)), 'test', use_time).cpu().numpy() for i in range(0, ds.n_users, 1024)])
        os.makedirs(os.path.dirname(emb_path(ds.name, tag, seed)), exist_ok=True)
        np.savez_compressed(emb_path(ds.name, tag, seed), W=W, Uv=Uv, Ut=Ut)
    return m.score_fn(use_time), info


def run_stellar(ds, model, seed, cfg):
    """Fusion of the temporal channel (TPR) and the KG-SGW structural channel."""
    tpr_name = cfg.get('tpr', 'tpr'); sgw_name = cfg.get('sgw', 'sgw')
    T = np.load(emb_path(ds.name, tpr_name, seed)); S = np.load(emb_path(ds.name, sgw_name, seed))
    base = {'val': FU.Channel('tpr', T['Uv'], T['W']), 'test': FU.Channel('tpr', T['Ut'], T['W'])}
    sgw = FU.Channel('sgw', S['U'], S['I'])
    res = {'val': [sgw], 'test': [sgw]}
    fv = FU.context_features(ds, 'val'); ft = FU.context_features(ds, 'test')
    mu, sd_ = fv.mean(0), fv.std(0) + 1e-6
    phi = {'val': torch.as_tensor((fv - mu) / sd_, device=DEVICE), 'test': torch.as_tensor((ft - mu) / sd_, device=DEVICE)}
    gated = cfg.get('gated', True)
    hgf, curve = FU.fit_hgf(ds, base['val'], res['val'], phi['val'], seed, gated=gated, l2=cfg.get('l2', 1e-3))

    def score(users, phase):
        return FU.fused_scores(hgf, base[phase], res[phase], phi[phase], users)
    w = hgf.w.detach().cpu().numpy()
    g_test = hgf.gates(phi['test']).detach().cpu().numpy()[:, 0]
    info = {'gate_w': w.tolist(), 'temp': float(torch.exp(hgf.log_temp).cpu()), 'hgf_curve': curve,
            'gate_mean_test': float(g_test.mean()), 'gate_std_test': float(g_test.std()),
            'feature_names': ['bias', 'log_len', 'mean_logpop', 'std_logpop', 'tail_share', 'log_gap']}
    os.makedirs(os.path.join(RES, ds.name, model), exist_ok=True)
    np.save(os.path.join(RES, ds.name, model, f'gates_seed{seed}.npy'), g_test)
    return score, info


def older_twin(dsn, tag, seed):
    import subprocess
    me = os.getpid()
    for l in subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()[1:]:
        w = l.split()
        if len(w) < 3 or 'run.py' not in l or int(w[0]) >= me: continue
        try:
            if w[w.index('--dataset') + 1] == dsn and w[w.index('--tag') + 1] == tag and int(w[w.index('--seed') + 1]) == seed:
                return True
        except (ValueError, IndexError):
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset'); ap.add_argument('--model'); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cfg', default='{}'); ap.add_argument('--tag', default=None); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(a.cfg); tag = a.tag or a.model
    while older_twin(a.dataset, tag, a.seed):   # the same run already in progress (started by another driver): wait for it
        time.sleep(60)
    if done(a.dataset, tag, a.seed) and not a.force:
        log('already done', a.dataset, tag, a.seed); return
    set_seed(a.seed)
    ds = Dataset(a.dataset)
    log(f'== {a.dataset} {tag} seed {a.seed} cfg {cfg}')
    t0 = time.time()
    info = {}
    if a.model == 'pop':
        m = Pop(ds).fit(); score = m.score
    elif a.model == 'itemknn':
        m = ItemKNN(ds, **cfg).fit(); score = m.score
    elif a.model == 'ease':
        m = EASE(ds, **cfg).fit(); score = m.score
    elif a.model in ('bpr', 'lightgcn', 'kgat', 'kgin') or a.model in SGW_VARIANTS:
        score, info = run_graph(ds, a.model, a.seed, cfg, tag)
    elif a.model in ('sasrec', 'bert4rec', 'bsarec') or a.model in TPR_VARIANTS:
        score, info = run_seq(ds, a.model, a.seed, cfg, tag)
    elif a.model.startswith('stellar'):
        score, info = run_stellar(ds, tag, a.seed, cfg)
    else:
        raise ValueError(a.model)
    train_time = time.time() - t0
    t1 = time.time()
    if a.model in ('pop', 'itemknn', 'ease'):
        rv = evaluate(ds, score, 'val', keep_lists=False); info['val_NDCG@10'] = float(rv['NDCG@10'].mean())
    res = evaluate(ds, score, 'test')
    infer_time = time.time() - t1
    summ = summarize(ds, res)
    info.update(train_time_s=train_time, infer_time_s=infer_time, cfg=cfg)
    save_result(a.dataset, tag, a.seed, res, summ, info)
    log(f'== DONE {a.dataset} {tag} seed {a.seed}: test NDCG@10 {summ["NDCG@10"]:.4f} HR@10 {summ["HR@10"]:.4f} '
        f'val {info.get("val_NDCG@10", float("nan")):.4f} train {train_time:.0f}s')


if __name__ == '__main__':
    main()
