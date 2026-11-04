"""LaTeX tables from results/tables/<ds>_report.json (produced by src/report.py)."""
import json, os, math, csv
TAB = '../results/tables'
OUT = '../paper_out/tables'
os.makedirs(OUT, exist_ok=True)
DS = [('ml1m', 'MovieLens-1M')]
R = {d: json.load(open(f'{TAB}/{d}_report.json')) for d, _ in DS if os.path.exists(f'{TAB}/{d}_report.json')}
DSA = [(d, n) for d, n in DS if d in R]
NAME = {'pop': 'Popularity', 'itemknn': 'ItemKNN', 'ease': 'EASE', 'bpr': 'BPR-MF', 'lightgcn': 'LightGCN',
        'kgat': 'KGAT', 'kgin': 'KGIN', 'sasrec': 'SASRec', 'bert4rec': 'BERT4Rec', 'bsarec': 'BSARec',
        'sgw': 'KG-SGW', 'tpr': 'TPR', 'stellar': 'STELLAR-KG', 'stellar_fixed': 'STELLAR-KG, constant gate',
        'stellar_nokg': 'STELLAR-KG without KG', 'sgw_nokg': 'KG-SGW without KG (bipartite)',
        'sgw_band': 'KG-SGW with band-pass bank', 'sgw_fixrel': r'KG-SGW with $\omega_r=1$',
        'tpr_nokg': 'TPR without tokens', 'tpr_nopcl': 'TPR without PCL', 'tpr_nocir': 'TPR without CIR',
        'tpr_notime': 'TPR without time decay and RoPE', 'tpr_tok_sgw_nokg': 'TPR, bipartite KG-SGW tokens', 'tpr_tok_lightgcn': 'TPR, LightGCN tokens', 'tpr_norope': 'TPR without RoPE', 'tpr_nodecay': 'TPR without time decay'}
BASE = ['pop', 'itemknn', 'ease', 'bpr', 'lightgcn', 'kgat', 'kgin', 'sasrec', 'bert4rec', 'bsarec']
OURS = ['sgw', 'tpr', 'stellar']


def row(d, m):
    for r in R[d]['main']:
        if r['model'] == m: return r
    return None


def f4(x): return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.4f}'


def pfmt(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return '--'
    return r'$<10^{-4}$' if p < 1e-4 else f'{p:.4f}' if p < 0.01 else f'{p:.3f}'


def main_table(metrics=(('HR@10', 'HR@10'), ('HR@20', 'HR@20'), ('NDCG@10', 'NDCG@10'), ('NDCG@20', 'NDCG@20'), ('MRR@10', 'MRR@10'))):
    cols = 'l' + 'c' * len(metrics)
    L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
         r'\caption{Full-catalogue ranking accuracy on MovieLens-1M (mean and standard deviation over seeds 42, 1 and 7). The best value is set in bold and the best baseline is underlined. All models are trained on the same splits and selected on the same validation targets.}',
         r'\label{tab:main}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{' + cols + '}', r'\toprule']
    L.append('Model & ' + ' & '.join(n for _, n in metrics) + r' \\')
    for d, dn in DSA:
        L.append(r'\midrule')
        best, bestb = {}, {}
        for k, _ in metrics:
            vals = {m: row(d, m)[k] for m in BASE + OURS if row(d, m)}
            best[k] = max(vals, key=vals.get)
            vb = {m: v for m, v in vals.items() if m in BASE}
            bestb[k] = max(vb, key=vb.get)
        for gi, grp in enumerate((BASE, OURS)):
            if gi: L.append(r'\cmidrule(lr){1-' + str(len(metrics) + 1) + '}')
            for m in grp:
                r = row(d, m); cells = []
                for k, _ in metrics:
                    if r is None: cells.append('--'); continue
                    s_ = f"{r[k]:.4f}"
                    if best[k] == m: s_ = r'\textbf{' + s_ + '}'
                    elif bestb[k] == m: s_ = r'\underline{' + s_ + '}'
                    cells.append(s_ + r'{\tiny$\pm$' + f"{r[k + '_std']:.4f}" + '}')
                L.append(NAME[m] + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_main.tex', 'w').write('\n'.join(L) + '\n')


def stat_table():
    """STELLAR-KG and TPR against the strongest baselines; Wilcoxon on per-user NDCG@10, Holm-adjusted."""
    L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
         r'\caption{Paired comparison of STELLAR-KG with each baseline and with its own components on per-user NDCG@10 (averaged over seeds). $\Delta$ is the mean difference with a 95\% bootstrap interval, $p$ the Holm-adjusted Wilcoxon signed-rank $p$-value, $d_z$ the standardised paired effect size, and W the number of seeds on which STELLAR-KG is better.}',
         r'\label{tab:stats}', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lccccc}', r'\toprule',
         r'Compared with & $\Delta$ NDCG@10 & 95\% CI & $p_{\mathrm{Holm}}$ & $d_z$ & W \\', r'\midrule']
    for d, n in DSA:
        rows = R[d]['stellar_vs']
        first = True
        for f in rows:
            L.append(f"{NAME[f['b']]} & {f['delta']:+.4f} & [{f['ci'][0]:+.4f}, {f['ci'][1]:+.4f}] & {pfmt(f.get('p_holm'))} & {f['dz']:+.3f} & {f['seed_wins']}/{f['n_seeds']}" + r' \\')
            first = False
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L += [r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_stats.tex', 'w').write('\n'.join(L) + '\n')


def ablation_table():
    pairs = [('tpr', 'tpr_tok_sgw_nokg'), ('tpr', 'tpr_tok_lightgcn'), ('tpr_tok_sgw_nokg', 'tpr_nokg'), ('tpr_tok_lightgcn', 'tpr_nokg'),
             ('stellar', 'stellar_nokg'), ('stellar', 'stellar_fixed'), ('stellar', 'tpr'),
             ('sgw', 'sgw_nokg'), ('sgw', 'sgw_fixrel'), ('sgw', 'sgw_band'), ('sgw', 'lightgcn'),
             ('tpr', 'tpr_nokg'), ('tpr', 'tpr_nopcl'), ('tpr', 'tpr_nocir'), ('tpr', 'tpr_notime'), ('tpr', 'sasrec')]
    L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
         r'\caption{Ablations. Each row compares a variant (right) with its parent model (left) on NDCG@10; values are means over seeds 42, 1 and 7, $\Delta$ is the parent minus the variant, and $p$ the Holm-adjusted Wilcoxon $p$-value on per-user NDCG@10 (family of all rows). Differences are computed before rounding. The rows with replaced tokens test whether the gain from the knowledge tokens requires the knowledge graph: with bipartite KG-SGW tokens, TPR uses the item representations of KG-SGW trained without item--entity edges, and with LightGCN tokens those of LightGCN.}',
         r'\label{tab:ablation}', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{ll' + 'ccc' * len(DSA) + '}', r'\toprule',
         'Parent & Variant & ' + ' & '.join(['NDCG@10', r'$\Delta$', '$p$'] * len(DSA)) + r' \\', r'\midrule']
    for a, b in pairs:
        cells = []
        for d, _ in DSA:
            f = next((x for x in R[d]['pairs'] if x['a'] == a and x['b'] == b), None)
            cells += [f4(f['mean_b']), f"{f['delta']:+.4f}", pfmt(f.get('p_holm'))] if f else ['--'] * 3
        if all(c == '--' for c in cells): continue
        L.append(f'{NAME[a]} & {NAME[b]} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_ablation.tex', 'w').write('\n'.join(L) + '\n')


def beyond_table():
    ms = BASE + ['sgw', 'tpr', 'tpr_nocir', 'stellar']
    C = {d: {x['model']: {k: float(v) for k, v in x.items() if k not in ('model',) and v not in ('', None)} for x in csv.DictReader(open(f'{TAB}/{d}_main.csv'))} for d, _ in DSA}
    keys = [('ARP@10', 'ARP', '{:.1f}'), ('APLT@10', 'APLT', '{:.3f}'), ('Coverage@10', 'Cov.', '{:.3f}'),
            ('Gini@10', 'Gini', '{:.3f}'), ('NDCG@10_env_low', r'N@10$_{\mathrm{low}}$', '{:.4f}')]
    L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
         r'\caption{Popularity and exposure of the top-10 lists (means over seeds): average recommendation popularity (ARP, training interactions per recommended item, lower is less popularity-biased), average share of long-tail items (APLT), catalogue coverage, Gini coefficient of exposure (lower is more even), and NDCG@10 on users whose test item lies in the least popular tertile of items (330 of the 5,950 test users).}',
         r'\label{tab:beyond}', r'\setlength{\tabcolsep}{2.5pt}', r'\begin{tabular}{l' + 'c' * (len(keys) * len(DSA)) + '}', r'\toprule',
         'Model & ' + ' & '.join(h for _ in DSA for _, h, _ in keys) + r' \\', r'\midrule']
    for m in ms:
        cells = []
        for d, _ in DSA:
            r = C[d].get(m)
            cells += [fmt.format(r[k]) if r and r.get(k) is not None else '--' for k, _, fmt in keys]
        L.append(NAME[m] + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_beyond.tex', 'w').write('\n'.join(L) + '\n')


def time_table():
    ms = ['ease', 'lightgcn', 'kgat', 'kgin', 'sgw', 'sasrec', 'bert4rec', 'bsarec', 'tpr', 'stellar']
    L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
         r'\caption{Training and full-catalogue inference time in minutes (means over seeds, two CPU threads per run), including periodic validation. STELLAR-KG: fusion step only, on top of its two encoders.}',
         r'\label{tab:time}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{l' + 'cc' * len(DSA) + '}', r'\toprule',
         'Model & ' + ' & '.join(['Train', 'Infer.'] * len(DSA)) + r' \\', r'\midrule']
    for m in ms:
        cells = []
        for d, _ in DSA:
            r = row(d, m)
            cells += [f"{r['train_time_s'] / 60:.1f}", f"{r['infer_time_s'] / 60:.2f}"] if r else ['--', '--']
        L.append(NAME[m] + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_time.tex', 'w').write('\n'.join(L) + '\n')


def sampled_table():
    ms = BASE + OURS
    L = [r'\begin{table}[!htbp]', r'\centering', r'\footnotesize',
         r'\caption{HR@10 and NDCG@10 against 99 sampled negatives drawn once per user with a fixed seed and shared by all models (means over seeds), reported for comparison with earlier work that uses sampled metrics.}',
         r'\label{tab:sampled}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{l' + 'cc' * len(DSA) + '}', r'\toprule',
         'Model & ' + ' & '.join(['HR@10', 'NDCG@10'] * len(DSA)) + r' \\', r'\midrule']
    for m in ms:
        cells = []
        for d, _ in DSA:
            r = row(d, m)
            cells += [f"{r['s_HR@10']:.4f}", f"{r['s_NDCG@10']:.4f}"] if r else ['--', '--']
        L.append(NAME[m] + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_sampled.tex', 'w').write('\n'.join(L) + '\n')


def strat_table():
    """Stratified tests on MovieLens-1M (results/strat/ml1m_strat.json, produced by src/stratify.py)."""
    S = json.load(open('../results/strat/ml1m_strat.json'))['dimensions']
    dims = [('history', 'History length', 'quintiles'), ('gender', 'Gender', ''), ('age', 'Age', ''),
            ('gap', 'Gap before target', ''), ('period', 'Test period', ''), ('popularity', 'Target popularity', 'tertiles'),
            ('knowledge', 'Target Satori links', '')]
    pairs = [('stellar-sasrec', 'STELLAR-KG $-$ SASRec'), ('tpr-tpr_nokg', 'KG tokens $-$ no tokens'), ('tpr-tpr_tok_sgw_nokg', 'KG $-$ bipartite tokens'), ('stellar-tpr', 'STELLAR-KG $-$ TPR')]
    def sg(x): return f'{x:+.4f}' if abs(x) >= 5e-5 or x == 0 else f'{x:+.5f}'
    L = [r'\begin{table}[!t]', r'\centering', r'\footnotesize',
         r'\caption{Stratified analysis on MovieLens-1M for four key comparisons (per-user NDCG@10 averaged over seeds); the second and third compare TPR with knowledge-graph tokens against TPR without tokens and against TPR with tokens from KG-SGW on the bipartite graph. For each stratification, Strata is the number of strata, $+$ the number of strata with a positive mean difference, $\Delta_{\mathrm{bal}}$ the difference in a balanced population in which every stratum has equal weight, with a 95\% stratified-bootstrap interval, $p_{\mathrm{s}}$ the stratified signed-rank test with van Elteren weights and $p_{\mathrm{het}}$ Cochran\textquoteright s $Q$ test of heterogeneity across strata. The stratified tests are alternative tests of the same overall hypothesis and are reported without correction; the heterogeneity tests are Holm-adjusted over the seven stratifications.}',
         r'\label{tab:strat}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lccccc}', r'\toprule',
         r'Stratification & Strata & $+$ & $\Delta_{\mathrm{bal}}$ [95\% CI] & $p_{\mathrm{s}}$ & $p_{\mathrm{het}}$ \\']
    for gi, (key, pn) in enumerate(pairs):
        L += [r'\midrule', r'\multicolumn{6}{l}{\textit{' + pn + r'}} \\']
        for dim, dn, _ in dims:
            P = S[dim]['pairs'][key]
            L.append(f"{dn} & {P['K']} & {P['n_positive']} & {sg(P['equal_weight'])} [{sg(P['equal_weight_ci'][0])}, {sg(P['equal_weight_ci'][1])}] & {pfmt(P['vanelteren_p'])} & {pfmt(P['Q_p_holm'])}" + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_strat.tex', 'w').write('\n'.join(L) + '\n')


AMZ = [('beauty', 'Beauty'), ('toys', 'Toys')]


def amazon_tables():
    A = {d: json.load(open(f'../results/supplementary_amazon/{d}_report.json')) for d, _ in AMZ}
    def r_(d, m): return next((x for x in A[d]['main'] if x['model'] == m), None)
    ms = BASE + ['sgw', 'tpr', 'tpr_nokg', 'tpr_nopcl', 'tpr_nocir', 'tpr_notime', 'stellar_nokg', 'stellar_fixed', 'stellar']
    L = [r'\begin{table}[!h]', r'\centering', r'\footnotesize',
         r'\caption{Supplementary results on Amazon Beauty and Toys (NDCG@10 and HR@10, means over seeds 42, 1 and 7; full-catalogue ranking).}',
         r'\label{tab:amazon}', r'\begin{tabular}{lcccc}', r'\toprule',
         r' & \multicolumn{2}{c}{Beauty} & \multicolumn{2}{c}{Toys} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Model & NDCG@10 & HR@10 & NDCG@10 & HR@10 \\', r'\midrule']
    for gi, grp in enumerate((BASE, ms[len(BASE):])):
        if gi: L.append(r'\cmidrule(lr){1-5}')
        for m in grp:
            cells = []
            for d, _ in AMZ:
                r = r_(d, m); cells += [f"{r['NDCG@10']:.4f}", f"{r['HR@10']:.4f}"] if r else ['--', '--']
            L.append(NAME[m] + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_amazon.tex', 'w').write('\n'.join(L) + '\n')
    comps = [('sv', 'bsarec'), ('sv', 'sasrec'), ('sv', 'bert4rec'), ('sv', 'itemknn'), ('sv', 'ease'), ('sv', 'kgin'), ('pairs', ('stellar', 'tpr')),
             ('pairs', ('stellar', 'stellar_nokg')), ('pairs', ('tpr', 'tpr_nokg')), ('pairs', ('tpr', 'tpr_notime')), ('pairs', ('tpr', 'sasrec'))]
    L = [r'\begin{table}[!h]', r'\centering', r'\footnotesize',
         r'\caption{Paired comparisons on the Amazon data: $\Delta$ is the difference in mean NDCG@10 (first model minus second) and $p$ the Holm-adjusted Wilcoxon signed-rank $p$-value on per-user NDCG@10, adjusted within the comparisons of STELLAR-KG with other models (first six rows) or within the ablations (remaining rows), as for MovieLens-1M.}',
         r'\label{tab:amazon_stats}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lcccc}', r'\toprule',
         r' & \multicolumn{2}{c}{Beauty} & \multicolumn{2}{c}{Toys} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Comparison & $\Delta$ & $p_{\mathrm{Holm}}$ & $\Delta$ & $p_{\mathrm{Holm}}$ \\', r'\midrule']
    for kind, c in comps:
        cells = []
        for d, _ in AMZ:
            if kind == 'sv':
                f = next(x for x in A[d]['stellar_vs'] if x['b'] == c); lab = f'STELLAR-KG $-$ {NAME[c]}'
            else:
                f = next(x for x in A[d]['pairs'] if x['a'] == c[0] and x['b'] == c[1]); lab = f'{NAME[c[0]]} $-$ {NAME[c[1]]}'
            cells += [f"{f['delta']:+.4f}", pfmt(f.get('p_holm'))]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{OUT}/tab_amazon_stats.tex', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main_table(); stat_table(); ablation_table(); beyond_table(); time_table(); sampled_table(); strat_table(); amazon_tables()
    print('tables written for', [d for d, _ in DSA])


def _fit_width(names=('tab_ablation', 'tab_beyond')):
    for n in names:
        p = f'{OUT}/{n}.tex'; s = open(p).read()
        if r'\resizebox' in s: continue
        s = s.replace(r'\begin{tabular}', r'\resizebox{\textwidth}{!}{\begin{tabular}', 1).replace(r'\end{tabular}', r'\end{tabular}}', 1)
        open(p, 'w').write(s)


