"""Result figures from the per-dataset report JSON files (results/tables/<ds>_report.json)."""
import sys, json, os
import numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, '.')
import style; style.apply()
TAB = '../results/tables'
OUT = '../paper_out/figures'; os.makedirs(OUT, exist_ok=True)
DS = [('ml1m', 'MovieLens-1M')]
R = {d: json.load(open(f'{TAB}/{d}_report.json')) for d, _ in DS if os.path.exists(f'{TAB}/{d}_report.json')}
RELNAME = {'genre(MovieLens)': 'MovieLens genre', 'genre': 'Satori genre', 'release_decade': 'release decade', 'rating': 'content rating'}
NAMES = {'stellar': 'STELLAR-KG', 'tpr': 'TPR', 'tpr_nopcl': 'TPR w/o PCL', 'sasrec': 'SASRec', 'bsarec': 'BSARec',
         'sgw': 'KG-SGW', 'lightgcn': 'LightGCN', 'ease': 'EASE'}


def fig_groups():
    """NDCG@10 by quintile of user history length (PCL / sparsity analysis)."""
    models = ['stellar', 'tpr', 'tpr_nopcl', 'bsarec', 'sasrec', 'ease']
    fig, ax = plt.subplots(1, len(R), figsize=(7.4, 2.7), gridspec_kw={'wspace': 0.32})
    ax = np.atleast_1d(ax)
    for a, (d, title) in zip(ax, [x for x in DS if x[0] in R]):
        g = R[d]['user_groups']; b = g['bounds']
        x = np.arange(len(b))
        for j, m in enumerate(models):
            if m in g:
                a.plot(x, g[m], marker='o', ms=3.5, color=style.PAL[j], label=NAMES[m],
                       ls='--' if m == 'tpr_nopcl' else '-')
        a.set_xticks(x); a.set_xticklabels([f'{lo}-{hi}' for lo, hi, _ in b], rotation=35, ha='right')
        a.set_title(title, loc='left'); a.set_xlabel('history length (interactions)')
    ax[0].set_ylabel('NDCG@10')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=6)
    fig.savefig(f'{OUT}/Fig4_user_groups.pdf'); fig.savefig(f'{OUT}/Fig4_user_groups.png', dpi=140)


def fig_diag():
    """Learned relation weights of KG-SGW and HGF gate by history length."""
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={'wspace': 0.35, 'width_ratios': [1.6, 1]})
    labels, vals, errs, cols = [], [], [], []
    for k, (d, title) in enumerate([x for x in DS if x[0] in R]):
        rw = R[d]['diagnostics'].get('sgw', {}).get('relation_weights', {})
        for r, (mu, sd) in rw.items():
            labels.append(RELNAME.get(r, r)); vals.append(mu); errs.append(sd); cols.append(style.PAL[k])
    y = np.arange(len(labels))
    ax[0].barh(y, vals, xerr=errs, color=cols, height=0.7, error_kw={'lw': 0.8})
    ax[0].set_xscale('log'); ax[0].set_xlim(1e-2, 50)
    ax[0].axvline(1.0, color=style.INK2, lw=0.8, ls=':')
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels); ax[0].invert_yaxis()
    ax[0].set_xlabel(r'learned relation weight $\omega_r$'); ax[0].set_title('(a) KG-SGW relation weights', loc='left')
    for k, (d, title) in enumerate([x for x in DS if x[0] in R]):
        gl = R[d]['diagnostics'].get('stellar', {}).get('gate_by_len_group')
        if gl:
            ax[1].plot(range(len(gl)), gl, marker='o', ms=3.5, color=style.PAL[k], label=title)
    from matplotlib.patches import Patch
    ax[1].axhline(0, color=style.INK2, lw=0.8, ls=':'); ax[1].set_ylim(0, 0.55)
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f'{lo}-{hi}' for lo, hi, _ in R['ml1m']['user_groups']['bounds']], rotation=35, ha='right'); ax[1].set_xlabel('history length (interactions)'); ax[1].set_ylabel(r'mean gate $g(\psi_u)$')
    ax[1].set_title('(b) HGF gate', loc='left')
    fig.savefig(f'{OUT}/Fig3_diagnostics.pdf'); fig.savefig(f'{OUT}/Fig3_diagnostics.png', dpi=140)


STRAT = '../results/strat/ml1m_strat.json'
DIMS = [('history', 'History length'), ('gender', 'Gender'), ('age', 'Age'), ('gap', 'Gap before target'),
        ('period', 'Test period'), ('popularity', 'Target popularity'), ('knowledge', 'Target Satori links')]


def fig_strata():
    """Forest plot: paired NDCG@10 differences with 95% bootstrap intervals in every stratum."""
    S = json.load(open(STRAT))['dimensions']
    panels = [('stellar-sasrec', '(a) STELLAR-KG $-$ SASRec'), ('tpr-tpr_nokg', '(b) TPR $-$ TPR without tokens'),
              ('stellar-tpr', '(c) STELLAR-KG $-$ TPR')]
    labels, ys, y = [], [], 0
    seps = []
    for dim, dn in DIMS:
        for lv in S[dim]['levels']:
            labels.append(f'{dn}: ' + lv.replace('>=', '≥ ').replace('<', '< ').replace('>', '> ')); ys.append(y); y += 1
        seps.append(y - 0.5); y += 0.6
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 5.0), sharey=True, gridspec_kw={'wspace': 0.08})
    for a, (key, title) in zip(ax, panels):
        k = 0
        for dim, dn in DIMS:
            for r in S[dim]['pairs'][key]['strata']:
                sig = r['p_holm'] < 0.05
                c = style.PAL[0] if r['delta'] >= 0 else style.PAL[1]
                a.plot(r['ci'], [ys[k]] * 2, color=c, lw=1.1)
                a.plot([r['delta']], [ys[k]], marker='o', ms=3.8, color=c, mfc=c if sig else 'white', mew=1.0)
                k += 1
        a.axvline(0, color=style.INK2, lw=0.8, ls=':')
        a.set_title(title, loc='left', fontsize=8.5); a.set_xlabel(r'$\Delta$ NDCG@10')
        a.grid(axis='y', visible=False)
    ax[0].set_yticks(ys); ax[0].set_yticklabels(labels); ax[0].invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', color=style.INK2, mfc=style.INK2, ls='', label=r'$p_{\mathrm{Holm}}<0.05$ within the dimension'),
                        Line2D([], [], marker='o', color=style.INK2, mfc='white', ls='', label='not significant')],
               loc='upper center', bbox_to_anchor=(0.55, 0.0), ncol=2)
    fig.savefig(f'{OUT}/Fig4_strata.pdf'); fig.savefig(f'{OUT}/Fig4_strata.png', dpi=140)


if __name__ == '__main__':
    fig_strata(); fig_diag(); print('ok', list(R))
