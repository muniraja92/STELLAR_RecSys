import sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, '../src'); sys.path.insert(0, '.')
from models_kg import kernel_bank, cheb_coeffs, cheb_eval
import style; style.apply()
lam = np.linspace(0, 2, 801)
heat = [((r'$e^{-\lambda}$' if s == 1 else f'$e^{{-{s:g}\\lambda}}$'), lambda l, s=s: np.exp(-s * l)) for s in (1.0, 2.0, 4.0)]
band = kernel_bank((1.0, 2.0, 4.0), 4.0)
bnames = [r'$e^{-4\lambda}$', r'$g(\lambda)$', r'$g(2\lambda)$', r'$g(4\lambda)$']
lg = np.mean([(1 - lam) ** m for m in range(4)], 0)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={'wspace': 0.42})
for j, (n, f) in enumerate(heat):
    ax[0].plot(lam, f(lam), color=style.PAL[j], label=n)
    ax[0].plot(lam, cheb_eval(cheb_coeffs(f, 8), lam), color=style.PAL[j], ls=(0, (2, 2)), lw=1.0)
for j, (n, f) in enumerate(band):
    ax[1].plot(lam, f(lam), color=style.PAL[j], label=bnames[j])
for a in ax[:2]:
    a.plot(lam, lg, color=style.INK2, lw=1.2, ls='-.', label='LightGCN')
    a.set_xlabel(r'graph frequency $\lambda$'); a.set_ylim(-0.03, 1.05)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.4, columnspacing=0.8)
ax[0].set_ylabel('kernel response')
ax[0].set_title('(a) heat-kernel wavelets', loc='left'); ax[1].set_title('(b) band-pass bank (ablation)', loc='left')
for k, (lab, bank, col) in enumerate([('heat-kernel wavelets', heat, style.PAL[0]), ('band-pass bank', band, style.PAL[1])]):
    Ks = range(2, 15)
    errs = [max(np.abs(cheb_eval(cheb_coeffs(f, K), lam) - f(lam)).max() for _, f in bank) for K in Ks]
    ax[2].plot(list(Ks), errs, marker='o', ms=3.5, color=col, label=lab)
ax[2].axvline(8, color=style.INK2, lw=0.8, ls=':')
ax[2].set_yscale('log'); ax[2].set_xlabel('Chebyshev order $K$'); ax[2].set_ylabel('max. abs. error')
ax[2].set_title('(c) approximation error', loc='left'); ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1)
plt.savefig('../paper_out/figures/Fig2_kernels.pdf'); plt.savefig('./Fig2_kernels.png', dpi=140)
