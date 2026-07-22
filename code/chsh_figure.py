"""Generate chsh_results.png: three-panel figure for 'CHSH by Counting'.

Panel 1: counted correlator E vs analyzer separation, coherent class vs
         sector-recorded control, against the quantum reference -cos(Delta).
Panel 2: counted |S| against the local bound (2) and Tsirelson (2*sqrt(2)).
Panel 3: residue | |S| - 2*sqrt(2) | vs N, log-log, with a 1/N guide.

Series colors validated (CVD + contrast): blue #3466C4, orange #C85A19.
Identity is never color-alone: every series is direct-labeled.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chsh_by_counting import (joint_counts_coherent, joint_counts_decohered,
                              correlator, chsh)

BLUE, ORANGE = "#3466C4", "#C85A19"
INK, MUTED, GRID = "#333333", "#666666", "#dddddd"
TSIRELSON = 2 * np.sqrt(2)
N = 4_200_000
OPT = (0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4)

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 3.7))

# ---- Panel 1: counted correlator vs analyzer separation -------------------
deltas = np.linspace(0, np.pi, 25)
E_coh, E_dec = [], []
for d in deltas:
    a, b = d / 2, -d / 2  # separation enters symmetrically
    E_coh.append(correlator(joint_counts_coherent(N, a, b)))
    E_dec.append(correlator(joint_counts_decohered(N, a, b)))
dd = np.linspace(0, np.pi, 300)
ax1.plot(np.degrees(dd), -np.cos(dd), color=MUTED, lw=1, ls="--", zorder=1)
ax1.plot(np.degrees(deltas), E_coh, "o", color=BLUE, ms=4.5, zorder=3)
ax1.plot(np.degrees(deltas), E_dec, "s", color=ORANGE, ms=4, zorder=2)
ax1.text(8, 0.62, "coherent class\n(counted)", color=BLUE, fontsize=9)
ax1.text(97, -0.86, "sector record inserted\n(counted)", color=ORANGE, fontsize=9)
ax1.text(139, 0.28, r"$-\cos\Delta$", color=MUTED, fontsize=9)
ax1.set_xlabel(r"analyzer separation $\Delta = a-b$ (deg)")
ax1.set_ylabel(r"correlator $E$ (integer counts)")
ax1.set_xticks([0, 45, 90, 135, 180])
ax1.grid(color=GRID, lw=0.6, zorder=0)

# ---- Panel 2: |S| vs bounds ----------------------------------------------
S_coh, _ = chsh(N, joint_counts_coherent, OPT)
S_dec, _ = chsh(N, joint_counts_decohered, OPT)
bars = ax2.bar([0, 1], [abs(S_coh), abs(S_dec)], width=0.52,
               color=[BLUE, ORANGE], zorder=3)
ax2.axhline(2.0, color=INK, lw=1, ls=":", zorder=2)
ax2.axhline(TSIRELSON, color=MUTED, lw=1, ls="--", zorder=2)
ax2.text(1.42, 2.03, "local bound 2", fontsize=8.5, color=INK, ha="right")
ax2.text(1.42, TSIRELSON + 0.03, r"Tsirelson $2\sqrt{2}$", fontsize=8.5,
         color=MUTED, ha="right")
for x, v in [(0, abs(S_coh)), (1, abs(S_dec))]:
    ax2.text(x, v + 0.07, f"{v:.4f}", ha="center", fontsize=9, color=INK)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["coherent\nclass", "sector record\ninserted"])
ax2.set_ylabel(r"counted $|S|$")
ax2.set_ylim(0, 3.25)
ax2.grid(axis="y", color=GRID, lw=0.6, zorder=0)

# ---- Panel 3: residue scaling --------------------------------------------
Ns = np.array([2_000, 20_000, 200_000, 2_000_000, 20_000_000])
res, above = [], []
for n_ in Ns:
    S_, _ = chsh(int(n_), joint_counts_coherent, OPT)
    res.append(abs(abs(S_) - TSIRELSON))
    above.append(abs(S_) > TSIRELSON)
res = np.array(res)
guide = res[0] * (Ns[0] / Ns.astype(float))
ax3.loglog(Ns, guide, color=MUTED, lw=1, ls="--", zorder=1)
for filled in (True, False):
    m = np.array(above) == filled
    ax3.loglog(Ns[m], res[m], "o", ms=5.5, zorder=3, color=BLUE,
               markerfacecolor=BLUE if filled else "white",
               markeredgecolor=BLUE)
slope = np.polyfit(np.log10(Ns), np.log10(res), 1)[0]
ax3.text(4e3, 2e-6, r"$\propto 1/N$", color=MUTED, fontsize=9)
ax3.text(0.03, 0.06,
         f"filled: above Tsirelson\nopen: below\nfit slope {slope:.2f}",
         transform=ax3.transAxes, fontsize=8.5, color=INK, va="bottom")
ax3.set_xlabel(r"class population $N$")
ax3.set_ylabel(r"$|\,|S| - 2\sqrt{2}\,|$")
ax3.grid(color=GRID, lw=0.6, which="major", zorder=0)

fig.tight_layout(w_pad=2.2)
fig.savefig("chsh_results.png", dpi=220, bbox_inches="tight")
print("wrote chsh_results.png")
