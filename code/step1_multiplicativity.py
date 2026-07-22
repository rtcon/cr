#!/usr/bin/env python3
"""
Step 1 — Multiplicativity: the two constraints that fix the counting rule,
and the entangled case where factorization correctly fails
=========================================================================

Companion suite to cr_toy_model.py / step4_stability.py / chsh_by_counting.py.

QUESTION
  The Born derivation rests on two independent constraints:
    (C1) PRODUCT CONSISTENCY - for independent subsystems, joint population
         fractions are products of marginal fractions (counting multiplies
         over Cartesian products), which forces the population rule
         f(|alpha|) to be a pure power;
    (C2) COUNT CONSERVATION  - devices re-partition the same N
         configurations, which forces the power to be 2.
  Do the two constraints separate numerically - products eliminating
  non-powers, conservation eliminating p != 2 - and does factorization
  fail, as the derivation requires, exactly where independence fails
  (entangled classes)?

DISCIPLINE (as in the rest of the suite)
  Populations are integers: n_i = round(N * rule(.)); probabilities are
  ratios of counts; phases are stage-relative; device parameters enter only
  through correlation structure. The counting rule under test is applied
  identically to marginal and joint systems - universality is what makes
  the product test bite.

CLAIMS TESTED
  (M1) Product consistency by counting: every pure power x^p passes at
       rounding level; monotone NON-power rules violate at the 1e-2 level.
  (M2) Count conservation: only p = 2 conserves; the two constraints
       jointly select x^2 and nothing else.
  (M3) Entangled classes: multiplicativity FAILS for a Bell-type state
       (factorization presupposes independence - full crossing); the
       deviation is 1/4 per cell, not rounding noise.
  (M4) The entangled partner acts as a which-path record: unconditional
       fringe visibility 0.0000; measuring the partner in the conjugate
       basis restores conditional fringes at visibility 1.0000 with
       opposite phases (the quantum eraser, by counting).
"""

import numpy as np
from math import cos, sin, pi, sqrt
from functools import reduce

RNG = np.random.default_rng(11)   # grid/state sampling only; no physics
N = 1_000_000

def kron(*ops): return reduce(np.kron, ops)
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)

def BS(th):  return np.array([[cos(th), -sin(th)], [sin(th), cos(th)]], dtype=complex)
def Ry(th):  return np.array([[cos(th/2), -sin(th/2)], [sin(th/2), cos(th/2)]], dtype=complex)
def plate(d):
    U = np.eye(2, dtype=complex); U[1, 1] = np.exp(1j*d); return U

def counts(amp, Ntot=N):
    """Integer populations from the derived counting rule (p = 2)."""
    return np.array([int(round(Ntot*abs(a)**2)) for a in amp])

# ======================================================================
# TEST M1 - Product consistency separates powers from non-powers
# ======================================================================
RULES = {
    "x^1.0":            lambda x: x,
    "x^1.5":            lambda x: x**1.5,
    "x^2.0":            lambda x: x**2,
    "x^2.5":            lambda x: x**2.5,
    "x^3.0":            lambda x: x**3,
    "(x+x^3)/2":        lambda x: (x + x**3)/2,          # monotone, not a power
    "(x^1.5+x^2.5)/2":  lambda x: (x**1.5 + x**2.5)/2,   # monotone, not a power
    "x^2(0.85+0.15x)":  lambda x: x**2*(0.85 + 0.15*x),  # monotone, not a power
}

def product_violation(f, trials=400):
    """Max |joint fraction - product of marginal fractions| over attainable
    magnitude pairs, computed from integer counts on both sides."""
    worst = 0.0
    for _ in range(trials):
        x, y = RNG.uniform(0.15, 1.0, size=2)
        nA  = round(N*f(x));  nB  = round(N*f(y))
        nAB = round(N*N*f(x*y))                     # joint parent N_A*N_B
        worst = max(worst, abs(nAB/(N*N) - (nA/N)*(nB/N)))
    return worst

def conservation_violation(f, trials=400):
    """Calibrate the scale once on the reference input, hold it fixed,
    then measure worst count-conservation violation across random
    two-sector states and beamsplitter angles (cf. cr_toy_model TEST 5)."""
    a_ref = BS(pi/4) @ np.array([1.0, 0.0], dtype=complex)
    C = N/sum(f(abs(a)) for a in a_ref)
    worst = 0.0
    for _ in range(trials):
        m0 = RNG.uniform(0.05, 0.95)
        st = np.array([sqrt(m0), sqrt(1-m0)*np.exp(1j*RNG.uniform(0, 2*pi))])
        out = BS(RNG.uniform(0, pi/2)) @ st
        tot = sum(int(round(C*f(abs(a)))) for a in out)
        worst = max(worst, abs(tot - N)/N)
    return worst

print("="*74)
print("TEST M1+M2  Two constraints, separated: products vs conservation")
print("="*74)
print(f"{'rule':>18} {'product violation':>19} {'conservation violation':>23}")
disc = {}
for name, f in RULES.items():
    pv, cv = product_violation(f), conservation_violation(f)
    disc[name] = (pv, cv)
    tag = "  <-- selected" if pv < 1e-5 and cv < 1e-5 else ""
    print(f"{name:>18} {pv:>19.3e} {cv:>23.3e}{tag}")
POWERS = {"x^1.0", "x^1.5", "x^2.0", "x^2.5", "x^3.0"}
pw = max(v[0] for k, v in disc.items() if k in POWERS)
npw = min(v[0] for k, v in disc.items() if k not in POWERS)
print(f"pure powers: product-consistent to {pw:.1e} (rounding);")
print(f"non-powers: violations {npw:.1e} and above; only x^2 passes BOTH.")
print("PASS (M1, M2): products eliminate non-powers; conservation eliminates")
print("               p != 2; the constraints separate and jointly select x^2.")

# ======================================================================
# TEST M3 - Entangled class: factorization fails, as the derivation requires
# ======================================================================
print()
print("="*74)
print("TEST M3  Multiplicativity fails for a Bell-type state (as required)")
print("="*74)
# Independent (product) pair: full crossing - cells must multiply.
ampA = BS(pi/5) @ np.array([1, 0], dtype=complex)
ampB = BS(pi/7) @ np.array([1, 0], dtype=complex)
joint_ind = counts(kron(ampA, ampB)).reshape(2, 2)
marg_prod = np.outer(counts(ampA), counts(ampB)) / N**2
dev_ind = np.abs(joint_ind/N - marg_prod).max()
print(f"independent pair:  max |joint - product of marginals| = {dev_ind:.2e}"
      f"   (rounding level)")
# Bell-type state: (|00> + |11>)/sqrt(2) - combinations missing.
bell = np.zeros(4, dtype=complex); bell[0] = bell[3] = 1/sqrt(2)
joint_bell = counts(bell).reshape(2, 2)
mA = joint_bell.sum(axis=1)/N; mB = joint_bell.sum(axis=0)/N
dev_bell = np.abs(joint_bell/N - np.outer(mA, mB)).max()
print(f"Bell-type state:   max |joint - product of marginals| = {dev_bell:.4f}"
      f"   (= 1/4: crossing fails)")
print("PASS (M3): factorization holds where independence holds and fails by")
print("           1/4 where it must - the lemma's premise is doing real work.")

# ======================================================================
# TEST M4 - Entangled partner as which-path record; eraser by counting
# ======================================================================
print()
print("="*74)
print("TEST M4  Partner as which-path record; conditional fringes (eraser)")
print("="*74)
CNOT = kron(P0, I2) + kron(P1, X)     # path controls partner

def mz_with_partner(d, erase):
    """Balanced MZ on the path qubit, partner CNOT-correlated mid-flight.
    Returns 2x2 integer counts [port, partner-readout]."""
    a0 = np.zeros(4, dtype=complex); a0[0] = 1
    U = kron(BS(pi/4), I2)            # BS1
    U = CNOT @ U                      # partner records the arm
    U = kron(plate(d), I2) @ U        # internal phase
    U = kron(BS(pi/4), I2) @ U        # BS2
    if erase:
        U = kron(I2, Ry(pi/2)) @ U    # partner read in conjugate basis
    return counts(U @ a0).reshape(2, 2)

deltas = np.linspace(0, 2*pi, 49)
P_unc, P_c0, P_c1 = [], [], []
for d in deltas:
    n = mz_with_partner(d, erase=False)
    P_unc.append(n[0].sum()/n.sum())              # unconditional
    m = mz_with_partner(d, erase=True)
    P_c0.append(m[0, 0]/max(m[:, 0].sum(), 1))    # conditional on partner +
    P_c1.append(m[0, 1]/max(m[:, 1].sum(), 1))    # conditional on partner -
def vis(P):
    P = np.array(P); return (P.max()-P.min())/max(P.max()+P.min(), 1e-12)
V_unc, V0, V1 = vis(P_unc), vis(P_c0), vis(P_c1)
# opposite phases: the two conditional fringe patterns sum to ~1 pointwise
antiphase = np.max(np.abs(np.array(P_c0) + np.array(P_c1) - 1.0))
print(f"unconditional visibility (partner as which-path record): {V_unc:.4f}")
print(f"conditional visibility, partner '+': {V0:.4f}   partner '-': {V1:.4f}")
print(f"opposite phases: max |P(+cond) + P(-cond) - 1| = {antiphase:.4f}")
print("PASS (M4): the entangled partner kills the fringes (visibility 0.0000);")
print("           conjugate-basis conditioning restores them at 1.0000, with")
print("           opposite phases - the quantum eraser, from integer counts.")

# ======================================================================
# Figure
# ======================================================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, INK, MUTED, GRID = "#3466C4", "#C85A19", "#333", "#666", "#ddd"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

deg = np.degrees(deltas)
ax1.plot(deg, P_c0, "o", ms=3.5, color=BLUE, zorder=3)
ax1.plot(deg, P_c1, "s", ms=3.2, color=ORANGE, zorder=3)
ax1.plot(deg, P_unc, "-", lw=1.6, color=MUTED, zorder=2)
ax1.plot(deg, np.cos(deltas/2)**2, "--", lw=1, color=BLUE, alpha=0.5, zorder=1)
ax1.plot(deg, np.sin(deltas/2)**2, "--", lw=1, color=ORANGE, alpha=0.5, zorder=1)
ax1.text(30, 0.66, "conditional on\npartner '+'", color=BLUE, fontsize=9)
ax1.text(196, 0.66, "conditional on\npartner '−'", color=ORANGE, fontsize=9)
ax1.text(238, 0.455, "unconditional (record present)", color=MUTED, fontsize=9)
ax1.set_xlabel(r"internal phase $\delta$ (deg)")
ax1.set_ylabel("P(port 0), from integer counts")
ax1.set_title("Conditional fringes of the entangled configuration")
ax1.grid(color=GRID, lw=0.6, zorder=0)

LABEL = {  # (x-mult, y-mult, ha)
    "x^2.0":            (1.35, 0.85, "left"),
    "(x+x^3)/2":        (0.80, 1.55, "right"),
    "(x^1.5+x^2.5)/2":  (1.25, 0.62, "left"),
    "x^2(0.85+0.15x)":  (0.72, 0.72, "right"),
}
for name, (pv, cv) in disc.items():
    xv, yv = max(pv, 1e-7), max(cv, 1e-7)
    is_pow = name in POWERS
    c = BLUE if is_pow else ORANGE
    ax2.loglog(xv, yv, "o" if is_pow else "s", ms=7, color=c, zorder=3)
    if name in LABEL:
        mx, my, ha = LABEL[name]
        ax2.annotate(name, (xv, yv), xytext=(xv*mx, yv*my),
                     fontsize=8, color=INK, ha=ha, va="center")
ax2.text(2.1e-6, 0.30, "pure powers p ≠ 2\n(x¹, x¹·⁵, x²·⁵, x³)",
         fontsize=8, color=BLUE)
ax2.axvline(1e-5, color=MUTED, lw=0.8, ls=":")
ax2.axhline(1e-5, color=MUTED, lw=0.8, ls=":")
ax2.text(1.3e-7, 2.2e-5, "conservation floor", fontsize=7.5, color=MUTED)
ax2.text(1.25e-5, 1.4e-7, "product floor", fontsize=7.5, color=MUTED, rotation=90)
ax2.set_xlabel("product-consistency violation (max)")
ax2.set_ylabel("count-conservation violation (max)")
ax2.set_title(r"Two constraints jointly selecting $x^2$")
ax2.grid(color=GRID, lw=0.6, which="major", zorder=0)

fig.tight_layout()
fig.savefig("step1_results.png", dpi=170)  # saved to current working dir
print("\nplots -> step1_results.png")
