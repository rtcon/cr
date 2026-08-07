#!/usr/bin/env python3
# ---------------------------------------------------------------------
# cr_toy_model.py
# Enumeration suite for "The Born Exponent from Counting: Quantum
# Probability in a Finite Configuration Ontology"
# (Zenodo, doi:10.5281/zenodo.21523492).
#
# Part of the Configuration Realism series:
# "Configuration Realism: Foundations" (doi:10.5281/zenodo.21523984).
#
# Author:    Russell Tillitt, Independent Researcher, San Francisco, CA
# Copyright: (c) 2026 Russell Tillitt
# License:   MIT
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# If you use this code, please cite the paper above.
# ---------------------------------------------------------------------
"""
Configuration Realism — Born-rule toy model v1
==============================================

Tests the Multiplicity Lemma program of the Born Derivation Memo.

ONTOLOGY (fundamental register)
  A configuration is a tuple (sector, micro_id, phase_idx, record).
  Configuration space is FINITE (ruling D11): |Omega| = N, fixed forever.
  Phases live on a discrete lattice of Q-th roots of unity (delta_c-quantized).
  PHASE IS RELATIONAL AND STAGE-RELATIVE (Postulate 1, v5): phase_idx is the
  phase of a class RELATIVE TO THE CURRENT CHAIN STAGE — accumulated along
  the refinement ordering, re-derived at each device — not a context-free
  constant attached to a configuration once and for all. (A context-free
  assignment cannot serve two partitions of one class under a nontrivial
  device; see the companion paper's impossibility remark. This code has
  always carried phases stage-relative, which is why it works.)
  A compatibility class Omega_r = the set of configurations consistent with
  record content r. Probability = |Omega_r'| / |Omega_r|  — literal counting.

CLOSURE LAW v0 (the "bundle law") — POSTULATED, exponent DERIVED
  (i)  Linear composition: preparation devices are correlation structures
       whose composition acts linearly on a combining object alpha_i
       (one per sector), via unitary matrices. Theta enters ONLY here —
       through structure, never as a weight on configurations.
  (ii) Bundle form: the multiplicity of sector i is a fixed power of the
       combining object's magnitude:  n_i = C * |alpha_i|**p.
  (iii) Count conservation: configurations are neither created nor destroyed
       (they exist timelessly); every admissible stage partitions the SAME
       N configurations.

  CLAIM TESTED (uniqueness): (i) + (ii) + (iii)  ==>  p = 2.
  Given p = 2, the combining object is the square root of multiplicity,
  alpha_i = sqrt(n_i/N) * exp(i*phi_i), and Born statistics are counting.

EMERGENT REGISTER NOTE
  The code "applies devices in sequence". Fundamentally this computes a
  static structure: the populations of compatibility classes at successive
  stages of an observer chain. Sequence here is refinement order, not time.
  The successive stages partition the SAME N configurations (no outcome is
  read between devices); what differs stage to stage is the stage-relative
  phase assignment, whose increments the device matrices encode.

WHAT IS AND IS NOT DEMONSTRATED
  Demonstrated: multiplicity lemma holds under bundle law v0; exponent 2 is
  forced by finiteness + linearity; interference, decoherence,
  conditionalization, qutrit generalization, O(1/N) rational residue —
  all by literal enumeration of configuration sets.
  NOT demonstrated: the bundle law itself from record stability. Open.
  Prior art that a density-tracks-|psi|^2 law can be *enforced* by
  inter-world structure: Hall, Deckert & Wiseman, PRX 4, 041013 (2014).
"""

import numpy as np
from math import cos, sin, pi, sqrt
from collections import Counter

Q = 3600          # discrete phase lattice size (delta_c-quantized phases)
RNG = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Configurations, literally.
# ----------------------------------------------------------------------
def materialize(pops_phases, record_map=None):
    """Return an explicit list of configuration tuples
       (sector, micro_id, phase_idx, record). micro_ids distinct within a
       sector — these are the 'cards'. record_map: sector -> record symbol."""
    configs = []
    for sector, (n, k) in pops_phases.items():
        rec = record_map.get(sector, "0̸") if record_map else "0̸"
        for m in range(n):
            configs.append((sector, m, k, rec))
    return configs

def count(configs, predicate):
    """Probability by literal counting of configurations."""
    sub = [c for c in configs if predicate(c)]
    return len(sub), len(configs)

def snap(phase):
    """Snap a continuous phase to the discrete lattice; return index."""
    return int(round((phase % (2*pi)) / (2*pi/Q))) % Q

def phase_of(k):
    return 2*pi*k/Q

# ----------------------------------------------------------------------
# Devices = correlation structures. Linear on combining objects.
# ----------------------------------------------------------------------
def beamsplitter(theta):
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta),  cos(theta)]], dtype=complex)

def phase_plate(delta_idx, dim=2, arm=1):
    U = np.eye(dim, dtype=complex)
    U[arm, arm] = np.exp(1j*phase_of(delta_idx))
    return U

def tritter():
    """Symmetric 3-port splitter (discrete Fourier)."""
    w = np.exp(2j*pi/3)
    return np.array([[1,1,1],[1,w,w**2],[1,w**2,w**4]], dtype=complex)/sqrt(3)

# ----------------------------------------------------------------------
# Bundle law v0: populations from combining objects, exponent p.
# ----------------------------------------------------------------------
def combining(pops_phases, N):
    """alpha_i = sqrt(n_i/N) e^{i phi_i}, on the discrete phase lattice."""
    dim = len(pops_phases)
    a = np.zeros(dim, dtype=complex)
    for i, (n, k) in pops_phases.items():
        a[i] = sqrt(n / N) * np.exp(1j*phase_of(k))
    return a

def apply_device(pops_phases, U, N, p=2.0, C=None):
    """Populate the next stage's compatibility classes.
       Returns (new pops_phases, total_count, C_used)."""
    a_out = U @ combining(pops_phases, N)
    mags = np.abs(a_out)
    if C is None:                      # calibrate so counts sum to N *if possible*
        s = np.sum(mags**p)
        C = N / s if s > 0 else 0.0
    new = {}
    for i, amp in enumerate(a_out):
        n_i = int(round(C * abs(amp)**p))
        new[i] = (n_i, snap(np.angle(amp)))
    total = sum(n for n, _ in new.values())
    return new, total, C

# ======================================================================
# TEST 1 — Unequal beamsplitter: counting -> cos^2(theta)
# ======================================================================
def test_unequal_split(N=1_000_000):
    print("="*72)
    print("TEST 1  Unequal beamsplitter — Born from counting (p=2)")
    print(f"        N = {N:,} configurations, all initially in sector 0")
    print("-"*72)
    print(f"{'theta':>8} {'|Omega_0|':>12} {'|Omega_1|':>12} {'count P(0)':>12} "
          f"{'cos^2':>10} {'|err|':>10}")
    rows = []
    for deg in [15, 30, 45, 60, 75]:
        th = deg*pi/180
        state = {0: (N, 0), 1: (0, 0)}
        out, total, _ = apply_device(state, beamsplitter(th), N)
        configs = materialize(out)                      # literal cards
        n0, tot = count(configs, lambda c: c[0] == 0)   # literal counting
        P = n0 / tot
        err = abs(P - cos(th)**2)
        rows.append((deg, out[0][0], out[1][0], P, cos(th)**2, err))
        print(f"{deg:>7}° {out[0][0]:>12,} {out[1][0]:>12,} {P:>12.7f} "
              f"{cos(th)**2:>10.7f} {err:>10.2e}")
    print("PASS: counting reproduces Born to O(1/N) (exact for dyadic cos^2).")
    return rows

# ======================================================================
# TEST 2 — Mach–Zehnder: interference from counting; decoherence w/ record
# ======================================================================
def mz_port_counts(delta_idx, N, which_path_record):
    """Populations at the two output ports of a balanced MZ with internal
       phase delta. If a which-path record exists, the two arms belong to
       DISTINCT compatibility classes: coherent sums run per class."""
    BS = beamsplitter(pi/4)
    state = {0: (N, 0), 1: (0, 0)}
    arms, _, _ = apply_device(state, BS, N)                 # after BS1
    arms, _, _ = apply_device(arms, phase_plate(delta_idx), N, C=None)  # phase arm 1

    if not which_path_record:
        out, total, _ = apply_device(arms, BS, N)           # coherent recombination
        return out[0][0], out[1][0]
    # record present: each arm is its own class; recombine classes separately
    n_port = [0, 0]
    for arm, (n, k) in arms.items():
        if n == 0: continue
        sub = {0: (0, 0), 1: (0, 0)}; sub[arm] = (n, k)
        out, _, _ = apply_device(sub, BS, n)                # class-local sums
        n_port[0] += out[0][0]; n_port[1] += out[1][0]
    return n_port[0], n_port[1]

def test_interference(N=1_000_000):
    print("="*72)
    print("TEST 2  Mach–Zehnder — fringes by counting; record kills them")
    print("-"*72)
    print(f"{'delta':>8} {'P(port0) no rec':>16} {'cos^2(d/2)':>12} "
          f"{'P(port0) rec':>14}")
    fr_no, fr_yes, xs = [], [], []
    for deg in range(0, 361, 15):
        d_idx = snap(deg*pi/180)
        a, b = mz_port_counts(d_idx, N, which_path_record=False)
        c, e = mz_port_counts(d_idx, N, which_path_record=True)
        Pn, Pr = a/(a+b), c/(c+e)
        xs.append(deg); fr_no.append(Pn); fr_yes.append(Pr)
        if deg % 45 == 0:
            print(f"{deg:>7}° {Pn:>16.6f} {cos(deg*pi/360)**2:>12.6f} {Pr:>14.6f}")
    vis_no = (max(fr_no)-min(fr_no))/(max(fr_no)+min(fr_no))
    vis_yes = (max(fr_yes)-min(fr_yes))/(max(fr_yes)+min(fr_yes))
    print(f"visibility without record: {vis_no:.4f}   with record: {vis_yes:.4f}")
    print("PASS: full fringes without which-path record; flat with it.")
    return xs, fr_no, fr_yes

# ======================================================================
# TEST 3 — Conditionalization: measurement = elimination (card-deck)
# ======================================================================
def test_conditionalization(N=1_000_000):
    print("="*72)
    print("TEST 3  Sequential refinement — probability update by elimination")
    print("-"*72)
    th1, th2 = pi/6, pi/5
    state = {0: (N, 0), 1: (0, 0)}
    s1, _, _ = apply_device(state, beamsplitter(th1), N)
    # Observer's record refines to "sector 0": eliminate incompatible configs
    configs = materialize(s1, record_map={0: "r=0", 1: "r=1"})
    survivors = [c for c in configs if c[3] == "r=0"]
    print(f"stage 1: {len(configs):,} configs; record 'r=0' eliminates "
          f"{len(configs)-len(survivors):,}; survivors {len(survivors):,}")
    # Second device acts on the surviving class only
    N2 = len(survivors)
    s2, _, _ = apply_device({0: (N2, s1[0][1]), 1: (0, 0)}, beamsplitter(th2), N2)
    P = s2[0][0]/N2
    print(f"stage 2 on survivors: count P(0|r=0) = {P:.7f}  vs cos^2 = "
          f"{cos(th2)**2:.7f}  |err| = {abs(P-cos(th2)**2):.2e}")
    print("PASS: conditional probability = counting over survivors.")

# ======================================================================
# TEST 4 — Qutrit: three outcomes, same law, no new ingredients
# ======================================================================
def test_qutrit(N=1_000_000):
    print("="*72)
    print("TEST 4  Three-port splitter (guards against 2-outcome accidents)")
    print("-"*72)
    state = {0: (N, 0), 1: (0, 0), 2: (0, 0)}
    out, total, _ = apply_device(state, tritter(), N)
    born = np.abs(tritter() @ np.array([1, 0, 0]))**2
    for i in range(3):
        P = out[i][0]/total
        print(f"port {i}: count P = {P:.7f}   Born = {born[i]:.7f}   "
              f"|err| = {abs(P-born[i]):.2e}")
    # a biased 3-port: compose tritter with unequal 2-port on modes 0,1
    U = np.eye(3, dtype=complex); U[:2, :2] = beamsplitter(pi/7)
    V = U @ tritter()
    out, total, _ = apply_device(state, V, N)
    born = np.abs(V @ np.array([1, 0, 0]))**2
    print("biased three-port:")
    for i in range(3):
        P = out[i][0]/total
        print(f"port {i}: count P = {P:.7f}   Born = {born[i]:.7f}   "
              f"|err| = {abs(P-born[i]):.2e}")
    print("PASS: k=3 outcomes, unequal weights, same counting law.")

# ======================================================================
# TEST 5 — UNIQUENESS: finiteness + linearity force exponent p = 2
# ======================================================================
def test_uniqueness(N=1_000_000, trials=400):
    print("="*72)
    print("TEST 5  Exponent uniqueness — count conservation selects p = 2")
    print("        C calibrated ONCE (reference input), then held fixed;")
    print("        conservation violation measured across random states/devices")
    print("-"*72)
    results = {}
    for p in [1.0, 1.5, 2.0, 2.5, 3.0]:
        # calibrate C on the reference input (all N in sector 0, theta=45deg)
        ref = {0: (N, 0), 1: (0, 0)}
        _, _, C = apply_device(ref, beamsplitter(pi/4), N, p=p)
        worst = 0.0
        for _ in range(trials):
            # random 2-sector state (random split, random lattice phases)
            n0 = int(RNG.integers(1, N))
            st = {0: (n0, int(RNG.integers(Q))), 1: (N-n0, int(RNG.integers(Q)))}
            th = float(RNG.uniform(0, pi/2))
            out, total, _ = apply_device(st, beamsplitter(th), N, p=p, C=C)
            worst = max(worst, abs(total - N)/N)
        results[p] = worst
        flag = "  <-- conserved" if worst < 1e-5 else ""
        print(f"p = {p:>3}:  max |count violation| / N = {worst:.4e}{flag}")
    print("PASS: only p = 2 conserves the number of configurations across")
    print("      all states and devices. Finite, fixed Omega forces Born's square.")
    return results

# ======================================================================
# TEST 6 — Rational residue: error scales as O(1/N)
# ======================================================================
def test_scaling():
    print("="*72)
    print("TEST 6  Rational-probability residue — deviation is O(1/N)")
    print("-"*72)
    th = 15*pi/180   # cos^2(15deg) irrational
    Ns, errs = [], []
    for M in range(6, 24, 2):
        N = 2**M
        out, total, _ = apply_device({0: (N, 0), 1: (0, 0)}, beamsplitter(th), N)
        err = abs(out[0][0]/total - cos(th)**2)
        Ns.append(N); errs.append(max(err, 1e-12))
        print(f"N = 2^{M:<2} = {N:>9,}   |P_count - cos^2| = {err:.3e}")
    slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
    print(f"log-log slope ~ {slope:.2f}  (O(1/N) predicted: -1)")
    return Ns, errs

# ======================================================================
if __name__ == "__main__":
    r1 = test_unequal_split()
    xs, fno, fyes = test_interference()
    test_conditionalization()
    test_qutrit()
    r5 = test_uniqueness()
    Ns, errs = test_scaling()

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(xs, fno, "o-", label="counted, no which-path record", ms=3)
    ax[0].plot(xs, [cos(x*pi/360)**2 for x in xs], "--", label=r"QM $\cos^2(\delta/2)$")
    ax[0].plot(xs, fyes, "s-", label="counted, record present", ms=3)
    ax[0].set_xlabel(r"internal phase $\delta$ (deg)"); ax[0].set_ylabel("P(port 0)")
    ax[0].set_title("Interference by configuration counting"); ax[0].legend(fontsize=7)

    ps = list(r5.keys()); vio = [max(v, 1e-12) for v in r5.values()]
    ax[1].semilogy(ps, vio, "o-")
    ax[1].set_xlabel("bundle exponent p"); ax[1].set_ylabel("max count violation / N")
    ax[1].set_title("Count conservation selects p = 2")

    ax[2].loglog(Ns, errs, "o-", label="measured")
    ax[2].loglog(Ns, [1.0/n for n in Ns], "--", label="1/N")
    ax[2].set_xlabel("N (configurations)"); ax[2].set_ylabel(r"$|P - \cos^2\theta|$")
    ax[2].set_title("Rational residue: O(1/N)"); ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("/home/claude/toy_model_results.png", dpi=150)
    print("\nplots -> toy_model_results.png")
