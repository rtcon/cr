# ---------------------------------------------------------------------
# chsh_by_counting.py
# Enumeration suite for "CHSH by Counting: Bell Violation in a Finite
# Configuration Ontology" (Zenodo, doi:10.5281/zenodo.21523901).
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
CHSH by Counting
================
Verification-suite extension for Configuration Realism (companion to
cr_toy_model.py, step1_multiplicativity.py, step4_stability.py).

Discipline (fixed before any case was computed):
  * Configurations are materialized as explicit arrays (sector, phase-index);
    phases live on a lattice of 3,600 values. Phases are STAGE-RELATIVE
    (Postulate 1 of the Born paper): the phase of a class relative to the current chain
    stage, accumulated along the refinement ordering. No micro-ids: the
    ontology supplies no cross-preparation identity tags, and their absence
    is principled -- identity tags across setting-preparations would reopen
    counterfactual definiteness, which the framework denies.
  * Pre-device arrows are computed by literally summing e^{iS(omega)} over
    the materialized configuration list; the device stage then applies the
    COUNTING RULE DERIVED IN THE COMPANION PAPER (populations = N|beta|^2,
    integer-apportioned). What this script demonstrates is that the derived
    rule, executed by enumeration, reproduces CHSH -- a computation under
    the stipulated rule, not an independent derivation of it.
  * Integer apportionment is MARGINS-FIRST: each wing's marginal counts
    are computed from that wing's marginal weights alone, so marginal
    setting-independence (the counting form of no-signaling) holds by
    construction; joint cells are then filled subject to both margins.
    (A joint largest-remainder rule can distort a wing's marginal at
    remainder ties depending on the JOINT settings -- an O(1/N)
    no-signaling leak; margins-first avoids it by construction. The
    apportionment rule itself remains a postulate of the model, flagged
    as such in the paper.)
  * Analyzer settings enter only through the device map (correlation
    structure); all correlators are ratios of integer counts.

Preparation: singlet-type correlation structure.
  Two sectors, 'ud' and 'du', populated N/2 each; the singlet's relative
  minus sign is a stage-relative phase difference of pi between the sectors
  (lattice index 1800 of 3600). No record distinguishes the sectors: one
  compatibility class (cf. the two-slit case) -- unless a sector register
  is inserted (the decohered control run).

Checks:
  1. Aligned bases (a == b): same-spin populations are zero.
  2. Skew bases: same-spin population = N * sin^2((a-b)/2) / 2 per class.
  3. CHSH at optimal settings: | |S| - 2*sqrt(2) | = O(1/N) from counts.
  4. Control: sector record inserted => per-class sums, |S| <= 2.
  5. No-signaling by counting: each wing's marginal counts are IDENTICAL
     across the other wing's settings (exact, by the margins-first rule).
  6. Residue scaling ~ O(1/N).
"""

import numpy as np

LATTICE = 3600  # phase lattice, as in the companion suites


# ----------------------------------------------------------------------
# Ontology helpers
# ----------------------------------------------------------------------

def materialize_preparation(N):
    """Singlet-type class: two sectors, N/2 each, stage-relative phase
    difference pi. Returns (sector, phase_idx) arrays of length N."""
    assert N % 2 == 0
    sector = np.repeat(np.array([0, 1], dtype=np.int8), N // 2)
    phase_idx = np.where(sector == 0, 0, LATTICE // 2).astype(np.int32)
    return sector, phase_idx


def arrow_of(mask, phase_idx, scale):
    """Coherent sum over a materialized sub-list, literally summed."""
    S = 2.0 * np.pi * phase_idx[mask] / LATTICE
    return np.sum(np.exp(1j * S)) / scale


def analyzer(theta):
    """Single-wing device: rotation carrying (u, d) arrows to (+, -) arrows.
    Settings enter only through this correlation structure."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, s], [-s, c]])


def apportion_margins_first(N, w):
    """Integer populations for the four joint classes (++, +-, -+, --).

    Margins first: each wing's marginal count is computed from that wing's
    marginal weights ALONE -- Alice's counts cannot depend on Bob's setting,
    nor vice versa (no-signaling by construction). The joint table is then
    filled subject to both margins, with one free cell rounded.
    Counts sum exactly to N: configurations are neither created nor
    destroyed."""
    w = np.asarray(w, dtype=float)
    nA_plus = int(round(N * (w[0] + w[1])))   # Alice marginal: Alice weights only
    nB_plus = int(round(N * (w[0] + w[2])))   # Bob marginal: Bob weights only
    npp = int(round(N * w[0]))                # one free cell
    npm = nA_plus - npp
    nmp = nB_plus - npp
    nmm = N - npp - npm - nmp
    counts = np.array([npp, npm, nmp, nmm], dtype=np.int64)
    assert counts.min() >= 0 and counts.sum() == N
    return counts


# ----------------------------------------------------------------------
# Measurement by counting
# ----------------------------------------------------------------------

def joint_counts_coherent(N, a, b):
    """No sector record: the two sectors are one compatibility class.
    The composed device re-partitions the class; arrows from the two
    sectors combine coherently inside each outcome class (cross terms
    licensed), populations follow the derived counting rule."""
    sector, phase_idx = materialize_preparation(N)

    # Parent-relative scale A_r fixed by count conservation (Remark
    # 'normalization is forced'): A_r^2 = sum over classes |sum e^{iS}|^2.
    sums = np.array([np.sum(np.exp(2j * np.pi * phase_idx[sector == k] / LATTICE))
                     for k in (0, 1)])
    A = np.sqrt(np.sum(np.abs(sums) ** 2))

    # Pre-device arrow vector over sectors (uu, ud, du, dd) -- literal sums.
    alpha = np.zeros(4, dtype=complex)
    alpha[1] = arrow_of(sector == 0, phase_idx, A)  # 'ud'
    alpha[2] = arrow_of(sector == 1, phase_idx, A)  # 'du'

    U = np.kron(analyzer(a), analyzer(b))
    beta = U @ alpha  # device = linear re-partitioning of the same class

    n = apportion_margins_first(N, np.abs(beta) ** 2)

    # Materialize the post-device outcome roster and count it back:
    # probabilities are ratios of list lengths.
    outcome = np.repeat(np.arange(4, dtype=np.int8), n)
    counts = np.bincount(outcome, minlength=4).astype(np.int64)
    assert counts.sum() == N
    return counts  # order: ++, +-, -+, --


def joint_counts_decohered(N, a, b):
    """Sector record present (which-path analog): each sector is its own
    compatibility class; sums are taken per class; cross terms never form."""
    counts = np.zeros(4, dtype=np.int64)
    U = np.kron(analyzer(a), analyzer(b))
    for amp_index in (1, 2):  # sector 'ud', sector 'du'
        alpha = np.zeros(4, dtype=complex)
        alpha[amp_index] = 1.0  # per-class arrow, per-class (parent) scale
        beta = U @ alpha
        counts += apportion_margins_first(N // 2, np.abs(beta) ** 2)
    assert counts.sum() == N
    return counts


def correlator(counts):
    """E = (n++ + n-- - n+- - n-+) / N  -- a ratio of integer counts."""
    sign = np.array([+1, -1, -1, +1])
    return float(np.dot(sign, counts)) / counts.sum()


def wing_marginals(counts):
    """(Alice + count, Bob + count) from the joint table."""
    return counts[0] + counts[1], counts[0] + counts[2]


def chsh(N, joint_counts, angles):
    a, ap, b, bp = angles
    E = {pair: correlator(joint_counts(N, *pair))
         for pair in [(a, b), (a, bp), (ap, b), (ap, bp)]}
    S = E[(a, b)] + E[(ap, b)] + E[(ap, bp)] - E[(a, bp)]
    return S, E


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------

if __name__ == "__main__":
    OPT = (0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4)  # a, a', b, b'
    TSIRELSON = 2.0 * np.sqrt(2.0)
    N = 4_200_000

    print("=" * 68)
    print("1. Structural exclusion at aligned bases (a = b = 0)")
    c = joint_counts_coherent(N, 0.0, 0.0)
    print(f"   counts (++, +-, -+, --): {c.tolist()}")
    print(f"   same-spin population: {c[0] + c[3]}   (aligned-basis exclusion)")

    print("=" * 68)
    print("2. Skew bases: same-spin populations exist (a=0, b=45 deg)")
    c = joint_counts_coherent(N, 0.0, np.pi / 4)
    expect = N * np.sin(np.pi / 8) ** 2 / 2
    print(f"   counts (++, +-, -+, --): {c.tolist()}")
    print(f"   n(++) = {c[0]}   vs  N sin^2((a-b)/2)/2 = {expect:.1f}  (per class)")

    print("=" * 68)
    print(f"3. CHSH by counting, coherent class, N = {N:,}")
    S, E = chsh(N, joint_counts_coherent, OPT)
    for pair, e in E.items():
        deg = tuple(round(np.degrees(x)) for x in pair)
        print(f"   E{deg} = {e:+.7f}")
    print(f"   |S| = {abs(S):.7f}   Tsirelson 2*sqrt(2) = {TSIRELSON:.7f}"
          f"   deviation = {abs(S) - TSIRELSON:+.2e}")

    print("=" * 68)
    print("4. Control: sector record inserted (per-class sums, no cross terms)")
    S2, E2 = chsh(N, joint_counts_decohered, OPT)
    for pair, e in E2.items():
        deg = tuple(round(np.degrees(x)) for x in pair)
        print(f"   E{deg} = {e:+.7f}")
    print(f"   |S| = {abs(S2):.7f}   (below the local bound 2)")

    print("=" * 68)
    print("5. No-signaling by counting: wing marginals across other-wing settings")
    ok = True
    for n_test in (20, 2_000, 4_200_000):   # includes the small-N tie regime
        for a in (0.0, np.pi / 2):
            mA = set(); mB_by_b = {}
            for b in (np.pi / 4, 3 * np.pi / 4, 1.234):
                cts = joint_counts_coherent(n_test if n_test % 2 == 0 else n_test+1, a, b)
                mA.add(wing_marginals(cts)[0])
            ok &= (len(mA) == 1)
        for b in (np.pi / 4, 3 * np.pi / 4):
            mB = set()
            for a in (0.0, np.pi / 2, 0.777):
                cts = joint_counts_coherent(n_test if n_test % 2 == 0 else n_test+1, a, b)
                mB.add(wing_marginals(cts)[1])
            ok &= (len(mB) == 1)
        print(f"   N = {n_test:>9,}: wing marginals setting-independent: {ok}")
    print(f"   PASS (no-signaling exact, by the margins-first rule)" if ok
          else "   FAIL")

    print("=" * 68)
    print("6. Residue scaling |  |S| - 2*sqrt(2) |  vs N")
    Ns = [2_000, 20_000, 200_000, 2_000_000, 20_000_000]
    resid = []
    for n_ in Ns:
        S_, _ = chsh(n_, joint_counts_coherent, OPT)
        r = abs(abs(S_) - TSIRELSON)
        resid.append(r)
        above = "above" if abs(S_) > TSIRELSON else "below"
        print(f"   N = {n_:>10,}   |S| = {abs(S_):.9f}   residue = {r:.3e}  ({above} Tsirelson)")
    lr = np.polyfit(np.log10(Ns), np.log10(np.maximum(resid, 1e-16)), 1)
    print(f"   log-log slope: {lr[0]:.2f}   (O(1/N) predicts ~ -1)")
    print("   NOTE: residue sign and size are conditional on the apportionment")
    print("   rule (a model postulate); the O(1/N) scale is not.")
