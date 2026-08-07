# ---------------------------------------------------------------------
# schrodinger_by_counting.py
# Enumeration suite for "Schrodinger by Counting: Wavepacket Dynamics
# in a Finite Configuration Ontology" (Zenodo, 2026).
#   [TODO: add doi once deposited]
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
Schrödinger by Counting — enumeration model in the discipline of the
Configuration Realism companion suites (cr_toy_model.py / chsh_by_counting.py).

Discipline, stated in advance:
  * The counting rule is fixed before any case is computed: populations are
    n_x = apportion(N |beta_x|^2), integer-apportioned by largest remainder
    (ties broken by cell index; the apportionment rule is a MODEL POSTULATE,
    as in the CHSH companion).
  * Probability is a ratio of integer counts, P(x) = n_x / N. Nothing else.
  * Devices are correlation structures: local, unitary, with phase increments
    drawn from the 3,600-value lattice (0.1 degree), as in the companions.
    Elementary moves act only between ADJACENT cells (nearest-neighbour
    beamsplitters) — the locality of Postulate 4's move set.
  * Stage-relative class phases are DERIVED at each stage from the device's
    action on the previous stage's arrows (never assigned context-free).
  * The stage state is the record-level census: an integer population n_x and
    a derived arrow direction per cell class. The instantiated phase texture
    is an admissible one, not a derived one (companion open problem).
  * Sum(n_x) = N is asserted EXACTLY at every stage: the counting form of
    unitarity (norm preservation as conservation of cardinality).

What is computed:
  Suite F  — free packet: counted spreading sigma^2(t) vs the Schrödinger
             prediction sigma0^2 + (t / (2 m* sigma0))^2, m* = 1/(2 eta).
  Suite D  — drifting packet: counted <x>(t) vs group velocity 2 eta sin k0.
  Suite H  — harmonic well: counted <x>(t) vs x_c + A cos(w t), w = sqrt(2 eta kappa).
  Suite E  — the literal edge: counted profile terminates (population zero)
             where the continuum Gaussian merely thins — prediction P1 in
             position space.
  Suite R  — residue: L1 distance between counted P and the exact discrete
             amplitudes at fixed t, swept over N: expected O(1/N).

The device (one refinement step = one 'tick'):
  U = [potential phase layer] o [odd-pair beamsplitters] o [even-pair beamsplitters]
  Beamsplitter angle eta and per-cell potential phases phi_x = -V_x are
  rounded ONCE to the 3,600 lattice and held fixed (device increments live on
  the lattice; derived arrow directions are exact sums, as in the companions).
"""

import numpy as np
import json

PHASE_LATTICE = 3600  # 0.1 degree

def lat(angle_rad):
    """Round a device phase increment to the 3,600-value lattice."""
    step = 2 * np.pi / PHASE_LATTICE
    return np.round(angle_rad / step) * step

# ----------------------------------------------------------------------
# The counting rule (fixed in advance)
# ----------------------------------------------------------------------

def apportion(weights, N):
    """Largest-remainder apportionment of N among integer populations with
    target proportions `weights` (which sum to 1 up to float error).
    Ties broken by cell index (deterministic). MODEL POSTULATE."""
    target = weights * N
    base = np.floor(target).astype(np.int64)
    short = int(N - base.sum())
    if short > 0:
        frac = target - base
        # stable argsort descending by fraction, ascending by index on ties
        order = np.argsort(-frac, kind="stable")
        base[order[:short]] += 1
    return base

# ----------------------------------------------------------------------
# The device
# ----------------------------------------------------------------------

def bs_pairs(psi, eta, offset):
    """Nearest-neighbour beamsplitter layer on pairs (offset, offset+1), ...
    Periodic. [a,b] -> [cos(eta) a + i sin(eta) b, i sin(eta) a + cos(eta) b]."""
    L = psi.shape[0]
    out = psi.copy()
    c, s = np.cos(eta), np.sin(eta)
    idx_a = np.arange(offset, L, 2) % L
    idx_b = (idx_a + 1) % L
    a, b = psi[idx_a], psi[idx_b]
    out[idx_a] = c * a + 1j * s * b
    out[idx_b] = 1j * s * a + c * b
    return out

def make_step(L, eta_lat, phi_lat=None):
    """One refinement step acting on the arrow vector."""
    phase = np.exp(1j * phi_lat) if phi_lat is not None else None
    def step(psi):
        psi = bs_pairs(psi, eta_lat, 0)
        psi = bs_pairs(psi, eta_lat, 1)
        if phase is not None:
            psi = psi * phase
        return psi
    return step

# ----------------------------------------------------------------------
# The two evolutions
# ----------------------------------------------------------------------

def counted_at_measurement(alpha0, N, step, T):
    """MODEL A (primary; the ontology's reading). The T-step chain is ONE
    composite device standing between the preparation stage and the
    measurement stage: no record distinguishes cells at intermediate stages,
    so no cell partition exists there and no intermediate census is taken
    (classes are record-defined; Postulate 3). Arrows are carried exactly by
    the closure structure; the counting rule is applied ONCE per readout
    stage: n_x(t) = apportion(N |psi_x(t)|^2). Returns (T+1, L) populations,
    each row an independent terminal measurement of the same chain."""
    psi = alpha0.astype(np.complex128)
    L = alpha0.shape[0]
    traj = np.empty((T + 1, L), dtype=np.int64)
    w = np.abs(psi) ** 2
    traj[0] = apportion(w / w.sum(), N)
    for t in range(1, T + 1):
        psi = step(psi)
        w = np.abs(psi) ** 2
        n = apportion(w / w.sum(), N)
        assert n.sum() == N
        traj[t] = n
    return traj

def counted_census(alpha0, N, step, T):
    """MODEL B (exploratory contrast; goes beyond what the ontology
    licenses). An integer census is taken at EVERY stage — as if a partition
    existed there — and the next stage's arrow magnitudes are the census read
    back through the derived rule, |alpha_x| = sqrt(n_x/N); directions are
    the derived complex sums. Tail classes (populations of a few) inject
    amplitude granularity ~1/sqrt(N n_x), so this variant's residues scale
    WORSE than 1/N — which is the counting model's own demonstration that
    partitions must not be imputed to unrecorded stages."""
    L = alpha0.shape[0]
    n = apportion(np.abs(alpha0) ** 2, N)
    theta = np.angle(alpha0)
    traj = np.empty((T + 1, L), dtype=np.int64)
    traj[0] = n
    for t in range(1, T + 1):
        alpha = np.sqrt(n / N) * np.exp(1j * theta)
        beta = step(alpha)
        w = np.abs(beta) ** 2
        w = w / w.sum()          # guard float drift; unitarity holds to 1e-15
        n = apportion(w, N)
        assert n.sum() == N      # counting form of unitarity: exact, every stage
        theta = np.angle(beta)
        traj[t] = n
    return traj

def exact_evolution(alpha0, step, T):
    """The same device chain on exact complex arrows (no apportionment):
    the reference the counted populations are compared against."""
    psi = alpha0.astype(np.complex128)
    traj = np.empty((T + 1, alpha0.shape[0]))
    traj[0] = np.abs(psi) ** 2
    for t in range(1, T + 1):
        psi = step(psi)
        traj[t] = np.abs(psi) ** 2
    return traj

# ----------------------------------------------------------------------
# Preparations (correlation structures)
# ----------------------------------------------------------------------

def gaussian_packet(L, x0, sigma0, k0=0.0):
    x = np.arange(L)
    psi = np.exp(-((x - x0) ** 2) / (4 * sigma0 ** 2)) * np.exp(1j * k0 * x)
    return psi / np.linalg.norm(psi)

def moments(P, L):
    x = np.arange(L)
    m1 = (P * x).sum(axis=-1)
    m2 = (P * x ** 2).sum(axis=-1)
    return m1, m2 - m1 ** 2

# ----------------------------------------------------------------------
# Suites
# ----------------------------------------------------------------------

def run_all(outdir="."):
    results = {}
    L = 128
    eta = lat(np.deg2rad(11.5))          # beamsplitter angle, on-lattice
    m_star = 1.0 / (2 * eta)             # effective mass of the hopping limit

    # ---- Suite F: free spreading -------------------------------------
    N = 4_200_000
    sigma0, x0, T = 6.0, 64, 360
    a0 = gaussian_packet(L, x0, sigma0)
    stepF = make_step(L, eta)
    cnt = counted_at_measurement(a0, N, stepF, T)
    ex = exact_evolution(a0, stepF, T)
    Pc, Pe = cnt / N, ex
    t = np.arange(T + 1)
    _, var_c = moments(Pc, L)
    _, var_e = moments(Pe, L)
    var_s = sigma0 ** 2 + (t / (2 * m_star * sigma0)) ** 2   # Schrödinger
    results["F"] = dict(
        N=N, T=T, sigma0=sigma0, eta=float(eta), m_star=float(m_star),
        var_counted_final=float(var_c[-1]),
        var_exact_final=float(var_e[-1]),
        var_schrodinger_final=float(var_s[-1]),
        max_L1_vs_exact=float(np.abs(Pc - Pe).sum(axis=1).max()),
    )
    F_data = (t, var_c, var_e, var_s, Pc, Pe)

    # ---- Suite D: drift ----------------------------------------------
    k0 = 0.3
    x0d, Td = 40, 300
    a0d = gaussian_packet(L, x0d, sigma0, k0=k0)
    cntd = counted_at_measurement(a0d, N, stepF, Td)
    exd = exact_evolution(a0d, stepF, Td)
    td = np.arange(Td + 1)
    mean_c, _ = moments(cntd / N, L)
    mean_e, _ = moments(exd, L)
    v_group = 2 * eta * np.sin(k0)
    results["D"] = dict(
        k0=k0, v_group=float(v_group),
        v_counted=float((mean_c[-1] - mean_c[0]) / Td),
        v_exact=float((mean_e[-1] - mean_e[0]) / Td),
    )
    D_data = (td, mean_c, mean_e, x0d, v_group)

    # ---- Suite H: harmonic well --------------------------------------
    kappa = 0.001
    omega = np.sqrt(2 * eta * kappa)
    period = 2 * np.pi / omega
    xc, amp = 64, 8
    sig_coh = (m_star * omega) ** -0.5           # coherent-state width
    Th = int(round(2 * period))
    xg = np.arange(L)
    phi = lat(-0.5 * kappa * (xg - xc) ** 2)     # potential increments, on-lattice
    stepH = make_step(L, eta, phi)
    a0h = gaussian_packet(L, xc - amp, sig_coh)
    cnth = counted_at_measurement(a0h, N, stepH, Th)
    exh = exact_evolution(a0h, stepH, Th)
    th = np.arange(Th + 1)
    mh_c, _ = moments(cnth / N, L)
    mh_e, _ = moments(exh, L)
    pred = xc - amp * np.cos(omega * th)
    results["H"] = dict(
        kappa=kappa, omega=float(omega), period=float(period), T=Th,
        max_dev_counted_vs_cos=float(np.abs(mh_c - pred).max()),
        max_dev_exact_vs_cos=float(np.abs(mh_e - pred).max()),
        max_L1_vs_exact=float(np.abs(cnth / N - exh).sum(axis=1).max()),
    )
    H_data = (th, mh_c, pred)

    # ---- Suite E: the literal edge -----------------------------------
    tE = 200
    occ = np.nonzero(cnt[tE])[0]
    results["E"] = dict(
        t=tE, N=N,
        support=[int(occ.min()), int(occ.max())],
        support_cells=int(occ.size),
        outermost_populations=[int(cnt[tE][occ.min()]), int(cnt[tE][occ.max()])],
        continuum_tail_at_support_edge=float(Pe[tE][occ.max()]),
    )
    E_data = (cnt[tE] / N, Pe[tE], occ)

    # ---- Suite R: residue scaling ------------------------------------
    Ns = [10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7]
    TR = 200
    a0r = gaussian_packet(L, x0, sigma0)
    exr = exact_evolution(a0r, stepF, TR)[-1]
    L1_A, L1_B = [], []
    for Nr in Ns:
        cA = counted_at_measurement(a0r, Nr, stepF, TR)[-1] / Nr
        cB = counted_census(a0r, Nr, stepF, TR)[-1] / Nr
        L1_A.append(float(np.abs(cA - exr).sum()))
        L1_B.append(float(np.abs(cB - exr).sum()))
    slope_A = np.polyfit(np.log10(Ns), np.log10(L1_A), 1)[0]
    slope_B = np.polyfit(np.log10(Ns), np.log10(L1_B), 1)[0]
    results["R"] = dict(Ns=Ns, L1_modelA=L1_A, L1_modelB=L1_B,
                        slope_modelA=float(slope_A), slope_modelB=float(slope_B), T=TR)

    with open(f"{outdir}/schrodinger_by_counting_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results, (F_data, D_data, H_data, E_data, (Ns, L1_A, L1_B, slope_A, slope_B)), (L, eta, m_star, N)

if __name__ == "__main__":
    res, _, _ = run_all()
    print(json.dumps(res, indent=2))
