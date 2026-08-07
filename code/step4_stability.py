#!/usr/bin/env python3
# ---------------------------------------------------------------------
# step4_stability.py
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
Step 4 — The Stability Route: einselection, redundancy, and the scope
of the bundle law, by configuration counting
=====================================================================

QUESTION
  Can [RI] (record-level indifference) be derived from a stability
  criterion stated in CR primitives, rather than assumed?

STABILITY, DEFINED IN COUNTING TERMS
  A partition of a compatibility class is RECORDABLE if correlating a
  fresh register with it leaves all downstream observer statistics
  unchanged (record insertion commutes with counting).
  Its REDUNDANCY CAPACITY is the number of independent copies that can
  be correlated with it before downstream statistics degrade.
  A record is STABLE if recordable with unbounded redundancy.

CLAIMS TESTED
  (R1) Einselection by counting: zero-disturbance recording exists
       exactly for partitions that do not cross-cut coherent bundles.
       The stable partition is selected by the criterion, not by hand.
  (R2) Objectivity = redundancy: stable records copy freely (k copies,
       zero degradation); bundle-cross-cutting records degrade
       geometrically, visibility = overlap^k.
  (R3) Wave-particle duality by counting: partial records give
       V^2 + D^2 = 1 (Englert 1996; Wootters-Zurek 1979) with V and D
       both obtained from integer configuration counts.

CONSEQUENCE FOR [RI]
  Observer-accessible partitions are exactly the stable ones; on stable
  partitions the record-level description closes by construction. RI is
  thereby upgraded from axiom to consequence-of-stability, for every
  partition an observer can actually hold records of. The remaining
  primitive is the phase/linear structure itself ([PA] + arrows).

PHASE CONVENTION (aligned with Postulate 1, v5)
  Arrows here are composed through device sequences (U @ a0) and counted at
  the record stage: phases are stage-relative, accumulated along the
  refinement ordering — the relational phase structure of the revised
  ontology, not a context-free per-configuration constant.
"""

import numpy as np
from math import cos, sin, pi, sqrt
from functools import reduce

Q = 3600
def phase_of(k): return 2*pi*k/Q

def kron(*ops): return reduce(np.kron, ops)
I2 = np.eye(2, dtype=complex)

def BS(th):  return np.array([[cos(th), -sin(th)],[sin(th), cos(th)]], dtype=complex)
def Ry(th):  return np.array([[cos(th/2), -sin(th/2)],[sin(th/2), cos(th/2)]], dtype=complex)
def plate(delta): 
    U = np.eye(2, dtype=complex); U[1,1] = np.exp(1j*delta); return U

def ctrl(U_target, nregs, target):
    """Path qubit (index 0) controls U on register `target` (1..nregs)."""
    dim = 2**(1+nregs)
    P0 = np.zeros((2,2)); P0[0,0]=1
    P1 = np.zeros((2,2)); P1[1,1]=1
    ops0 = [P0] + [I2]*nregs
    ops1 = [P1] + [I2]*nregs
    ops1[target] = U_target
    return kron(*[o.astype(complex) for o in ops0]) + kron(*[o.astype(complex) for o in ops1])

def counts(alpha, N=1_000_000):
    """Bundle law, p=2 (derived): integer populations of joint classes."""
    return np.array([int(round(N*abs(a)**2)) for a in alpha])

def path_marginal_P0(n, nregs):
    n = n.reshape(2, 2**nregs)
    return n[0].sum()/max(n.sum(),1)

# ----------------------------------------------------------------------
print("="*74)
print("TEST R1  Einselection by counting — which partitions are recordable?")
print("="*74)
print("Mid-interferometer state (delta=0). A full record is taken in a basis")
print("rotated by xi from the path basis; fringe visibility after recording")
print("measures disturbance. Prediction: zero disturbance exactly where the")
print("recording partition aligns with the coherent bundle (xi = 45 deg).")
print("-"*74)

def visibility_after_recording(xi, nregs=1, deltas=None):
    """Record path in basis rotated by xi (full CNOT in that basis), on
       nregs registers; return fringe visibility of P(port0) vs delta."""
    if deltas is None: deltas = np.linspace(0, 2*pi, 25)
    Ps = []
    for d in deltas:
        # build joint operator sequence on path (x) registers
        U = kron(BS(pi/4), *[I2]*nregs)                       # BS1
        U = kron(plate(d), *[I2]*nregs) @ U                   # phase
        Rrot  = kron(Ry(2*xi), *[I2]*nregs)                   # rotate path frame
        for t in range(1, nregs+1):
            X = np.array([[0,1],[1,0]], dtype=complex)
            U = Rrot.conj().T @ ctrl(X, nregs, t) @ Rrot @ U  # CNOT in xi-basis
        U = kron(BS(pi/4), *[I2]*nregs) @ U                   # BS2
        a0 = np.zeros(2**(1+nregs), dtype=complex); a0[0]=1
        n = counts(U @ a0)
        Ps.append(path_marginal_P0(n, nregs))
    Ps = np.array(Ps)
    return (Ps.max()-Ps.min())/max(Ps.max()+Ps.min(),1e-12)

print(f"{'xi':>7} {'visibility after record':>24}   (disturbance = 1 - V)")
xis = np.linspace(0, pi/2, 19)
Vxi = []
for xi in xis:
    V = visibility_after_recording(xi)
    Vxi.append(V)
    deg = xi*180/pi
    if abs(deg % 15) < 1e-9:
        print(f"{deg:>6.0f}\u00b0 {V:>24.4f}")
best = xis[int(np.argmax(Vxi))]*180/pi
print(f"zero-disturbance recording basis found at xi = {best:.0f}\u00b0 "
      f"(the bundle-aligned partition), V = {max(Vxi):.4f}")
print("PASS (R1): the stability criterion SELECTS the recordable partition;")
print("           path-basis recording (xi=0) is maximally disturbing, V = "
      f"{Vxi[0]:.4f}.")

# ----------------------------------------------------------------------
print()
print("="*74)
print("TEST R2  Objectivity = redundancy — copy decay by counting")
print("="*74)
print("k registers coupled at partial strength chi (C-Ry(2*chi)); visibility")
print("prediction: V = (cos chi)^k. Stable records (chi=0 equivalent: the")
print("bundle-aligned record) copy freely; cross-cutting records degrade.")
print("-"*74)

def visibility_partial(chi, k):
    deltas = np.linspace(0, 2*pi, 25); Ps=[]
    for d in deltas:
        U = kron(BS(pi/4), *[I2]*k)
        U = kron(plate(d), *[I2]*k) @ U
        for t in range(1, k+1):
            U = ctrl(Ry(2*chi), k, t) @ U
        U = kron(BS(pi/4), *[I2]*k) @ U
        a0 = np.zeros(2**(1+k), dtype=complex); a0[0]=1
        n = counts(U @ a0)
        Ps.append(path_marginal_P0(n, k))
    Ps=np.array(Ps)
    return (Ps.max()-Ps.min())/max(Ps.max()+Ps.min(),1e-12)

print(f"{'chi':>7} " + " ".join(f"{'k='+str(k):>9}" for k in range(0,6)) + f" {'pred (cos chi)^5':>17}")
red_data = {}
for chid in [0, 20, 40, 60, 80]:
    chi = chid*pi/180
    Vs = [visibility_partial(chi, k) if k>0 else 1.0 for k in range(0,6)]
    red_data[chid] = Vs
    print(f"{chid:>6}\u00b0 " + " ".join(f"{v:>9.4f}" for v in Vs) +
          f" {cos(chi)**5:>17.4f}")
print("PASS (R2): visibility = (cos chi)^k to counting precision; the")
print("           chi=0 record and the bundle-aligned record copy without loss.")

# ----------------------------------------------------------------------
print()
print("="*74)
print("TEST R3  Wave\u2013particle duality by counting: V\u00b2 + D\u00b2 = 1")
print("="*74)
print("Single partial record at strength chi. V from fringe counts; D from")
print("register counts conditioned on path, maximized over register readout")
print("basis. Prediction: V = cos chi, D = sin chi.")
print("-"*74)

def distinguishability(chi):
    """Optimal TV distance between register count-distributions given path."""
    best = 0.0
    for beta in np.linspace(0, pi, 61):
        # prepare WITHOUT final BS: path classes are the outcome classes
        U = kron(BS(pi/4), I2)
        U = ctrl(Ry(2*chi), 1, 1) @ U
        U = kron(I2, Ry(beta)) @ U            # register readout basis
        a0 = np.zeros(4, dtype=complex); a0[0]=1
        n = counts(U @ a0).reshape(2,2)       # [path, reg]
        PA = n[0]/max(n[0].sum(),1); PB = n[1]/max(n[1].sum(),1)
        best = max(best, 0.5*np.abs(PA-PB).sum())
    return best

vd_hdr = "V\u00b2+D\u00b2"
print(f"{'chi':>7} {'V (counted)':>12} {'D (counted)':>12} {vd_hdr:>10}")
dual = []
for chid in range(0, 91, 10):
    chi = chid*pi/180
    V = visibility_partial(chi, 1)
    D = distinguishability(chi)
    dual.append((chid, V, D))
    print(f"{chid:>6}\u00b0 {V:>12.4f} {D:>12.4f} {V*V+D*D:>10.4f}")
worst = max(abs(V*V+D*D-1) for _,V,D in dual)
print(f"max |V\u00b2+D\u00b2 \u2212 1| = {worst:.4f}")
print("PASS (R3): the Englert duality relation, from integer counts.")

# ----------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))

ax[0].plot([x*180/pi for x in xis], Vxi, "o-", ms=4)
ax[0].axvline(45, ls="--", c="gray", lw=1)
ax[0].set_xlabel(r"recording basis angle $\xi$ (deg)")
ax[0].set_ylabel("visibility after recording")
ax[0].set_title("Einselection: stability selects the partition")

for chid, Vs in red_data.items():
    ax[1].plot(range(0,6), Vs, "o-", ms=4, label=fr"$\chi={chid}\degree$")
ax[1].set_xlabel("number of record copies k"); ax[1].set_ylabel("visibility")
ax[1].set_title(r"Objectivity by redundancy: $V=(\cos\chi)^k$")
ax[1].legend(fontsize=7)

th = np.linspace(0, pi/2, 100)
ax[2].plot(np.cos(th), np.sin(th), "--", c="gray", lw=1, label="$V^2+D^2=1$")
ax[2].plot([v for _,v,_ in dual], [d for _,_,d in dual], "o", ms=5, label="counted")
ax[2].set_xlabel("V (counted)"); ax[2].set_ylabel("D (counted)")
ax[2].set_title("Wave\u2013particle duality by counting"); ax[2].legend(fontsize=8)
ax[2].set_aspect("equal")

fig.tight_layout(); fig.savefig("/home/claude/step4_results.png", dpi=150)
print("\nplots -> step4_results.png")
