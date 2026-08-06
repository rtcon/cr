# Schrödinger by Counting — Enumeration Results

**Model:** `schrodinger_by_counting.py` · **Figure:** `schrodinger_results.png` · **Raw numbers:** `schrodinger_by_counting_results.json`

Discipline as in the companion suites: counting rule fixed before any case was computed; devices are local (nearest-neighbour beamsplitters, angle η = 11.5° on the 3,600-value phase lattice; potential phases per-cell increments on the same lattice); populations integer-apportioned by largest remainder (apportionment rule a model postulate, as in the CHSH paper); Σnₓ = N asserted exactly at every readout — the counting form of unitarity, tested exactly.

**The construction.** One refinement step = even-pair beamsplitters ∘ odd-pair beamsplitters ∘ (optional) per-cell potential phase layer, on L = 128 cells, periodic. For small η this brickwork realizes the hopping Hamiltonian H = −2η cos k + V(x): effective mass m* = 1/(2η) — literally the inverse hopping rate, "resistance to record restructuring" as §13.1's dictionary asserts — and V(x) a cell-dependent closure phase cost. Both identifications fall out of the move set rather than being posited separately.

## Results (N = 4.2 × 10⁶ except where swept)

| Suite | Quantity | Counted | Prediction | Agreement |
|---|---|---|---|---|
| F (free) | σ²(360) | 183.866 | Schrödinger σ₀² + (t/2m*σ₀)²: 181.03 | 1.6% (lattice/Trotter corr.); counted vs exact arrows: 8.6 × 10⁻⁶ (L1) |
| D (drift) | v | 0.1195212 | exact-device 0.1195214; 2η sin k₀ = 0.11863 | 7 sig. figs vs device; 0.75% Trotter corr. vs formula |
| H (harmonic) | ⟨x⟩(t) | oscillates | x_c − A cos(ωt), ω = √(2ηκ) | max dev 1.6 cells / amplitude 8 over 2 periods (band anharmonicity at k_max ≈ 0.4); counted vs exact: 6.2 × 10⁻⁶ (L1) |
| E (edge) | support at t=200 | cells [19, 109]; outermost populations = 1 | continuum Gaussian: nonzero everywhere | tail terminates exactly where N·P < 1 (continuum tail at edge: 1.7 × 10⁻⁷ ≈ 0.7/N) |
| R (residue) | L1 vs exact at t=200 | slope **−0.94** over N = 10³–10⁷ | O(1/N) | Model A confirms the P1 scaling for dynamics |

## Two findings beyond "it works"

**1. The literal edge (Suite E).** In continuum QM a Gaussian packet's tails are nonzero at every distance. In the counting model the profile has finite support: populations fall to a few, then one, then zero, and the support edge sits exactly where the Born value crosses 1/N — no configuration in the class places the particle beyond it. This is prediction P1 rendered in position space: the counting world's particle is, with population zero, definitely *not* almost everywhere. Candidate for a sentence in §15.

**2. Partitions only where records are (Suite R) — the conceptual result.** Two counting models were run. **Model A** treats the T-step chain as one composite device between preparation and measurement — no record distinguishes cells at intermediate stages, so per Postulate 3 no cell partition exists there, and the counting rule is applied once, at the measurement stage. Residues scale as O(1/N) (slope −0.94). **Model B** imputes an integer census to every intermediate stage; tail classes with populations of a few inject amplitude granularity ~1/√(N·nₓ), and residues degrade to slope −0.62. The record-relative definition of classes — a philosophical-looking clause of the ontology — is thereby worth a factor of ~√N in dynamical accuracy, and the enumeration exhibits it. The ontology's own bookkeeping discipline (no partition without a record) is what protects the O(1/N) family of predictions under time evolution.

## Status, in ledger terms

- Brickwork device chain realizes hopping H with m* = 1/(2η), V as phase cost — **demonstrated by construction**.
- Counted populations track exact-device dynamics to O(1/N) at measurement stages — **computed by enumeration** (slope −0.94).
- Exact-device dynamics tracks Schrödinger predictions (spreading law, group velocity, harmonic period) — **computed**, with stated lattice/Trotter corrections (the finite-step face of the continuum limit).
- Position-cell structure of the refinement graph — **assumed**, not derived (named input, [AS]/[LC] style); where cells come from remains open.
- Phase texture admissible — **assumed** (companion open problem, inherited).
- Continuum limit (formal) — **open**, as before; what this enumeration removes is the need for it in order to state and check the discrete claim.

Upgrade path for §13.3: from C+ ("posit a kernel") to *computed by enumeration on a stipulated cell lattice* — the same epistemic move CHSH-by-counting made for Bell.
