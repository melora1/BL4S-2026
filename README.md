# BL4S 2026 — Muon Scattering Tomography Simulation
## Files

| File | Description |
|------|-------------|
| `config.py` | All tunable parameters — edit only this file |
| `simulate.py` | Physics engine and analysis pipeline |
| `highland_validation.png` | Branch A: Highland formula vs. simulation |
| `tomo_comparison.png` | Branch B: PoCA tomographic reconstruction |
| `sensitivity.png` | ±15% Highland bias sensitivity study |
| `results.npz` | Numerical outputs |

---

## Setup

```bash
pip install numpy matplotlib scipy
python simulate.py
```

Outputs are written to the working directory.

---

## Beam parameters

| Parameter | Value |
|-----------|-------|
| Momenta | 1.0, 2.0, 3.5, 6.0 GeV/c |
| Events per momentum | 60,000 |
| Magnet B×L | 1.0 T × 1.0 m (MNP17) |
| Detector resolution | 200 µm (MicroMegas) |
| Station z-positions | −65, −25, +25, +65 cm |

---

## Analysis branches

**Branch A — Highland validation**
Compares the simulated RMS scattering angle against the Highland formula across all four momenta. Positive residuals (+1–12%) at higher momenta reflect Molière tails not captured by the Gaussian approximation.

**Branch B — PoCA tomography**
Reconstructs a 50×50 voxel image of the Al/Cu/Pb target using the Point of Closest Approach algorithm. Weighted image uses inverse-momentum-squared weighting to enhance high-angle scatter contrast.

**Sensitivity study**
Repeats Branch B with ±15% artificial bias on the Highland σ. A ±15% input bias produces roughly ∓30% change in the Pb ROI voxel mean.

---

## Target geometry

- Outer shell: Al (X₀ = 8.90 cm), half-width 12.5 cm
- Inner block: Cu (X₀ = 1.44 cm), half-width 7.5 cm
- Embedded cylinder: Pb (X₀ = 0.56 cm), r = 2.0 cm, offset (+3, +2) cm

---

## Notes

- The Pb cylinder is partially unresolved at 60k events/momentum due to Cu volume dominance — this is a known physics limitation, not a code bug.
- All parameters are set in `config.py`. Do not edit `simulate.py` unless modifying the physics engine.
  
