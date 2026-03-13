"""
config.py — All simulation parameters for the BL4S muon scattering tomography experiment.
Change ONLY this file to modify any aspect of the simulation.
"""

import numpy as np

# ─── Beam ────────────────────────────────────────────────────────────────────
BEAM_MOMENTA_GEV = [1.0, 2.0, 3.5, 6.0]        # GeV/c, one run per entry
EVENTS_PER_MOMENTUM = 60000  # number of events to simulate per momentum (for each target configuration)                    
MUON_MASS_GEV = 0.10566                           # GeV/c²

# ─── Magnet (momentum tagger) ─────────────────────────────────────────────
MAGNET_B_TESLA   = 1    # dipole field strength  [T]
MAGNET_L_METRES  = 1    # effective magnetic length [m]
MAGNET_SIGMA_RAD = 5e-4   # angular measurement resolution on deflection [rad]

# ─── Tracking stations ───────────────────────────────────────────────────
# z positions in cm, relative to target centre.  Two upstream, two downstream.
STATION_Z_CM = [-65.0, -25.0, +25.0, +65.0]
STATION_SPATIAL_RES_CM = 0.020   

# ─── Target geometry ─────────────────────────────────────────────────────
# Outer aluminium shell
AL_SHELL_HALF = 12.5   # cm  (25×25×25 cm box → ±12.5 cm)

# Inner copper block
CU_BLOCK_HALF = 7.5    # cm  (15×15×15 cm block → ±7.5 cm)

# Lead cylinder
PB_RADIUS_CM  = 2.0    # cm
PB_OFFSET_X   = +3.0   # cm  transverse offset from target centre
PB_OFFSET_Y   = +2.0   # cm
PB_HALF_Z     = 7.5    # cm  (same height as Cu block)

# Radiation lengths (PDG 2024)
X0_AL_CM  =  8.90   # cm
X0_CU_CM  =  1.44   # cm
X0_PB_CM  =  0.56   # cm

# ─── Highland model ──────────────────────────────────────────────────────
HIGHLAND_K    = 13.6e-3   # GeV  (the 13.6 MeV constant)
# Artificial bias for sensitivity study  (+0.15 → +15%, -0.15 → −15%, 0 → nominal)
HIGHLAND_BIAS = 0.0       # fractional bias ε applied as θ₀ → θ₀·(1+ε)

# ─── Event selection ─────────────────────────────────────────────────────
MAX_SPACE_ANGLE_RAD = 0.200   # 200 mrad hard cut (hadronic interactions / δ-rays)

# ─── Reconstruction grid ─────────────────────────────────────────────────
VOXEL_N     = 50       # number of voxels per side
VOXEL_HALF  = 15.0     # cm  → voxels span ±15 cm → 6 mm voxel size

# ─── Molière tail model ──────────────────────────────────────────────────
# Single-scattering Gaussian core fraction; (1 − GAUSS_CORE_FRAC) goes into tail.
# Simple two-Gaussian model: core σ = θ₀, tail σ = TAIL_SIGMA_FACTOR·θ₀
GAUSS_CORE_FRAC    = 0.98
TAIL_SIGMA_FACTOR  = 3.5

# ─── Output ──────────────────────────────────────────────────────────────
OUTPUT_DIR         = "output"     # directory for all plots / npz files
RANDOM_SEED        = 42
