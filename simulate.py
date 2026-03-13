import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import parameters as C

# ─── Reproducibility ────────────────────────────────────────────────────────
rng = np.random.default_rng(C.RANDOM_SEED)
os.makedirs(C.OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def ray_box_path(x0, y0, dx, dy, half):
    """
    Path length of ray  P(t) = (x0 + dx*t, y0 + dy*t, t)  inside cube |x|,|y|,|z| <= half.
    x0, y0 are positions at z=0.  t = z (beam axis).
    Returns path length in cm, or 0 if ray misses the box.
    """
    t_min, t_max = -half, +half   # z-slab of box
    # x constraint
    if abs(dx) > 1e-12:
        tx1 = (-half - x0) / dx
        tx2 = ( half - x0) / dx
        t_min = max(t_min, min(tx1, tx2))
        t_max = min(t_max, max(tx1, tx2))
    else:
        if abs(x0) > half:
            return 0.0
    # y constraint
    if abs(dy) > 1e-12:
        ty1 = (-half - y0) / dy
        ty2 = ( half - y0) / dy
        t_min = max(t_min, min(ty1, ty2))
        t_max = min(t_max, max(ty1, ty2))
    else:
        if abs(y0) > half:
            return 0.0
    if t_max <= t_min:
        return 0.0
    return (t_max - t_min) * np.sqrt(1 + dx**2 + dy**2)


def ray_cylinder_path(x0, y0, dx, dy, cx, cy, r, half_z):
    """
    Path length of ray P(t)=(x0+dx*t, y0+dy*t, t) inside a vertical cylinder
    centred at (cx,cy) with radius r and half-height half_z.
    t = z (beam axis), x0/y0 positions at z=0.
    """
    ox = x0 - cx;  oy = y0 - cy
    # Quadratic: (ox+dx*t)^2 + (oy+dy*t)^2 = r^2
    a = dx**2 + dy**2
    if a < 1e-18:
        # ray nearly parallel to z axis; transverse position fixed
        if ox**2 + oy**2 < r**2:
            return 2 * half_z * np.sqrt(1 + dx**2 + dy**2)
        return 0.0
    b = 2*(ox*dx + oy*dy)
    c = ox**2 + oy**2 - r**2
    disc = b**2 - 4*a*c
    if disc < 0:
        return 0.0
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2*a)
    t2 = (-b + sq) / (2*a)
    t1c = max(t1, -half_z);  t2c = min(t2, +half_z)
    if t2c <= t1c:
        return 0.0
    return (t2c - t1c) * np.sqrt(1 + dx**2 + dy**2)


def compute_x_over_X0(x_us, y_us, dx, dy):
    """
    Total x/X0 for a muon given position (x_us,y_us) at z=STATION_Z_CM[1]=-25 cm
    and slopes (dx,dy).  Propagates to z=0 (target centre) for geometry calcs.
    """
    # Position at z=0 (target centre)
    dz = 0.0 - C.STATION_Z_CM[1]   # = +25 cm
    x0 = x_us + dx * dz
    y0 = y_us + dy * dz

    x_al_full = ray_box_path(x0, y0, dx, dy, C.AL_SHELL_HALF)
    x_cu_box  = ray_box_path(x0, y0, dx, dy, C.CU_BLOCK_HALF)
    x_pb_cyl  = ray_cylinder_path(x0, y0, dx, dy,
                                   C.PB_OFFSET_X, C.PB_OFFSET_Y,
                                   C.PB_RADIUS_CM, C.PB_HALF_Z)
    x_al_net = max(x_al_full - x_cu_box, 0.0)
    x_cu_net = max(x_cu_box  - x_pb_cyl, 0.0)
    return (x_al_net / C.X0_AL_CM +
            x_cu_net / C.X0_CU_CM +
            x_pb_cyl / C.X0_PB_CM)


# ════════════════════════════════════════════════════════════════════════════
#  HIGHLAND FORMULA
# ════════════════════════════════════════════════════════════════════════════

def theta0_highland(p_gev, x_over_X0, bias=0.0):
    """
    Highland projected RMS scattering angle [rad].
    bias: fractional systematic offset (HIGHLAND_BIAS).
    """
    beta = p_gev / np.sqrt(p_gev**2 + C.MUON_MASS_GEV**2)
    log_term = 1 + 0.038 * np.log(np.maximum(x_over_X0, 1e-10))
    th0 = (C.HIGHLAND_K / (beta * p_gev)) * np.sqrt(x_over_X0) * log_term
    return th0 * (1 + bias)


def theta_space_highland(p_gev, x_over_X0, bias=0.0):
    return np.sqrt(2) * theta0_highland(p_gev, x_over_X0, bias)


# ════════════════════════════════════════════════════════════════════════════
#  TRACK SIMULATION (single momentum setting)
# ════════════════════════════════════════════════════════════════════════════

def simulate_momentum(p_nom_gev, n_events):
    """
    Simulate `n_events` muons at nominal momentum p_nom_gev.

    Returns dict with per-event arrays:
      p_true, p_meas         [GeV/c]
      x0, y0                 [cm]  transverse position at target entry
      dx_in, dy_in           [rad] upstream slope
      dx_out, dy_out         [rad] downstream slope
      dtheta_space           [rad] measured space angle
      x_X0                   [1]   true x/X0 traversed
      poca_x/y/z             [cm]  PoCA coordinates
      pass_cut               [bool]
    """
    # ── True momentum spread (1% Gaussian smear around nominal) ──────────
    p_true = rng.normal(p_nom_gev, 0.01 * p_nom_gev, n_events)
    p_true = np.clip(p_true, 0.1, 20.0)

    # ── Measured momentum from magnet deflection ──────────────────────────
    delta_true = 0.3 * C.MAGNET_B_TESLA * C.MAGNET_L_METRES / p_true  # rad
    delta_meas = delta_true + rng.normal(0, C.MAGNET_SIGMA_RAD, n_events)
    delta_meas = np.where(np.abs(delta_meas) < 1e-6,
                          np.sign(delta_meas + 1e-9) * 1e-6, delta_meas)
    p_meas = 0.3 * C.MAGNET_B_TESLA * C.MAGNET_L_METRES / np.abs(delta_meas)

    # ── Beam spot & incoming direction ───────────────────────────────────
    beam_sigma_xy  = 1.0   # cm transverse size
    beam_sigma_ang = 2e-3  # rad angular divergence
    x0   = rng.normal(0, beam_sigma_xy,  n_events)
    y0   = rng.normal(0, beam_sigma_xy,  n_events)
    dx_in = rng.normal(0, beam_sigma_ang, n_events)
    dy_in = rng.normal(0, beam_sigma_ang, n_events)

    # ── x/X0 per event ───────────────────────────────────────────────────
    x_X0 = np.array([compute_x_over_X0(x0[i], y0[i], dx_in[i], dy_in[i])
                     for i in range(n_events)])
    x_X0 = np.maximum(x_X0, 1e-6)

    # ── Scattering angle sampling (two-Gaussian core+tail model) ─────────
    th0 = theta0_highland(p_true, x_X0, bias=0.0)  # no bias in truth
    # projected angles: two-component Gaussian
    def sample_angle(sigma_core, n):
        core_mask = rng.random(n) < C.GAUSS_CORE_FRAC
        angles = np.where(core_mask,
                          rng.normal(0, sigma_core, n),
                          rng.normal(0, C.TAIL_SIGMA_FACTOR * sigma_core, n))
        return angles

    dthx = sample_angle(th0, n_events)
    dthy = sample_angle(th0, n_events)

    dx_out = dx_in + dthx
    dy_out = dy_in + dthy

    # ── Apply hit resolution to tracking stations ─────────────────────────
    z_up1, z_up2 = C.STATION_Z_CM[0], C.STATION_Z_CM[1]
    z_dn1, z_dn2 = C.STATION_Z_CM[2], C.STATION_Z_CM[3]

    def smeared_track(z1, z2, x_true_at_z1, slope):
        """Returns (reconstructed slope, smeared hit position at z1)."""
        h1 = x_true_at_z1 + rng.normal(0, C.STATION_SPATIAL_RES_CM, n_events)
        h2 = x_true_at_z1 + slope*(z2 - z1) + rng.normal(0, C.STATION_SPATIAL_RES_CM, n_events)
        slope_reco = (h2 - h1) / (z2 - z1)
        return slope_reco, h1

    # Upstream: true position at z_up1 is x0 + dx_in*z_up1
    dx_in_reco,  h1x_up = smeared_track(z_up1, z_up2, x0 + dx_in*z_up1,  dx_in)
    dy_in_reco,  h1y_up = smeared_track(z_up1, z_up2, y0 + dy_in*z_up1,  dy_in)

    # Downstream: true position at z_dn1 is x0 + dx_out*z_dn1
    dx_out_reco, h1x_dn = smeared_track(z_dn1, z_dn2, x0 + dx_out*z_dn1, dx_out)
    dy_out_reco, h1y_dn = smeared_track(z_dn1, z_dn2, y0 + dy_out*z_dn1, dy_out)

    # Upstream ray anchor at z=0 (extrapolate forward from z_up1)
    x1 = h1x_up - dx_in_reco * z_up1
    y1 = h1y_up - dy_in_reco * z_up1

    # Downstream ray anchor at z=0 (extrapolate backward from z_dn1)
    x2 = h1x_dn - dx_out_reco * z_dn1
    y2 = h1y_dn - dy_out_reco * z_dn1

    # ── Space angle ───────────────────────────────────────────────────────
    ddx = dx_out_reco - dx_in_reco
    ddy = dy_out_reco - dy_in_reco
    dtheta_space = np.sqrt(ddx**2 + ddy**2)

    # ── PoCA calculation ──────────────────────────────────────────────────
    # Ray 1 (upstream): anchor x1,y1 at z=0, direction (dx_in_reco, dy_in_reco, 1)
    # Ray 2 (downstream): anchor x2,y2 at z=0, direction (dx_out_reco, dy_out_reco, 1)
    # PoCA = midpoint of shortest segment connecting the two rays.
    d1x = dx_in_reco;  d1y = dy_in_reco;  d1z = np.ones(n_events)
    d2x = dx_out_reco; d2y = dy_out_reco; d2z = np.ones(n_events)
    w0x = x1 - x2;    w0y = y1 - y2;    w0z = np.zeros(n_events)
    a = d1x*d1x + d1y*d1y + d1z*d1z
    b = d1x*d2x + d1y*d2y + d1z*d2z
    c = d2x*d2x + d2y*d2y + d2z*d2z
    d = d1x*w0x + d1y*w0y + d1z*w0z
    e = d2x*w0x + d2y*w0y + d2z*w0z
    denom = a*c - b*b
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    sc = (b*e - c*d) / denom
    tc = (a*e - b*d) / denom
    poca_x = 0.5 * ((x1 + sc*d1x) + (x2 + tc*d2x))
    poca_y = 0.5 * ((y1 + sc*d1y) + (y2 + tc*d2y))
    poca_z = 0.5 * ((   + sc*d1z) + (    + tc*d2z))

    # ── Event selection ───────────────────────────────────────────────────
    pass_cut = dtheta_space < C.MAX_SPACE_ANGLE_RAD

    return dict(
        p_true=p_true, p_meas=p_meas,
        x0=x0, y0=y0,
        dx_in=dx_in_reco, dy_in=dy_in_reco,
        dx_out=dx_out_reco, dy_out=dy_out_reco,
        dtheta_space=dtheta_space,
        x_X0=x_X0,
        poca_x=poca_x, poca_y=poca_y, poca_z=poca_z,
        pass_cut=pass_cut,
    )


# ════════════════════════════════════════════════════════════════════════════
#  FULL SIMULATION (all momenta)
# ════════════════════════════════════════════════════════════════════════════

print("Running simulation...")
all_events = []
for p in C.BEAM_MOMENTA_GEV:
    print(f"  p = {p} GeV/c  ({C.EVENTS_PER_MOMENTUM} events)")
    ev = simulate_momentum(p, C.EVENTS_PER_MOMENTUM)
    ev["p_nom"] = p
    all_events.append(ev)

# Concatenate all momenta
def concat(key):
    return np.concatenate([ev[key] for ev in all_events])

p_true       = concat("p_true")
p_meas       = concat("p_meas")
x_X0         = concat("x_X0")
dtheta_space = concat("dtheta_space")
poca_x       = concat("poca_x")
poca_y       = concat("poca_y")
poca_z       = concat("poca_z")
pass_cut     = concat("pass_cut")
p_nom_arr    = np.concatenate([[ev["p_nom"]]*C.EVENTS_PER_MOMENTUM for ev in all_events])

# ════════════════════════════════════════════════════════════════════════════
#  BRANCH A — Highland validation
# ════════════════════════════════════════════════════════════════════════════

print("Branch A: Highland validation...")
theta_rms_meas  = []
theta_rms_pred  = []
x_X0_mean_vals  = []
residuals       = []

for ev in all_events:
    sel = ev["pass_cut"]
    dt  = ev["dtheta_space"][sel]
    xX  = ev["x_X0"][sel]
    pm  = ev["p_meas"][sel]
    pt  = ev["p_true"][sel]

    # Projected angle ≈ space_angle / sqrt(2)
    theta_proj_meas = dt / np.sqrt(2)
    rms_meas = np.sqrt(np.mean(theta_proj_meas**2))

    # Highland prediction at mean x/X0 and mean momentum
    xX_mean  = np.mean(xX)
    p_mean   = np.mean(pt)
    rms_pred = theta0_highland(p_mean, xX_mean, bias=C.HIGHLAND_BIAS)

    residual = (rms_meas - rms_pred) / rms_pred * 100  # percent

    theta_rms_meas.append(rms_meas)
    theta_rms_pred.append(rms_pred)
    x_X0_mean_vals.append(xX_mean)
    residuals.append(residual)

    print(f"  p_nom={ev['p_nom']:.1f} GeV/c  "
          f"θ_RMS meas={rms_meas*1e3:.2f} mrad  "
          f"pred={rms_pred*1e3:.2f} mrad  "
          f"residual={residual:+.1f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
momenta = C.BEAM_MOMENTA_GEV
ax.plot(momenta, np.array(theta_rms_pred)*1e3, 'k--o', label='Highland prediction')
ax.plot(momenta, np.array(theta_rms_meas)*1e3, 'rs', markersize=7, label='Simulated measurement')
ax.set_xlabel("Muon momentum [GeV/c]")
ax.set_ylabel("θ₀ RMS [mrad]")
ax.set_title("Highland Validation: θ_RMS vs Momentum")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.axhline(0, color='k', lw=0.8)
ax.bar(momenta, residuals, width=0.3, color=['#2196F3','#4CAF50','#FF9800','#9C27B0'], alpha=0.8)
ax.set_xlabel("Muon momentum [GeV/c]")
ax.set_ylabel("Residual [%]  (meas − pred) / pred")
ax.set_title("Highland Residuals")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(C.OUTPUT_DIR, "highland_validation.png"), dpi=150)
plt.close()
print("  Saved highland_validation.png")


# ════════════════════════════════════════════════════════════════════════════
#  BRANCH B — Tomographic reconstruction
# ════════════════════════════════════════════════════════════════════════════

print("Branch B: Tomographic reconstruction...")

# Voxel grid
edges   = np.linspace(-C.VOXEL_HALF, C.VOXEL_HALF, C.VOXEL_N + 1)
centres = 0.5 * (edges[:-1] + edges[1:])

# Select passing events
sel  = pass_cut
px   = poca_x[sel]
py   = poca_y[sel]
pz   = poca_z[sel]
dth  = dtheta_space[sel]
pm   = p_meas[sel]
pt   = p_true[sel]
xX   = x_X0[sel]

# Clamp PoCA to grid
in_grid = ((np.abs(px) < C.VOXEL_HALF) &
           (np.abs(py) < C.VOXEL_HALF) &
           (np.abs(pz) < C.VOXEL_HALF))
px = px[in_grid]; py = py[in_grid]; pz = pz[in_grid]
dth = dth[in_grid]; pm = pm[in_grid]; xX = xX[in_grid]

# Weights for momentum-weighted PoCA
theta_pred = theta_space_highland(pm, xX, bias=C.HIGHLAND_BIAS)
theta_pred = np.where(theta_pred < 1e-6, 1e-6, theta_pred)
weights    = (dth / theta_pred)**2

def fill_voxels(px, py, pz, w=None):
    H, _ = np.histogramdd(
        np.column_stack([px, py, pz]),
        bins=[edges, edges, edges],
        weights=w
    )
    return H

voxel_unweighted = fill_voxels(px, py, pz)
voxel_weighted   = fill_voxels(px, py, pz, w=weights)

def plot_slice(ax, grid_2d, title, cmap='hot'):
    im = ax.imshow(grid_2d.T, origin='lower',
                   extent=[-C.VOXEL_HALF, C.VOXEL_HALF,
                            -C.VOXEL_HALF, C.VOXEL_HALF],
                   cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="PoCA density")
    # Mark Pb cylinder centre
    circle = plt.Circle((C.PB_OFFSET_X, C.PB_OFFSET_Y), C.PB_RADIUS_CM,
                         color='cyan', fill=False, lw=1.5, linestyle='--', label='Pb (true)')
    ax.add_patch(circle)
    ax.legend(loc='upper right', fontsize=8)

mid_z = C.VOXEL_N // 2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_slice(axes[0], voxel_unweighted[:, :, mid_z], "Unweighted PoCA (z-slice)")
plot_slice(axes[1], voxel_weighted[:, :, mid_z],   "Momentum-weighted PoCA (z-slice)")
plt.suptitle("Muon Scattering Tomography — Central z-slice", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(C.OUTPUT_DIR, "tomo_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved tomo_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
#  SENSITIVITY STUDY — Highland bias ±15%
# ════════════════════════════════════════════════════════════════════════════

print("Sensitivity study: Highland bias ±15% ...")
bias_levels = [-0.15, 0.0, +0.15]
labels      = ["−15%", "Nominal", "+15%"]
colors      = ["#E53935", "#43A047", "#1E88E5"]
voxel_maps  = {}

for bias in bias_levels:
    th_pred_b = theta_space_highland(pm, xX, bias=bias)
    th_pred_b = np.where(th_pred_b < 1e-6, 1e-6, th_pred_b)
    w_b       = (dth / th_pred_b)**2
    voxel_maps[bias] = fill_voxels(px, py, pz, w=w_b)

# Compare: mean voxel value in Pb region vs rest
def roi_mean(vmap, cx, cy, r_cm):
    """Mean voxel value inside cylindrical ROI."""
    vals = []
    for ix, xc in enumerate(centres):
        for iy, yc in enumerate(centres):
            if (xc - cx)**2 + (yc - cy)**2 < r_cm**2:
                vals.extend(vmap[ix, iy, :].tolist())
    return np.mean(vals) if vals else 0.0

roi_pb   = [roi_mean(voxel_maps[b], C.PB_OFFSET_X, C.PB_OFFSET_Y, C.PB_RADIUS_CM) for b in bias_levels]
roi_bulk = [roi_mean(voxel_maps[b], 0, 0, C.CU_BLOCK_HALF * 0.5) for b in bias_levels]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for i, (bias, lbl, col) in enumerate(zip(bias_levels, labels, colors)):
    axes[i].imshow(voxel_maps[bias][:, :, mid_z].T, origin='lower',
                   extent=[-C.VOXEL_HALF, C.VOXEL_HALF]*2,
                   cmap='hot', interpolation='nearest')
    circle = plt.Circle((C.PB_OFFSET_X, C.PB_OFFSET_Y), C.PB_RADIUS_CM,
                         color='cyan', fill=False, lw=1.5, linestyle='--')
    axes[i].add_patch(circle)
    axes[i].set_title(f"Highland bias {lbl}\nPb ROI mean={roi_pb[i]:.1f}", color=col)
    axes[i].set_xlabel("x [cm]"); axes[i].set_ylabel("y [cm]")

plt.suptitle("Sensitivity Study: Effect of Highland Model Bias on Weighted PoCA Image", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(C.OUTPUT_DIR, "sensitivity.png"), dpi=150, bbox_inches='tight')
plt.close()

print("  Bias level | Pb ROI mean | Bulk ROI mean | Pb/Bulk ratio")
for bias, lbl, rp, rb in zip(bias_levels, labels, roi_pb, roi_bulk):
    ratio = rp / rb if rb > 0 else float('nan')
    print(f"  {lbl:>6s}     {rp:10.2f}   {rb:12.2f}   {ratio:.3f}")
print("  Saved sensitivity.png")

# ════════════════════════════════════════════════════════════════════════════
#  SAVE NUMERICAL RESULTS
# ════════════════════════════════════════════════════════════════════════════

np.savez(os.path.join(C.OUTPUT_DIR, "results.npz"),
         beam_momenta      = C.BEAM_MOMENTA_GEV,
         theta_rms_meas    = theta_rms_meas,
         theta_rms_pred    = theta_rms_pred,
         residuals_pct     = residuals,
         voxel_unweighted  = voxel_unweighted,
         voxel_weighted    = voxel_weighted,
         sensitivity_biases= bias_levels,
         sensitivity_pb_roi= roi_pb,
         sensitivity_bulk  = roi_bulk,
)
print(f"\nAll outputs written to ./{C.OUTPUT_DIR}/")
print("Done.")
