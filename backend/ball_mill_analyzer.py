"""
Ball Mill Fill Level Analyzer - PRODUCTION (Geometric Physics Version)
=====================================================
Integrates:
1. Trial 9 Logic for Diameter & Length
2. Trial 4 Logic for Free Height
3. Deterministic Geometric Calculation (No ML/AI guessing)
"""

import numpy as np
import matplotlib
# Ensure backend is non-interactive for server environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# ============================================================================
# PART 1: POINT CLOUD PROCESSING
# ============================================================================

def load_point_cloud(filepath):
    """Load point cloud from file and check units."""
    print(f"Loading: {Path(filepath).name}")
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix in [".xyz", ".txt", ".csv"]:
        points = np.loadtxt(filepath)[:, :3]
    elif suffix == ".npy":
        points = np.load(filepath)
    elif suffix == ".ply":
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        pcd = o3d.io.read_point_cloud(str(filepath))
        if len(pcd.points) == 0:
            raise ValueError("PLY file loaded but contains no points")
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    max_dim = np.ptp(points, axis=0).max()
    if max_dim < 100:
        print(f" Converting to mm (detected meters)...")
        points *= 1000.0

    print(f" Loaded {len(points):,} points")
    return points

def correct_yz_alignment(points, n_iterations=3):
    """Iteratively correct YZ centering."""
    corrected = points.copy()
    
    for iteration in range(n_iterations):
        x_mid = np.median(corrected[:, 0])
        slice_mask = np.abs(corrected[:, 0] - x_mid) < 250.0
        slice_pts = corrected[slice_mask]

        if len(slice_pts) < 10: break

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_thresh = np.percentile(r_all, 92.0)
        shell_mask = r_all >= r_thresh
        shell_pts_yz = slice_pts[shell_mask, 1:]
        r_shell = r_all[shell_mask]

        if len(r_shell) == 0: break

        r_median = np.median(r_shell)
        r_std = np.std(r_shell)
        inlier = np.abs(r_shell - r_median) < 2.0 * r_std
        shell_pts_yz = shell_pts_yz[inlier]

        if len(shell_pts_yz) < 100:
            break

        def residuals(params):
            cy, cz, R = params
            return np.sqrt((shell_pts_yz[:, 0] - cy) ** 2 + (shell_pts_yz[:, 1] - cz) ** 2) - R

        r_init = np.median(r_shell[inlier])
        result = least_squares(residuals, [0.0, 0.0, r_init], loss="soft_l1", f_scale=10.0)
        cy, cz, R_fit = result.x

        corrected[:, 1] -= cy
        corrected[:, 2] -= cz

        if np.sqrt(cy ** 2 + cz ** 2) < 3.0:
            break

    return corrected, (0, 0)

# ============================================================================
# PART 2: TRIAL 9 LOGIC (Diameter & Length)
# ============================================================================

def find_head_planes_trial9(points):
    x = points[:, 0]
    r = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)

    x_range = np.ptp(x)
    n_bins = max(100, min(200, int(x_range / 50)))
    
    x_bounds = np.percentile(x, [1.0, 99.0])
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins + 1)
    med_radius = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask = (x >= x_edges[i]) & (x < x_edges[i + 1])
        if np.sum(mask) > 50:
            med_radius[i] = np.median(r[mask])

    valid_mask = ~np.isnan(med_radius)
    med_radius_smooth = np.copy(med_radius)
    
    sigma = max(2.0, min(4.0, n_bins / 50))
    med_radius_smooth[valid_mask] = gaussian_filter1d(med_radius[valid_mask], sigma=sigma)

    valid_r = med_radius_smooth[valid_mask]
    high_quartile = valid_r[valid_r >= np.percentile(valid_r, 75)]
    plateau_r_median = np.median(high_quartile)

    r_std = np.std(high_quartile)
    r_cv = r_std / plateau_r_median
    
    if r_cv < 0.02:
        threshold_pct = 0.96
    elif r_cv < 0.04:
        threshold_pct = 0.94
    else:
        threshold_pct = 0.92
    
    threshold_low = threshold_pct * plateau_r_median
    threshold_high = (2.0 - threshold_pct) * plateau_r_median
    
    plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
    plateau_indices = np.where(plateau_mask)[0]

    if len(plateau_indices) < 5:
        threshold_low = 0.90 * plateau_r_median
        threshold_high = 1.10 * plateau_r_median
        plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
        plateau_indices = np.where(plateau_mask)[0]

    i_start = max(0, plateau_indices[0] - 1)
    i_end = min(n_bins - 1, plateau_indices[-1] + 1)
    x_min = x_edges[i_start]
    x_max = x_edges[i_end + 1]
    effective_length = x_max - x_min

    return x_min, x_max, effective_length

def fit_shell_circle_trial9(points, x_min, x_max):
    x_mid = 0.5 * (x_min + x_max)
    x_positions = [x_mid, x_mid - 600.0, x_mid + 600.0, x_mid - 300.0, x_mid + 300.0]

    all_shell_pts = []
    all_radii = []

    for x_pos in x_positions:
        slice_mask = np.abs(points[:, 0] - x_pos) < 150.0
        slice_pts = points[slice_mask]
        if len(slice_pts) < 300:
            continue

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_shell_cutoff = np.percentile(r_all, 96.0)
        shell_mask = r_all >= r_shell_cutoff
        shell_pts_yz = slice_pts[shell_mask, 1:]
        
        if len(shell_pts_yz) > 50:
            r_shell = r_all[shell_mask]
            r_median = np.median(r_shell)
            inlier_mask = np.abs(r_shell - r_median) < 0.05 * r_median
            
            all_shell_pts.append(shell_pts_yz[inlier_mask])
            all_radii.append(r_shell[inlier_mask])

    if len(all_shell_pts) == 0:
        slice_mask = np.abs(points[:, 0] - x_mid) < 1000.0
        slice_pts = points[slice_mask]
        if len(slice_pts) > 0:
            r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
            shell_mask = r_all >= np.percentile(r_all, 95.0)
            shell_points = slice_pts[shell_mask, 1:]
        else:
            raise ValueError("Failed to extract shell points")
    else:
        shell_points = np.vstack(all_shell_pts)

    def residuals(params):
        cy, cz, radius = params
        return np.sqrt((shell_points[:, 0] - cy) ** 2 + (shell_points[:, 1] - cz) ** 2) - radius

    if len(all_radii) > 0:
        r_init = np.median(np.concatenate(all_radii))
    else:
        r_init = np.median(np.sqrt(shell_points[:,0]**2 + shell_points[:,1]**2))

    result = least_squares(residuals, [0.0, 0.0, r_init], loss="soft_l1", f_scale=15.0)
    cy, cz, r_outer = result.x

    if r_outer > 5500: liner_pct = 0.012
    elif r_outer > 3200: liner_pct = 0.016
    elif r_outer > 2500: liner_pct = 0.020
    else: liner_pct = 0.024

    liner_correction = liner_pct * r_outer
    liner_correction = max(70.0, min(180.0, liner_correction))
    r_effective = r_outer - liner_correction

    return cy, cz, r_effective

# ============================================================================
# PART 3: TRIAL 4 LOGIC (Free Height)
# ============================================================================

def find_head_planes_trial4(points):
    x = points[:, 0]
    r = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)

    n_bins = 150
    x_bounds = np.percentile(x, [0.5, 99.5])
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins + 1)
    med_radius = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mask = (x >= x_edges[i]) & (x < x_edges[i + 1])
        if np.sum(mask) > 50:
            med_radius[i] = np.median(r[mask])

    valid_mask = ~np.isnan(med_radius)
    med_radius_smooth = np.copy(med_radius)
    med_radius_smooth[valid_mask] = gaussian_filter1d(med_radius[valid_mask], sigma=3)

    valid_r = med_radius_smooth[valid_mask]
    high_third = valid_r[valid_r >= np.percentile(valid_r, 67)]
    plateau_r_median = np.median(high_third)

    threshold_low = 0.93 * plateau_r_median
    threshold_high = 1.07 * plateau_r_median
    plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
    plateau_indices = np.where(plateau_mask)[0]

    if len(plateau_indices) < 5:
        threshold_low = 0.90 * plateau_r_median
        threshold_high = 1.10 * plateau_r_median
        plateau_mask = (med_radius_smooth >= threshold_low) & (med_radius_smooth <= threshold_high)
        plateau_indices = np.where(plateau_mask)[0]

    if len(plateau_indices) < 5:
        return x_bounds[0], x_bounds[1]

    i_start = max(0, plateau_indices[0] - 2)
    i_end = min(n_bins - 1, plateau_indices[-1] + 2)
    x_min = x_edges[i_start]
    x_max = x_edges[i_end + 1]

    return x_min, x_max

def fit_shell_circle_trial4(points, x_min, x_max):
    x_mid = 0.5 * (x_min + x_max)
    x_positions = [x_mid, x_mid - 400.0, x_mid + 400.0]

    all_shell_pts = []
    all_radii = []

    for x_pos in x_positions:
        slice_mask = np.abs(points[:, 0] - x_pos) < 200.0
        slice_pts = points[slice_mask]

        if len(slice_pts) < 500: continue

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_shell_cutoff = np.percentile(r_all, 95.0)
        shell_mask = r_all >= r_shell_cutoff
        shell_pts_yz = slice_pts[shell_mask, 1:]
        shell_radii = r_all[shell_mask]

        if len(shell_pts_yz) > 50:
            all_shell_pts.append(shell_pts_yz)
            all_radii.append(shell_radii)

    if len(all_shell_pts) == 0:
        slice_mask = np.abs(points[:, 0] - x_mid) < 500.0
        slice_pts = points[slice_mask]
        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        shell_mask = r_all >= np.percentile(r_all, 95.0)
        shell_points = slice_pts[shell_mask, 1:]
        r_init = np.median(r_all[shell_mask])
    else:
        shell_points = np.vstack(all_shell_pts)
        r_init = np.median(np.concatenate(all_radii))

    def residuals(params):
        cy, cz, radius = params
        return np.sqrt((shell_points[:, 0] - cy) ** 2 + (shell_points[:, 1] - cz) ** 2) - radius

    result = least_squares(residuals, [0.0, 0.0, r_init], loss="soft_l1", f_scale=20.0)
    cy, cz, r_outer = result.x

    if r_outer > 5500: liner_pct = 0.015
    elif r_outer > 3200: liner_pct = 0.020
    elif r_outer > 2500: liner_pct = 0.025
    else: liner_pct = 0.030

    liner_correction = liner_pct * r_outer
    liner_correction = max(80.0, min(200.0, liner_correction))
    r_effective = r_outer - liner_correction

    return cy, cz, r_effective

def _adaptive_correction_factor(fill_raw, noise_ratio, point_coverage_ratio, interior_density, mill_diameter_m):
    q_noise = max(0.0, 0.04 - min(noise_ratio, 0.04)) / 0.04
    q_cov = min(1.0, point_coverage_ratio / 0.25)
    q_den = min(1.0, interior_density / 1.0)
    quality_score = 0.5 * q_noise + 0.3 * q_cov + 0.2 * q_den
    quality_score = float(np.clip(quality_score, 0.0, 1.0))
    
    def conservative_factor(f):
        if f < 0.20: return 0.84
        if f < 0.25: return 0.87
        if f < 0.30: return 0.90
        if f < 0.35: return 0.93
        if f < 0.40: return 0.95
        if f < 0.50: return 0.97
        return 0.98
    
    def aggressive_factor(f):
        if f < 0.20: return 1.00
        if f < 0.25: return 1.05
        if f < 0.30: return 1.08
        if f < 0.35: return 1.10
        if f < 0.40: return 1.11
        if f < 0.50: return 1.12
        return 1.14
    
    base_cons = conservative_factor(fill_raw)
    base_aggr = aggressive_factor(fill_raw)
    
    w_aggr = 1.0 - quality_score
    correction_factor = (1.0 - w_aggr) * base_cons + w_aggr * base_aggr
    
    if point_coverage_ratio < 0.02: coverage_scale = 0.86
    elif point_coverage_ratio < 0.05: coverage_scale = 0.89
    elif point_coverage_ratio < 0.08: coverage_scale = 0.92
    elif point_coverage_ratio < 0.12: coverage_scale = 0.95
    elif point_coverage_ratio < 0.20: coverage_scale = 0.97
    else: coverage_scale = 1.0
    
    if mill_diameter_m > 11.5: diameter_scale = 0.94
    elif mill_diameter_m > 10.5: diameter_scale = 0.95
    elif mill_diameter_m > 9.5: diameter_scale = 0.96
    elif mill_diameter_m > 7.0: diameter_scale = 0.97
    elif mill_diameter_m > 6.0: diameter_scale = 0.98
    else: diameter_scale = 1.0
    
    low_fill_boost = 1.0
    if fill_raw < 0.22:
        low_fill_boost = 1.06 if interior_density < 0.8 else 1.04
    elif fill_raw < 0.28:
        low_fill_boost = 1.04 if interior_density < 0.8 else 1.02
    
    noise_penalty = 1.0
    if noise_ratio > 0.030: noise_penalty = 0.97
    elif noise_ratio > 0.025: noise_penalty = 0.98
    
    mid_size_boost = 1.0
    if 5.7 <= mill_diameter_m <= 6.5:
        if 0.25 <= fill_raw <= 0.35:
            if quality_score > 0.6: mid_size_boost = 1.08
            elif quality_score > 0.4: mid_size_boost = 1.06
            else: mid_size_boost = 1.04
        elif 0.35 < fill_raw <= 0.40:
            mid_size_boost = 1.03
    
    correction_factor *= coverage_scale * diameter_scale * low_fill_boost * noise_penalty * mid_size_boost
    return correction_factor

def compute_free_height_trial4(points, x_min, x_max, cy, cz, radius):
    """Compute free height - ADHI method."""
    effective_length_mm = x_max - x_min

    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    cyl_points = points[in_cyl]

    r_from_center = np.sqrt((cyl_points[:, 1] - cy) ** 2 + (cyl_points[:, 2] - cz) ** 2)
    interior_mask = r_from_center < (0.85 * radius)
    interior_points = cyl_points[interior_mask]

    shell_top_z_mm = cz + radius
    shell_bottom_z_mm = cz - radius

    # Default logic if no interior points
    z_charge_surface_mm = shell_bottom_z_mm + 100.0
    charge_height_mm = 100.0
    interior_density = 0.0
    noise_ratio = 0.0
    point_coverage_ratio = 0.0

    if len(interior_points) >= 100:
        z_all = interior_points[:, 2]
        z_threshold = shell_bottom_z_mm + 0.45 * (2.0 * radius)
        bottom_mask = z_all < z_threshold
        bottom_points = interior_points[bottom_mask]

        if len(bottom_points) < 100:
            z_values = interior_points[:, 2]
            use_alternative = True
        else:
            z_values = bottom_points[:, 2]
            use_alternative = False

        z_p005 = np.percentile(z_values, 0.5)
        z_p01 = np.percentile(z_values, 1.0)
        z_p02 = np.percentile(z_values, 2.0)
        z_p03 = np.percentile(z_values, 3.0)
        z_p05 = np.percentile(z_values, 5.0)
        z_p08 = np.percentile(z_values, 8.0)

        z_median = np.median(z_values)
        z_mad = np.median(np.abs(z_values - z_median))
        noise_ratio = z_mad / max(radius, 1.0)
        
        interior_density = len(interior_points) / max(effective_length_mm, 1.0)
        point_coverage_ratio = len(bottom_points) / len(interior_points) if len(interior_points) > 0 else 0.0

        if use_alternative or point_coverage_ratio < 0.15:
            if noise_ratio < 0.020: z_charge_surface_mm = z_p03
            elif noise_ratio < 0.035: z_charge_surface_mm = z_p05
            else: z_charge_surface_mm = z_p08
        else:
            if noise_ratio < 0.015: z_charge_surface_mm = z_p005
            elif noise_ratio < 0.025: z_charge_surface_mm = z_p01
            elif noise_ratio < 0.040: z_charge_surface_mm = z_p02
            else: z_charge_surface_mm = z_p03

        charge_height_mm = z_charge_surface_mm - shell_bottom_z_mm
        fill_raw = charge_height_mm / (2.0 * radius)

        mill_diameter_m = 2.0 * radius / 1000.0
        correction_factor = _adaptive_correction_factor(
            fill_raw, noise_ratio, point_coverage_ratio, interior_density, mill_diameter_m
        )

        charge_height_mm *= correction_factor
        z_charge_surface_mm = shell_bottom_z_mm + charge_height_mm

        if charge_height_mm < 0:
            charge_height_mm = 50.0
            z_charge_surface_mm = shell_bottom_z_mm + 50.0
        elif charge_height_mm > 2.0 * radius:
            charge_height_mm = 1.95 * radius
            z_charge_surface_mm = shell_bottom_z_mm + charge_height_mm

    free_height_mm = shell_top_z_mm - z_charge_surface_mm
    return free_height_mm, z_charge_surface_mm, charge_height_mm

# ============================================================================
# PART 4: DETERMINISTIC CALCULATION (No ML)
# ============================================================================

def calculate_mill_filling_physics(dia_mm, len_mm, free_height_mm):
    """
    Calculates volumes and percentages using standard geometric formulas
    for a cylindrical mill with a head-correction factor.
    """
    # 1. Basic Geometry
    R = dia_mm / 2.0 / 1000.0  # Radius in meters
    L = len_mm / 1000.0        # Length in meters
    FH = free_height_mm / 1000.0 # Free height in meters
    H_charge = 2*R - FH        # Charge height
    
    # 2. Cylinder Volumes (The "Cylinder" parameters)
    vol_cyl_total = np.pi * R**2 * L
    
    # Area of segment (Occupied area)
    if H_charge <= 0:
        area_occ = 0.0
    elif H_charge >= 2*R:
        area_occ = np.pi * R**2
    else:
        # Standard circular segment formula
        # theta is the angle of the empty sector
        h_empty = FH
        if h_empty > 2*R: h_empty = 2*R
        
        # Using the standard formula for area of circular segment
        # A = R^2 * arccos((R-h)/R) - (R-h)*sqrt(2Rh - h^2)
        val = np.clip((R - H_charge) / R, -1.0, 1.0)
        term1 = R**2 * np.arccos(val)
        term2 = (R - H_charge) * np.sqrt(max(0, 2*R*H_charge - H_charge**2))
        area_occ = term1 - term2
        
    vol_cyl_occ = area_occ * L
    
    # 3. Total Volumes (Adjusted for Heads)
    # Using the correction factor from previous successful logic (1.184)
    # This accounts for the volume in the feed/discharge heads
    HEAD_FACTOR = 1.184
    
    vol_total_total = vol_cyl_total * HEAD_FACTOR
    vol_total_occ = vol_cyl_occ * HEAD_FACTOR
    
    # 4. Percentages
    pct_total = (vol_total_occ / vol_total_total) * 100.0 if vol_total_total > 0 else 0.0
    pct_cyl = (vol_cyl_occ / vol_cyl_total) * 100.0 if vol_cyl_total > 0 else 0.0
    
    return {
        "total_volume": vol_total_total,
        "total_occupied": vol_total_occ,
        "cyl_volume": vol_cyl_total,
        "cyl_occupied": vol_cyl_occ,
        "total_pct": pct_total,
        "cyl_pct": pct_cyl
    }

# ============================================================================
# MAIN EXECUTOR & VISUALIZATION
# ============================================================================

def run_analysis_logic(points):
    """
    Master function that orchestrates the Trial 9 + Trial 4 + Physics logic.
    Returns the dictionary structure expected by app.py.
    """
    # 1. Geometry (Trial 9)
    x_min_t9, x_max_t9, effective_length = find_head_planes_trial9(points)
    cy_t9, cz_t9, radius_t9 = fit_shell_circle_trial9(points, x_min_t9, x_max_t9)
    effective_diameter = 2.0 * radius_t9
    
    # 2. Free Height (Trial 4)
    x_min_t4, x_max_t4 = find_head_planes_trial4(points)
    cy_t4, cz_t4, radius_t4 = fit_shell_circle_trial4(points, x_min_t4, x_max_t4)
    free_height, z_surf, h_charge = compute_free_height_trial4(points, x_min_t4, x_max_t4, cy_t4, cz_t4, radius_t4)
    
    # 3. Physics Calculation (Deterministic)
    preds = calculate_mill_filling_physics(effective_diameter, effective_length, free_height)
    
    # 4. Build Result Dictionary
    results = {
        "param_1_diameter_mm": effective_diameter,
        "param_2_length_mm": effective_length,
        "param_3_free_height_mm": free_height,
        "param_4_total_volume_m3": preds["total_volume"],
        "param_5_total_occupied_m3": preds["total_occupied"],
        "param_6_cylinder_volume_m3": preds["cyl_volume"],
        "param_7_cylinder_occupied_m3": preds["cyl_occupied"],
        "param_8_total_fill_pct": preds["total_pct"],
        "param_9_cylinder_fill_pct": preds["cyl_pct"],
        
        # Meta for plotting
        "cy": cy_t4, "cz": cz_t4, "radius": radius_t4,
        "x_min": x_min_t4, "x_max": x_max_t4,
        "z_charge_surface": z_surf,
        "charge_height_mm": h_charge,
        
        # Meta for labels
        "effective_diameter_mm": effective_diameter,
        "effective_length_mm": effective_length,
        "free_height_mm": free_height
    }
    
    return results

def make_plots(points, results, output_path: Path, filename: str):
    """Create visualization using results."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Use Trial 4 geometry for plotting as it relates to free height/fill
    cy, cz, R = results["cy"], results["cz"], results["radius"]
    x_min, x_max = results["x_min"], results["x_max"]
    z_charge_surface = results["z_charge_surface"]

    # 1. YZ Cross-section
    ax1 = fig.add_subplot(gs[0, 0])
    x_mid = 0.5 * (x_min + x_max)
    slice_mask = np.abs(points[:, 0] - x_mid) < 200.0
    slice_pts = points[slice_mask]
    
    if len(slice_pts) > 10000: 
        slice_pts = slice_pts[np.random.choice(len(slice_pts), 10000, replace=False)]

    ax1.scatter(slice_pts[:, 1], slice_pts[:, 2], s=0.8, c="gray", alpha=0.3)
    ax1.add_patch(Circle((cy, cz), R, fill=False, color="red", linewidth=2.5))
    
    if abs(z_charge_surface - cz) < R:
        xc = np.sqrt(max(0.0, R ** 2 - (z_charge_surface - cz) ** 2))
        ax1.plot([cy - xc, cy + xc], [z_charge_surface, z_charge_surface], "b-", linewidth=2.5)
    
    ax1.set_title("Cross-section (YZ Plane - Trial 4)", fontweight="bold")
    ax1.axis("equal")

    # 2. XZ Side View
    ax2 = fig.add_subplot(gs[0, 1])
    ds = points[::max(1, len(points) // 12000)]
    ax2.scatter(ds[:, 0], ds[:, 2], s=0.4, c="gray", alpha=0.25, rasterized=True)
    ax2.axvline(x_min, color="red", linestyle="--", linewidth=2.5)
    ax2.axvline(x_max, color="red", linestyle="--", linewidth=2.5)
    ax2.set_title("Side View (XZ - Trial 4)", fontweight="bold")

    # 3. Results Table
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    table_data = [
        ["Parameter", "Value"],
        ["Effective Diameter (mm)", f"{results['param_1_diameter_mm']:.0f}"],
        ["Effective Length (mm)", f"{results['param_2_length_mm']:.0f}"],
        ["Free Height Over Load (mm)", f"{results['param_3_free_height_mm']:.0f}"],
        ["Total Internal Volume (m³)", f"{results['param_4_total_volume_m3']:.1f}"],
        ["Total Volume Occupied (m³)", f"{results['param_5_total_occupied_m3']:.1f}"],
        ["Cylinder Volume (m³)", f"{results['param_6_cylinder_volume_m3']:.1f}"],
        ["Cylinder Occupied (m³)", f"{results['param_7_cylinder_occupied_m3']:.1f}"],
        ["Total Mill Filling (%)", f"{results['param_8_total_fill_pct']:.2f}%"],
        ["Cylinder Filling (%)", f"{results['param_9_cylinder_fill_pct']:.2f}%"],
    ]
    table = ax3.table(cellText=table_data, cellLoc="left", loc="center", colWidths=[0.65, 0.35])
    table.scale(1.0, 2.3)
    for col in range(2):
        table[(0, col)].set_facecolor("#1565C0")
        table[(0, col)].set_text_props(weight="bold", color="white")
    ax3.set_title("Calculation Results (Deterministic)", fontweight="bold", pad=20)

    # 4. Fill Schematic
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_aspect("equal")
    ax4.add_patch(Circle((0.0, 0.0), R, fill=False, color="black", linewidth=3.5))
    zl = z_charge_surface - cz
    if 0.0 < results["charge_height_mm"] < 2.0 * R:
        th = np.linspace(0.0, 2.0 * np.pi, 600)
        xc, yc = R * np.cos(th), R * np.sin(th)
        m = yc <= zl
        if np.any(m):
            xc_f, yc_f = xc[m], yc[m]
            chord = float(np.sqrt(max(0.0, R ** 2 - zl ** 2)))
            ax4.fill(np.concatenate(([-chord], xc_f, [chord])), np.concatenate(([zl], yc_f, [zl])), color="#FF9800", alpha=0.8)
    ax4.set_xlim(-1.65 * R, 1.65 * R)
    ax4.set_ylim(-1.65 * R, 1.65 * R)
    ax4.set_title("Fill Schematic", fontweight="bold")

    plt.suptitle(f"Ball Mill Fill Level Analysis - {filename}", fontsize=14, fontweight="bold", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()