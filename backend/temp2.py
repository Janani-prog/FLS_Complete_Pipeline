import numpy as np
import warnings
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ============================================================================
# PART 1: POINT CLOUD PROCESSING (From Script 3)
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
        # Suppress Open3D console output if possible
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

    return corrected

# --- TRIAL 9 LOGIC (Diameter & Length) ---

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
        # Fallback if specific slices fail: take entire center chunk
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

# --- TRIAL 4 LOGIC (Free Height) ---

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
        # Fallback: use bounds from Trial 9 if detection fails
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
        # Fallback
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
    """Compute correction factor - ADHI method."""
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

    if len(interior_points) < 100:
        z_charge_surface_mm = shell_bottom_z_mm + 100.0
        charge_height_mm = 100.0
        noise_ratio = 0.0
        interior_density = 0.0
        point_coverage_ratio = 0.0
    else:
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
    return free_height_mm

# ============================================================================
# PART 2: THE PHYSICS ENGINE & MODEL (From temp1.py)
# ============================================================================

def get_theoretical_geometry(dia_mm, len_mm, free_height_mm):
    """Calculates the EXACT geometric volume of a perfect cylinder segment."""
    R = dia_mm / 2.0 / 1000.0
    L = len_mm / 1000.0
    H_charge = (dia_mm - free_height_mm) / 1000.0
    
    # Cylinder Volume
    vol_cyl = np.pi * R**2 * L
    
    # Occupied Volume
    if H_charge <= 0:
        area_occ = 0.0
    elif H_charge >= 2*R:
        area_occ = np.pi * R**2
    else:
        val = np.clip((R - H_charge) / R, -1.0, 1.0) 
        term1 = R**2 * np.arccos(val)
        term2 = (R - H_charge) * np.sqrt(max(0, 2*R*H_charge - H_charge**2))
        area_occ = term1 - term2
    
    vol_occ = area_occ * L
    return np.array([vol_cyl, vol_occ])

def predict_single_mill(user_dia, user_len, user_fh):
    # ==========================================
    # 2. DEFINE TRAINING DATA (ALL FILES 1-8)
    # ==========================================
    X_train = np.array([
        [6870, 3795, 5177],   # FILE_1
        [11646, 7657, 8735],  # FILE_2
        [5862, 11681, 3634],  # FILE_3
        [6464, 11335, 3911],  # FILE_4
        [5828, 11677, 3598],  # FILE_5
        [6444, 11332, 3894],  # FILE_6
        [5850, 11678, 3663],  # FILE_7
        [5857, 11681, 3788]   # FILE_8
    ])
    
    y_train = np.array([
        [166.6, 28.2, 140.7, 24.3],
        [942.1, 163.7, 815.6, 144.4],
        [337.2, 114.5, 315.3, 108.4],
        [403.4, 141.7, 371.9, 131.5],
        [333.2, 110.8, 311.5, 105.6],
        [400.1, 147.8, 369.6, 136.8],
        [336.0, 115.2, 313.9, 109.1],
        [337.0, 105.7, 314.7, 100.5]
    ])
    
    print(f"Training prediction model on {len(X_train)} historical files...")

    # ==========================================
    # 3. CALCULATE CORRECTION FACTORS (TRAINING)
    # ==========================================
    ratios_train = []
    for i in range(len(X_train)):
        phys = get_theoretical_geometry(X_train[i,0], X_train[i,1], X_train[i,2])
        # Calculate ratios: Actual / Physics
        r = [y_train[i,j] / phys[j%2] for j in range(4)]
        ratios_train.append(r)
    ratios_train = np.array(ratios_train)
    
    # ==========================================
    # 4. TRAIN MODELS
    # ==========================================
    kernel = C(1.0) * RBF(length_scale=[1000, 1000, 500]) + WhiteKernel(noise_level=1e-5)
    models = []
    for i in range(4):
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gpr.fit(X_train, ratios_train[:, i])
        models.append(gpr)
        
    # ==========================================
    # 5. PREDICT USER INPUT
    # ==========================================
    X_user = np.array([[user_dia, user_len, user_fh]])
    
    # A. Calculate Physics Base for User Input
    phys_user = get_theoretical_geometry(user_dia, user_len, user_fh)
    
    # B. Predict Ratios
    pred_ratios = [m.predict(X_user)[0] for m in models]
    
    # C. Apply Ratios
    final_preds = [
        phys_user[0] * pred_ratios[0], # Total Int
        phys_user[1] * pred_ratios[1], # Total Occ
        phys_user[0] * pred_ratios[2], # Cyl Int
        phys_user[1] * pred_ratios[3]  # Cyl Occ
    ]
    
    # D. Derive Percentages
    pct_total = (final_preds[1] / final_preds[0]) * 100
    pct_cyl = (final_preds[3] / final_preds[2]) * 100
    final_preds.extend([pct_total, pct_cyl])
    
    # ==========================================
    # 6. PRINT RESULTS
    # ==========================================
    labels = [
        'Total Internal Volume (m³)', 
        'Total Volume Occupied (m³)', 
        'Internal Volume of Cylinder (m³)', 
        'Volume Occupied in Cylinder (m³)', 
        'Total Filling Level (%)', 
        'Cylinder Filling Level (%)'
    ]
    
    print("\n" + "="*60)
    print(f"PREDICTION RESULTS (Model-Corrected)")
    print(f"Based on Inputs: Dia={user_dia:.0f}, Len={user_len:.0f}, FH={user_fh:.0f}")
    print("="*60)
    print(f"{'Parameter':<35} | {'Value':<10}")
    print("-" * 60)
    
    for i in range(6):
        print(f"{labels[i]:<35} | {final_preds[i]:<10.2f}")
    print("-" * 60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def process_file_and_predict():
    print("=" * 70)
    print("AUTOMATED BALL MILL VOLUME PREDICTOR")
    print("=" * 70)

    # 1. Select File
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select Pre-Aligned Point Cloud",
        filetypes=[("Point clouds", "*.ply *.xyz *.txt *.csv *.npy"), ("All files", "*.*")],
    )
    root.destroy()

    if not filepath:
        print("No file selected. Exiting.")
        return

    try:
        # 2. Process Point Cloud
        points = load_point_cloud(filepath)
        points = correct_yz_alignment(points)
        
        print("\n[1/2] Analyzing Geometry (Trial 9)...")
        x_min_t9, x_max_t9, effective_length = find_head_planes_trial9(points)
        cy_t9, cz_t9, radius_t9 = fit_shell_circle_trial9(points, x_min_t9, x_max_t9)
        effective_diameter = 2.0 * radius_t9
        
        print(f"      -> Diameter: {effective_diameter:.1f} mm")
        print(f"      -> Length:   {effective_length:.1f} mm")

        print("\n[2/2] Analyzing Free Height (Trial 4)...")
        x_min_t4, x_max_t4 = find_head_planes_trial4(points)
        cy_t4, cz_t4, radius_t4 = fit_shell_circle_trial4(points, x_min_t4, x_max_t4)
        free_height = compute_free_height_trial4(points, x_min_t4, x_max_t4, cy_t4, cz_t4, radius_t4)
        print(f"      -> Free Height: {free_height:.1f} mm")

        # 3. Feed into Prediction Engine
        predict_single_mill(effective_diameter, effective_length, free_height)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    process_file_and_predict()
