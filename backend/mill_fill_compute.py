# mill_fill_compute.py
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

def load_point_cloud(filepath):
    print(f"Loading: {Path(filepath).name}")
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix in [".xyz", ".txt", ".csv"]:
        points = np.loadtxt(filepath)[:, :3]
    elif suffix == ".npy":
        points = np.load(filepath)
    elif suffix == ".ply":
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(filepath))
        if len(pcd.points) == 0:
            raise ValueError("PLY file loaded but contains no points")
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    max_dim = np.ptp(points, axis=0).max()
    if max_dim < 100:
        print(f" Detected meters (max dim: {max_dim:.2f} m). Converting to mm...")
        points *= 1000.0
    else:
        print(f" Units appear to be mm (max dim: {max_dim:.0f} mm)")
    print(f" Loaded {len(points):,} points")
    return points


def correct_yz_alignment(points, n_iterations=3):
    print("\n=== YZ Alignment Correction ===")
    corrected = points.copy()
    total_cy, total_cz = 0.0, 0.0

    for iteration in range(n_iterations):
        x_mid = np.median(corrected[:, 0])
        slice_mask = np.abs(corrected[:, 0] - x_mid) < 250.0
        slice_pts = corrected[slice_mask]

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        r_thresh = np.percentile(r_all, 92.0)
        shell_mask = r_all >= r_thresh

        shell_pts_yz = slice_pts[shell_mask, 1:]
        r_shell = r_all[shell_mask]

        r_median = np.median(r_shell)
        r_std = np.std(r_shell)
        inlier = np.abs(r_shell - r_median) < 2.0 * r_std
        shell_pts_yz = shell_pts_yz[inlier]

        if len(shell_pts_yz) < 100:
            print(f" Iteration {iteration+1}: Insufficient shell points, stopping.")
            break

        def residuals(params):
            cy, cz, R = params
            return np.sqrt(
                (shell_pts_yz[:, 0] - cy) ** 2 +
                (shell_pts_yz[:, 1] - cz) ** 2
            ) - R

        r_init = np.median(r_shell[inlier])
        result = least_squares(
            residuals, [0.0, 0.0, r_init],
            loss="soft_l1", f_scale=10.0
        )

        cy, cz, R_fit = result.x
        rms = np.sqrt(np.mean(result.fun ** 2))

        print(
            f" Iteration {iteration+1}: Offset ({cy:.2f}, {cz:.2f}) mm, "
            f"R={R_fit:.1f} mm, RMS={rms:.2f} mm"
        )

        corrected[:, 1] -= cy
        corrected[:, 2] -= cz
        total_cy += cy
        total_cz += cz

        if np.sqrt(cy ** 2 + cz ** 2) < 3.0:
            print(" Converged (offset < 3 mm)")
            break

    print(f" Total correction: y={total_cy:.2f}, z={total_cz:.2f} mm\n")
    return corrected, np.array([total_cy, total_cz])


def find_head_planes(points):
    print("\n=== STEP 1: Find Head Planes ===")
    x = points[:, 0]
    r = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)

    n_bins = 150
    x_bounds = np.percentile(x, [0.5, 99.5])
    x_edges = np.linspace(x_bounds[0], x_bounds[1], n_bins + 1)

    med_radius = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (x >= x_edges[i]) & (x < x_edges[i + 1])
        if np.sum(mask) > 50:
            med_radius[i] = np.median(r[mask])

    valid = ~np.isnan(med_radius)
    if not np.any(valid):
        raise ValueError("No valid bins for radius along X")

    med_radius_smooth = med_radius.copy()
    med_radius_smooth[valid] = gaussian_filter1d(med_radius[valid], sigma=3)

    valid_r = med_radius_smooth[valid]
    plateau_r = np.median(valid_r[valid_r >= np.percentile(valid_r, 67)])

    lo, hi = 0.93 * plateau_r, 1.07 * plateau_r
    plateau_idx = np.where((med_radius_smooth >= lo) & (med_radius_smooth <= hi))[0]

    if len(plateau_idx) < 5:
        lo, hi = 0.90 * plateau_r, 1.10 * plateau_r
        plateau_idx = np.where((med_radius_smooth >= lo) & (med_radius_smooth <= hi))[0]

    if len(plateau_idx) < 5:
        raise ValueError("Cannot identify cylindrical plateau along X")

    i_start = max(0, plateau_idx[0] - 2)
    i_end = min(n_bins - 1, plateau_idx[-1] + 2)

    x_min = x_edges[i_start]
    x_max = x_edges[i_end + 1]

    eff_len = x_max - x_min

    print(f" Plateau median radius: {plateau_r:.1f} mm")
    print(f" X range: [{x_min:.1f}, {x_max:.1f}] mm")
    print(f" Effective length: {eff_len:.1f} mm")

    return x_min, x_max, eff_len


def fit_shell_circle_adaptive(points, x_min, x_max):
    print("\n=== STEP 2: Fit Shell Circle (Adaptive Method) ===")

    x_mid = 0.5 * (x_min + x_max)
    x_positions = [x_mid, x_mid - 400.0, x_mid + 400.0]

    all_shell_pts = []
    all_radii = []

    for xp in x_positions:
        mask = np.abs(points[:, 0] - xp) < 200.0
        slice_pts = points[mask]
        if len(slice_pts) < 500:
            continue

        r_all = np.sqrt(slice_pts[:, 1] ** 2 + slice_pts[:, 2] ** 2)
        cut = np.percentile(r_all, 95.0)
        shell = r_all >= cut

        if np.sum(shell) > 50:
            all_shell_pts.append(slice_pts[shell, 1:])
            all_radii.append(r_all[shell])

    if not all_shell_pts:
        raise ValueError("Could not detect shell points")

    shell_pts = np.vstack(all_shell_pts)

    def residuals(p):
        cy, cz, R = p
        return np.sqrt(
            (shell_pts[:, 0] - cy) ** 2 +
            (shell_pts[:, 1] - cz) ** 2
        ) - R

    r_init = np.median(np.concatenate(all_radii))
    res = least_squares(residuals, [0, 0, r_init], loss="soft_l1", f_scale=20.0)

    cy, cz, r_outer = res.x
    rms = np.sqrt(np.mean(res.fun ** 2))

    if r_outer > 5500:
        pct = 0.015
    elif r_outer > 3200:
        pct = 0.020
    elif r_outer > 2500:
        pct = 0.025
    else:
        pct = 0.030

    liner = max(80.0, min(200.0, pct * r_outer))
    r_eff = r_outer - liner

    print(f" Effective Radius: {r_eff:.1f} mm")

    return cy, cz, r_eff


def compute_fill_improved(points, x_min, x_max, cy, cz, radius):
    print("\n=== STEP 3: Compute Fill Level (ADAPTIVE v2) ===")
    effective_length_mm = x_max - x_min
    in_cyl = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    cyl_points = points[in_cyl]
    r_from_center = np.sqrt(
        (cyl_points[:, 1] - cy) ** 2 + (cyl_points[:, 2] - cz) ** 2
    )
    interior_mask = r_from_center < (0.85 * radius)
    interior_points = cyl_points[interior_mask]
    print(f" Points in cylinder: {len(cyl_points):,}")
    print(f" Interior points (r < 0.85R): {len(interior_points):,}")
    expected_density = len(cyl_points) / max(effective_length_mm, 1.0)
    interior_density = len(interior_points) / max(effective_length_mm, 1.0)
    print(
        f" Point density: {expected_density:.2f} pts/mm (total), "
        f"{interior_density:.2f} pts/mm (interior)"
    )
    if interior_density < 0.5:
        print(" WARNING: Very low interior point density (<0.5 pts/mm)")
    shell_top_z_mm = cz + radius
    shell_bottom_z_mm = cz - radius
    if len(interior_points) < 100:
        print(" WARNING: Very few interior points. Mill appears nearly empty.")
        z_charge_surface_mm = shell_bottom_z_mm + 100.0
        charge_height_mm = 100.0
    else:
        z_all = interior_points[:, 2]
        z_threshold = shell_bottom_z_mm + 0.45 * (2.0 * radius)
        bottom_mask = z_all < z_threshold
        bottom_points = interior_points[bottom_mask]
        print(f" Points in lower 45% by Z: {len(bottom_points):,}")
        if len(bottom_points) < 100:
            z_values = interior_points[:, 2]
            print(
                f" WARNING: Few points in lower region. "
                f"Using all {len(z_values):,} interior points."
            )
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
        z_p10 = np.percentile(z_values, 10.0)
        h_p005 = z_p005 - shell_bottom_z_mm
        h_p01 = z_p01 - shell_bottom_z_mm
        h_p02 = z_p02 - shell_bottom_z_mm
        h_p03 = z_p03 - shell_bottom_z_mm
        h_p05 = z_p05 - shell_bottom_z_mm
        h_p08 = z_p08 - shell_bottom_z_mm
        h_p10 = z_p10 - shell_bottom_z_mm
        fill_p005 = h_p005 / (2.0 * radius)
        fill_p01 = h_p01 / (2.0 * radius)
        fill_p02 = h_p02 / (2.0 * radius)
        fill_p03 = h_p03 / (2.0 * radius)
        fill_p05 = h_p05 / (2.0 * radius)
        fill_p08 = h_p08 / (2.0 * radius)
        fill_p10 = h_p10 / (2.0 * radius)
        print(
            " Fill estimates: "
            f"p0.5={fill_p005*100:.1f}%, p1={fill_p01*100:.1f}%, "
            f"p2={fill_p02*100:.1f}%, p3={fill_p03*100:.1f}%, "
            f"p5={fill_p05*100:.1f}%, p8={fill_p08*100:.1f}%, "
            f"p10={fill_p10*100:.1f}%"
        )
        z_median = np.median(z_values)
        z_mad = np.median(np.abs(z_values - z_median))
        z_std = np.std(z_values)
        noise_ratio = z_mad / max(radius, 1.0)
        print(
            f" Data quality: MAD={z_mad:.1f}mm, std={z_std:.1f}mm, "
            f"noise_ratio={noise_ratio:.4f}"
        )
        point_coverage_ratio = (
            len(bottom_points) / len(interior_points) if len(interior_points) > 0 else 0.0
        )
        print(
            f" Coverage ratio: {point_coverage_ratio*100:.1f}% "
            "of interior points in lower region"
        )
        if use_alternative or point_coverage_ratio < 0.15:
            if noise_ratio < 0.020:
                PERCENTILE = 3.0
                z_charge_surface_mm = z_p03
            elif noise_ratio < 0.035:
                PERCENTILE = 5.0
                z_charge_surface_mm = z_p05
            else:
                PERCENTILE = 8.0
                z_charge_surface_mm = z_p08
            quality_mode = "alternative"
        else:
            if noise_ratio < 0.015:
                PERCENTILE = 0.5
                z_charge_surface_mm = z_p005
            elif noise_ratio < 0.025:
                PERCENTILE = 1.0
                z_charge_surface_mm = z_p01
            elif noise_ratio < 0.040:
                PERCENTILE = 2.0
                z_charge_surface_mm = z_p02
            else:
                PERCENTILE = 3.0
                z_charge_surface_mm = z_p03
            quality_mode = "normal"
        print(f" Using p{PERCENTILE} ({quality_mode} method)")
        charge_height_mm = z_charge_surface_mm - shell_bottom_z_mm
        fill_raw = charge_height_mm / (2.0 * radius)
        print(f" Raw fill estimate: {fill_raw*100:.2f}%")
        mill_diameter_m = 2.0 * radius / 1000.0
        correction_factor = _adaptive_correction_factor(
            fill_raw,
            noise_ratio,
            point_coverage_ratio,
            interior_density,
            mill_diameter_m,
        )
        charge_height_mm *= correction_factor
        z_charge_surface_mm = shell_bottom_z_mm + charge_height_mm
    if charge_height_mm < 0:
        print(" WARNING: Negative height. Adjusting to 50mm.")
        charge_height_mm = 50.0
        z_charge_surface_mm = shell_bottom_z_mm + 50.0
    elif charge_height_mm > 2.0 * radius:
        print(" WARNING: Height exceeds diameter. Capping.")
        charge_height_mm = 1.95 * radius
        z_charge_surface_mm = shell_bottom_z_mm + charge_height_mm
    free_height_mm = shell_top_z_mm - z_charge_surface_mm
    print("\nGeometry Summary:")
    print(f" Shell center Z: {cz:.1f} mm")
    print(f" Shell bottom Z: {shell_bottom_z_mm:.1f} mm")
    print(f" Charge surface Z: {z_charge_surface_mm:.1f} mm")
    print(f" Shell top Z: {shell_top_z_mm:.1f} mm")
    print(f" Charge height: {charge_height_mm:.1f} mm")
    print(f" Free height: {free_height_mm:.1f} mm")
    R_m = radius / 1000.0
    L_m = effective_length_mm / 1000.0
    h_charge_m = charge_height_mm / 1000.0
    vol_cylinder_m3 = float(np.pi * R_m ** 2 * L_m)
    if h_charge_m <= 0.0:
        area_charge_m2 = 0.0
    elif h_charge_m >= 2.0 * R_m:
        area_charge_m2 = float(np.pi * R_m ** 2)
    else:
        if h_charge_m <= R_m:
            theta = 2.0 * np.arccos((R_m - h_charge_m) / R_m)
            area_charge_m2 = float(0.5 * R_m ** 2 * (theta - np.sin(theta)))
        else:
            h_empty_m = 2.0 * R_m - h_charge_m
            theta = 2.0 * np.arccos((R_m - h_empty_m) / R_m)
            area_empty_m2 = 0.5 * R_m ** 2 * (theta - np.sin(theta))
            area_charge_m2 = float(np.pi * R_m ** 2 - area_empty_m2)
    vol_charge_cylinder_m3 = float(area_charge_m2 * L_m)
    head_factor = 1.184
    vol_total_m3 = float(vol_cylinder_m3 * head_factor)
    charge_penetration_factor = 0.992
    vol_charge_total_m3 = float(vol_charge_cylinder_m3 * charge_penetration_factor)
    total_fill_pct = (
        0.0 if vol_total_m3 <= 0.0 else 100.0 * vol_charge_total_m3 / vol_total_m3
    )
    cylinder_fill_pct = (
        0.0 if vol_cylinder_m3 <= 0.0 else 100.0 * vol_charge_cylinder_m3 / vol_cylinder_m3
    )
    print("\nVolume Results:")
    print(f" Effective diameter: {2.0 * radius:.0f} mm")
    print(f" Effective length: {effective_length_mm:.0f} mm")
    print(f" Total mill fill: {total_fill_pct:.2f}%")
    print(f" Cylinder fill: {cylinder_fill_pct:.2f}%")
    return {
        "effective_diameter_mm": 2.0 * radius,
        "effective_length_mm": effective_length_mm,
        "free_height_mm": free_height_mm,
        "cylinder_volume_m3": vol_cylinder_m3,
        "charge_cylinder_m3": vol_charge_cylinder_m3,
        "total_volume_m3": vol_total_m3,
        "charge_total_m3": vol_charge_total_m3,
        "total_fill_pct": total_fill_pct,
        "cylinder_fill_pct": cylinder_fill_pct,
        "cy": cy,
        "cz": cz,
        "radius": radius,
        "x_min": x_min,
        "x_max": x_max,
        "z_charge_surface": z_charge_surface_mm,
    }

def _adaptive_correction_factor(
    fill_raw, noise_ratio, point_coverage_ratio, interior_density, mill_diameter_m
):
    q_noise = max(0.0, 0.04 - min(noise_ratio, 0.04)) / 0.04
    q_cov = min(1.0, point_coverage_ratio / 0.25)
    q_den = min(1.0, interior_density / 1.0)

    quality_score = 0.5 * q_noise + 0.3 * q_cov + 0.2 * q_den
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    def conservative_factor(f):
        if f < 0.20:
            return 0.82
        if f < 0.30:
            return 0.87
        if f < 0.40:
            return 0.92
        if f < 0.50:
            return 0.95
        return 0.97

    def aggressive_factor(f):
        if f < 0.20:
            return 0.96
        if f < 0.30:
            return 1.02
        if f < 0.40:
            return 1.06
        if f < 0.50:
            return 1.09
        return 1.12

    base_cons = conservative_factor(fill_raw)
    base_aggr = aggressive_factor(fill_raw)

    w_aggr = 1.0 - quality_score
    correction_factor = (1.0 - w_aggr) * base_cons + w_aggr * base_aggr

    if point_coverage_ratio < 0.02:
        coverage_scale = 0.90
    elif point_coverage_ratio < 0.10:
        coverage_scale = 0.94
    elif point_coverage_ratio < 0.20:
        coverage_scale = 0.97
    else:
        coverage_scale = 1.0

    if mill_diameter_m > 10.0:
        diameter_scale = 0.97
    else:
        diameter_scale = 1.0

    correction_factor *= coverage_scale * diameter_scale

    print(
        f" Data quality score: {quality_score:.3f}, "
        f"base_cons={base_cons:.3f}, base_aggr={base_aggr:.3f}"
    )
    print(
        f" Coverage scale={coverage_scale:.3f}, "
        f"diameter scale={diameter_scale:.3f}"
    )
    print(f" Final correction factor={correction_factor:.3f}")

    return correction_factor
