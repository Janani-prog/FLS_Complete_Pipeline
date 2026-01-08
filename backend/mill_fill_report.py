# mill_fill_report.py
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from mill_fill_compute import (
    load_point_cloud,
    correct_yz_alignment,
    find_head_planes,
    fit_shell_circle_adaptive,
    compute_fill_improved,
)


def make_plots(points, results, output_path: Path, filename: str):
    print("\n=== STEP 4: Generate Plots ===")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    cy = results["cy"]
    cz = results["cz"]
    R = results["radius"]
    x_min = results["x_min"]
    x_max = results["x_max"]
    z_charge_surface = results["z_charge_surface"]
    ax1 = fig.add_subplot(gs[0, 0])
    x_mid = 0.5 * (x_min + x_max)
    slice_mask = np.abs(points[:, 0] - x_mid) < 150.0
    slice_pts = points[slice_mask]
    if len(slice_pts) > 10000:
        idx = np.random.choice(len(slice_pts), 10000, replace=False)
        slice_pts = slice_pts[idx]
    ax1.scatter(
        slice_pts[:, 1],
        slice_pts[:, 2],
        s=0.8,
        c="gray",
        alpha=0.3,
        label="Point cloud",
    )
    circle = Circle(
        (cy, cz),
        R,
        fill=False,
        color="red",
        linewidth=2.5,
        label=f"Effective Shell (D={2*R:.0f} mm)",
    )
    ax1.add_patch(circle)
    ax1.plot(cy, cz, "r+", markersize=14, markeredgewidth=2.5)
    if abs(z_charge_surface - cz) < R:
        x_chord = np.sqrt(max(0.0, R ** 2 - (z_charge_surface - cz) ** 2))
        ax1.plot(
            [cy - x_chord, cy + x_chord],
            [z_charge_surface, z_charge_surface],
            "b-",
            linewidth=2.5,
            label="Charge surface",
        )
    ax1.set_xlabel("Y (mm)", fontsize=11)
    ax1.set_ylabel("Z (mm)", fontsize=11)
    ax1.set_title("Cross-section (YZ Plane)", fontweight="bold", fontsize=12)
    ax1.axis("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax2 = fig.add_subplot(gs[0, 1])
    step = max(1, len(points) // 12000)
    ds = points[::step]
    ax2.scatter(
        ds[:, 0],
        ds[:, 2],
        s=0.4,
        c="gray",
        alpha=0.25,
        rasterized=True,
        label="Point cloud",
    )
    ax2.axvline(x_min, color="red", linestyle="--", linewidth=2.5, label="Head planes")
    ax2.axvline(x_max, color="red", linestyle="--", linewidth=2.5)
    z_mid = cz
    ax2.annotate(
        "",
        xy=(x_min, z_mid),
        xytext=(x_max, z_mid),
        arrowprops=dict(arrowstyle="<->", lw=2.5, color="red"),
    )
    ax2.text(
        0.5 * (x_min + x_max),
        z_mid - 450.0,
        f"{results['effective_length_mm']:.0f} mm",
        ha="center",
        fontsize=11,
        color="red",
        weight="bold",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="red",
            alpha=0.95,
            linewidth=1.5,
        ),
    )
    ax2.set_xlabel("X (Length, mm)", fontsize=11)
    ax2.set_ylabel("Z (Height, mm)", fontsize=11)
    ax2.set_title("Side View (XZ)", fontweight="bold", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    table_data = [
        ["Parameter", "Calculated"],
        ["Effective Diameter (mm)", f"{results['effective_diameter_mm']:.0f}"],
        ["Effective Length (mm)", f"{results['effective_length_mm']:.0f}"],
        ["Free Height Over Load (mm)", f"{results['free_height_mm']:.0f}"],
        ["Total Internal Vol (m³)", f"{results['total_volume_m3']:.1f}"],
        ["Total Vol Occupied by Load (m³)", f"{results['charge_total_m3']:.1f}"],
        ["Internal Vol of Cylinder (m³)", f"{results['cylinder_volume_m3']:.1f}"],
        ["Vol Occupied in Cylinder (m³)", f"{results['charge_cylinder_m3']:.1f}"],
        ["Total Mill Filling (%)", f"{results['total_fill_pct']:.2f}"],
        ["Cylinder Filling (%)", f"{results['cylinder_fill_pct']:.2f}"],
    ]
    table = ax3.table(
        cellText=table_data, cellLoc="left", loc="center", colWidths=[0.6, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.1)
    for col in range(2):
        cell = table[(0, col)]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(weight="bold", color="white", fontsize=10)
    ax3.set_title("Calculation Results", fontweight="bold", fontsize=12, pad=20)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_aspect("equal")
    shell = Circle((0.0, 0.0), R, fill=False, color="black", linewidth=3.5)
    ax4.add_patch(shell)
    z_level = z_charge_surface - cz
    h_charge = z_level + R
    h_free = results["free_height_mm"]
    if 0.0 < h_charge < 2.0 * R:
        theta = np.linspace(0.0, 2.0 * np.pi, 600)
        x_circ = R * np.cos(theta)
        y_circ = R * np.sin(theta)
        mask = y_circ <= z_level
        x_fill = x_circ[mask]
        y_fill = y_circ[mask]
        if len(x_fill) > 0:
            x_chord = float(np.sqrt(max(0.0, R ** 2 - z_level ** 2)))
            x_fill = np.concatenate(([-x_chord], x_fill, [x_chord]))
            y_fill = np.concatenate(([z_level], y_fill, [z_level]))
            ax4.fill(
                x_fill,
                y_fill,
                color="#FF9800",
                alpha=0.8,
                label="Charge",
                zorder=2,
            )
    if abs(z_level) < R:
        x_chord = float(np.sqrt(max(0.0, R ** 2 - z_level ** 2)))
        ax4.plot(
            [-x_chord, x_chord],
            [z_level, z_level],
            "b--",
            linewidth=2.5,
            label="Charge surface",
            zorder=3,
        )
    ax4.annotate(
        "",
        xy=(R * 1.25, R),
        xytext=(R * 1.25, z_level),
        arrowprops=dict(arrowstyle="<->", lw=2.5, color="#1976D2"),
    )
    ax4.text(
        R * 1.42,
        0.5 * (R + z_level),
        f"Free\n{h_free:.0f} mm",
        fontsize=11,
        color="#1976D2",
        weight="bold",
        ha="left",
        va="center",
    )
    ax4.annotate(
        "",
        xy=(-R * 1.25, -R),
        xytext=(-R * 1.25, z_level),
        arrowprops=dict(arrowstyle="<->", lw=2.5, color="#2E7D32"),
    )
    ax4.text(
        -R * 1.42,
        0.5 * (-R + z_level),
        f"Charge\n{h_charge:.0f} mm",
        fontsize=11,
        color="#2E7D32",
        weight="bold",
        ha="right",
        va="center",
    )
    ax4.axhline(0.0, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
    ax4.axvline(0.0, color="gray", linestyle=":", alpha=0.4, linewidth=1.0)
    ax4.plot(0.0, 0.0, "k+", markersize=12, markeredgewidth=2.5)
    ax4.set_xlim(-1.65 * R, 1.65 * R)
    ax4.set_ylim(-1.65 * R, 1.65 * R)
    ax4.set_title("Fill Schematic", fontweight="bold", fontsize=12)
    ax4.legend(loc="upper right", fontsize=10)
    plt.suptitle(
        f"Ball Mill Fill Level Analysis - ADAPTIVE v2 - {filename}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")

def main():
    print("=" * 72)
    print("BALL MILL FILL LEVEL ANALYZER - REPORT MODE")
    print("=" * 72)

    root = tk.Tk()
    root.withdraw()

    filepath = filedialog.askopenfilename(
        title="Select Pre-Aligned Point Cloud (X=axis)",
        filetypes=[("Point clouds", "*.ply *.xyz *.txt *.csv *.npy")]
    )
    root.destroy()

    if not filepath:
        print("No file selected. Exiting.")
        return

    points = load_point_cloud(filepath)
    points, _ = correct_yz_alignment(points)
    x_min, x_max, _ = find_head_planes(points)
    cy, cz, radius = fit_shell_circle_adaptive(points, x_min, x_max)
    results = compute_fill_improved(points, x_min, x_max, cy, cz, radius)

    out_dir = Path(r"C:\FLS\output\mill_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{Path(filepath).stem}_analysis.png"
    make_plots(points, results, out_path, Path(filepath).name)

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
