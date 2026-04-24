import numpy as np
import math
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _double_factorial(n: int) -> int:
    """Compute double factorial n!!"""
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


def _assoc_legendre_no_cs(l: int, m: int, x):
    """Associated Legendre polynomial P_l^m(x) without Condon-Shortley factor."""
    if m < 0 or m > l:
        raise ValueError("Require 0 <= m <= l")
    
    # P_m^m
    P_mm = _double_factorial(2 * m - 1) * np.power(np.maximum(1.0 - x * x, 0.0), m / 2.0)
    if l == m:
        return P_mm
    
    # P_{m+1}^m
    P_m1m = (2 * m + 1) * x * P_mm
    if l == m + 1:
        return P_m1m
    
    # Upward recurrence
    P_lm_minus2 = P_mm
    P_lm_minus1 = P_m1m
    for ell in range(m + 2, l + 1):
        P_lm = ((2 * ell - 1) * x * P_lm_minus1 - (ell + m - 1) * P_lm_minus2) / (ell - m)
        P_lm_minus2, P_lm_minus1 = P_lm_minus1, P_lm
    return P_lm_minus1


def _norm_lm(l: int, m: int) -> float:
    """Normalization factor for spherical harmonics."""
    log_ratio = math.lgamma(l - m + 1) - math.lgamma(l + m + 1)
    return math.sqrt((2 * l + 1) / (4 * math.pi) * math.exp(log_ratio))


def _complex_Y_lm(l: int, m: int, theta_grid, phi_grid):
    """Compute complex spherical harmonic Y_l^m."""
    if m < 0 or m > l:
        raise ValueError("Require 0 <= m <= l")
    x = np.cos(theta_grid)
    P = _assoc_legendre_no_cs(l, m, x)
    N = _norm_lm(l, m)
    return (N * P) * np.exp(1j * m * phi_grid)


def compute_real_SH(l: int, m: int, theta_grid, phi_grid):
    """
    Compute real spherical harmonic Y_l^m.
    
    Parameters:
    -----------
    l : int
        Band number (l >= 0)
    m : int
        Order (-l <= m <= l)
    theta_grid : ndarray
        Polar angle (0 to π)
    phi_grid : ndarray
        Azimuthal angle (0 to 2π)
    
    Returns:
    --------
    Y : ndarray
        Real spherical harmonic values
    """
    m_abs = abs(m)
    
    if m_abs == 0:
        Y = np.real(_complex_Y_lm(l, 0, theta_grid, phi_grid))
    else:
        Y_pos = _complex_Y_lm(l, m_abs, theta_grid, phi_grid)
        if m > 0:
            Y = np.sqrt(2) * np.real(Y_pos)
        else:
            Y = np.sqrt(2) * np.imag(Y_pos)
    
    return Y


def precompute_all_SH(theta_grid, phi_grid, max_band):
    """
    Pre-compute all spherical harmonics up to max_band.
    This avoids redundant computation when processing multiple channels.
    
    Returns:
    --------
    sh_basis : dict
        Dictionary mapping (l, m) to pre-computed Y_l^m arrays
    """
    sh_basis = {}
    for l in range(max_band + 1):
        for m in range(-l, l + 1):
            sh_basis[(l, m)] = compute_real_SH(l, m, theta_grid, phi_grid)
    return sh_basis


def compute_SH_coefficients(target, theta_grid, phi_grid, max_band=3, verbose=True, sh_basis=None):
    """
    Decompose a spherical function into SH coefficients.
    
    This is the spherical analog of 2D Fourier decomposition in 0302!
    
    Parameters:
    -----------
    sh_basis : dict, optional
        Pre-computed SH basis functions. If None, will compute on-the-fly.
    """
    coeffs = {}
    
    # Integration weights: sin(theta) * dtheta * dphi
    dtheta = theta_grid[1, 0] - theta_grid[0, 0]
    dphi = phi_grid[0, 1] - phi_grid[0, 0]
    weights = np.sin(theta_grid) * dtheta * dphi
    
    if verbose:
        print(f"Computing coefficients for bands 0-{max_band}...")
    
    for l in range(max_band + 1):
        for m in range(-l, l + 1):
            # Use pre-computed or compute on-the-fly
            if sh_basis is not None and (l, m) in sh_basis:
                Y_lm = sh_basis[(l, m)]
            else:
                Y_lm = compute_real_SH(l, m, theta_grid, phi_grid)
            
            # Project: integrate target * Y_lm over sphere
            coeff = np.sum(target * Y_lm * weights)
            coeffs[(l, m)] = coeff
        if verbose:
            print(f"  Band {l} complete ({2*l+1} coefficients)")
    
    return coeffs


def compute_SH_coefficients_rgb(target_R, target_G, target_B, theta_grid, phi_grid, max_band=3, verbose=True):
    """Compute SH coefficients for RGB channels in one pass.

    Computes each basis function Y_l^m once and projects all three channels.
    This avoids recomputing the basis 3× when `sh_basis` is not precomputed.
    """
    coeffs_R = {}
    coeffs_G = {}
    coeffs_B = {}

    # Integration weights: sin(theta) * dtheta * dphi
    dtheta = theta_grid[1, 0] - theta_grid[0, 0]
    dphi = phi_grid[0, 1] - phi_grid[0, 0]
    weights = np.sin(theta_grid) * dtheta * dphi

    if verbose:
        print(f"Computing RGB coefficients for bands 0-{max_band}...")

    for l in range(max_band + 1):
        for m in range(-l, l + 1):
            Y_lm = compute_real_SH(l, m, theta_grid, phi_grid)
            wY = Y_lm * weights
            coeffs_R[(l, m)] = np.sum(target_R * wY)
            coeffs_G[(l, m)] = np.sum(target_G * wY)
            coeffs_B[(l, m)] = np.sum(target_B * wY)
        if verbose:
            print(f"  Band {l} complete ({2*l+1} coefficients)")

    return coeffs_R, coeffs_G, coeffs_B


def reconstruct_from_SH(coeffs, theta_grid, phi_grid, max_band, sh_basis=None):
    """
    Reconstruct spherical function from SH coefficients.
    
    This is the inverse operation: sum(c_l^m * Y_l^m)
    
    Parameters:
    -----------
    sh_basis : dict, optional
        Pre-computed SH basis functions. If None, will compute on-the-fly.
    """
    reconstruction = np.zeros_like(theta_grid, dtype=float)
    
    for l in range(max_band + 1):
        for m in range(-l, l + 1):
            if (l, m) in coeffs:
                # Use pre-computed or compute on-the-fly
                if sh_basis is not None and (l, m) in sh_basis:
                    Y_lm = sh_basis[(l, m)]
                else:
                    Y_lm = compute_real_SH(l, m, theta_grid, phi_grid)
                reconstruction += coeffs[(l, m)] * Y_lm
    
    return reconstruction


def grid_to_mesh_indices(h: int, w: int):
    """Generate triangle indices for an h×w grid for Mesh3d, wrapping the seam."""
    tris = []
    for r in range(h - 1):
        base = r * w
        next_base = (r + 1) * w
        for c in range(w):
            c_next = (c + 1) % w
            v0 = base + c
            v1 = base + c_next
            v2 = next_base + c
            v3 = next_base + c_next
            tris.append((v0, v1, v2))
            tris.append((v2, v1, v3))
    i_idx, j_idx, k_idx = zip(*tris)
    return np.array(i_idx), np.array(j_idx), np.array(k_idx)


def main():
    """Main execution function for demonstration."""
    # Load Blue Marble image
    img = Image.open('Blue_Marble_2002.png')
    print(f"Original image size: {img.size}")

    # Resize for faster computation (optional)
    target_resolution = 100
    img_resized = img.resize((target_resolution * 2, target_resolution), Image.Resampling.LANCZOS)
    print(f"Resized to: {img_resized.size}")

    # Convert to numpy array (values 0-255)
    img_array = np.array(img_resized).astype(float) / 255.0  # Normalize to [0, 1]
    print(f"Image array shape: {img_array.shape}")

    # Extract RGB channels
    target_R = img_array[:, :, 0]
    target_G = img_array[:, :, 1]
    target_B = img_array[:, :, 2]

    # Create spherical grid matching image dimensions
    resolution_theta = img_array.shape[0]
    resolution_phi = img_array.shape[1]

    theta = np.linspace(0, np.pi, resolution_theta)  # Polar angle (top to bottom)
    phi = np.linspace(0, 2 * np.pi, resolution_phi, endpoint=False)  # Azimuthal angle (left to right)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Convert to Cartesian for 3D plotting
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    print(f"\nSpherical grid: {resolution_theta} × {resolution_phi}")
    print(f"RGB value ranges: R=[{target_R.min():.3f}, {target_R.max():.3f}], "
          f"G=[{target_G.min():.3f}, {target_G.max():.3f}], "
          f"B=[{target_B.min():.3f}, {target_B.max():.3f}]")

    # Compute coefficients for RGB.
    # NOTE: Band 100 implies 10,201 basis functions; precomputing & storing all Y_l^m
    # on a dense grid can consume a lot of RAM. This script computes basis functions
    # on-the-fly and reuses them across channels to keep memory usage reasonable.
    max_band_to_compute = 100

    print(f"\nDecomposing RGB channels into spherical harmonics (bands 0-{max_band_to_compute})...\n")

    coeffs_R, coeffs_G, coeffs_B = compute_SH_coefficients_rgb(
        target_R, target_G, target_B,
        theta_grid, phi_grid,
        max_band=max_band_to_compute,
        verbose=True,
    )

    print(f"\n✅ Computed {len(coeffs_R)} coefficients per channel")
    print(f"   Total: {3 * len(coeffs_R)} coefficients for RGB")
    print(f"   (3DGS uses only bands 0-3 = 16 RGB vectors = 48 floats)")
    print(f"   (We computed up to band {max_band_to_compute} = {len(coeffs_R)} coefficients per channel)")
    print(f"   (Band {max_band_to_compute} = {(max_band_to_compute+1)**2}×3 = {3*(max_band_to_compute+1)**2} floats total!)")

    # Show top coefficients for green channel
    print("\nTop 10 Green channel coefficients by magnitude:")
    sorted_coeffs = sorted(coeffs_G.items(), key=lambda x: abs(x[1]), reverse=True)
    for (l, m), c in sorted_coeffs[:10]:
        print(f"  c_{l}^{m:2d} = {c:7.4f}")

    # Reconstruct RGB channels incrementally per band.
    # This computes each Y_l^m once and updates the reconstruction, instead of
    # recomputing all lower bands for every slider step.
    print("\nReconstructing Earth image with different numbers of bands...\n")

    # Plotly becomes very heavy if we add a separate mesh trace for every band.
    # For high max_band values, visualize a curated subset while still computing
    # /printing metrics for every band.
    bands_to_visualize = list(range(0, min(max_band_to_compute, 20) + 1))
    if max_band_to_compute > 20:
        bands_to_visualize += [30, 40, 50, 60, 70, 80, 90, max_band_to_compute]
    bands_to_visualize = sorted({b for b in bands_to_visualize if 0 <= b <= max_band_to_compute})
    bands_to_visualize_set = set(bands_to_visualize)
    print(f"\nVisualizing {len(bands_to_visualize)} bands: {bands_to_visualize}")

    reconstructions_RGB = {}

    recon_R_lin = np.zeros_like(theta_grid, dtype=float)
    recon_G_lin = np.zeros_like(theta_grid, dtype=float)
    recon_B_lin = np.zeros_like(theta_grid, dtype=float)

    for l in range(max_band_to_compute + 1):
        print(f"Band {l}:")

        for m in range(-l, l + 1):
            Y_lm = compute_real_SH(l, m, theta_grid, phi_grid)
            recon_R_lin += coeffs_R[(l, m)] * Y_lm
            recon_G_lin += coeffs_G[(l, m)] * Y_lm
            recon_B_lin += coeffs_B[(l, m)] * Y_lm

        # Clip only for visualization + error metrics (keep linear sums for next bands)
        recon_R = np.clip(recon_R_lin, 0, 1)
        recon_G = np.clip(recon_G_lin, 0, 1)
        recon_B = np.clip(recon_B_lin, 0, 1)

        recon_RGB = np.stack([recon_R, recon_G, recon_B], axis=-1)
        if l in bands_to_visualize_set:
            reconstructions_RGB[l] = recon_RGB

        mse_R = np.mean((target_R - recon_R) ** 2)
        mse_G = np.mean((target_G - recon_G) ** 2)
        mse_B = np.mean((target_B - recon_B) ** 2)
        avg_mse = (mse_R + mse_G + mse_B) / 3
        rmse = np.sqrt(avg_mse)

        num_coeffs = (l + 1) ** 2
        print(f"  RMSE = {rmse:.6f}, Coeffs/channel = {num_coeffs}")

    print("\n✅ Reconstructions complete!")
    print(f"\n💡 Notice how RMSE decreases with more bands:")
    print(f"   Band 3 (3DGS) vs Band {max_band_to_compute}: "
          f"{np.sqrt(np.mean((target_G - reconstructions_RGB[3][:,:,1])**2)):.6f} → "
          f"{np.sqrt(np.mean((target_G - reconstructions_RGB[max_band_to_compute][:,:,1])**2)):.6f}")

    # Create interactive visualization
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]],
        subplot_titles=('Original Earth Image', 'SH Reconstruction'),
        horizontal_spacing=0.05
    )

    # Prepare mesh indices and flattened coordinates for per-vertex RGB
    i_idx, j_idx, k_idx = grid_to_mesh_indices(*x.shape)
    flat_x, flat_y, flat_z = x.ravel(), y.ravel(), z.ravel()

    # Brightness scaling for visualization only (doesn't affect SH coefficients)
    brightness_factor = 1.2

    # Left: Original (always visible) with RGB vertex colors
    fig.add_trace(go.Mesh3d(
        x=flat_x, y=flat_y, z=flat_z,
        i=i_idx, j=j_idx, k=k_idx,
        vertexcolor=np.clip(img_array.reshape(-1, 3) * brightness_factor, 0, 1),
        name='Original',
        showscale=False,
        lighting=dict(ambient=0.7, diffuse=0.4),
        flatshading=False
    ), row=1, col=1)

    # Right: Add reconstructions for all computed bands (RGB)
    for max_band in bands_to_visualize:
        recon_RGB = reconstructions_RGB[max_band]
        
        fig.add_trace(go.Mesh3d(
            x=flat_x, y=flat_y, z=flat_z,
            i=i_idx, j=j_idx, k=k_idx,
            vertexcolor=np.clip(recon_RGB.reshape(-1, 3) * brightness_factor, 0, 1),
            name=f'Band {max_band}',
            visible=(max_band == 0),  # Start with band 0 visible
            showscale=False,
            lighting=dict(ambient=0.7, diffuse=0.4),
            flatshading=False
        ), row=1, col=2)

    # Create slider steps
    steps = []

    for i, max_band in enumerate(bands_to_visualize):
        num_coeffs = (max_band + 1)**2
        
        # Original always visible (index 0), then toggle reconstructions
        visible = [True] + [False] * len(bands_to_visualize)
        visible[i + 1] = True
        
        recon_RGB = reconstructions_RGB[max_band]
        mse_R = np.mean((target_R - recon_RGB[:,:,0])**2)
        mse_G = np.mean((target_G - recon_RGB[:,:,1])**2)
        mse_B = np.mean((target_B - recon_RGB[:,:,2])**2)
        avg_rmse = np.sqrt((mse_R + mse_G + mse_B) / 3)
        
        title_suffix = " (3DGS Standard)" if max_band == 3 else ""
        
        steps.append(dict(
            method="update",
            args=[{"visible": visible},
                  {"title": f"Bands 0-{max_band}{title_suffix} | {num_coeffs} coeffs/channel | RMSE={avg_rmse:.5f}"}],
            label=f"Band {max_band}"
        ))

    # Configure layout with slider
    fig.update_layout(
        sliders=[dict(
            active=0,
            yanchor="top",
            y=0.02,
            xanchor="left",
            x=0.05,
            currentvalue=dict(
                prefix="Reconstruction: ",
                visible=True,
                xanchor="left",
                font=dict(size=14)
            ),
            pad=dict(b=10, t=10),
            len=0.9,
            steps=steps
        )],
        title="Bands 0-0 | 1 coeffs/channel | RMSE=0.00000",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        scene2=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            camera_projection=dict(type='perspective')
        ),
        width=1400, height=600,
        uirevision='constant'
    )

    fig.show()

    print("\n🎛️ Use the slider to compare original (left) vs SH reconstruction (right)!")
    print("🌍 Watch how the reconstruction improves with more bands:")
    print("   • Band 0: Average blue color (DC component)")
    print("   • Band 1: Major land/ocean boundaries appear")
    print("   • Band 2: Continents become clearer")
    print("   • Band 3: Coastal details (3DGS stops here - 16 coeffs/ch)")
    print("   • Band 4-6: Sharper coastlines, cloud details")
    print("   • Band 7-9: Fine textures, mountain ranges")
    print("   • Band 10-15: Very fine details")
    print(f"   • Band 20-{max_band_to_compute}: Extremely fine details ({(max_band_to_compute+1)**2} coeffs/ch!)")

    print("\n💡 Trade-off comparison:")
    print("   • Band 3 (3DGS):  16×3 = 48 floats")
    print(f"   • Band 6:         49×3 = 147 floats (3.1× larger)")
    print(f"   • Band 12:       169×3 = 507 floats (10.6× larger!)")
    print(f"   • Band {max_band_to_compute}:       {(max_band_to_compute+1)**2}×3 = {3*(max_band_to_compute+1)**2} floats ({3*(max_band_to_compute+1)**2/48:.1f}× larger!)")


if __name__ == "__main__":
    main()
