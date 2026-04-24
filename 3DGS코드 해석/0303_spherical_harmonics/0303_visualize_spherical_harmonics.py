"""
Spherical Harmonics Visualizer

A fast interactive 3D visualizer for real spherical harmonics basis functions.
This standalone script is optimized for performance compared to Jupyter notebooks.

Usage:
    python visualize_spherical_harmonics.py              # Run examples
    python visualize_spherical_harmonics.py --interactive # Run with sliders (requires dash)

Features:
- Interactive 3D visualization with Plotly
- Multiple visualization modes (individual harmonics, combinations)
- High-resolution rendering for smooth surfaces
- Support for bands 0-3 (l=0,1,2,3)
- Interactive sliders for each coefficient (with --interactive flag)
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import math
import sys

# Configure Plotly to open in browser automatically
pio.renderers.default = "browser"


def _double_factorial(n: int) -> int:
    """Compute double factorial n!!"""
    if n <= 0:
        return 1
    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


def _assoc_legendre(l: int, m: int, x):
    """
    Associated Legendre polynomial P_l^m(x) including the Condon–Shortley (-1)^m phase.
    This matches the convention used in many physics texts and 3D Gaussian Splatting.
    
    Parameters:
    -----------
    l : int
        Degree of the polynomial
    m : int
        Order of the polynomial (0 <= m <= l)
    x : np.ndarray
        Input values (typically cos(theta))
    
    Returns:
    --------
    P : np.ndarray
        Associated Legendre polynomial values
    """
    if m < 0 or m > l:
        raise ValueError("Require 0 <= m <= l")
    
    phase = (-1)**m

    # P_m^m (base recurrence)
    P_mm = _double_factorial(2 * m - 1) * np.power(np.maximum(1.0 - x * x, 0.0), m / 2.0)
    if l == m:
        return P_mm * phase
    
    # P_{m+1}^m
    P_m1m = (2 * m + 1) * x * P_mm
    if l == m + 1:
        return P_m1m * phase
    
    # upward recurrence for l >= m+2
    P_lm_minus2 = P_mm
    P_lm_minus1 = P_m1m
    for ell in range(m + 2, l + 1):
        P_lm = ((2 * ell - 1) * x * P_lm_minus1 - (ell + m - 1) * P_lm_minus2) / (ell - m)
        P_lm_minus2, P_lm_minus1 = P_lm_minus1, P_lm
    
    return P_lm_minus1 * phase


def _norm_lm(l: int, m: int) -> float:
    """
    Compute normalization factor for spherical harmonics.
    
    N_l^m = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!)
    """
    log_ratio = math.lgamma(l - m + 1) - math.lgamma(l + m + 1)
    return math.sqrt((2 * l + 1) / (4 * math.pi) * math.exp(log_ratio))


def _complex_Y_lm(l: int, m: int, theta_grid, phi_grid):
    """
    Compute complex spherical harmonic Y_l^m(theta, phi).
    
    Parameters:
    -----------
    l : int
        Degree (band number)
    m : int
        Order (0 <= m <= l for this function)
    theta_grid : np.ndarray
        Polar angle grid (angle from z-axis), shape (n_theta, n_phi)
    phi_grid : np.ndarray
        Azimuthal angle grid (angle in xy-plane), shape (n_theta, n_phi)
    
    Returns:
    --------
    Y : np.ndarray (complex)
        Complex spherical harmonic values
    """
    if m < 0 or m > l:
        raise ValueError("Require 0 <= m <= l for _complex_Y_lm")
    x = np.cos(theta_grid)
    P = _assoc_legendre(l, m, x)
    N = _norm_lm(l, m)
    return (N * P) * np.exp(1j * m * phi_grid)


def compute_real_spherical_harmonic(l: int, m: int, theta_grid, phi_grid):
    """
    Compute real spherical harmonic Y_l^m(theta, phi).
    
    Real spherical harmonics are defined as:
    - m > 0: sqrt(2) * Re(Y_l^m)
    - m = 0: Y_l^0
    - m < 0: sqrt(2) * Im(Y_l^{|m|})
    
    Parameters:
    -----------
    l : int
        Degree (band number, l >= 0)
    m : int
        Order (-l <= m <= l)
    theta_grid : np.ndarray
        Polar angle grid (angle from z-axis)
    phi_grid : np.ndarray
        Azimuthal angle grid (angle in xy-plane)
    
    Returns:
    --------
    Y : np.ndarray (float)
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


def run_interactive_dash_app(resolution=100, port=8050):
    """
    Launch an interactive Dash app with individual sliders for each spherical harmonic coefficient.
    
    This function creates a web-based interface with separate sliders for:
    - Band 0: Y_0^0
    - Band 1: Y_1^-1, Y_1^0, Y_1^1
    - Band 2: Y_2^-2, Y_2^-1, Y_2^0, Y_2^1, Y_2^2
    
    Parameters:
    -----------
    resolution : int
        Number of points along theta and phi axes (default: 100)
    port : int
        Port number for the Dash app (default: 8050)
    
    Requirements:
    -------------
    pip install dash
    """
    try:
        from dash import Dash, dcc, html, Input, Output, State
        import dash
    except ImportError:
        print("ERROR: Dash is not installed.")
        print("Please install it with: pip install dash")
        print("\nAlternatively, run without --interactive flag for preset-based controls.")
        return
    
    # Pre-compute spherical harmonics
    print(f"Pre-computing spherical harmonics at resolution {resolution}...")
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, resolution)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    
    # Pre-compute all harmonics
    harmonics = {}
    
    # Band 0
    harmonics[(0, 0)] = compute_real_spherical_harmonic(0, 0, theta_grid, phi_grid)
    
    # Band 1
    for m in [-1, 0, 1]:
        harmonics[(1, m)] = compute_real_spherical_harmonic(1, m, theta_grid, phi_grid)
    
    # Band 2
    for m in [-2, -1, 0, 1, 2]:
        harmonics[(2, m)] = compute_real_spherical_harmonic(2, m, theta_grid, phi_grid)
    
    print("Initialization complete!\n")
    
    # Create Dash app
    app = Dash(__name__)
    
    # Define the layout
    app.layout = html.Div([
        html.H1("Interactive Spherical Harmonics Visualizer",
                style={'textAlign': 'center', 'marginBottom': 10, 'marginTop': 10}),
        
        html.Div([
            # Left panel with 3D plot
            html.Div([
                dcc.Graph(id='sh-plot', style={'height': '88vh'})
            ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Right panel with controls (scrollable)
            html.Div([
                html.Div([
                    html.H3("Coefficients", style={'marginTop': 5, 'marginBottom': 15}),
                    
                    html.Div([
                        html.Label("Render Mode:", style={'fontSize': 13, 'marginBottom': '5px', 'fontWeight': 'bold'}),
                        dcc.RadioItems(
                            id='render-mode',
                            options=[
                                {'label': ' Magnitude as Radius', 'value': 'magnitude'},
                                {'label': ' Color on Sphere', 'value': 'color'}
                            ],
                            value='magnitude',
                            style={'fontSize': 12}
                        ),
                    ], style={'marginBottom': 20, 'padding': 10, 'backgroundColor': '#e8e8e8', 'borderRadius': 5}),
                    
                    html.H4("Band 0 (DC)", style={'marginTop': 10, 'marginBottom': 5, 'fontSize': 16}),
                    html.Div([
                        html.Label("Y₀⁰:", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-0-0', min=-3, max=3, step=0.1, value=1.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 20}),
                    
                    html.H4("Band 1 (Linear)", style={'marginTop': 10, 'marginBottom': 5, 'fontSize': 16}),
                    html.Div([
                        html.Label("Y₁⁻¹ (y):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-1--1', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 15}),
                    html.Div([
                        html.Label("Y₁⁰ (z):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-1-0', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 15}),
                    html.Div([
                        html.Label("Y₁¹ (x):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-1-1', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 20}),
                    
                    html.H4("Band 2 (Quadratic)", style={'marginTop': 10, 'marginBottom': 5, 'fontSize': 16}),
                    html.Div([
                        html.Label("Y₂⁻² (xy):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-2--2', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 15}),
                    html.Div([
                        html.Label("Y₂⁻¹ (yz):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-2--1', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 15}),
                    html.Div([
                        html.Label("Y₂⁰ (3z²-1):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-2-0', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 15}),
                    html.Div([
                        html.Label("Y₂¹ (xz):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-2-1', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 15}),
                    html.Div([
                        html.Label("Y₂² (x²-y²):", style={'fontSize': 13, 'marginBottom': '5px'}),
                        dcc.Slider(id='slider-2-2', min=-3, max=3, step=0.1, value=0.0,
                                  marks={i: str(i) for i in range(-3, 4)},
                                  tooltip={"placement": "bottom", "always_visible": True},
                                  updatemode='drag'),
                    ], style={'marginBottom': 20}),
                    
                    html.Button('Reset to DC', id='reset-button', n_clicks=0,
                               style={'width': '100%', 'padding': 8, 'marginTop': 10}),
                    
                    html.P("Drag to rotate | Scroll to zoom",
                          style={'textAlign': 'center', 'color': 'gray', 'fontSize': 11, 'marginTop': 15}),
                    html.P("Drag sliders to adjust coefficients",
                          style={'textAlign': 'center', 'color': 'gray', 'fontSize': 10})
                ], style={'padding': 15, 'height': '88vh', 'overflowY': 'auto'})
                
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'backgroundColor': '#f0f0f0', 'height': '88vh'}),
        ], style={'display': 'flex'})
    ])
    
    # Callback to update plot based on slider values
    @app.callback(
        [Output('sh-plot', 'figure'),
         Output('slider-0-0', 'value'), Output('slider-1--1', 'value'), Output('slider-1-0', 'value'),
         Output('slider-1-1', 'value'), Output('slider-2--2', 'value'), Output('slider-2--1', 'value'),
         Output('slider-2-0', 'value'), Output('slider-2-1', 'value'), Output('slider-2-2', 'value')],
        [Input('slider-0-0', 'value'), Input('slider-1--1', 'value'), Input('slider-1-0', 'value'),
         Input('slider-1-1', 'value'), Input('slider-2--2', 'value'), Input('slider-2--1', 'value'),
         Input('slider-2-0', 'value'), Input('slider-2-1', 'value'), Input('slider-2-2', 'value'),
         Input('reset-button', 'n_clicks'), Input('render-mode', 'value')]
    )
    def update_all(c00, c1m1, c10, c11, c2m2, c2m1, c20, c21, c22, reset_clicks, render_mode):
        
        ctx = dash.callback_context
        if not ctx.triggered:
            trigger_id = 'none'
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Handle reset
        if trigger_id == 'reset-button':
            c00, c1m1, c10, c11 = 1.0, 0.0, 0.0, 0.0
            c2m2, c2m1, c20, c21, c22 = 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Combine all harmonics with their coefficients
        coeffs = {
            (0, 0): c00,
            (1, -1): c1m1, (1, 0): c10, (1, 1): c11,
            (2, -2): c2m2, (2, -1): c2m1, (2, 0): c20, (2, 1): c21, (2, 2): c22
        }
        
        Y_sum = np.zeros_like(theta_grid, dtype=float)
        for (l, m), coeff in coeffs.items():
            Y_sum += coeff * harmonics[(l, m)]
        
        # Convert to Cartesian coordinates based on render mode
        if render_mode == 'magnitude':
            r_sum = np.abs(Y_sum)
            r_sum = np.maximum(r_sum, 0.01)
        else:  # render_mode == 'color'
            r_sum = np.ones_like(Y_sum)
        
        x = r_sum * np.sin(theta_grid) * np.cos(phi_grid)
        y = r_sum * np.sin(theta_grid) * np.sin(phi_grid)
        z = r_sum * np.cos(theta_grid)
        
        # Create figure
        fig = go.Figure(data=[
            go.Surface(
                x=x, y=y, z=z,
                surfacecolor=Y_sum,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Value"),
                cmin=-3, cmax=3
            )
        ])
        
        # Build title with non-zero coefficients
        terms = []
        for (l, m), c in coeffs.items():
            if abs(c) > 0.01:
                terms.append(f"{c:.1f}·Y_{l}^{m}")
        
        mode_desc = "Magnitude as Radius" if render_mode == 'magnitude' else "Color on Sphere"
        title = f"Spherical Harmonics ({mode_desc}): " + (" + ".join(terms) if terms else "")
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        # Return figure and slider values (for reset functionality)
        return fig, c00, c1m1, c10, c11, c2m2, c2m1, c20, c21, c22
    
    # Run the app
    print(f"Starting Dash app on http://localhost:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    # Open browser automatically
    import webbrowser
    from threading import Timer
    Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    
    app.run(debug=False, port=port)


def main():
    """Main function - launches interactive visualization with sliders."""
    
    print("=" * 60)
    print("Interactive Spherical Harmonics Visualizer")
    print("=" * 60)
    print("\nLaunching Dash app with individual sliders for each coefficient...")
    print("Control coefficients for bands 0, 1, and 2 in real-time.\n")
    
    try:
        run_interactive_dash_app(resolution=100, port=8050)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")


if __name__ == "__main__":
    main()
