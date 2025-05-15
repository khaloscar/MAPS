"""
Planet specific codes. Such as boundary models etc.
"""

import numpy as np


# ============================== General ==============================
def add_planet_in_plot(ax, z_offset=0., xr=False):
    import matplotlib.pyplot as plt
    circle = plt.Circle((0, 0 - z_offset), 1, color='k', fill=False, lw=2)
    if xr:
        circle = plt.Circle((0, 0 - z_offset), 1, color='k', fill=False, lw=1, ls='--')
        circle2 = plt.Circle((0, 0 + z_offset), 1, color='k', fill=False, lw=1, ls='--')
        ax.add_artist(circle2)
    ax.add_artist(circle)


def bow_shock_conic_section(theta, p, epsilon, X_0):
    # Function to calculate X and rho from theta
    r = p * epsilon / (1 + epsilon * np.cos(theta))
    X = r * np.cos(theta) + X_0
    rho = r * np.sin(theta)
    return X[X < r + X_0], rho[X < r + X_0]  # Remove the unwanted artefacts


def magnetopause_shue(theta, r_ss, alpha):
    # Define the function for r(θ) based on the Shue et al. (1997) model
    return r_ss * (2 / (1 + np.cos(theta))) ** alpha


def fix_spatial_plot_limits(x_lims=[-5, 5], y_lims=[-5, 5], z_lims=[-5, 5], r_lims=[0, 6], ax_mso_xy=None, ax_mso_xz=None, ax_mso_yz=None, ax_mso_xr=None):
    # Fix the ranges of the orbital plots
    if ax_mso_xy is not None:
        ax_mso_xy.set_xlim(x_lims)
        ax_mso_xy.set_ylim(y_lims)

    if ax_mso_xz is not None:
        ax_mso_xz.set_xlim(x_lims)
        ax_mso_xz.set_ylim(z_lims)

    if ax_mso_yz is not None:
        ax_mso_yz.set_xlim(y_lims)
        ax_mso_yz.set_ylim(z_lims)

    if ax_mso_xr is not None:
        ax_mso_xr.set_xlim(x_lims)
        ax_mso_xr.set_ylim(r_lims)


# ============================== Venus ==============================
def add_venus_bow_shock_imb_plot(ax, x0=-6, labels=False, alpha=None, coords='VSO', xr=False):
    import matplotlib.patches as patch
    if alpha is None:
        alpha = 1.0

    # Generate theta values from -pi to pi for a full plot
    theta_values = np.linspace(0, np.pi, 1000)

    # === Add the bow shock ===
    # Bow shock parameters from Signoles et al. (2023)
    p = 1.586  # Semi-latus rectum
    epsilon = 1.052  # Eccentricity
    X_0 = 0.688  # Focal point

    # Calculate X and rho for each theta
    X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
    if labels:
        ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray', label='Bow shock (Signoles et al., 2023)')
    else:
        ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')
    ax.plot(X_values[X_values > x0], -rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')

    # === Add the magnetic pileup boundary ===
    # parameters from Signoles et al. (2023) and Martinecz et al. 2008
    radius = 1.115  # IMB radius
    k = -0.097  # Slope for the nightside boundary

    # Add the circular arc for the dayside (X > 0)
    radius = 1.115
    IMB_circle = patch.Arc((0, 0), 2 * radius, 2 * radius, angle=0, theta1=-90, theta2=90, color='gray', ls='--', lw=1.5, alpha=alpha)
    ax.add_patch(IMB_circle)

    # Straight line for X < 0
    X_line = np.linspace(x0, 0, 500)  # X < 0
    rho_line = k * X_line + radius

    # Plot the straight line for the nightside
    if labels:
        ax.plot(X_line, rho_line, alpha=alpha, ls='--', c='gray', label='ICB (Martinecz et al., 2008; Signoles et al., 2023)')
    else:
        ax.plot(X_line, rho_line, alpha=alpha, ls='--', c='gray')
    ax.plot(X_line, -rho_line, alpha=alpha, ls='--', c='gray')

    # === Add the planet ===
    add_planet_in_plot(ax, xr)


# ============================== Mars ==============================
def add_mars_bow_shock_mpb_plot(ax, x0=-6, labels=False, alpha=None, coords='MSO', xr=False, MPB_model='Edberg2008'):
    if alpha is None:
        alpha = 1.0

    # Generate theta values from -pi to pi for a full plot
    theta_values = np.linspace(0, np.pi, 1000)

    # === Add the bow shock ===
    # Bow shock parameters from Edberg et al. (2008)
    p = 2.10  # Semi-latus rectum
    epsilon = 1.05  # Eccentricity
    X_0 = 0.55  # Focal point

    # Calculate X and rho for each theta
    X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
    if labels:
        ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray', label='Bow shock (Edberg et al., 2008)')
    else:
        ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')
    ax.plot(X_values[X_values > x0], -rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')

    # === Add the magnetic pileup boundary ===
    if MPB_model == 'Trotignon2006':
        # Magnetic pileup boundary parameters from Trotignon et al. (2006)
        # Dayside
        p = 1.08  # Semi-latus rectum
        epsilon = 0.77  # Eccentricity
        X_0 = 0.64  # Focal point

        # Calculate X and rho for each theta
        X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
        ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')

        # Nightside
        p = 0.528  # Semi-latus rectum
        epsilon = 1.009  # Eccentricity
        X_0 = 1.6  # Focal point

        # Calculate X and rho for each theta
        X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
        ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')

    elif MPB_model == 'Edberg2008':
        p = 0.90  # Semi-latus rectum
        epsilon = 0.92  # Eccentricity
        X_0 = 0.86  # Focal point

        # Calculate X and rho for each theta
        X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
        if labels:
            ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray', label='MPB (Edberg et al., 2008)')
        else:
            ax.plot(X_values[X_values > x0], rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')
        ax.plot(X_values[X_values > x0], -rho_values[X_values > x0], alpha=alpha, ls='--', c='gray')

    # === Add the planet ===
    add_planet_in_plot(ax, xr)


# ============================== Mercury ==============================
def add_mercury_bow_shock_magnetopause_plot(ax, labels=False, alpha=None, coords='MSO', xr=False):
    if alpha is None:
        alpha = 1.0

    if coords == 'MSM':
        z_offset = 0.196
        bs_offset = 0.
    elif 'MSO':
        z_offset = 0.
        bs_offset = 0.196

    # Generate theta values from -pi to pi for a full plot
    theta_values = np.linspace(0, 2 * np.pi, 1000)

    # === Add the bow shock ===
    # Bow shock parameters from Winslow et al. (2013)
    p = 2.75  # Semi-latus rectum, scaled by Mercury's radius (in km)
    epsilon = 1.04  # Eccentricity
    X_0 = 0.5  # Focal point
    # Calculate X and rho for each theta
    X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
    if labels:
        ax.plot(X_values, rho_values + bs_offset, label='Bow shock (Winslow et al., 2013)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values, rho_values + bs_offset, alpha=alpha, ls='--', c='gray')

    # === Add the magnetopause ===
    # Magnetopause parameters from Winslow et al. (2013) using the Shue et al. (1997) model
    r_ss = 1.45  # Subsolar standoff distance
    alpha_flare = 0.5  # Tail flaring parameter

    # Compute the corresponding r(θ) values
    r_values = magnetopause_shue(theta_values, r_ss, alpha_flare)

    # Convert polar coordinates (r, θ) to Cartesian coordinates (X, ρ)
    X_values_mp = r_values * np.cos(theta_values)
    rho_values_mp = r_values * np.sin(theta_values) + bs_offset

    if labels:
        ax.plot(X_values_mp, rho_values_mp, label='Magnetopause (Winslow et al., 2013)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values_mp, rho_values_mp, alpha=alpha, ls='--', c='gray')

    # === Add the planet ===
    add_planet_in_plot(ax, z_offset, xr)


def add_bow_shock_magnetopause_plot_yz(ax, labels=False, alpha=None, coords='MSO'):
    if alpha is None:
        alpha = 1.0

    if coords == 'MSM':
        z_offset = 0.196
    else:
        z_offset = 0.

    # === Add the planet ===
    add_planet_in_plot(ax, z_offset)


# ============================== Jupiter ==============================
def add_jupiter_bow_shock_magnetopause_plot(ax, labels=False, alpha=None, coords='JSO', xr=False):
    if alpha is None:
        alpha = 1.0

    # Generate theta values from -pi to pi for a full plot
    theta_values = np.linspace(0, 2 * np.pi, 1000)

    # === Add the bow shock ===
    # Bow shock parameters from Edberg et al. (2024) and Huddleston et al. (1998)
    p = 167  # Semi-latus rectum, scaled by Jupiter's radius (in km)
    epsilon = 1.21  # Eccentricity
    X_0 = 0.  # Focal point
    # Calculate X and rho for each theta
    X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
    if labels:
        ax.plot(X_values, rho_values, label='Bow shock (Huddleston et al., 1998)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values, rho_values, alpha=alpha, ls='--', c='gray')

    # # === Add the magnetopause ===
    # Bow shock parameters from Huddleston et al. (1998)
    p = 113  # Semi-latus rectum
    epsilon = 0.92  # Eccentricity
    X_0 = 0.  # Focal point
    # Calculate X and rho for each theta
    X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
    if labels:
        ax.plot(X_values, rho_values, label='Magnetopause (Huddleston et al., 1998)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values, rho_values, alpha=alpha, ls='--', c='gray')

    # === Add the planet ===
    add_planet_in_plot(ax, xr)


# ============================== Saturn ==============================
def add_saturn_bow_shock_magnetopause_plot(ax, labels=False, alpha=None, coords='JSO', xr=False):
    if alpha is None:
        alpha = 1.0

    # Generate theta values from -pi to pi for a full plot
    theta_values = np.linspace(0, 2 * np.pi, 1000)

    # === Add the bow shock ===
    # Bow shock parameters from Edberg et al. (2024) and Masters et al. (2008)
    p = 51  # Semi-latus rectum
    epsilon = 1.05  # Eccentricity
    X_0 = 0.  # Focal point
    # Calculate X and rho for each theta
    X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
    if labels:
        ax.plot(X_values, rho_values, label='Bow shock (Masters et al., 2008)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values, rho_values, alpha=alpha, ls='--', c='gray')

    # === Add the magnetopause ===
    X_values, rho_values = saturn_magnetopause(n_points=1000)
    if labels:
        ax.plot(X_values, rho_values, label='Magnetopause (Kanani et al., 2010)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values, rho_values, alpha=alpha, ls='--', c='gray')
    ax.plot(X_values, -rho_values, alpha=alpha, ls='--', c='gray')

    # === Add the planet ===
    add_planet_in_plot(ax, xr)


def dynamic_pressure_nPa(n_cm3, v_kms):
    """
    Calculate solar wind dynamic pressure in nPa
    from number density (in cm^-3) and velocity (in km/s).
    """
    m_p = 1.6726e-27  # proton mass in kg
    n_m3 = n_cm3 * 1e6  # convert cm^-3 to m^-3
    v_ms = v_kms * 1e3  # convert km/s to m/s

    P_d_Pa = n_m3 * m_p * v_ms**2  # dynamic pressure in Pa
    P_d_nPa = P_d_Pa * 1e9  # convert to nPa

    return P_d_nPa


def saturn_magnetopause(Pdyn=0.02, n_points=100):
    """
    Returns the magnetopause boundary (x, y) in Rs for Saturn based on Kanani et al. (2010).
    """

    # Parameters from Kanani et al. (2010)
    a1 = 10.3
    a2 = 0.2
    a3 = 0.73
    a4 = 0.4

    Rs0 = a1 * Pdyn**(-a2)
    K = a3 + a4 * Pdyn

    # Generate theta values but ignore theta=pi for division by zero warning
    epsilon = 1e-6
    theta = np.linspace(0, np.pi - epsilon, n_points)

    # Shape of the magnetopause
    r = Rs0 * (2 / (1 + np.cos(theta)))**K

    # Convert to Cartesian (assuming symmetry in x-y plane)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


# ============================== Earth ==============================
def add_earth_bow_shock_magnetopause_plot(ax, labels=False, alpha=None, coords='JSO', xr=False):
    if alpha is None:
        alpha = 1.0

    RE = 6378  # km, Earth radius
    # Generate theta values from -pi to pi for a full plot
    theta_values = np.linspace(0, 2 * np.pi, 1000)

    # === Add the bow shock ===
    # Bow shock parameters from Edberg et al. (2024) and Farris et al. (1991)
    epsilon = 0.81  # Eccentricity
    p = 13.7 * (1 + epsilon) / epsilon  # Semi-latus rectum
    X_0 = 0.  # Focal point

    # Calculate X and rho for each theta
    X_values, rho_values = bow_shock_conic_section(theta_values, p, epsilon, X_0)
    if labels:
        ax.plot(X_values, rho_values, label='Bow shock (Farris et al., 1991)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values, rho_values, alpha=alpha, ls='--', c='gray')

    # === Add the magnetopause ===
    X_values, rho_values = earth_magnetopause()
    if labels:
        ax.plot(X_values, rho_values, label='Magnetopause (Shue et al., 1997)', alpha=alpha, ls='--', c='gray')
    else:
        ax.plot(X_values, rho_values, alpha=alpha, ls='--', c='gray')
    ax.plot(X_values, -rho_values, alpha=alpha, ls='--', c='gray')

    # === Add the planet ===
    add_planet_in_plot(ax, xr)


def earth_magnetopause(Pdyn=2.0, Bz=0, n_points=1000):
    """
    Returns the magnetopause boundary (x, y) in RE for Earth based on Shue et al. (1997).
    """

    # Parameters from Shue et al. (1997)
    if Bz >= 0.:
        R0 = (11.4 + 0.013 * Bz) * Pdyn**(-1 / 6.6)
    else:
        R0 = (11.4 + 0.14 * Bz) * Pdyn**(-1 / 6.6)
    a = (0.58 - 0.010 * Bz) * (1 + 0.010 * Pdyn)

    # Generate theta values but ignore theta=pi for division by zero warning
    epsilon = 1e-6
    theta = np.linspace(0, np.pi - epsilon, n_points)

    # Shape of the magnetopause
    r = R0 * (2 / (1 + np.cos(theta)))**a

    # Convert to Cartesian (assuming symmetry in x-y plane)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y
