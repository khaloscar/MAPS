"""
Plot the average boundary models of all planets possible
"""

import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# User defined packages
import planets as hp_planets


def main():
    folder = 'processed_data/mpc09/'
    os.makedirs(folder, exist_ok=True)

    # ======== Mercury ========
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Fix all the subplots
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_xr = fig.add_subplot(gs[1, 1])

    ax_xy.set_xlabel('X MSO')
    ax_xy.set_ylabel('Y MSO')

    ax_xz.set_xlabel('X MSO')
    ax_xz.set_ylabel('Z MSO')

    ax_yz.set_xlabel('Y MSO')
    ax_yz.set_ylabel('Z MSO')

    ax_xr.set_xlabel('X MSO')
    ax_xr.set_ylabel('R MSO')

    # Add the boundaries in the plots
    for ax in [ax_xy, ax_xz, ax_xr]:
        ax.set_aspect('equal')
        hp_planets.add_mercury_bow_shock_magnetopause_plot(ax, alpha=0.5, labels=True)

    ax_yz.set_aspect('equal')
    hp_planets.add_planet_in_plot(ax_yz)

    ax_xr.legend()

    # Fix the ranges of the orbital plots
    hp_planets.fix_spatial_plot_limits(x_lims=[-5, 5], y_lims=[-5, 5], z_lims=[-5, 5], r_lims=[0, 6], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr)

    fig.savefig(folder + 'mpc09-mercury.png', bbox_inches='tight')
    plt.close('all')

    # ======== Venus ========
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Fix all the subplots
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_xr = fig.add_subplot(gs[1, 1])

    ax_xy.set_xlabel('X VSO')
    ax_xy.set_ylabel('Y VSO')

    ax_xz.set_xlabel('X VSO')
    ax_xz.set_ylabel('Z VSO')

    ax_yz.set_xlabel('Y VSO')
    ax_yz.set_ylabel('Z VSO')

    ax_xr.set_xlabel('X VSO')
    ax_xr.set_ylabel('R VSO')

    # Add the boundaries in the plots
    for ax in [ax_xy, ax_xz, ax_xr]:
        ax.set_aspect('equal')
        hp_planets.add_venus_bow_shock_imb_plot(ax, alpha=0.5, labels=True)

    ax_yz.set_aspect('equal')
    hp_planets.add_planet_in_plot(ax_yz)

    ax_xr.legend()

    # Fix the ranges of the orbital plots
    hp_planets.fix_spatial_plot_limits(x_lims=[-5, 5], y_lims=[-5, 5], z_lims=[-5, 5], r_lims=[0, 6], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr)

    fig.savefig(folder + 'mpc09-venus.png', bbox_inches='tight')
    plt.close('all')

    # ======== Earth ========
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Fix all the subplots
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_xr = fig.add_subplot(gs[1, 1])

    ax_xy.set_xlabel('X GSE')
    ax_xy.set_ylabel('Y GSE')

    ax_xz.set_xlabel('X GSE')
    ax_xz.set_ylabel('Z GSE')

    ax_yz.set_xlabel('Y GSE')
    ax_yz.set_ylabel('Z GSE')

    ax_xr.set_xlabel('X GSE')
    ax_xr.set_ylabel('R GSE')

    # Add the boundaries in the plots
    for ax in [ax_xy, ax_xz, ax_xr]:
        ax.set_aspect('equal')
        hp_planets.add_earth_bow_shock_magnetopause_plot(ax, alpha=0.5, labels=True)

    ax_yz.set_aspect('equal')
    hp_planets.add_planet_in_plot(ax_yz)

    ax_xr.legend()

    # Fix the ranges of the orbital plots
    hp_planets.fix_spatial_plot_limits(x_lims=[-25, 25], y_lims=[-25, 25], z_lims=[-25, 25], r_lims=[0, 25], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr)

    fig.savefig(folder + 'mpc09-earth.png', bbox_inches='tight')
    plt.close('all')

    # ======== Mars ========
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Fix all the subplots
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_xr = fig.add_subplot(gs[1, 1])

    ax_xy.set_xlabel('X MSO')
    ax_xy.set_ylabel('Y MSO')

    ax_xz.set_xlabel('X MSO')
    ax_xz.set_ylabel('Z MSO')

    ax_yz.set_xlabel('Y MSO')
    ax_yz.set_ylabel('Z MSO')

    ax_xr.set_xlabel('X MSO')
    ax_xr.set_ylabel('R MSO')

    # Add the boundaries in the plots
    for ax in [ax_xy, ax_xz, ax_xr]:
        ax.set_aspect('equal')
        hp_planets.add_mars_bow_shock_mpb_plot(ax, alpha=0.5, labels=True)

    ax_yz.set_aspect('equal')
    hp_planets.add_planet_in_plot(ax_yz)

    ax_xr.legend()

    # Fix the ranges of the orbital plots
    hp_planets.fix_spatial_plot_limits(x_lims=[-5, 5], y_lims=[-5, 5], z_lims=[-5, 5], r_lims=[0, 6], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr)

    fig.savefig(folder + 'mpc09-mars.png', bbox_inches='tight')
    plt.close('all')

    # ======== Jupiter ========
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Fix all the subplots
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_xr = fig.add_subplot(gs[1, 1])

    ax_xy.set_xlabel('X JSO')
    ax_xy.set_ylabel('Y JSO')

    ax_xz.set_xlabel('X JSO')
    ax_xz.set_ylabel('Z JSO')

    ax_yz.set_xlabel('Y JSO')
    ax_yz.set_ylabel('Z JSO')

    ax_xr.set_xlabel('X JSO')
    ax_xr.set_ylabel('R JSO')

    # Add the boundaries in the plots
    for ax in [ax_xy, ax_xz, ax_xr]:
        ax.set_aspect('equal')
        hp_planets.add_jupiter_bow_shock_magnetopause_plot(ax, alpha=0.5, labels=True)

    ax_yz.set_aspect('equal')
    hp_planets.add_planet_in_plot(ax_yz)

    ax_xr.legend()

    # Fix the ranges of the orbital plots
    hp_planets.fix_spatial_plot_limits(x_lims=[-180, 180], y_lims=[-180, 180], z_lims=[-180, 180], r_lims=[0, 180], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr)

    fig.savefig(folder + 'mpc09-jupiter.png', bbox_inches='tight')
    plt.close('all')

    # ======== Saturn ========
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # Fix all the subplots
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_xr = fig.add_subplot(gs[1, 1])

    ax_xy.set_xlabel('X KSO')
    ax_xy.set_ylabel('Y KSO')

    ax_xz.set_xlabel('X KSO')
    ax_xz.set_ylabel('Z KSO')

    ax_yz.set_xlabel('Y KSO')
    ax_yz.set_ylabel('Z KSO')

    ax_xr.set_xlabel('X KSO')
    ax_xr.set_ylabel('R KSO')

    # Add the boundaries in the plots
    for ax in [ax_xy, ax_xz, ax_xr]:
        ax.set_aspect('equal')
        hp_planets.add_saturn_bow_shock_magnetopause_plot(ax, alpha=0.5, labels=True)

    ax_yz.set_aspect('equal')
    hp_planets.add_planet_in_plot(ax_yz)

    ax_xr.legend()

    # Fix the ranges of the orbital plots
    hp_planets.fix_spatial_plot_limits(x_lims=[-80, 30], y_lims=[-100, 100], z_lims=[-100, 100], r_lims=[0, 100], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr)

    fig.savefig(folder + 'mpc09-saturn.png', bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    main()
