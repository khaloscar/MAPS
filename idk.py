import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

def get_middle_idx_slice(n_edges):

    if n_edges % 2 == 0:

        edge_slice = slice(n_edges//2-1, n_edges//2+1)
        bin_slice = slice(edge_slice.start,edge_slice.stop-1)
    
    elif n_edges % 2 == 1:

        edge_slice = slice(n_edges//2-1, n_edges//2+2)
        bin_slice = slice(edge_slice.start,edge_slice.stop-1)

    return edge_slice, bin_slice

def add_thickness(edge_slice, bin_slice, thickness=1, direction='up'):
    if direction == 'up':
        print('up')
        edge_slice = slice(edge_slice.start, edge_slice.stop+thickness)
        bin_slice = slice(bin_slice.start, bin_slice.stop+thickness)

        
    if direction == 'down':
        print('down')
        edge_slice = slice(edge_slice.start-thickness, edge_slice.stop)
        bin_slice = slice(bin_slice.start-thickness, bin_slice.stop)

    if direction == 'sym':
        print('sym')
        edge_slice = slice(edge_slice.start-thickness, edge_slice.stop+thickness)
        bin_slice = slice(bin_slice.start-thickness, bin_slice.stop+thickness)
    return edge_slice, bin_slice

def plot_histogram_data(hist_xyz_nrmeas, hist_xyz_den, species, edges, filename, filepath_name):



    cmap = plt.cm.inferno
    cmap.set_under(cmap(0))
    
    xedges, yedges, zedges, redges = edges[0], edges[1], edges[2], edges[3]
    # Fix average density for each histogram and then extract for plotting

    histogram_xyz_average = hist_xyz_den / hist_xyz_nrmeas
    print(histogram_xyz_average.shape)
    idx = int(xedges.shape[0] / 2)
    idy = int(yedges.shape[0] / 2)
    idz = int(zedges.shape[0] / 2)

    # Skapa meshgrid av y- och z-edges för att beräkna r
    Yc, Zc = np.meshgrid((yedges[:-1] + yedges[1:]) / 2, (zedges[:-1] + zedges[1:]) / 2, indexing='ij')
    R = np.sqrt(Yc**2 + Zc**2)

    # Initiera hist_xr
    hist_xr = np.full((len(xedges)-1, len(redges)-1), np.nan)

    # Gå igenom varje x-bin
    for ix in range(histogram_xyz_average.shape[0]):
        slice_yz = histogram_xyz_average[ix, :, :]  # yz-slice vid fix x
        r_vals = R.flatten()
        dens_vals = slice_yz.flatten()

        # Filtrera bort NaN
        valid = ~np.isnan(dens_vals)
        r_vals = r_vals[valid]
        dens_vals = dens_vals[valid]

        # Bin densities i r för varje x-bin
        if len(r_vals) > 0:
            hist_sum, _ = np.histogram(r_vals, bins=redges, weights=dens_vals)
            hist_count, _ = np.histogram(r_vals, bins=redges)
            with np.errstate(invalid='ignore', divide='ignore'):
                hist_avg = hist_sum / hist_count
            hist_xr[ix, :] = hist_avg

   

    """     hist_xy = histogram_xyz_average[:, :, idz]
    hist_xz = histogram_xyz_average[:, idy, :]
    hist_yz = histogram_xyz_average[idx, :, :] """

    """     hist_xy = np.nanmean(histogram_xyz_average[:, :, idz-t:idz+t], axis=2)
    hist_xz = np.nanmean(histogram_xyz_average[:, idy-t:idy+t, :], axis=1)
    hist_yz = np.nanmean(histogram_xyz_average[idx-t:idx+t, :, :], axis=0) """

    hist_xy = histogram_xyz_average[:, :, idz]
    hist_xz = histogram_xyz_average[:, idy, :]
    hist_yz = histogram_xyz_average[idx, :, :]

    print(hist_xy.shape)


    # Use the smallest/largest density as colorbar
    """     cmin = np.nanmin([np.nanmin(hist_xy[hist_xy > 0.]), np.nanmin(hist_xz[hist_xz > 0.]), np.nanmin(hist_yz[hist_yz > 0.])])
    cmax = np.nanmax([np.nanmax(hist_xy[hist_xy > 0.]), np.nanmax(hist_xz[hist_xz > 0.]), np.nanmax(hist_yz[hist_yz > 0.])]) """

    
    """     cmin = np.nanmin([safe_nanmin(hist_xy), safe_nanmin(hist_xz), safe_nanmin(hist_yz)])
    cmax = np.nanmax([safe_nanmax(hist_xy), safe_nanmax(hist_xz), safe_nanmax(hist_yz)]) """

    vals = np.hstack((
        hist_xy[hist_xy > 0],
        hist_xz[hist_xz > 0],
        hist_yz[hist_yz > 0],
        hist_xr[hist_xr > 0]
    ))

    if vals.size == 0:
        print(f'INGEN DATA O PLOTTA')
        return
    
    cmin, cmax = vals.min(), vals.max()

    # Plot all the data
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[10, 10, 1])

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_yz = fig.add_subplot(gs[0, 1])
    ax_xr = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[:, -1])

    n_edges = len(yedges)
    edges_idx_sice, bin_idx_slice =  get_middle_idx_slice(n_edges)
    edges_idx_sice, bin_idx_slice = add_thickness(edges_idx_sice, bin_idx_slice, thickness=2, direction='sym')

    idx = np.where(xedges>0)[0]
    print(hist_xy.T.shape)
    print(f'lola {idx-1}')
    ax_xy.pcolormesh(xedges[idx], yedges[edges_idx_sice], hist_xy.T[bin_idx_slice,idx[:-1]], cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    ax_xz.pcolormesh(xedges, zedges, hist_xy.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    ax_yz.pcolormesh(yedges, zedges, hist_yz.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    ax_xr.pcolormesh(xedges, redges, hist_xr.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))

    """     hp.add_bow_shock_magnetopause_plot(ax_xy)
    hp.add_bow_shock_magnetopause_plot(ax_xz)
    hp.add_bow_shock_magnetopause_plot_yz(ax_yz)
    # hp.add_bow_shock_magnetopause_plot(ax_xr) """

    ax_xy.set_xlabel('MSO X [RM]')
    ax_xz.set_xlabel('MSO X [RM]')
    ax_yz.set_xlabel('MSO Y [RM]')
    ax_xr.set_xlabel('MSO X [RM]')

    ax_xy.set_ylabel('MSO Y [RM]')
    ax_xz.set_ylabel('MSO Z [RM]')
    ax_yz.set_ylabel('MSO Z [RM]')
    ax_xr.set_ylabel('MSO R [RM]')

    # Fix the ranges of the orbital plots
    ax_xy.set_xlim(xedges[0], xedges[-1])
    ax_xy.set_ylim(yedges[0], yedges[-1])

    ax_xz.set_xlim(xedges[0], xedges[-1])
    ax_xz.set_ylim(zedges[0], zedges[-1])

    ax_yz.set_xlim(yedges[0], yedges[-1])
    ax_yz.set_ylim(zedges[0], zedges[-1])

    ax_xr.set_xlim(xedges[0], xedges[-1])
    ax_xr.set_ylim(redges[0], redges[-1])

    for ax in [ax_xy, ax_xz, ax_yz, ax_xr]:  # , ax_xr
        ax.set_aspect('equal')

    # Add a colorbar with the counts
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cb, orientation='vertical')
    # cb.ax.set_yticklabels([timeDt_valid[0].strftime('%H:%M'), timeDt_valid[-1].strftime('%H:%M')])  # vertically oriented colorbar
    cb.set_label('some info')

    plt.suptitle(filepath_name)

    # Save figure
    plt.show()
    fig.savefig(filename, bbox_inches='tight')
    plt.close('all')

def load_histogram(filename):
    data = np.load(f'{filename}.npz')
    hist_nrmeas = data['hist_nrmeas']
    hist_den = data['hist_den']
    edges = (
    data['xedges'],
    data['yedges'],
    data['zedges'],
    data['redges'])
    return hist_nrmeas, hist_den, edges

radius = 5
n_bins = 50

species = 'olabola'

filename_histogram = f'Saved/c3_h_dens/Data/__bins__50__radius__25'

hist_xyz_nrmeas, hist_xyz_dens, edges = load_histogram(filename_histogram)

plot_histogram_data(hist_xyz_nrmeas, hist_xyz_dens, species, edges, 'abobror', 'ngtbgt')

