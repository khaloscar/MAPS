import speasy as spz
from speasy import amda
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import amda_datahandler as amddh

amda_tree = spz.inventories.tree.amda


def fix_histogram_placeholder(edges_n_bins):
    hist_xyz_nrmeas = np.zeros(edges_n_bins)
    hist_xyz_den = np.zeros(edges_n_bins)
    return hist_xyz_nrmeas, hist_xyz_den

def parameter_to_histogram(dataframe, parameter):

    # Remove Nan, None and zero-values from dataframe
    # Preserves indexing, be it row-number or datetime
    # Could clean earlier perhaps??
    dataframe = dataframe[dataframe[parameter].notna() & (dataframe[parameter] != 0)]

    if len(dataframe) > 0:
        # Get the relevant times, this supposes that dataframe is indexed by datetime
        # Does conversion from 
        
        # Extract position, convert to jupiter radius
        # Dataset pos already in jupiter radius
        # Here, get positional data and density data, cleaned df
        # whatever
        print()

def sum_histogram_data(hist_xyz_nrmeas, hist_xyz_den, density, sc_position_ntp, edges):
    hist_nrmeas, _ = np.histogramdd(sc_position_ntp.values, bins=edges)
    hist_den, _ = np.histogramdd(sc_position_ntp.values, bins=edges, weights=density.values)
    hist_xyz_nrmeas += hist_nrmeas
    hist_xyz_den += hist_den
    return hist_xyz_nrmeas, hist_xyz_den

def safe_nanmin(arr):
    arr = arr[~np.isnan(arr) & (arr > 0)]
    return np.nanmin(arr) if arr.size > 0 else np.nan

def safe_nanmax(arr):
    arr = arr[~np.isnan(arr) & (arr > 0)]
    return np.nanmax(arr) if arr.size > 0 else np.nan

def merge_dataframes(pos_df, param_df):

    # This code supposes that pos_df
    # has higher temporal resolution than
    # param_df 
    df_merged = pd.merge_asof(
                        param_df.sort_index(),
                        pos_df.sort_index(),
                        left_index=True,
                        right_index=True,
                        allow_exact_matches=True,
                        direction='nearest'
                    )

    return df_merged

def clean_dataframe(dataframe, parameter):
    # Takes in column name and removes Nan, None, and zero-values
    dataframe = dataframe[dataframe[parameter].notna() & (dataframe[parameter] != 0)]
    return dataframe

def plot_histogram_data(hist_xyz_nrmeas, hist_xyz_den, species, edges, filename):

    cmap = plt.cm.inferno
    cmap.set_under(cmap(0))
    
    xedges, yedges, zedges = edges[0], edges[1], edges[2]
    # Fix average density for each histogram and then extract for plotting

    histogram_xyz_average = hist_xyz_den / hist_xyz_nrmeas
    print(histogram_xyz_average.shape)
    idx = int(xedges.shape[0] / 2)
    idy = int(yedges.shape[0] / 2)
    idz = int(zedges.shape[0] / 2)
    print(idx)
    print(idy)
    print(idz)
    t = 1

   

    """     hist_xy = histogram_xyz_average[:, :, idz]
    hist_xz = histogram_xyz_average[:, idy, :]
    hist_yz = histogram_xyz_average[idx, :, :] """

    """     hist_xy = np.nanmean(histogram_xyz_average[:, :, idz-t:idz+t], axis=2)
    hist_xz = np.nanmean(histogram_xyz_average[:, idy-t:idy+t, :], axis=1)
    hist_yz = np.nanmean(histogram_xyz_average[idx-t:idx+t, :, :], axis=0) """

    hist_xy = histogram_xyz_average[:, :, idz]
    hist_xz = histogram_xyz_average[:, idy, :]
    hist_yz = histogram_xyz_average[idx, :, :]

    # Use the smallest/largest density as colorbar
    """     cmin = np.nanmin([np.nanmin(hist_xy[hist_xy > 0.]), np.nanmin(hist_xz[hist_xz > 0.]), np.nanmin(hist_yz[hist_yz > 0.])])
    cmax = np.nanmax([np.nanmax(hist_xy[hist_xy > 0.]), np.nanmax(hist_xz[hist_xz > 0.]), np.nanmax(hist_yz[hist_yz > 0.])]) """

    
    """     cmin = np.nanmin([safe_nanmin(hist_xy), safe_nanmin(hist_xz), safe_nanmin(hist_yz)])
    cmax = np.nanmax([safe_nanmax(hist_xy), safe_nanmax(hist_xz), safe_nanmax(hist_yz)]) """

    vals = np.hstack((
        hist_xy[hist_xy > 0],
        hist_xz[hist_xz > 0],
        hist_yz[hist_yz > 0]
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
    # ax_xr = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[:, -1])

    ax_xy.pcolormesh(xedges, yedges, hist_xy.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    ax_xz.pcolormesh(xedges, zedges, hist_xz.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    ax_yz.pcolormesh(yedges, zedges, hist_yz.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    # ax_xr.pcolormesh(xedges, redges, hist_xr.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))

    """     hp.add_bow_shock_magnetopause_plot(ax_xy)
    hp.add_bow_shock_magnetopause_plot(ax_xz)
    hp.add_bow_shock_magnetopause_plot_yz(ax_yz)
    # hp.add_bow_shock_magnetopause_plot(ax_xr) """

    ax_xy.set_xlabel('MSO X [RM]')
    ax_xz.set_xlabel('MSO X [RM]')
    ax_yz.set_xlabel('MSO Y [RM]')
    # ax_xr.set_xlabel('MSO X [RM]')

    ax_xy.set_ylabel('MSO Y [RM]')
    ax_xz.set_ylabel('MSO Z [RM]')
    ax_yz.set_ylabel('MSO Z [RM]')
    # ax_xr.set_ylabel('MSO R [RM]')

    # Fix the ranges of the orbital plots
    ax_xy.set_xlim(xedges[0], xedges[-1])
    ax_xy.set_ylim(yedges[0], yedges[-1])

    ax_xz.set_xlim(xedges[0], xedges[-1])
    ax_xz.set_ylim(zedges[0], zedges[-1])

    ax_yz.set_xlim(yedges[0], yedges[-1])
    ax_yz.set_ylim(zedges[0], zedges[-1])

    # ax_xr.set_xlim(xedges[0], xedges[-1])
    # ax_xr.set_ylim(redges[0], redges[-1])

    for ax in [ax_xy, ax_xz, ax_yz]:  # , ax_xr
        ax.set_aspect('equal')

    # Add a colorbar with the counts
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cb, orientation='vertical')
    # cb.ax.set_yticklabels([timeDt_valid[0].strftime('%H:%M'), timeDt_valid[-1].strftime('%H:%M')])  # vertically oriented colorbar
    cb.set_label('some info')

    plt.suptitle(f'infoinfo')

    # Save figure
    fig.savefig(filename, bbox_inches='tight')
    plt.close('all')


# Basic logic 
# Get data  x
# Convert to df x
# merge temporally x
# put into bins and plot x
# maybe add a loop to do it daily or monthly or smth

def main():

    run_time_t0 = dt.datetime.now()

    #        amda_tree.Parameters.MAVEN.Ephemeris.maven_orb_marsobs1s.mav_xyz_mso1s,
    #    amda_tree.Parameters.MAVEN.NGIMS.mav_ngims_kp.mav_ngimskp_he

    """     amda_dir = [
        amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
        amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n
    ] """

    amda_dir = [
        amda_tree.Parameters.MAVEN.Ephemeris.maven_orb_marsobs1s.mav_xyz_mso1s,
        amda_tree.Parameters.MAVEN.SWIA.mav_swia_kp.mav_swiakp_n
    ]

    species = amda_dir[1].name
    print(f'Species name: {species}')
    now = dt.datetime.now()
    start_date = dt.datetime(2016,1,1)
    stop_date = dt.datetime(2018,12,30)
    timedelta = dt.timedelta(days=25)
    

    start_date, stop_date = amddh.retrieve_restrictive_time_boundaries(amda_dir)
    #stop_date = start_date + timedelta
    filepath = 'Data_plots'

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    filename = f'{filepath}/oxygen_density__average_dens_plot_{start_date.strftime("%Y%m%d__%H%M%S")}--{stop_date.strftime("%Y%m%d__%H%M%S")}__created__{now.strftime("%Y%m%d__%H%M%S")}'

    """     print('Getting files:')
    sc_pos = spz.get_data(amda_dir[0], start_date, stop_date).to_dataframe()
    dens  = spz.get_data(amda_dir[1], start_date, stop_date).to_dataframe()
    dens = clean_dataframe(dens, species) """

    print('Calculating max distance')
    radius = 5
    n_bins = 50

    """     print('Merging frames')
    pos_dens_df = merge_dataframes(sc_pos, dens)
    print(f'has shape {pos_dens_df.shape}') """
     
    # Boundary edges for the bins/grids
    xedges = np.linspace(-radius,radius,n_bins)
    yedges = np.linspace(-radius,radius,n_bins)
    zedges = np.linspace(-radius,radius,n_bins)
    edges = (xedges, yedges, zedges)
    edges_n_bins = (xedges.shape[0]-1,
                yedges.shape[0]-1,
                zedges.shape[0]-1
                )


    hist_xyz_nrmeas, hist_xyz_dens = fix_histogram_placeholder(edges_n_bins)

    """     start_date = dt.datetime(2015,1,1)
    stop_date = dt.datetime(2015,10,1)
    tdt = dt.timedelta(weeks=104)
    stop_date = start_date+tdt """
    time_delta = dt.timedelta(weeks=4)
    t0 = start_date
    t1 = t0+time_delta
    while t0 < stop_date:

        sc_pos = spz.get_data(amda_dir[0], t0, t1).to_dataframe()
        dens  = spz.get_data(amda_dir[1], t0, t1).to_dataframe()
        dens = clean_dataframe(dens, species)

        print('Merging frames')
        pos_dens_df = merge_dataframes(sc_pos, dens)
        print(f'has shape {pos_dens_df.shape}')

        hist_xyz_nrmeas, hist_xyz_dens = sum_histogram_data(hist_xyz_nrmeas,
                                                            hist_xyz_dens,
                                                            pos_dens_df[species],
                                                            pos_dens_df[['x', 'y', 'z']],
                                                            edges)
        
        t0 = t1
        t1 += time_delta
        # plotta sista fucking jäveln också
    
    plot_histogram_data(hist_xyz_nrmeas, hist_xyz_dens, species, edges, filename)

    run_time_t2 = dt.datetime.now()
    delta = run_time_t2 - run_time_t0
    print(f'Runtime: {delta}')



if __name__ == '__main__':
    main()