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
import planets as hp_planets

amda_tree = spz.inventories.tree.amda


def fix_histogram_placeholder(edges_n_bins):
    hist_xyz_nrmeas = np.zeros(edges_n_bins)
    hist_xyz_den = np.zeros(edges_n_bins)
    return hist_xyz_nrmeas, hist_xyz_den

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

def plot_histogram_data(hist_xyz_nrmeas, hist_xyz_den, hist_xr_nrmeas, hist_xr_dens, species, edges, filename, filepath_name, info, eph_name, times):

    cmap = plt.cm.inferno
    cmap.set_under(cmap(0))
    
    xedges, yedges, zedges, redges = edges[0], edges[1], edges[2], edges[3]
    # Fix average density for each histogram and then extract for plotting

    histogram_xyz_average = hist_xyz_den / hist_xyz_nrmeas
    histogram_xr_average = hist_xr_dens / hist_xr_nrmeas
    print(histogram_xyz_average.shape)
    idx = int(xedges.shape[0] / 2)
    idy = int(yedges.shape[0] / 2)
    idz = int(zedges.shape[0] / 2)

    """     # Skapa meshgrid av y- och z-edges för att beräkna r
    Yc, Zc = np.meshgrid((yedges[:-1] + yedges[1:]) / 2, (zedges[:-1] + zedges[1:]) / 2, indexing='ij')
    R = np.sqrt(Yc**2 + Zc**2)

    # Initiera hist_xr
    hist_xr = np.full((len(xedges)-1, len(redges)-1), np.nan)
    hist_xr_nr = np.full((len(xedges)-1, len(redges)-1), np.nan)

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
            hist_xr_nr[ix,:] = hist_count """

   

    """     hist_xy = histogram_xyz_average[:, :, idz]
    hist_xz = histogram_xyz_average[:, idy, :]
    hist_yz = histogram_xyz_average[idx, :, :] """

    """     hist_xy = np.nanmean(histogram_xyz_average[:, :, idz-t:idz+t], axis=2)
    hist_xz = np.nanmean(histogram_xyz_average[:, idy-t:idy+t, :], axis=1)
    hist_yz = np.nanmean(histogram_xyz_average[idx-t:idx+t, :, :], axis=0) """

    hist_xy = histogram_xyz_average[:, :, idz]
    hist_xz = histogram_xyz_average[:, idy, :]
    hist_yz = histogram_xyz_average[idx, :, :]
    

    t = 2
    hist_xy = np.nanmean(histogram_xyz_average[:, :, idz-t:idz+t], axis=2)
    hist_xz = np.nanmean(histogram_xyz_average[:, idy-t:idy+t, :], axis=1)
    hist_yz = np.nanmean(histogram_xyz_average[idx-t:idx+t, :, :], axis=0)

    hist_xy_nmeas = np.nanmean(hist_xyz_nrmeas[:, :, idz-t:idz+t], axis=2)
    hist_xz_nmeas = np.nanmean(hist_xyz_nrmeas[:, idy-t:idy+t, :], axis=1)
    hist_yz_nmeas = np.nanmean(hist_xyz_nrmeas[idx-t:idx+t, :, :], axis=0)

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
        histogram_xr_average[histogram_xr_average > 0]
    ))

    if vals.size == 0:
        print(f'INGEN DATA O PLOTTA')
        return
    
    cmin, cmax = vals.min(), vals.max()

    # Plot all the data
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, width_ratios=[10, 10, 1])

    """     ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_yz = fig.add_subplot(gs[0, 1])
    ax_xr = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[:, -1]) """

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xr = fig.add_subplot(gs[1, 0])
    ax_xy_nmeas = fig.add_subplot(gs[0, 1])
    ax_xr_nmeas = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[:, -1])


    ax_xy.pcolormesh(xedges, yedges, hist_xy.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax), shading='auto')
    ax_xr.pcolormesh(xedges, redges, histogram_xr_average.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax),shading='auto')
    ax_xy_nmeas.pcolormesh(xedges, yedges, hist_xy_nmeas.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax), shading='auto')
    ax_xr_nmeas.pcolormesh(xedges, redges, hist_xr_nrmeas.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax),shading='auto')




    redge_slice, r_bin_slice = get_middle_idx_slice(len(redges))
    #redge_slice, r_bin_slice = add_thickness(redge_slice, r_bin_slice, thickness=2, direction='up')
    xedge_slice, x_bin_slice = get_middle_idx_slice(len(xedges))
    """ ax_xy.pcolormesh(xedges, yedges, hist_xy.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax), shading='auto')
    ax_xz.pcolormesh(xedges, zedges, hist_xz.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax),shading='auto')
    ax_yz.pcolormesh(yedges, zedges, hist_yz.T, cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax),shading='auto')
    ax_xr.pcolormesh(xedges[xedge_slice.start+14:-1], redges[2:redge_slice.stop-2], hist_xr.T[2:r_bin_slice.stop-2,x_bin_slice.start+14:-1], cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax),shading='auto') """
    avgavg = np.nanmean(histogram_xr_average.T[2:r_bin_slice.stop-2,x_bin_slice.start+14:-1])
    if filepath_name == 'gll_pls_fitt':
        avgavg = avgavg*11604.5250061657/10**6
        print(f'Avg for {filepath_name}: {avgavg} MK')
    else:
        print(f'Avg for {filepath_name}: {avgavg}')
    """     hp.add_bow_shock_magnetopause_plot(ax_xy)
    hp.add_bow_shock_magnetopause_plot(ax_xz)
    hp.add_bow_shock_magnetopause_plot_yz(ax_yz)
    # hp.add_bow_shock_magnetopause_plot(ax_xr) """



    ax_xy.set_xlabel(r'$JSO\ X\ [\mathrm{R_J}]$')
    ax_xy_nmeas.set_xlabel(r'$JSO\ X\ [\mathrm{R_J}]$')
    ax_xr_nmeas.set_xlabel(r'$JSO\ X\ [\mathrm{R_J}]$')
    ax_xr.set_xlabel(r'$JSO\ X\ [\mathrm{R_J}]$')

    ax_xy.set_ylabel(r'$JSO\ Y\ [\mathrm{R_J}]$')
    ax_xy_nmeas.set_ylabel(r'$JSO\ Y\ [\mathrm{R_J}]$')
    ax_xr_nmeas.set_ylabel(r'$JSO\ R\ [\mathrm{R_J}]$')
    ax_xr.set_ylabel(r'$JSO\ R\ [\mathrm{R_J}]$')

    # Fix the ranges of the orbital plots
    ax_xy.set_xlim(xedges[0], xedges[-1])
    ax_xy.set_ylim(yedges[0], yedges[-1])

    ax_xy_nmeas.set_xlim(xedges[0], xedges[-1])
    ax_xy_nmeas.set_ylim(zedges[0], zedges[-1])

    ax_xr_nmeas.set_xlim(xedges[0], xedges[-1])
    ax_xr_nmeas.set_ylim(redges[0], redges[-1])

    ax_xr.set_xlim(xedges[0], xedges[-1])
    ax_xr.set_ylim(redges[0], redges[-1])

    print(f'Filepath name: {eph_name}')
    for ax in [ax_xy, ax_xy_nmeas,ax_xr_nmeas, ax_xr]:  # , ax_xr
        ax.set_aspect('equal')
        if eph_name == 'juno_eph_orb_jso' or eph_name == 'gll_xyz_jso':
            hp_planets.add_jupiter_bow_shock_magnetopause_plot(ax, alpha=0.5, labels=True)
        if eph_name == 'cass_xyz_kso_1s':
            hp_planets.add_saturn_bow_shock_magnetopause_plot(ax, alpha=0.5, labels=True)
    
    """ if eph_name == 'juno_eph_orb_jso' or eph_name == 'gll_xyz_jso':

        ax_xy_nmeas.set_aspect('equal')
        hp_planets.add_planet_in_plot(ax_xy_nmeas)

        ax_xr.legend()

        # Fix the ranges of the orbital plots
        hp_planets.fix_spatial_plot_limits(x_lims=[-100, 100], y_lims=[-100, 100], z_lims=[-100, 100], r_lims=[0, 100], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr)

    if eph_name == 'cass_xyz_kso_1s':
        ax_yz.set_aspect('equal')
        hp_planets.add_planet_in_plot(ax_yz)

        ax_xr.legend()

        # Fix the ranges of the orbital plots
        hp_planets.fix_spatial_plot_limits(x_lims=[-35, 35], y_lims=[-35, 35], z_lims=[-35, 35], r_lims=[0, 35], ax_mso_xy=ax_xy, ax_mso_xz=ax_xz, ax_mso_yz=ax_yz, ax_mso_xr=ax_xr) """


    # Add a colorbar with the counts
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.LogNorm(vmin=cmin, vmax=cmax))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cb, orientation='vertical')
    # cb.ax.set_yticklabels([timeDt_valid[0].strftime('%H:%M'), timeDt_valid[-1].strftime('%H:%M')])  # vertically oriented colorbar
    cb.set_label('some info')

    #plt.suptitle(filepath_name+f' w/ bins {info[0]} and edges {info[1]}')
    start_str = times[0]
    stop_str = times[1]

    if filepath_name == 'gll_pls_fitn':
        plt.suptitle(r'Jupiter average proton density [$cm^{-3}$]'+ f' from {start_str} to {stop_str}')
    if filepath_name == 'gll_pls_fitt':
        plt.suptitle(r'Jupiter average temperature [$eV$]'+f' from {start_str} to {stop_str}')
    if filepath_name == 'gll_pls_fitv':
        plt.suptitle(r'Jupiter average speed  [$km/s$]'+ f' from {start_str} to {stop_str}')
    if filepath_name == 'gmmr_magnitude':
        plt.suptitle(r'Jupiter average b-field density [$nT$]'+ f'from {start_str} to {stop_str}')

    # Save figure
    fig.savefig(filename, bbox_inches='tight')
    plt.close('all')
    return (avgavg, filepath_name)


# Basic logic 
# Get data  x
# Convert to df x
# merge temporally x
# put into bins and plot x
# maybe add a loop to do it daily or monthly or smth

def main():
    
    avg_val_array = []
    radius = 100
    n_bins = 51
    
    # Boundary edges for the bins/grids
    xedges = np.linspace(-radius,radius,n_bins)
    yedges = np.linspace(-radius,radius,n_bins)
    zedges = np.linspace(-radius,radius,n_bins)
    redges = np.linspace(0,radius,n_bins)
    edges = (xedges, yedges, zedges, redges)
    edges_n_bins = (
                xedges.shape[0]-1,
                yedges.shape[0]-1,
                zedges.shape[0]-1
            )

    run_time_t0 = dt.datetime.now()

    print(f'Initializing directories')

    dir_clut4 = [amda_tree.Parameters.Cluster.Cluster_4.Ephemeris.clust4_orb_all.c4_xyz_gse,
    amda_tree.Parameters.Cluster.Cluster_4.CIS_CODIF.clust4_cis_prp.c4_h_dens,
    amda_tree.Parameters.Cluster.Cluster_4.CIS_CODIF.clust4_cis_prp.c4_o_dens
    ]

    dir_clut1 = [amda_tree.Parameters.Cluster.Cluster_1.Ephemeris.clust1_orb_all.c1_xyz_gse,
    amda_tree.Parameters.Cluster.Cluster_1.CIS_CODIF.clust1_cis_prp.c1_h_dens,
    amda_tree.Parameters.Cluster.Cluster_1.CIS_CODIF.clust1_cis_prp.c1_o_dens,
    #amda_tree.Parameters.Cluster.Cluster_1.CIS_CODIF.clust1_cis_prp.c1_h_t
    amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_t,
    amda_tree.Parameters.Cluster.Cluster_1.CIS_HIA.clust1_hia_mom.c1_hia_press
    ]

    dir_clut3 = [amda_tree.Parameters.Cluster.Cluster_3.Ephemeris.clust3_orb_all.c3_xyz_gse,
    amda_tree.Parameters.Cluster.Cluster_3.CIS_CODIF.clust3_cis_prp.c3_h_dens,
    amda_tree.Parameters.Cluster.Cluster_3.CIS_CODIF.clust3_cis_prp.c3_o_dens
    ]

    dir_juno = [amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
        amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n,
        amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_p,
        amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_t,
        amda_tree.Parameters.Juno.FGM.orbit_jupiter.juno_fgm_orb1.juno_fgm_orb1_mag,
        amda_tree.Parameters.Juno.FGM.orbit_jupiter.juno_fgm_orb60.juno_fgm_orb60_mag,
    ]

    dir_cassini_saturn = [amda_tree.Parameters.Cassini.Ephemeris__Cassini.orbit_saturn.cass_orb_1s.cass_xyz_kso_1s,
        amda_tree.Parameters.Cassini.MAG.orbit_saturn.cass_mag_orb1.cass_bmag_orb1s,
        amda_tree.Parameters.Cassini.RPWS.cass_fpe_sat.cass_sat_ne,
    ]

    dir_cassini_jup = [amda_tree.Parameters.Cassini.Ephemeris__Cassini.cass_ephem_eqt.cass_xyz_eqt,
        amda_tree.Parameters.Cassini.RPWS.cass_fpe_jup.cass_jup_ne,
        amda_tree.Parameters.Cassini.MAG.cass_mag_jup.cass_bmag_jup,
        amda_tree.Parameters.Cassini.CAPS.cass_caps_elemo.cass_caps_e_n,
        amda_tree.Parameters.Cassini.CAPS.cass_caps_elemo.cass_caps_e_t,

    ]

    dir_galileo = [amda_tree.Parameters.Galileo.Ephemeris___Galileo.gll_orbit_jup.gll_xyz_jso,
        amda_tree.Parameters.Galileo.MAG.gll_mag_msreal.gmmr_magnitude,
        amda_tree.Parameters.Galileo.PLS.gll_pls_fit.gll_pls_fitn,
        amda_tree.Parameters.Galileo.PLS.gll_pls_fit.gll_pls_fitt,
        amda_tree.Parameters.Galileo.PLS.gll_pls_fit.gll_pls_fitv,
        #amda_tree.Parameters.Galileo.Jovian_Magnetic_Field_Models.gll_jrm09_model.b_jrm09_can_tot_60_1_gll_xyz_iau,
        #amda_tree.Parameters.Galileo.Jovian_Magnetic_Field_Models.gll_jrm33_model.b_jrm33_con_tot_60_1_gll_xyz_iau
    ]

    dir_maven = [amda_tree.Parameters.MAVEN.Ephemeris.maven_orb_marsobs1s.mav_xyz_mso1s,
        amda_tree.Parameters.MAVEN.SWIA.mav_swia_kp.mav_swiakp_n
    ]

    amda_dirs = [dir_galileo]



    for dir in amda_dirs:
        pos_dir = dir.pop(0)

        for param_dir in dir:
            print(f'Collecting {param_dir.xmlid}')
            species = param_dir.name
            now = dt.datetime.now()
            start_date, stop_date = amddh.retrieve_restrictive_time_boundaries([pos_dir, param_dir])
            start_str, stop_str = start_date.strftime("%Y%m%d"), stop_date.strftime("%Y%m%d")

            eph_name = pos_dir.xmlid
            filepath_name = param_dir.xmlid
            filepath_plot = 'Plots'
            filepath_data = 'Data'
            filepath = 'Saved/'

            filepath += filepath_name+'/'
            filepath_plot = filepath + filepath_plot
            filepath_data = filepath + filepath_data
            if not os.path.exists(filepath):
                os.makedirs(filepath)

            if not os.path.exists(filepath_data):
                os.makedirs(filepath_data)

            if not os.path.exists(filepath_plot):
                os.makedirs(filepath_plot)


            filename_plot = f'{filepath_plot}/average_{start_date.strftime("%Y%m%d__%H%M%S")}--{stop_date.strftime("%Y%m%d__%H%M%S")}__created__{now.strftime("%Y%m%d__%H%M%S")}'
            filename_histogram = f'{filepath_data}/__bins__{n_bins}__radius__{radius}'
            
            
            """     print('Getting files:')
            sc_pos = spz.get_data(amda_dir[0], start_date, stop_date).to_dataframe()
            dens  = spz.get_data(amda_dir[1], start_date, stop_date).to_dataframe()
            dens = clean_dataframe(dens, species) """

            print('Calculating max distance')


            """     print('Merging frames')
            pos_dens_df = merge_dataframes(sc_pos, dens)
            print(f'has shape {pos_dens_df.shape}') """

            hist_xyz_nrmeas, hist_xyz_dens = fix_histogram_placeholder(edges_n_bins)
            hist_xr_nrmeas, hist_xr_dens = fix_histogram_placeholder((xedges.shape[0]-1, redges.shape[0]-1))


            """     start_date = dt.datetime(2015,1,1)
            stop_date = dt.datetime(2015,10,1)
            tdt = dt.timedelta(weeks=104)
            stop_date = start_date+tdt """
            time_delta = dt.timedelta(weeks=4)

            tot = stop_date - start_date

            t0 = start_date
            t1 = t0+time_delta
            iterations = 0
            print(f'Generating between {start_date} and {stop_date}')

            # If there is no histogram info present create:
            if not os.path.exists(filename_histogram+'.npz'):
                print(f'File: {filename_histogram}.npz does not exist\n',
                    'Generating...')
                while t0 < stop_date:
                    part = t1 - start_date
                    progress = part/tot 
                    print(f'Current progress: {progress:.2f}')
                    print(f'Time: {t0} to {t1}')


                    if os.path.exists(f'{filepath_data}/chunk_({iterations}).parquet'):
                        pos_dens_df = amddh.load_parquet(f'{filepath_data}/chunk_({iterations}).parquet')
                        if not 'r' in pos_dens_df:
                            pos_dens_df['r'] = (pos_dens_df['y']**2 + pos_dens_df['z']**2)**0.5
                            pos_dens_df.to_parquet(f'{filepath_data}/chunk_({iterations}).parquet')
                    else:
                        sc_pos = spz.get_data(pos_dir, t0, t1).to_dataframe()
                        if param_dir.xmlid == 'gll_pls_fitv':
                            species = 'v'
                            temp  = spz.get_data(param_dir, t0, t1).to_dataframe()
                            dens = pd.DataFrame({species: np.sqrt(temp['vr']**2 + temp['vth']**2 + temp['vph']**2)})
                
                        elif param_dir.xmlid == 'c1_h_t':
                            species = 't'
                            temp  = spz.get_data(param_dir, t0, t1).to_dataframe()
                            dens = pd.DataFrame({species: np.sqrt(temp['t_para']**2 + temp['t_perp']**2)})
                        
                        else:
                            dens = spz.get_data(param_dir, t0, t1).to_dataframe()
                    
                        dens = clean_dataframe(dens, species)

                        print('Merging frames')
                        pos_dens_df = merge_dataframes(sc_pos, dens)
                        print(f'has shape {pos_dens_df.shape}')

                        print(f'Save to parquet chunk')
                        if not 'r' in pos_dens_df:
                            pos_dens_df['r'] = (pos_dens_df['y']**2 + pos_dens_df['z']**2)**0.5
                        pos_dens_df.to_parquet(f'{filepath_data}/chunk_({iterations}).parquet')

                    # Spara denna bitch? Dvs, spara i chunks som förut..., men behåll chunksen?

                    print(f'Summing histogram data')
                    if param_dir.xmlid == 'gll_pls_fitv':
                        species = 'v'
                    elif param_dir.xmlid == 'c1_h_t':
                        species = 't'
                    
                    hist_xyz_nrmeas, hist_xyz_dens = sum_histogram_data(hist_xyz_nrmeas,
                                                                        hist_xyz_dens,
                                                                        pos_dens_df[species],
                                                                        pos_dens_df[['x', 'y', 'z']],
                                                                        (xedges,yedges,zedges))
                    
                    hist_xr_nrmeas, hist_xr_dens = sum_histogram_data(hist_xr_nrmeas,
                                                                      hist_xr_dens,
                                                                      pos_dens_df[species],
                                                                      pos_dens_df[['x', 'r']],
                                                                      (xedges, redges))
                    

                    
                    t0 = t1
                    t1 += time_delta
                    iterations += 1
                amddh.save_info(iterations,filepath_data)
                #if not os.path.exists(filepath_data+'/full.parquet'):
                #    amddh.combine_parquet_chunks(filepath_data+'/full',filepath_data+'/')
            
                amddh.save_histogram(hist_xyz_nrmeas, hist_xyz_dens, hist_xr_nrmeas, hist_xr_dens, edges, filename_histogram)

            

            print(f'loading histogram')
            print(f'{filename_histogram}')
            hist_xyz_nrmeas, hist_xyz_dens, hist_xr_nrmeas, hist_xr_dens, edges = amddh.load_histogram(filename_histogram)

            #pos_dens_df = amddh.load_parquet(f'{filepath_data}/full.parquet')
            #hist_xyz_nrmeas, hist_xyz_dens = sum_histogram_data(hist_xyz_nrmeas,
            #                                                            hist_xyz_dens,
            #                                                            pos_dens_df[species],
            #                                                            pos_dens_df[['x', 'y', 'z']],
            #                                                            edges)

            print(f'plotting histogram')
            avg_val = plot_histogram_data(hist_xyz_nrmeas, hist_xyz_dens, hist_xr_nrmeas, hist_xr_dens, species, edges, filename_plot, filepath_name, (n_bins, radius), eph_name, (start_str, stop_str))
            avg_val_array.append(avg_val)
        
        print()
        for e in avg_val_array:
            print(f'{e[1]} = {e[0]}')
        print()

        run_time_t2 = dt.datetime.now()
        delta = run_time_t2 - run_time_t0
        print(f'Runtime: {delta}')



if __name__ == '__main__':
    main()