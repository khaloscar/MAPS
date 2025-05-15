import pandas as pd
import numpy as np
import datetime as dt
import speasy as spz
import amda_datahandler as amddh
import os
amda_tree = spz.inventories.tree.amda

radius = 150
n_bins = 100

# Boundary edges for the bins/grids
xedges = np.linspace(-radius,radius,n_bins)
yedges = np.linspace(-radius,radius,n_bins//4)
zedges = np.linspace(-radius,radius,n_bins//4)
redges = np.linspace(0,radius,n_bins//4)
edges = (xedges, yedges, zedges, redges)
edges_n_bins = (xedges.shape[0]-1,
            yedges.shape[0]-1,
            zedges.shape[0]-1
            )


print(f'Initializing directories')

dir_clut4 = [amda_tree.Parameters.Cluster.Cluster_4.Ephemeris.clust4_orb_all.c4_xyz_gse,
    amda_tree.Parameters.Cluster.Cluster_4.CIS_CODIF.clust4_cis_prp.c4_h_dens,
    amda_tree.Parameters.Cluster.Cluster_4.CIS_CODIF.clust4_cis_prp.c4_o_dens
]

dir_clut1 = [amda_tree.Parameters.Cluster.Cluster_1.Ephemeris.clust1_orb_all.c1_xyz_gse,
    amda_tree.Parameters.Cluster.Cluster_1.CIS_CODIF.clust1_cis_prp.c1_h_dens,
    amda_tree.Parameters.Cluster.Cluster_1.CIS_CODIF.clust1_cis_prp.c1_o_dens
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

amda_dirs = [dir_clut4, dir_clut1, dir_clut3, dir_juno]

for dir in amda_dirs:
    pos_dir = dir.pop(0)

    for param_dir in dir:
        
        print(f'Collecting {param_dir.xmlid}')
        species = param_dir.name
        now = dt.datetime.now()
        start_date, stop_date = amddh.retrieve_restrictive_time_boundaries([pos_dir, param_dir])

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

        time_delta = dt.timedelta(weeks=4)
        tot = stop_date - start_date
        t0 = start_date
        t1 = t0+time_delta
        iterations = 0

        while t0 < stop_date:
            part = t1 - start_date
            progress = part/tot 
            print(f'Current progress: {progress:.2f}')
            print(f'Time: {t0} to {t1}')


            if os.path.exists(f'{filepath_data}/chunk_({iterations}).parquet'):
                
                pos_dens_df = amddh.load_parquet(f'{filepath_data}/chunk_({iterations}).parquet')
                print(pos_dens_df.head())
                pos_dens_df['r'] = (pos_dens_df['y']**2 + pos_dens_df['z']**2)**0.5
                print(pos_dens_df.head())
            else:
                print('NO PARQUET TO SHOW')
                print('NO PARQUET TO SHOW')
                print('NO PARQUET TO SHOW')
                print('NO PARQUET TO SHOW')
                print('NO PARQUET TO SHOW')
                print('NO PARQUET TO SHOW')

            t0 = t1
            t1 += time_delta
            iterations += 1


