import speasy as spz
from speasy import amda
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

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
    hist_nrmeas, _ = np.histogramdd(sc_position_ntp, bins=edges)
    hist_den, _ = np.histogramdd(sc_position_ntp, bins=edges, weights=density)
    hist_xyz_nrmeas += hist_nrmeas
    hist_xyz_den += hist_den
    return hist_xyz_nrmeas, hist_xyz_den

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

def clean_dataframe(dataframe):
    dataframe = dataframe[dataframe[parameter].notna() & (dataframe[parameter] != 0)]
    return dataframe

# Basic logic 
# Get data  x
# Convert to df x
# merge temporally x
# put into bins and plot x
# maybe add a loop to do it daily or monthly or smth

def main():

    # Boundary edges for the bins/grids
    xedges = np.linspace(-10,10,10)
    yedges = np.linspace(-10,10,10)
    zedges = np.linspace(-10,10,10)
    edges = (xedges, yedges, zedges)
    edges_n_bins = (xedges.shape[0]-1,
                yedges.shape[0]-1,
                zedges.shape[0]-1
                )
    
    hist_xyz_nrmeas, hist_xyz_dens = fix_histogram_placeholder(edges_n_bins)

    start_date = dt.datetime(2017,5,1)
    stop_date = dt.datetime(2017,5,20)

    amda_dir = [
        amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
        amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n
    ]

    start_date, stop_date = amddh.retrieve_restrictive_time_boundaries(amda_dir)
    time_delta = dt.timedelta(hours=24)
    start_date = stop_date-time_delta

    sc_pos = spz.get_data(amda_dir[0], start_date, stop_date).to_dataframe()
    dens  = spz.get_data(amda_dir[1], start_date, stop_date).to_dataframe()
    dens = clean_dataframe(dens)

    pos_dens_df = merge_dataframes(sc_pos, dens)

    hist_xyz_nrmeas, hist_xyz_dens = sum_histogram_data(hist_xyz_nrmeas,
                                                        hist_xyz_dens,
                                                        pos_dens_df['density'],
                                                        pos_dens_df[])




if __name__ == '__main__':
    main()