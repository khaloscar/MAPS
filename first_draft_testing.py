# Import relevant data from Juno
# keept it simple stupid
# position in JSO, thats cool bro

# Import libraries
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

def debugger(data):

    print('Printing attributes and values')
    for attribute, value in vars(amda_tree.Parameters.Juno.JADE.L5___ions.juno_jadel5_protmom).items():
        print(f"Attribute: {attribute}, Value: {value}")
    print('Initial printing done\nMoving on to imported data...\n\n')

    if isinstance(data, spz.products.variable.SpeasyVariable):
        data = [data]

    for var in data:
        print('\n############################\n')
        print(f'Printing info for: {var.name}')
        print(f'Var type: {var}')
        print(f'Shape is: {var.values.shape}')
        print(f'Columns contain: \n',var.columns)
        print(f'Time has the size: ',var.time.shape)
        print(f'Units: ', var.unit)
        print(f'Meta: ', var.meta)
        print(f'Nbytes: ', var.nbytes)
        print(f'\n End of line \n',
              '###################')
        
def load_juno_data():
    
    # Protons and its densities
    start_date = amda_tree.Parameters.Juno.JADE.L5___ions.juno_jadel5_protmom.start_date
    stop_date = amda_tree.Parameters.Juno.JADE.L5___ions.juno_jadel5_protmom.stop_date
    stop_date = dt.datetime(2016,7,26,0,0)

    jade_protmom_n = amda.get_parameter('jade_protmom_n',
                                        start_date, stop_date)
    
    start_date = amda_tree.Parameters.Juno.JADE.L5___ions.juno_jadel5_protmom.start_date
    stop_date = amda_tree.Parameters.Juno.JADE.L5___ions.juno_jadel5_protmom.stop_date
    stop_date = dt.datetime(2016,7,26,0,0)

    jade_protmom_lt = amda.get_parameter('jade_protmom_lt',
                                    start_date, stop_date)
    
    start_date = amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.start_date
    stop_date = amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.stop_date
    stop_date = dt.datetime(2016,7,26,0,0)
    
    xyz_jso = amda.get_parameter('juno_eph_orb_jso',
                                start_date, stop_date)
    
    print(jade_protmom_lt.values.size)

    return [xyz_jso, jade_protmom_n, jade_protmom_lt]


def plot_datasets(datasets):
    
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(
                            ncols=5, 
                            nrows=4, 
                            figure=fig,
                            width_ratios=[10, 10, 10, 10, 1], 
                            height_ratios=[10, 10, 10, 10, 10]
)

    # subplotting 
    ax_xyz_jso = fig.add_subplot(gs[0, :-1])
    ax_jade_protmom_lt = fig.add_subplot(gs[1, :-1], sharex=ax_xyz_jso)
    ax_jade_protmom_n = fig.add_subplot(gs[2, :-1], sharex=ax_xyz_jso)
    ax_jso_xy = fig.add_subplot(gs[3,0])
    ax_jso_xz = fig.add_subplot(gs[3,1])
    ax_jso_yz = fig.add_subplot(gs[3,2])
    ax_jso_xr = fig.add_subplot(gs[3,3])
    ax_jso_time = fig.add_subplot(gs[3,4])

    xyz_jso = datasets[0]
    jade_protmom_n = datasets[1]
    jade_protmom_lt = datasets[2]
    
    # Plot xyz data
    ax_xyz_jso.plot(xyz_jso.time, xyz_jso.values)

    # Plot local time
    ax_jade_protmom_lt.plot(jade_protmom_lt.time, jade_protmom_lt.values)

    # Plot number density
    ax_jade_protmom_n.plot(jade_protmom_n.time, jade_protmom_n.values)

    # Plot trajectories
    ratio = (xyz_jso.time - xyz_jso.time[0])/(xyz_jso.time[-1] - xyz_jso.time[0])
    cmap_time = plt.cm.bone_r
    color = [cmap_time(r) for r in ratio]

    ax_jso_xy.scatter(xyz_jso.values[:,0], xyz_jso.values[:,1], c=color)
    ax_jso_xy.set_xlabel('X JSO')
    ax_jso_xy.set_ylabel('Y JSO')

    ax_jso_xz.scatter(xyz_jso.values[:,0], xyz_jso.values[:,2], c=color)
    ax_jso_xz.set_xlabel('X JSO')
    ax_jso_xz.set_ylabel('Z JSO')

    ax_jso_yz.scatter(xyz_jso.values[:,1], xyz_jso.values[:,2], c=color)
    ax_jso_yz.set_xlabel('Y JSO')
    ax_jso_yz.set_ylabel('Z JSO')

    ax_jso_xr.scatter(xyz_jso.values[:,0], np.sqrt(np.nansum(xyz_jso.values[:, 1:]**2, axis=-1)), c=color)
    ax_jso_xr.set_xlabel('W JSO')
    ax_jso_xr.set_ylabel('R JSO')

    plt.show()

def main():

    # Import data
    # juno datetime range 2016/07/06 00:00:00 — 2025/10/20 00:00:00
    start_time = dt.datetime(2016,7,24,0,0)
    end_time = dt.datetime(2016,7,26,0,0)

    datasets = load_juno_data()
    plot_datasets(datasets)
    #debugger(datasets)

if __name__ == "__main__":
    main()