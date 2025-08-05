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

print('Printing attributes and values')
for attribute, value in vars(amda_tree.Parameters.Cluster.Cluster_1.Ephemeris.clust1_orb_all.c1_xyz_gse).items():
    print(f"Attribute: {attribute}, Value: {value}")
print('Initial printing done\nMoving on to imported data...\n\n')

""" start_time = dt.datetime(2017, 5, 1, 0, 0)
stop_time = dt.datetime(2017, 5, 2, 0, 0)

dataset_scpos = spz.get_data(amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
                            start_time, stop_time)

xyz_jso = amda.get_parameter('juno_eph_orb_jso',
                            start_time, stop_time)

print(xyz_jso)
print(xyz_jso.columns)
print(xyz_jso.time)
print(xyz_jso.values)
print(xyz_jso.shape)
print(dataset_scpos)
print(dataset_scpos.columns)
print(dataset_scpos.time)
print(dataset_scpos.values)
print(dataset_scpos.shape)

print(dataset_scpos.values[:,0].shape)
print(dataset_scpos['x'].values.shape)
a = dataset_scpos.values[:,0].reshape(-1, 1)-dataset_scpos['x']
print(np.sum(a))
#start_date = amda_tree.Parameters.MAVEN.Ephemeris.maven_orb_marsobs.start_date
#stop_date = amda_tree.Parameters.MAVEN.Ephemeris.maven_orb_marsobs.stop_date """