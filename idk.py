import pandas as pd
import numpy as np
import datetime as dt
import speasy as spz
import amda_datahandler as amddh
amda_tree = spz.inventories.tree.amda

start_date_density = dt.datetime(2015,1,1,0,1,1,15)
start_date_pos = dt.datetime(2015,1,1,0,0,1)
timeDelta = dt.timedelta(seconds=30)
time_stamps_density = [start_date_density + f*timeDelta for f in range(10)]
time_stamps_pos = [start_date_pos + f*timeDelta/30 for f in range(5*61)]
pos = [f for f in range(len(time_stamps_pos))]
density = [2*f for f in range(len(time_stamps_density))]

densities = pd.DataFrame({
    "density": density
})
densities.index = time_stamps_density

sc_pos = pd.DataFrame({
    "pos": pos
})
sc_pos.index = time_stamps_pos

df_merged = pd.merge_asof(
    densities.sort_index(),
    sc_pos.sort_index(),
    left_index=True,
    right_index=True,
    direction='nearest'
)

print(df_merged.head)

dens_extracted = []
pos_extracted = []
# Last density value will be missing
# real shame dude
for idxp in range(len(sc_pos.index)-1):
    t0 = sc_pos.index[idxp]
    t1 = sc_pos.index[idxp+1]
    
    for idxd in range(len(densities.index)):
        if t0 <= densities.index[idxd] < t1:
            dens_extracted.append(densities.density[idxd])
            pos_extracted.append(sc_pos.pos[t0])

print(dens_extracted)
print(pos_extracted)

""" amda_dir = [
    amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
    amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n
] """

amda_dir = [
    amda_tree.Parameters.MAVEN.Ephemeris.maven_orb_marsobs1s.mav_xyz_mso1s,
    amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n
]

start_date, stop_date = amddh.retrieve_restrictive_time_boundaries(amda_dir)
time_delta = dt.timedelta(hours=24)
start_date = stop_date-time_delta

sc_pos = spz.get_data(amda_dir[0], start_date-time_delta/24, stop_date+time_delta/24).to_dataframe()
dens  = spz.get_data(amda_dir[1], start_date, stop_date).to_dataframe()

print(sc_pos.head())
print(sc_pos.info())
print()
print()
print(dens.head())
print(dens.info())

df_merged = pd.merge_asof(
    dens.sort_index(),
    sc_pos.sort_index(),
    left_index=True,
    right_index=True,
    allow_exact_matches=True,
    direction='nearest'
)

df_merged['radius'] = (df_merged['x']**2 + df_merged['y']**2 + df_merged['z']**2)**0.5
radius = spz.get_data(amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_r, start_date, stop_date).to_dataframe()
df_merged = pd.merge_asof(
    df_merged.sort_index(),
    radius.sort_index(),
    left_index=True,
    right_index=True,
    direction='nearest'
)

print()
print()
print(df_merged.head())
print(df_merged.info())
print(df_merged[['x', 'y', 'z']])

# Får fram densitet o position för denna, dunderenkelt