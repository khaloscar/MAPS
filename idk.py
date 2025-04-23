import pandas as pd
import numpy as np
import datetime as dt
import speasy as spz
import amda_datahandler as amddh
amda_tree = spz.inventories.tree.amda

""" start_date_density = dt.datetime(2015,1,1,0,1,1,15)
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

amda_dir = [
    amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
    amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n
]

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

# Får fram densitet o position för denna, dunderenkelt """

def time_range_generator(start_date, stop_date):
    # Generates time range boundaries, as of now, yearly
    # ex: data available 2017/06/06 to 2019/06/06
    # gives 2017/06/06, 2018/01/01, 2019/01/01, 2019/06/06
    current_yr = start_date.year
    yrs =[start_date]
    while current_yr < stop_date.year:
        next_year = pd.to_datetime(f"{current_yr + 1}-01-01").tz_localize(start_date.tzinfo)
        yrs.append(next_year)
        current_yr += 1
    yrs.append(stop_date)
    print(f'Time boundaries: \n{yrs}')
    return yrs

def time_range_generator(start_date, stop_date, step='year'):
    # Generates time range boundaries based on the step: 'year', 'month', or 'day'
    if start_date > stop_date:
        raise ValueError("start_date must be before stop_date")
    
    current = start_date
    boundaries = [start_date]

    while current < stop_date:
        if step == 'year':
            next_step = pd.Timestamp(year=current.year + 1, month=1, day=1, tz=current.tzinfo)
        elif step == 'month':
            next_month = current.month + 1 if current.month < 12 else 1
            next_year = current.year if current.month < 12 else current.year + 1
            next_step = pd.Timestamp(year=next_year, month=next_month, day=1, tz=current.tzinfo)
        elif step == 'day':
            next_step = current + pd.Timedelta(days=1)
        else:
            raise ValueError("step must be 'year', 'month', or 'day'")
        
        if next_step >= stop_date:
            break
        boundaries.append(next_step)
        current = next_step

    boundaries.append(stop_date)
    print(f"Time boundaries ({step}s):\n{boundaries}")
    return boundaries

test_Date = dt.datetime(2017,1,12,13,55)

print(getattr(test_Date, 'year'))
print(getattr(test_Date, 'month'))
print(getattr(test_Date, 'day'))
print(getattr(test_Date, 'hour'))
print(getattr(test_Date, 'minute'))

start = dt.datetime(2017,2,3,2)
stop = dt.datetime(2017,2,6,6)

time_range_generator(start, stop, step='day')