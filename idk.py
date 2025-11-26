import pandas as pd
import numpy as np
import datetime as dt
import speasy as spz
import amda_datahandler as amddh
import os
amda_tree = spz.inventories.tree.amda

dirs = [
    amda_tree.Parameters.Solar_Wind_Propagation_Models.Jupiter.Tao_Model.tao_jup_dsc.jup_dsc_n,
    amda_tree.Parameters.Solar_Wind_Propagation_Models.Jupiter.Tao_Model.tao_jup_dsc.jup_dsc_t,
    amda_tree.Parameters.Solar_Wind_Propagation_Models.Jupiter.Tao_Model.tao_jup_dsc.jup_dsc_v.jup_dsc_v0,
]

start_date, stop_date = amddh.retrieve_restrictive_time_boundaries(dirs)
delta = dt.timedelta(weeks=20)
start_date = stop_date - delta

for dir in dirs[:-1]:
    print(f'{dir.xmlid}')
    df = spz.get_data(dir, start_date, stop_date).to_dataframe()
    print(df.describe())
vel_df = spz.get_data(dirs[-1], start_date, stop_date).to_dataframe()
print(vel_df.head)
#df = pd.DataFrame({'v' : })
