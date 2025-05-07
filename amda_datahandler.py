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


def save_data(amda_dir, save_dir="Saved_data/", start_date_arg=None, stop_date_arg=None):
    # Save data from given amda_dir, just define amda_dir = [amda1, amda2, ..., amdaN]
    # If given start and stop dates, then it will yield data for all datasets
    # for those dates
    # Not given, will yield full range for each dataset
    

    # Checking if Save_dir exists, else create
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    # Just get the data and save it, preferrably whole thingy
    # Able to handle pathing in amda tree

    for dir in amda_dir:
        dataset_name = dir.xmlid
        save_dir = save_dir+dataset_name+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if start_date_arg is not None and stop_date_arg is not None:
            start_date, stop_date = start_date_arg, stop_date_arg
            start_date_str = start_date.strftime("%Y%m%d")
            stop_date_str = stop_date.strftime("%Y%m%d")

            file_name = f"{dataset_name}_{start_date_str}-{stop_date_str}.parquet" # data between t0 and t1
            file_path = os.path.join(save_dir, file_name)
        else:
            start_date = pd.to_datetime(dir.start_date)
            stop_date = pd.to_datetime(dir.stop_date)
            start_date_str = start_date.strftime("%Y%m%d")
            stop_date_str = stop_date.strftime("%Y%m%d")

            file_name = f"{dataset_name}_full.parquet" # for full datasets
            file_path = os.path.join(save_dir, file_name)

        print(f'Saving data for {dataset_name}...')
        bool_file_present = check_if_already_saved(file_path)

        if not bool_file_present:
            print('Generating yearly time chunks...\n')
            print(f'From: {start_date}')
            print(f'To: {stop_date}')

            # Lets Just generate all the years:
            # Loop yearly from start_date to stop_date
            # ex: data available 2017/06/06 to 2019/06/06
            # gives 2017/06/06, 2018/01/01, 2019/01/01, 2019/06/06
            yrs = time_range_generator(start_date, stop_date)

            print(f'All is well, moving on...\n')

            # downloads data chunkwise
            # then combines to single .parquet
            # then deletes chunks
            print(f'Yielding data chunks')
            for yr in range(len(yrs)-1):
                t0, t1 = yrs[yr], yrs[yr+1]
                t0_str, t1_str = t0.strftime("%Y%m%d"), t1.strftime("%Y%m%d")

                dataset = spz.get_data(dir, t0, t1)
                name = dataset.name
                dataset = dataset.to_dataframe()
                dataset.index = dataset.index.tz_localize('UTC')
                chunk_file_path = f"{save_dir}{name}_chunk_{t0_str}-{t1_str}.parquet"

                print('Saving data chunk')
                dataset.to_parquet(chunk_file_path, index=True)
                print('Data chunk saved...')

            print(f'All chunks saved to {save_dir}\n\n')
            combine_parquet_chunks(file_path, save_dir)
            delete_parquet_chunks(name, save_dir)

        else:
            print(f'File already exist: {file_path} -- skipping download ')


def retrieve_restrictive_time_boundaries(amda_dir):
    # All amda datasets exist between some t0 and t1
    # gets the most restrictive datetime boundaries
    # across all datasets
    # useful when you want to minimize amnt. of data 
    # downloaded

    print('Retrieving datetime boundaries')
    # Convert to datetime for start and stop dates
    start_dates = []
    stop_dates = []
    for dir in amda_dir:
        start_dates.append(pd.to_datetime(dir.start_date))
        stop_dates.append(pd.to_datetime(dir.stop_date))

    print(f'Dates present:\n{start_dates}\n{stop_dates}')
    print(f'Most restrictive start date {max(start_dates)}')
    print(f'Most restrictive stop date {min(stop_dates)}')
    print()

    return (max(start_dates), min(stop_dates))

def vizualize(amda_dir, start_date, stop_date):

    for d in amda_dir:
        # Basic plotting for amda datasets between two dates
        dataset = spz.get_data(d,
                                start_date, stop_date)
        dataset.plot()
        plt.show()

def combine_parquet_chunks(file_path, save_dir):
    # Combines .parquet chunks into single parquet, then deletes
    # error prone if non related chunks are present

    print(f'Combining chunks')
    parquet_chunks = [f for f in os.listdir(save_dir) if f.endswith(".parquet") and 'chunk' in f]

    df_list = [pd.read_parquet(os.path.join(save_dir, f)) for f in sorted(parquet_chunks)]
    combined_df = pd.concat(df_list)

    combined_df.to_parquet(file_path+'.parquet')
    print(f'Chunks combined to: {file_path}')

def delete_parquet_chunks(dataset_name, save_dir):
    print(f'Deleting chunks...')
    for f in os.listdir(save_dir):
        if f.endswith(".parquet") and "chunk" in f and dataset_name in f:
            os.remove(os.path.join(save_dir, f))
            print(f"Deleted: {f}")

def check_if_already_saved(file_path):

    return os.path.exists(file_path)

def load_parquet(file_path):
    # loads parquet into pandas dataframe
    # indexing might not be datetime class
    # double check
    print(f'Loading parquet into dataframe: {file_path}')
    df = pd.read_parquet(file_path)
    print(f'Dataframe ready')
    return df

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

def save_histogram(hist_nrmeas, hist_den, edges, filename):
    np.savez_compressed(
    f'{filename}.npz',
    hist_nrmeas=hist_nrmeas,
    hist_den=hist_den,
    xedges=edges[0],
    yedges=edges[1],
    zedges=edges[2],
    redges=edges[3]
    )
    print(f'Hist data saved!')

def save_info(iterations, filepath):
    np.savez_compressed(
        filepath+'/info.npz',
        iterations)
    
def load_info(filepath):
    info = np.load(filepath+'info.npz')
    iterations = info['iterations']
    return iterations

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

def main():

    """     amda_dir = [
    amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
    amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n,
    ] """

    amda_dir = [
        amda_tree.Parameters.Juno.Ephemeris.orbit_jupiter.juno_ephem_orb1.juno_eph_orb_jso,
        amda_tree.Parameters.Juno.JADE.L5___electrons.juno_jadel5_elecmom.jade_elecmom_n,
        amda_tree.Parameters.MAVEN.Ephemeris.maven_orb_marsobs1s.mav_xyz_mso1s,
        amda_tree.Parameters.MAVEN.NGIMS.mav_ngims_kp.mav_ngimskp_he
    ]

    save_dir = 'Saved_data/'
    start_date, stop_date = retrieve_restrictive_time_boundaries(amda_dir)
    save_data(amda_dir, save_dir=save_dir)

    """ df_loaded = load_parquet('Saved_data/juno_eph_orb_jso_full.parquet')

    size_bytes = df_loaded.memory_usage(deep=True).sum()
    size_mb = size_bytes / (1024 ** 2)
    print(f"DataFrame size: {size_mb:.2f} MB") """

if __name__ == "__main__":
    main()