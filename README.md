# MAPS
Magnetosheath Analysis for Planetary Systems

amda_datahandler:
    Downloads an array of amda datasets and saves them locally
    Features: 
        1. Saving into .parquet
        2. Loading into pandas dataframe
        3. Finding most restrictive time boundaries of all sets
        4. bugs

average_plot:
    Produces 2D-statistical maps for pos and some parameter, as much as possible is automatically generated
    Featuers:
        Give pos dataset and paramdataset (say, density or temperature)
        Matches position and param time-axises, cleans dataset, and plots
    
    Possible features:
        - Do it chunkwise between two large time-ranges, perhaps monthly or yearly
        - ??
        - Make more beautiful plots
        - 