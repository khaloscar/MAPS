# MAPS
Magnetosheath Analysis for Planetary Systems

amda_datahandler:

    Downloads an array of amda datasets* and saves them locally.
    Features: 
        1. Saving into .parquet
        2. Loading into pandas dataframe
        3. Finding most restrictive time boundaries of all sets
        4. bugs
    
    * Must be of the form amda_tree.Parameters.{SATTELITE}.{MAP}.{DATA} etc, i.e, uses spz.get_data
        and no the other one. Use figure_out_tree to figure out amda path

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

figure_out_tree:

    Useful for figuring out amda structure and finding amda filepath to dataset of interest