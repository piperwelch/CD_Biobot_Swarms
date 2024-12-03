### In Silico Analysis

## Raw data 
- Stored in *results/BOT_ID/BOT_ID_res_\*_TX_run_\*_results_report.p* - single pickle file for each treatment for each bot
- Contents of pickle file is a **dictionary** (key: BOT_ID_res_\*_TX_run_\*, value: pandas data frame containing the trajectory for that run)
- Structure of pandas dataframe.. columns: t, x, y

## Rotate and shift the trajectories (data_analysis/rotated_trajectories.py)
- Data stored in *rotated_trajectories/BOT_ID/BOT_ID_res_\*_TX_run_\*_trajectories.p* - single pickle file for each treatment for each bot
- Columns added to raw trajectory data frames: x_shift, y_shift, x_rotate, y_rotate
- x_shift, y_shift = trajectory aftering shifting the initial position to (0,0)
- x_rotate, y_rotate = trajectory after rotation to match the initial in vitro **orientation** of the bot
- Use x_rotate, y_rotate as the coordinates of the trajectory in the following analyses
- *plot_trajectories.py* used to plot the trajectories from this data set

## Analysis of Behavioral Measures

# Heading Analysis 

Compare the average error in heading between the in silico bot and the in silico bot (true heading) for each treatment and across all bots

- Compute error in heading for each trajectory for each treatment for each bot and record resuls in a csv
    - *data_analysis/compute_error_in_headings.py*
    - Extract in vitro heading data stored in *heading_data/in_vitro/headings.csv*
    - Iterate through in silico trajectories and compute initial heading of the bot
    - Compute error between in silico heading and in vitro heading as the angle between headings (ranges between 0-180)
    - Store error in data frame with columns: TX, run, error, normalized error (between 0-1)
    - write of df as csv in *heading_errors/BOT_ID.csv*
- Exploratory data analysis of the heading errors of individual bots heading errors
    - *investigate_heading_errors.py*
    - plot histograms of the normalized errors - stored in *plots/BOT_ID/heading_errors_hist.png*
    - plot means and standard deviations of the normalized errors - stored in *plots/BOT_ID/heading_errors_mean_scatter.png*
- Aggregate data across all bots and compute statistics
    - *aggregate_heading_errors.py*
    - merge all errors associated with a specific treatment, then compute mean and standard deviation of the entire distribution
    - compute statistics on the mean and standard deviation and plot aggregate results
    - repeat the analysis but split by linear vs. circular bots

