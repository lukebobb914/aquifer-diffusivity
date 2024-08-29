import pandas as pd
pd.options.mode.chained_assignment = None  
import numpy as np
from scipy.signal import find_peaks
import datetime
from typing import Tuple


## ------------------------ Functions for cleaning well df --------------------------- ##
def clean_input_data(well_df: pd.DataFrame, well_x_col: str, well_y_col: str, window_length: int, min_dist: int, shore_df: pd.DataFrame, shore_x_col: str, shore_y_col: str, correction_factor: float) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple]:
    """ 
    returns a cleaned well df and a cleaned shore df ready to be analyzed 
    """
    well_df_cleaned, start_end_date = clean_well_df(well_df, well_x_col, well_y_col, window_length, min_dist)
    shore_df_cleaned = clean_shore_df(shore_df, shore_x_col, shore_y_col, correction_factor, *start_end_date)
    return shore_df_cleaned, well_df_cleaned, start_end_date

def clean_well_df(df: pd.DataFrame, x_col: str, y_col: str, window_length: int, min_dist: int) -> Tuple[pd.DataFrame, Tuple]: 
    """
    cleans the water level column and creates new hours column starting from the first entry returning the cleaned 
    df and the start/end date returning the cleaned df and optimal start/end date
    """
    # steps: 
    # 1) drop rows where Value col is na 
    # 2) convert Timestamp col to datetime type 
    # 3) find optimal start/end dates 
    # 4) filter df by optimal start/end dates
    # 5) create new hours column
    df = df.dropna(subset=[y_col])            
    df = df.reset_index(drop=True)                                                          # reset index
    df[x_col] = pd.to_datetime(df[x_col])                                                   # convert date to datetime 
    start_date, end_date = get_optimal_start_end(df, x_col, y_col, window_length, min_dist)           # get optimal start/end dates
    df_subset = filter_df_by_date(df, start_date, end_date, x_col)                          # filter for date range
    df_subset = df_subset.reset_index(drop=True)                                            # reset index
    df_subset = create_new_hours_col(df_subset, x_col, start_date)                          # convert to decimial hours from start date
    return df_subset, (start_date, end_date)

def get_optimal_start_end(df: pd.DataFrame, x_col: str, y_col: str, window_length: int, min_dist: int) -> Tuple[str, str]: 
    """ 
    returns the most optimal (most minimal differences in peaks) start/end dates based on inputted window_length which 
    denotes the time frame to look at; where peaks are identified by looking for max values within a min time period of 12 hours
    which denotes the time period for half of sine wave's cycle
    """
    # steps: 
    # 1) extract x and y values from col 
    # 2) get x_values of most optimal time range
    # 3) get most optimal start/end x_values 
    x_values = df[x_col].values          # extract x_values
    y_values = df[y_col].values          # extract y_values
    optimal_x_values, sd = get_optimal_x_values(x_values, y_values, window_length, min_dist)
    print(f"lowest SD = {sd}")
    print(f"optiaml start date: {optimal_x_values[0]}")
    return optimal_x_values[0], optimal_x_values[-1]

def get_optimal_x_values(x_values, y_values, window_length: int, min_dist: int) -> Tuple[np.ndarray, np.float64]: 
    """ 
    determines the best range with size window_length that has lowest sd at peak values and 
    returns an array of x-values from that most optimal time range where min_dist is the min 
    index distance away a max value will be identified 
    """
    best_window = np.array(['', ''])
    lowest_sd = float('inf')

    for i in range(len(x_values) - window_length + 1):            # minus window size to not go over bound 
        x_window_values = x_values[i:i + window_length]           # extract x-values within range
        y_window_values = y_values[i:i + window_length]           # extract y-values within range

        ## find the peaks 
        peak_inds, _ = find_peaks(y_window_values, distance=min_dist)  # get y_window_values indices of all peaks with min sep of 12 indices
        peak_values = y_window_values[peak_inds]                 # extract the actual value using peak_inds from y_window_values
        sd = np.std(peak_values)                                 # calc sd

        if sd < lowest_sd:
            lowest_sd = sd
            best_window = x_window_values                

    return best_window, lowest_sd

def filter_df_by_date(df: pd.DataFrame, start_date, end_date, date_col: str) -> pd.DataFrame: 
    """ 
    filters UTC date col (str type) by start and end dates 
    """
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
    return df

def create_new_hours_col(df: pd.DataFrame, date_col: str, start_datetime) -> pd.DataFrame:
    """ 
    returns a series of the datetime col values and converts it to decimal hours from start datetime 
    """
    if not isinstance(start_datetime, np.datetime64): 
        start_datetime = datetime.datetime(*start_datetime)

    df["Hours"] = df[date_col].apply(lambda x: datetime_to_hours(x, start_datetime))
    return df                                      
                                      
def datetime_to_hours(input_datetime, start_datetime) -> float: 
    """ 
    converts datetime to hours from start datetime
    """
    diff = input_datetime - start_datetime
    hours = diff.total_seconds() / 3600
    return hours

## ------------------ Functions to clean shore df --------------------------
def clean_shore_df(shore_df: pd.DataFrame, shore_x_col: str, shore_y_col: str, correction_factor: float, start_datetime: np.datetime64, end_datetime: np.datetime64) -> pd.DataFrame: 
    """ 
    cleans the water level column and creates new hours column starting from the first entry 
    """
    # steps: 
    # 1) remove ' UTC' from col values
    # 2) drop rows where Value col is na 
    # 3) convert date col to datetime type 
    # 4) filter df by start/end dates
    # 5) create new hours column
    # 6) apply correction factor to y_col 3.2
    # shore_df[shore_x_col] = shore_df[shore_x_col].str.replace(' UTC', '') 
    shore_df = shore_df.dropna(subset=[shore_y_col])
    shore_df[shore_x_col] = pd.to_datetime(shore_df[shore_x_col])                                             
    shore_df = filter_df_by_date(shore_df, start_datetime, end_datetime, shore_x_col)                           
    shore_df = shore_df.reset_index(drop=True)
    shore_df = create_new_hours_col(shore_df, shore_x_col, start_datetime)           
    shore_df[shore_y_col] = shore_df[shore_y_col] - correction_factor
    return shore_df   

