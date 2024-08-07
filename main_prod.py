#%%
import pandas as pd
pd.options.mode.chained_assignment = None  
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import datetime
import math
from typing import List, Tuple
from scipy.interpolate import PchipInterpolator

#%%
def main(stn_name: str, shore_stn: str, shore_df: pd.DataFrame, shore_x_col: str, shore_y_col: str, correction_factor: float, well_df: pd.DataFrame, well_x_col: str, well_y_col: str, window_length: int, initial_guess: List[float] | None, fit_fn, x: float, t: float, min_dist: int, z_thresh: float): 
    """ 
    calculates the simplified amplitude and time lag diffusivity (T/S) and plots the best fit sine curve and the final shore/river curve with the fitted gw sine curve given the following params: 
        1. stn name: name of gw station 
        2. shore_df: shore waterlevel df 
        3. shore_x_col: shore/river col x name 
        4. shore_y_col: shore/river col y name  
        5. correction factor: # to subtract from shore/river water level 
        6. well_df: well water level df 
        7. well_x_col: well col x name
        8. well_y_col: well col y name
        9. window_length: t in hours of plot's duration (can increase/decrease to find better optimal time period)
        10. initial_guess: initial guess of params for gw sine curve 
        11. fit_fn: sine function that will be used for curve fittin g
        12. x: closest straight line distance from well to shore/river [m]
        13. t: tidal period [s]
        14. min_dist: min # of indices for amplitude (increase if well curve's optimal time range consists of many small peaks/troughs within tidal period)
        15. z_thresh: is the max z-score that a t-lag can have to be included in the t-lag T/S calculations 
    """
    # steps: 
    # 1) import csv files 
    # 2) calculate T/S value 
    # 3) plot shore + fitted well 
    # shore_df, well_df = import_data(shore_fps, well_fp)
    shore_df_cleaned, well_df_cleaned, start_end_date = clean_input_data(well_df, well_x_col, well_y_col, window_length, min_dist, shore_df, shore_x_col, shore_y_col, correction_factor)
    initial_guess = compute_initial_guess(well_df_cleaned, well_y_col, initial_guess)
    params, plot, summary_df, sa_df, tl_df = analyze(stn_name, shore_df_cleaned, shore_y_col, well_df_cleaned, well_y_col, initial_guess, fit_fn, x, t, min_dist, z_thresh)
    final_graph = plot_final_graph(stn_name, params, fit_fn, shore_df_cleaned, shore_y_col, shore_stn)
    # result = f"----- OW{stn_name} Results ----- \nAmplitude Analysis\nhead effs: {efficiencies}\navg T/S = {avg_diff1}, sd = {sd1}\n\nTime Lag Analysis\nt lags: {t_lags}\navg T/S = {avg_diff2}, sd = {sd2}\n---------------------------\n"
    return params, plot, summary_df, sa_df, tl_df, start_end_date, final_graph, plot

# def import_data(shore_fps: Tuple[List[str], bool], well_fp: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """ 
#     imports the shore and well water level csv files and filter well data to 2 years where shore_fps is a tuple 
#     containing list of fps and bool where True when importing river data from aquarius and False when importing 
#     tide chart data
#     """
#     shore_df = import_shore_data(*shore_fps) 
#     well_df = import_aquarius_data(well_fp)
#     # well_df = well_df[(well_df[well_x_col] >= "2023-01-01")]

#     return shore_df, well_df

def import_shore_data(shore_fps: List) -> pd.DataFrame: 
    """ 
    imports yearly shore data (max/min) data and concatenates them into single df 
    """
    dfs = []
    for fp in shore_fps: 
        df = pd.read_csv(fp)
        dfs.append(df)

    reversed_dfs = dfs[::-1]
    shore_df = pd.concat(reversed_dfs, ignore_index=True)
    return shore_df

def import_aquarius_data(fp) -> pd.DataFrame: 
    """ 
    imports water level data from aquarius  
    """
    df = pd.read_csv(fp, skiprows=2)
    return df

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

## --------------------------- Functions for guessing initial params ---------------------------------------- 
def compute_initial_guess(well_df: pd.DataFrame, well_y_col: str, initial_guess: List[float] | None) -> List[float]: 
    """ 
    returns an initial guess of params for composite sine function, if initial_guess is not None then will return initial_guess 
    """
    # steps: 
    # 1) compute fourier transofrm 
    # 2) estimate dominant frequencies 
    # 3) guess amplitudes 
    # 4) guess y-offset
    # 5) set phase shift to 0
    if initial_guess is None: 
        x, y = well_df['Hours'].values, well_df[well_y_col].values
        xf, yf, N = compute_fourier_transform(x, y)
        w1, w2 = get_dom_frequency(xf, yf, N)
        a1, a2 = guess_amplitudes(y)
        y_offset = np.mean(y)  
        phi1, phi2 = 0, 0
        initial_guess = [a1, w1, phi1, a2, w2, phi2, y_offset]
    return initial_guess
   
def compute_fourier_transform(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]: 
    """ 
    returns the x/y transformed values using fourier transform
    """
    N = len(x)
    T = x[1] - x[0]  # Sample spacing
    yf = fft(y - np.mean(y))  # Remove mean to focus on frequency
    xf = fftfreq(N, T)[:N//2]
    return xf, yf, N

def get_dom_frequency(xf: np.ndarray, yf: np.ndarray, N: int) -> Tuple[float, float]: 
    """ 
    returns the estimated params for a composite sine function where if is_composite True then will 
    return a tuple with 2 floats or a single float if not composite 
    """
    dominant_frequencies = xf[np.argsort(2.0/N * np.abs(yf[:N//2]))[-2:]]
    w1_est = 2 * np.pi * dominant_frequencies[0]
    w2_est = 2 * np.pi * dominant_frequencies[1]
    return w1_est, w2_est

def guess_amplitudes(y: np.ndarray) -> Tuple[float, float]: 
    """ 
    returns guesses for A1 and A2 
    """
    A1_est = (np.max(y) - np.min(y)) / 2
    A2_est = (np.max(y) - np.min(y)) / 4
    return A1_est, A2_est


def analyze(stn_name: str, shore_df_cleaned: pd.DataFrame, shore_y_col: str, well_df_cleaned: pd.DataFrame, well_y_col: str, initial_guess: List[float], fit_fn, x: float, t: float, min_dist: int, z_thresh: float): 
    """     
    computes the diffusivity value given the shore and well water level data returning the diffusivity, curve parameters, and cleaned shore df 
    """
    # 1) clean shore data 
    # 2) clean well data 
    # 2) obtain sine composite params + plot best fit curve 
    # 3) get amplitudes of well + shore 
    # 4) get t at well/shore peaks 
    # 5) calculate T/S value using simp amp eqn 
    # 6) create summary results df
    # 7) create sa results df
    # 8) create tl results df 
    # well_df_cleaned, start_end_date = clean_well_df(well_df, well_x_col, well_y_col, window_length, min_dist)
    # shore_df_cleaned = clean_shore_df(shore_df, shore_x_col, shore_y_col, correction_factor, *start_end_date)
    params, plot = get_best_fit_params(stn_name, well_df_cleaned, 'Hours', well_y_col, initial_guess, fit_fn)
    efficiencies, well_amp_x, shore_amp_x, well_amp, shore_amp = get_all_head_efficiencies(fit_fn, params, shore_df_cleaned, shore_y_col, min_dist)
    t_lags, lot0_peaks, lotx_c_peaks, t_lags_peaks, lot0_troughs, lotx_c_troughs, t_lags_troughs = get_all_time_lags(fit_fn, params, shore_df_cleaned, shore_y_col, min_dist)
    avg_diff1, sd1, diff_values1, diff_removed_outliers1, diff_indices1 = calc_avg_diff_simp_amp(x, efficiencies, t, z_thresh)
    avg_diff2, sd2, diff_values2, diff_removed_outliers2, diff_indices2 = calc_avg_diff_t_lag(x, t_lags, t, z_thresh)
    summary_df = create_summary_df(avg_diff1, sd1, avg_diff2, sd2)
    sa_df = create_sa_results_df(well_amp_x, well_amp, shore_amp_x, shore_amp, efficiencies, diff_values1, diff_indices1)
    tl_df = create_tl_results_df(lotx_c_peaks, lot0_peaks, t_lags_peaks, lotx_c_troughs, lot0_troughs, t_lags_troughs, diff_values2, diff_indices2)
    return params, plot, summary_df, sa_df, tl_df

## ----------------- Functions fitting sine curve --------------------
def get_best_fit_params(stn_name: str, df: pd.DataFrame, x_col: str, y_col: str, initial_guess: List[float], fit_fn) -> np.ndarray: 
    """ 
    returns the parameters for best fitted curve based on initial guess and fit function 
    graphs the raw data + fitted curve 
    """
    x_values = df[x_col]
    y_values = df[y_col]
    params, plot = config_sine_fit(stn_name, x_values, y_values, initial_guess, fit_fn) 
    return params, plot 

def config_sine_fit(stn_name: str, x_values: pd.Series, y_values: pd.Series, initial_guess: List[float], fit_fn): 
    """ 
    computes best fit based on fit_fn and then plots a first plot of original y-values, second plot 
    of adjusted params based on best fit and third plot is plotted using initial_guess params
    """
    params, _ = curve_fit(fit_fn, x_values, y_values, p0=initial_guess)

    # Generate predictions
    x_generated = np.linspace(min(x_values), max(x_values)+20, 5000)
    y_fit = fit_fn(x_generated, *params)

    fig, ax = plt.subplots(figsize=(20,6))
    ax.plot(x_values, y_values, marker='o', linestyle='', color='green', label='raw')                                      # plot raw  data 
    ax.plot(x_generated, y_fit, marker='o', linestyle='', color='red', label='fitted')                                     # plot fitted curve
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel('Hours from start time')
    ax.set_ylabel('MASL Depth (m)')
    ax.set_title(f'{stn_name} Well')
    ax.legend()
    return params, fig

## --------------------------- Functions for Amplitude Analysis -----------------------------------
def get_all_head_efficiencies(fit_fn, params: np.ndarray, shore_df: pd.DataFrame, shore_y_col: str, min_dist: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ 
    returns a list of head efficiencies [Hx_1/h0_1, ...] along with the x-values of the peaks where amplitude was measured 
    Note: Amplitude is measured from trough to peak
    """
    shore_x_values = shore_df['Hours']
    well_amplitudes, well_x_peaks_c = get_well_amplitudes(fit_fn, params, min_dist, shore_x_values)
    shore_amplitudes, shore_x_peaks_c = get_shore_amplitudes(shore_df, shore_y_col, min_dist)
    efficiencies, well_x_c, shore_x_c, well_amplitudes, shore_amplitudes  = calc_head_efficiencies(well_amplitudes, well_x_peaks_c, shore_amplitudes, shore_x_peaks_c)

    print(f'efficiences: {efficiencies}')
    # print(f'efficiences filtered: {efficiencies_filtered}')
    return efficiencies, well_x_c, shore_x_c, well_amplitudes, shore_amplitudes 

# get well amplitudes: 
def get_well_amplitudes(fit_fn, params: np.ndarray, min_dist: int, shore_x_values: pd.Series) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    returns the well amplitudes at each matching peak and trough where peak has to be > than (in terms of x-value) trough 
    and also returns the x-value of the calculated amplitude's y_peak. Amplitude is calculated as vertical distance from 
    trough to nearest peak divided by 2
    """
    # steps: 
    # 1) generate cont x-values
    # 2) compute y-values using cont x-values 
    # 3) get indices of peaks + troughs 
    # 4) find y-values at peaks + troughs
    # 5) get matching peak/trough arrays 
    # 6) extract y_values of matching peak/trough arrays 
    # 7) calculate amplitude 
    x_values_cont = np.linspace(min(shore_x_values), max(shore_x_values)+20, 5000)                       # create nearly cont x-values 
    y_values = fit_fn(x_values_cont, *params)                       # compute y-values 
    peaks_ind, _ = find_peaks(y_values, distance=min_dist)          # find inds of peaks          
    troughs_ind, _ = find_peaks(-y_values, distance=min_dist)       # find inds of troughs 
    x_peaks = x_values_cont[peaks_ind]                              # get x-values of peaks 
    x_troughs = x_values_cont[troughs_ind]                          # get x-values of troughs 
    x_troughs_c, x_peaks_c = match_arrays(x_troughs, x_peaks)       # clean arrays to ensure peaks/troughs match 
    y_peaks_c = get_y_from_x(x_values_cont, y_values, x_peaks_c)    # extract y-values for each item in x_peaks_c
    y_troughs_c = get_y_from_x(x_values_cont, y_values, x_troughs_c)
    well_amplitudes = (y_peaks_c - y_troughs_c) / 2            
    return well_amplitudes, x_peaks_c

def match_arrays(prim_array: np.ndarray, sec_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    for each number in sec array, will match with a corresponding value from prim array that is < tx 
    and is the closest to tx compared to other values in prim_array, a single prim_array value can only be matched once 
    if there is no match for a given value in sec_array, then value form sec_array will be removed 
    """
    # steps: iterate thru ecah item in sec_array
    # 1) filter current array for items in prim_array < current s value 
    # 2) get ind of value closest to current s value 
    # 3) extract actual value using ind 
    # 4) append current tx to lotx
    # 5) append closest pc to prim_cleaned 
    # 6) update prim_array 
    prim_cleaned = []
    sec_cleaned = []
    for s in sec_array:
        less_than_s = prim_array[prim_array < s]     # filter array for prim_array values less than current s value
        if len(less_than_s) > 0:
            pc_ind = np.argmin(s - less_than_s)      # get ind of value that is clsosest to current s
            pc = less_than_s[pc_ind]                 # extract the value 
            sec_cleaned.append(s)                    # append to current tx to lotx 
            prim_cleaned.append(pc)                  # append closest_value to prim_cleaned
            prim_array = prim_array[pc_ind+1:]       # update prim_cleaned
    return np.array(prim_cleaned), np.array(sec_cleaned)

def get_y_from_x(x_array: np.ndarray, y_array: np.ndarray, x_values_array: np.ndarray) -> np.ndarray: 
    """ 
    returns an array of y-values given source x and y arrays and an array x_values consisting of x-values that you want to look up y-values for 
    """
    # steps: 
    # 1) obtain an array of the indices for each x-value contained in x_array 
    # 2) use array of indices to get the y-values for the desired x-values 
    src_indices = np.where(np.isin(x_array, x_values_array))[0]
    y_values_array = y_array[src_indices]
    return y_values_array 

## Functions to get shore amplitudes 
def get_shore_amplitudes(df: pd.DataFrame, y_col: str, min_dist: int) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    returns the shore amplitudes at each matching peak and trough where peak has to be > than (in terms of x-value) trough 
    and also returns the x-value of the calculated amplitude's y_peak
    """
    # steps: 
    # 1) get interpolated shore x/y values
    # 2) get indices of peaks + troughs 
    # 3) get x-values at peaks + troughs
    # 4) clean x_peaks/x_troughs to ensure they have matching pairs 
    # 5) get y-values for x_peaks/x_troughs 
    # 6) calc amplitudes 
    x_values, y_values = interpolate_shore_data(df, y_col)          # get x/y values 
    peaks_ind, _ = find_peaks(y_values, distance=min_dist)          # get inds of peaks    
    troughs_ind, _ = find_peaks(-y_values, distance=min_dist)       # get inds of troughs
    x_peaks = x_values[peaks_ind]                                   # get x-values of peaks
    x_troughs = x_values[troughs_ind]                               # get x-values of troughs
    x_troughs_c, x_peaks_c = match_arrays(x_troughs, x_peaks)       # clean arrays to ensure peaks/troughs match 
    y_peaks_c = get_y_from_x(x_values, y_values, x_peaks_c)         # extract y-values for each item in x_peaks_c
    y_troughs_c = get_y_from_x(x_values, y_values, x_troughs_c)
    shore_amplitudes = (y_peaks_c - y_troughs_c) / 2                # calc amplitude for each peak/trough pair
    return shore_amplitudes, x_peaks_c

def calc_head_efficiencies(well_amplitudes: np.ndarray, well_x_peaks_c: np.ndarray, shore_amplitudes: np.ndarray, shore_x_peaks_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """ 
    calculates head efficiencies Hx/h0 and the well/shore's x values for each well/shore amplitude matched pair and returns an array of efficiencies 
    """
    # steps: 
    # 1) return matched well/shore amplitudes 
    # 2) get the actual well/shore amplitudes given well_x and shore_x values 
    # 3) calculate head efficiency 
    shore_x_c, well_x_c = match_arrays(shore_x_peaks_c, well_x_peaks_c)
    well_amplitudes_c = get_y_from_x(well_x_peaks_c, well_amplitudes, well_x_c)
    shore_amplitudes_c = get_y_from_x(shore_x_peaks_c, shore_amplitudes, shore_x_c)
    efficiencies = well_amplitudes_c / shore_amplitudes_c

    print("------------ Amplitude Info --------------")
    print(f"well x-values: {well_x_peaks_c}, shore x-values: {shore_x_peaks_c}")
    print(f"well amplitudes: {well_amplitudes}, shore amplitudes: {shore_amplitudes}")
    print(f"well x-values matched: {well_x_c}, shore x-values matched: {shore_x_c}")
    print(f"well amplitudes matched: {well_amplitudes_c}, shore amplitudes matched: {shore_amplitudes_c}")
    return efficiencies, well_x_c, shore_x_c, well_amplitudes_c, shore_amplitudes_c

def interpolate_shore_data(shore_df_cleaned: pd.DataFrame, shore_y_col: str) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    interpolates shore data and returns the x/y values 
    """
    x = shore_df_cleaned['Hours']
    y = shore_df_cleaned[shore_y_col]
    pchip = PchipInterpolator(x, y)                 # interpolate 
    x_fit = np.linspace(min(x), max(x), 5000)        # generate x-values
    y_fit = pchip(x_fit)                            # compute y-values
    return x_fit, y_fit

def calc_avg_diff_simp_amp(x, eff: np.ndarray, t, z_thresh: float) -> Tuple[np.float64, np.float64, np.ndarray, np.ndarray, np.ndarray]: 
    """ 
    calculates T/S given each head efficiency and computes the avg to get avg T/S 
    """
    diff_values = []
    for e in eff: 
        diff = simplified_amp_eqn(x, e, t) 
        diff_values.append(diff)

    diff_values_array = np.array(diff_values)
    diff_removed_outliers, indices = keep_removing_outliers(diff_values_array, z_thresh)     # remove outliers 
    avg = np.mean(diff_removed_outliers)
    sd = np.std(diff_removed_outliers)
    print(f"\nT/S values unfiltered (simp): {diff_values}")
    print(f"T/S values (simp): {diff_removed_outliers}")
    return avg, sd, diff_values_array, diff_removed_outliers, indices

def simplified_amp_eqn(x, avg_eff, t) -> float: 
    """ 
    simplified amplitude analysis equation to compute for T/S (aquifer diffusivity) where: 
        x: straight line distance from monitoring well to hydrometric station (or shore) [m]
        avg_eff: mean(Hx_1/h0_1, Hx_2/h02, ...)
            hx: amplitude of well [m]
            h0: amplitdue of hydro station (or shore) [m]
        t: time period of tidal period [sec]
    """
    return (math.pi/t) * (-x / math.log(avg_eff))**2

## --------------------------- Functions to get time average time lag at peaks/troughs -----------------------------------
def get_all_time_lags(fit_fn, params: np.ndarray, shore_df: pd.DataFrame, shore_y_col: str, min_dist: int): 
    """ 
    returns an array of all time lags for peaks and troughs 
    """
    # steps: 
    # 1) get x-values at peaks/troughs in well curve 
    # 2) match each x-value in well peak/trough to corresponding peak/troughs in shore curve 
    # 4) calculate time lag for each pair 
    # 5) append trough/peaks t-lags together
    # 6) remove outliers 
    shore_x_values = shore_df['Hours']
    lotx_peaks, lotx_troughs = get_well_peak_trough_times(fit_fn, params, min_dist, shore_x_values)
    lot0_peaks, lotx_c_peaks, lot0_troughs, lotx_c_troughs = get_shore_peak_trough_times(shore_df, shore_y_col, min_dist, lotx_peaks, lotx_troughs)
    t_lags_peaks = lotx_c_peaks - lot0_peaks 
    t_lags_troughs = lotx_c_troughs - lot0_troughs 
    t_lags = np.append(t_lags_peaks, t_lags_troughs)
    # t_lags_filtered = remove_outliers(t_lags, z_thresh)

    print(f'peak tlags: {t_lags_peaks}, trough tlags: {t_lags_troughs}\n')
    # print(f'filtered t-lags: {t_lags_filtered}\n')
    return t_lags, lot0_peaks, lotx_c_peaks, t_lags_peaks, lot0_troughs, lotx_c_troughs, t_lags_troughs

## get well peak/trough times: 
def get_well_peak_trough_times(fit_fn, params: np.ndarray, min_dist: int, x: pd.Series) -> Tuple: 
    """ 
    returns array of x values at peak y-values and an array of x-values at trough y-values 
    """
    # steps: 
    # 1) generate more x-values to increase curve's resolution 
    # 2) compute y-values using generated x-values 
    # 3) get indices of peaks + troughs 
    # 4) find x-values at peaks + troughs
    x_values_cont = np.linspace(min(x), max(x)+20, 5000)                           # create nearly cont x-values 
    y_values = fit_fn(x_values_cont, *params)                          # compute y-values 

    peaks_ind, _ = find_peaks(y_values, distance=min_dist)             # get indices of peaks 
    x_peaks = x_values_cont[peaks_ind]                                 # find peak values

    troughs_ind, _ = find_peaks(-y_values, distance=min_dist)          # get indices of troughs 
    x_troughs = x_values_cont[troughs_ind]                             # find trough values
    return x_peaks, x_troughs

## get shore peak/trough times
def get_shore_peak_trough_times(df: pd.DataFrame, y_col: str, min_dist: int, lotx_peaks: np.ndarray, lotx_troughs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """ 
    gets the corresponding shore/well matched peak/trough times (x-values)
    """
    # steps: 
    # 1) get interpolated shore x/y values
    # 2) get indices of shore peaks 
    # 3) find x_values at shore peaks
    # 4) match tx with corresponding t0 if exists 
    x_values, y_values = interpolate_shore_data(df, y_col)

    peaks_ind, _ = find_peaks(y_values, distance=min_dist)      # find peak inds    
    x_peaks = x_values[peaks_ind]                               # find x_values at peaks
    lot0_peaks, lotx_peaks= match_arrays(x_peaks, lotx_peaks)

    troughs_ind, _ = find_peaks(-y_values, distance=min_dist)   # find trough inds    
    x_troughs = x_values[troughs_ind]                           # find x_values at peaks
    lot0_troughs, lotx_troughs = match_arrays(x_troughs, lotx_troughs)

    print('\n---------- Time Lag Info ---------------')
    print(f'Well peaks: {lotx_peaks}, Well troughs: {lotx_troughs}\nShore peaks: {x_peaks}\n Shore troughs: {x_troughs}')
    print(f'\nShore/Well Peaks: {lot0_peaks}; {lotx_peaks}\nShore/Well Troughs: {lot0_troughs}; {lotx_troughs}')
    return lot0_peaks, lotx_peaks, lot0_troughs, lotx_troughs

def keep_removing_outliers(nums: np.ndarray, z_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    continues to apply the remove_outlier function to input array till updated array == input array and returns also the original indices of items that 
    were filtered
    """
    original_inds = np.arange(len(nums))                                          # construct list of indices to track unfiltered diff_values
    while True: 
        updated_nums, indices = remove_outliers(nums, z_thresh)      # remove outliers 
        if np.array_equal(updated_nums, nums) or len(updated_nums) <= 1: 
            break
        print(f'input list: {nums}')
        print(f'updated list: {updated_nums}')
        nums = updated_nums                                 # update nums with updated_nums
        original_inds = original_inds[indices]              # filter array using indices (remove based on index)
    return nums, original_inds

def remove_outliers(nums: np.ndarray, z_thresh: float) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    given an array of numbers will remove numbers that have a z-score above the z-thresh and also the indices of the original nums
    """
    mean = np.mean(nums)
    std = np.std(nums)
    z_scores = (nums - mean) / std
    filtered_data = nums[np.abs(z_scores) < z_thresh]
    filtered_indices = np.where(np.abs(z_scores) < z_thresh)[0]     # get indices of ones that are under z_thresh

    print(f"z-scores: {z_scores}")
    return filtered_data, filtered_indices

def calc_avg_diff_t_lag(x, t_lags: np.ndarray, t, z_thresh) -> Tuple[np.float64, np.float64, np.ndarray, np.ndarray, np.ndarray]:
    """ 
    calculates each T/S given each time lag and computes the avg to get avg T/S 
    """
    diff_values = []
    for tl in t_lags: 
        diff = time_lag_eqn(x, tl, t) 
        diff_values.append(diff)
    diff_values_array = np.array(diff_values)
    diff_removed_outliers, indices = keep_removing_outliers(diff_values_array, z_thresh)    
    avg = np.mean(diff_removed_outliers)
    sd = np.std(diff_removed_outliers)
    print(f"\nT/S values unfiltered (t-lag): {diff_values}")
    print(f"T/S values (t-lag): {diff_removed_outliers}")
    return avg, sd, diff_values_array, diff_removed_outliers, indices

def time_lag_eqn(x, t_lag, t) -> float: 
    """ 
    time lag analysis eqn where: 
        x: straight line distance from monitoring well to hydrometric station (or shore) [m]
        tx: time of well peak 
        t0: time of shore/river peak 
        t: time period of tidal period [sec]
    """
    t_lag = t_lag * 3600
    return (x**2 * t) / (4 * np.pi * (t_lag**2))

## --------------------- functions to create summary dfs -------------------------------------------------
def create_summary_df(avg_diff1: np.float64, sd1: np.float64, avg_diff2: np.float64, sd2: np.float64) -> pd.DataFrame: 
    """ 
    creates summary df for averaged diffusivity values for SA and TL analysis    
    """
    data = {"Simplified Amplitude": [avg_diff1, sd1],"Time Lag": [avg_diff2, sd2]}
    df = pd.DataFrame(data, index=['Avg T/S (m²/s)', 'SD'])
    return df



def create_sa_results_df(well_amp_x: np.ndarray, well_amp: np.ndarray, shore_amp_x: np.ndarray, shore_amp: np.ndarray, efficiencies: np.ndarray, diff_values1: np.ndarray, diff_indices1: np.ndarray) -> pd.DataFrame: 
    """ 
    creates summary df for SA intermediate results 
    """
    diff_filtered_add_null = replace_missing_inds_with_null(diff_values1, diff_indices1)
    data = [well_amp_x, well_amp, shore_amp_x, shore_amp, efficiencies, diff_values1, diff_filtered_add_null]
    index = ['Well Hours (h)', 'Well Amplitdue (m)', 'Shore Hours (h)', 'Shore Amplitude (m)', 'Hx/H0', 'T/S (m²/s)', 'T/S filtered (m²/s)']
    for i in data: 
        print(i)
    df = pd.DataFrame(data, index=index)
    return df

def replace_missing_inds_with_null(input_array: np.ndarray, indices: np.ndarray) -> np.ndarray: 
    """ 
    given array of values and array of indices will replace any value where index is not included in indices with null value
    """
    full_indices = np.arange(len(input_array))              # create array of indices for input_array
    mask = np.isin(full_indices, indices)                   # create mask where assign True if index is in indices and False if not in indices
    update_array = np.where(mask, input_array, np.nan)      # replace with the correct value from input_array if True and if False will replace with NaN
    return update_array

def create_tl_results_df(lotx_c_peaks: np.ndarray, lot0_peaks: np.ndarray, t_lags_peaks: np.ndarray, lotx_c_troughs: np.ndarray, lot0_troughs: np.ndarray, t_lags_troughs: np.ndarray, diff_values2: np.ndarray, diff_indices2: np.ndarray) -> pd.DataFrame: 
    """
    creates summary df for TL intermediate results 
    """
    # steps; 
    # 1) get length of peaks 
    # 2) create filtered diff array 
    # 3) extract peaks diff and troughs diff 
    # 4) ensure all arrays are same length 
    peak_len = len(lotx_c_peaks)
    diff_filtered_add_null = replace_missing_inds_with_null(diff_values2, diff_indices2)

    # sep peaks/troughs
    diff_peaks = diff_values2[:peak_len]
    diff_troughs = diff_values2[peak_len:]
    diff_filtered_peaks= diff_filtered_add_null[0:peak_len]
    diff_filtered_troughs = diff_filtered_add_null[peak_len:]

    max_length = max(len(diff_peaks), len(diff_troughs))        # get max length 

    # ensure all arrays are same length 
    lotx_c_peaks_p = pad_array(lotx_c_peaks, max_length)
    lot0_peaks_p = pad_array(lot0_peaks, max_length)
    t_lags_peaks_p = pad_array(t_lags_peaks, max_length)

    diff_peaks_p = pad_array(diff_peaks, max_length)
    diff_filtered_peaks_p = pad_array(diff_filtered_peaks, max_length)


    lotx_c_troughs_p = pad_array(lotx_c_troughs, max_length)
    lot0_troughs_p = pad_array(lot0_troughs, max_length)
    t_lags_troughs_p = pad_array(t_lags_troughs, max_length)

    diff_troughs_p = pad_array(diff_troughs, max_length)
    diff_filtered_troughs_p = pad_array(diff_filtered_troughs, max_length)

    data = [lotx_c_peaks_p, lot0_peaks_p, t_lags_peaks_p, diff_peaks_p, diff_filtered_peaks_p, lotx_c_troughs_p, lot0_troughs_p, t_lags_troughs_p, diff_troughs_p, diff_filtered_troughs_p]
    index = ['Well Peak Hours (h)', 'Shore Peak Hours (h)', 'Time Lag Peaks (h)', 'T/S Peaks (m²/s)', 'T/S Peaks Filtered (m²/s)', 'Well Trough Hours (h)', 'Shore Trough Hours (h)', 'Time Lag Trough (h)', 'T/S Trough (m²/s)', 'T/S Trough Filtered (m²/s)']
    df = pd.DataFrame(data, index=index)
    return df

def pad_array(arr, length):
    return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)


def plot_final_graph(stn_name: str, params: np.ndarray, fit_fn, shore_df_cleaned: pd.DataFrame, shore_y_col: str, shore_stn: str):
    """ 
    plots the final graph using the parameters from curve fitting (high resolution) and the shore water level data 
    """
    # steps: 
    # 1) generate the high resolution well curve 
    # 2) extract shore x/y values
    # 3) extract fraser/tide chart stn name
    x = shore_df_cleaned['Hours']
    x_values_cont = np.linspace(min(x), max(x)+20, 5000)       # create nearly cont x-values 
    y_values = fit_fn(x_values_cont, *params)       # compute y-values 

    shore_x_values, shore_y_values = interpolate_shore_data(shore_df_cleaned, shore_y_col)
    # shore_label = get_shore_name(shore_fps)    

    plt.figure(figsize=(20, 6))
    plt.plot(x_values_cont, y_values, marker='o', linestyle='', color='green', label=f'{stn_name}')     
    plt.plot(shore_x_values, shore_y_values, marker='o', linestyle='', color='blue', label=f'{shore_stn}')  

    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    plt.xlabel('Hours from start time')
    plt.ylabel('MASL Depth (m)')
    plt.title(f'Water Level Time Series')
    plt.legend()
    plt.show()
    return plt

def get_shore_name(shore_fps: Tuple[List[str], bool]) -> str: 
    """ 
    returns the stn name of shore given filepath 
    """
    fp = shore_fps[0][0]
    is_river = shore_fps[1]
    if is_river: 
        fname = os.path.basename(fp)
        stn_name = os.path.splitext(fname)[0]
        return stn_name
    else: 
        fname = os.path.basename(fp)
        name = os.path.splitext(fname)[0]
        parts = name.split('_')
        stn_name = parts[2]
        return stn_name




#%% ## PARAMETERS: 
data_folder = r"\\spatialfiles.bcgov\srm\sry\Workarea\mjchan\transmissivity\data"
# shore_fp = os.path.join(data_folder, "predictions_07895_Mitlenatch_2023-08-14.csv")
# shore_fp1 = os.path.join(data_folder, "Annual_Predictions_Mitlenatch_2023.csv")
# shore_fp2 = os.path.join(data_folder, "Annual_Predictions_Mitlenatch_2024.csv")
# well_fp = os.path.join(data_folder, "MASL_OW511.csv")
well_x_col = 'Timestamp (UTC)'
well_y_col = 'Value (m)'
# shore_x_col = 'Date'
# shore_y_col = 'predictions (m)'
shore_x_col = 'Date and time (UTC)'
shore_y_col = 'Metres'
# window_length = 100
# correction_factor = 3.2
# initial_guess = [0.030, 23, 4.5, 0.9] 
# x = 154.8                               # dist to shore 
# t = 12.67 * 3600                        # time per tidal cycle


def fit_fn(t, A, B, C, D) -> float: 
    """ 
    sine composite function used by Elyse where: 
        A is amplitude 
        B is time period 
        C is phase 
        D is y-int
    """
    A2 = A / 2
    B2 = 24
    C2 = C - 5.5
    return A2*np.sin((2*np.pi / B2) * (t + C2)) + A*np.sin((2*np.pi / B) * (t + C)) + D

def composite_sine(x, A1, w1, phi1, A2, w2, phi2, offset):
    return A1 * np.sin(w1 * x + phi1) + A2 * np.sin(w2 * x + phi2) + offset


