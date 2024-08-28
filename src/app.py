#%%
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from main_prod import import_shore_data, main, import_aquarius_data, fit_fn, composite_sine
from typing import List, Tuple, Optional


#%%
def get_shore_data(shore_csvs):     
    """ 
    imports uploaded shore data and displays first couple of rows 
    """
    # upload shore csvs
    shore_df = import_shore_data(shore_csvs)
    st.write("Data Preview:")
    st.write(shore_df.head())
    # select correct x/y cols
    shore_x_col = st.selectbox('Select Shore Time Column', shore_df.columns)
    shore_y_col = st.selectbox('Select Shore Water Level Column', shore_df.columns)
    return shore_df, shore_x_col, shore_y_col
    
def get_well_data(well_csv):
    """ 
    imports uploaded well data and displays first couple of rows 
    """
    # upload well csv
    well_df = import_aquarius_data(well_csv)
    st.write("Data Preview:")
    st.write(well_df.head())

    # select correct x/y cols
    well_x_col = st.selectbox('Select Well Time Column', well_df.columns)
    well_y_col = st.selectbox('Select Well Water Level Column', well_df.columns)
    return well_df, well_x_col, well_y_col

def get_input_params(): 
    """ 
    gets user to enter input params for the analysis
    """
    st.markdown("### Input Parameters")
    stn_name = st.text_input('Obs Well Station Name', value='OW')
    shore_stn = st.text_input('Tidal/Hydrometric Station Name')
    correction_factor = st.number_input('Correction Factor', format="%.2f", step=0.1)
    window_length = st.number_input('Window Length (hours)', min_value=1, step=1, value=100)     
    x = st.number_input('Well to Shore Distance (m)', format="%.1f", step=1.0)                       
    t = st.number_input('Tidal Period (s)', format="%.1f", value=12.67*3600) 
    min_dist = st.slider('Minimum Distance (index)', min_value=0, max_value=24, step=1, value=8)         
    z_thresh = st.slider('Z Threshold', min_value=0.0, max_value=10.0, value=10.0, step=0.1)      
    initial_guess = get_initial_guess_params()
    return stn_name, shore_stn, correction_factor, window_length, x, t, min_dist, z_thresh, initial_guess        

def get_initial_guess_params(): 
    """ 
    gets the 4 initial params for sine initial guess 
    """
    st.markdown("### Guess of Sine Curve Parameters")
    computer_guess = st.toggle('Use algorithm to guess initial params', value=False)

    if not computer_guess: 
        col1, col2, col3, col4 = st.columns(4)              # Create four columns
        with col1:
            amp = st.number_input('Amplitude', value=0.030, step=0.01)
        with col2:
            tp = st.number_input('Time Period', value=23.0, step=1.0)
        with col3:
            ps = st.number_input('Phase Shift', value=4.5, step=0.1)
        with col4:
            ys = st.number_input('Y-shift', value=0.0, step=0.1)
        initial_guess = [amp, tp, ps, ys]
        return initial_guess
    else: 
        return None

def display_start_date(start_end_date: Tuple): 
    """ 
    displays the start date
    """
    start_date = start_end_date[0]
    fstart_date = pd.Timestamp(start_date).to_pydatetime().strftime('%Y-%m-%d')
    st.write(f'Start Date: {fstart_date}')

def display_df_as_table(df: pd.DataFrame, col_to_space_ratio: Optional[Tuple[int, int]]): 
    """ 
    displays df as table in streamlit dashboard based on the col_to_space_ratio 
    """
    if col_to_space_ratio is not None: 
        r1, r2 = col_to_space_ratio
        col1, col2 = st.columns([r1, r2])
        with col1: 
            st.table(df)
    else: 
        st.table(df)
        
#%%
#### config page 
st.set_page_config(
    page_title="Aquifer Diffusivity Calculator",
    # page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")
alt.themes.enable("dark")

st.title('Aquifer Diffusivity Calculator')

#### Get input params
with st.sidebar:
    # Title of the dashboard
    st.markdown("### Upload Water Level Data")

    # upload shore data
    shore_csvs = st.file_uploader("Upload Shore CSV files", type="csv", accept_multiple_files=True)
    if shore_csvs: 
        shore_df, shore_x_col, shore_y_col = get_shore_data(shore_csvs)

    # upload well data
    well_csv = st.file_uploader("Upload Well CSV file", type="csv", accept_multiple_files=False)
    if well_csv: 
        well_df, well_x_col, well_y_col = get_well_data(well_csv)

    # input analysis params
    if shore_csvs and well_csv: 
        stn_name, shore_stn, correction_factor, window_length, x, t, min_dist, z_thresh, initial_guess = get_input_params()

    

    run_button = st.button('Run Main Function')         # button to run analysis

#### Results
if run_button: 
    try: 
        # Call the main function
        if initial_guess is None: 
            fit_fn = composite_sine
        params, plot, summary_df, sa_df, tl_df, start_end_date, final_graph, plot = main(stn_name=stn_name,
                                                                                        shore_stn=shore_stn,
                                                                                        shore_df=shore_df, 
                                                                                        shore_x_col=shore_x_col,
                                                                                        shore_y_col=shore_y_col,
                                                                                        correction_factor=correction_factor,
                                                                                        well_df=well_df,
                                                                                        well_x_col=well_x_col,
                                                                                        well_y_col=well_y_col,
                                                                                        window_length=window_length,
                                                                                        initial_guess=initial_guess,
                                                                                        fit_fn=fit_fn,
                                                                                        x=x,
                                                                                        t=t,
                                                                                        min_dist=min_dist,
                                                                                        z_thresh=z_thresh )
        # Display Result Tables
        display_start_date(start_end_date)
        display_df_as_table(summary_df, (2, 1))
        display_df_as_table(sa_df, (30, 1))
        display_df_as_table(tl_df, (30, 1))

        # display plot
        st.markdown("### Final Plot") 
        st.pyplot(plot)
        st.pyplot(final_graph)    
    except Exception as e: 
        st.exception(e)






